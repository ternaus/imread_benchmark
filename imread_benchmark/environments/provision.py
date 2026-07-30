from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from imread_benchmark.environments.descriptor import (
    ENVIRONMENT_SCHEMA_VERSION,
    EnvironmentDescriptor,
    load_environment_descriptor,
    write_environment_descriptor,
)

_REVISION = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_PROBE_SCRIPT = """
import importlib.metadata
import json
import platform
import sys
import sysconfig

distributions = []
editable = []
for distribution in importlib.metadata.distributions():
    name = distribution.metadata["Name"]
    distributions.append([name, distribution.version])
    direct_url_text = distribution.read_text("direct_url.json")
    if direct_url_text:
        direct_url = json.loads(direct_url_text)
        if direct_url.get("dir_info", {}).get("editable") is True:
            editable.append(name)
print(json.dumps({
    "distributions": distributions,
    "editable": editable,
    "platform_tags": [sysconfig.get_platform()],
    "python": {
        "abi": str(sysconfig.get_config_var("SOABI") or sys.implementation.cache_tag or "unknown"),
        "implementation": platform.python_implementation().lower(),
        "version": platform.python_version(),
    },
}, sort_keys=True))
"""


@dataclass(frozen=True, slots=True)
class EnvironmentRequest:
    project_root: Path
    cache_root: Path
    dependency_group: str
    runner_revision: str
    python_executable: Path
    uv_command: tuple[str, ...] = ("uv",)
    lock_sha256: str = field(init=False)
    project_sha256: str = field(init=False)
    python_identity: dict[str, str] = field(init=False)
    platform_tags: tuple[str, ...] = field(init=False)
    environment_key: str = field(init=False)

    def __post_init__(self) -> None:
        project_root = self.project_root.resolve()
        cache_root = self.cache_root.resolve()
        python_executable = self.python_executable.resolve()
        if not self.dependency_group:
            raise ValueError("dependency_group must not be empty")
        if _REVISION.fullmatch(self.runner_revision) is None:
            raise ValueError("runner_revision must be a 40- or 64-character hexadecimal revision")
        if not self.uv_command:
            raise ValueError("uv_command must not be empty")
        lock_sha256 = _sha256_file(project_root / "uv.lock")
        project_sha256 = _sha256_file(project_root / "pyproject.toml")
        python_identity, platform_tags = _probe_python_identity(python_executable)
        key_identity = {
            "dependency_group": self.dependency_group,
            "lock_sha256": lock_sha256,
            "platform_tags": platform_tags,
            "project_sha256": project_sha256,
            "python": python_identity,
            "runner_revision": self.runner_revision,
            "schema_version": ENVIRONMENT_SCHEMA_VERSION,
        }
        object.__setattr__(self, "project_root", project_root)
        object.__setattr__(self, "cache_root", cache_root)
        object.__setattr__(self, "python_executable", python_executable)
        object.__setattr__(self, "lock_sha256", lock_sha256)
        object.__setattr__(self, "project_sha256", project_sha256)
        object.__setattr__(self, "python_identity", python_identity)
        object.__setattr__(self, "platform_tags", platform_tags)
        object.__setattr__(self, "environment_key", _digest(key_identity))


@dataclass(frozen=True, slots=True)
class ProvisionedEnvironment:
    root: Path
    python_executable: Path
    descriptor_path: Path
    environment_key: str
    environment_id: str
    cache_hit: bool = field(compare=False)


EnvironmentProbe = Callable[[Path, str], EnvironmentDescriptor]


def provision_environment(
    request: EnvironmentRequest,
    *,
    probe: EnvironmentProbe | None = None,
) -> ProvisionedEnvironment:
    destination = request.cache_root / request.environment_key
    if destination.exists():
        return _load_ready_environment(destination, request, cache_hit=True)
    request.cache_root.mkdir(parents=True, exist_ok=True)
    staging_parent = Path(tempfile.mkdtemp(prefix=f".{request.environment_key}.", dir=request.cache_root))
    staging = staging_parent / "environment"
    try:
        _run_frozen_sync(request, staging)
        python_executable = _venv_python(staging)
        if not python_executable.is_file():
            raise ValueError("uv sync did not create the expected environment Python executable")
        descriptor = (
            probe(python_executable, request.environment_key)
            if probe is not None
            else _probe_installed_environment(python_executable, request)
        )
        _validate_descriptor_for_request(descriptor, request)
        write_environment_descriptor(staging / "environment.json", descriptor)
        _write_ready_marker(staging, request, descriptor)
        try:
            staging.rename(destination)
        except FileExistsError:
            return _load_ready_environment(destination, request, cache_hit=True)
        return _load_ready_environment(destination, request, cache_hit=False)
    finally:
        shutil.rmtree(staging_parent, ignore_errors=True)


def load_provisioned_environment(
    root: str | Path,
    request: EnvironmentRequest,
    *,
    cache_hit: bool,
) -> ProvisionedEnvironment:
    return _load_ready_environment(Path(root).resolve(), request, cache_hit=cache_hit)


def _run_frozen_sync(request: EnvironmentRequest, destination: Path) -> None:
    environment = {
        **os.environ,
        "UV_LINK_MODE": "copy",
        "UV_PROJECT_ENVIRONMENT": str(destination),
    }
    command = (
        *request.uv_command,
        "sync",
        "--project",
        str(request.project_root),
        "--python",
        str(request.python_executable),
        "--no-group",
        "dev",
        "--frozen",
        "--no-editable",
        "--extra",
        request.dependency_group,
    )
    subprocess.run(command, check=True, env=environment)  # noqa: S603 - explicit provisioner command


def _probe_installed_environment(
    python_executable: Path,
    request: EnvironmentRequest,
) -> EnvironmentDescriptor:
    process = subprocess.run(  # noqa: S603 - probes the newly-created immutable Python environment
        (str(python_executable), "-c", _PROBE_SCRIPT),
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        document = json.loads(process.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError("installed environment probe did not return JSON") from exc
    if not isinstance(document, dict):
        raise TypeError("installed environment probe must return an object")
    editable = document.get("editable")
    if not isinstance(editable, list) or not all(isinstance(name, str) for name in editable):
        raise TypeError("installed environment probe returned invalid editable metadata")
    if editable:
        raise ValueError(f"editable distributions are forbidden: {', '.join(sorted(editable))}")
    return EnvironmentDescriptor.build(
        dependency_group=request.dependency_group,
        lock_sha256=request.lock_sha256,
        project_sha256=request.project_sha256,
        runner_revision=request.runner_revision,
        python=_probe_string_mapping(document, "python"),
        platform_tags=_probe_string_tuple(document, "platform_tags"),
        distributions=_probe_distributions(document),
        native_backends={},
    )


def _validate_descriptor_for_request(descriptor: EnvironmentDescriptor, request: EnvironmentRequest) -> None:
    expected = (
        descriptor.dependency_group == request.dependency_group
        and descriptor.lock_sha256 == request.lock_sha256
        and descriptor.project_sha256 == request.project_sha256
        and descriptor.runner_revision == request.runner_revision
        and descriptor.python == request.python_identity
        and descriptor.platform_tags == request.platform_tags
    )
    if not expected:
        raise ValueError("installed environment descriptor does not match provision request")


def _write_ready_marker(
    root: Path,
    request: EnvironmentRequest,
    descriptor: EnvironmentDescriptor,
) -> None:
    marker = {
        "environment_id": descriptor.environment_id,
        "environment_key": request.environment_key,
        "schema_version": ENVIRONMENT_SCHEMA_VERSION,
        "status": "ready",
    }
    (root / ".READY.json").write_text(json.dumps(marker, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _load_ready_environment(
    root: Path,
    request: EnvironmentRequest,
    *,
    cache_hit: bool,
) -> ProvisionedEnvironment:
    try:
        marker = json.loads((root / ".READY.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"environment cache entry is incomplete: {root}") from exc
    if not isinstance(marker, dict):
        raise TypeError("environment READY marker must be an object")
    descriptor_path = root / "environment.json"
    descriptor = load_environment_descriptor(descriptor_path)
    if (
        marker.get("schema_version") != ENVIRONMENT_SCHEMA_VERSION
        or marker.get("status") != "ready"
        or marker.get("environment_key") != request.environment_key
        or marker.get("environment_id") != descriptor.environment_id
    ):
        raise ValueError("environment READY marker does not match the cached descriptor")
    _validate_descriptor_for_request(descriptor, request)
    python_executable = _venv_python(root)
    if not python_executable.is_file():
        raise ValueError("environment cache entry has no Python executable")
    return ProvisionedEnvironment(
        root=root,
        python_executable=python_executable,
        descriptor_path=descriptor_path,
        environment_key=request.environment_key,
        environment_id=descriptor.environment_id,
        cache_hit=cache_hit,
    )


def _probe_python_identity(python_executable: Path) -> tuple[dict[str, str], tuple[str, ...]]:
    if python_executable == Path(sys.executable).resolve():
        abi = sysconfig.get_config_var("SOABI") or sys.implementation.cache_tag or "unknown"
        return (
            {
                "abi": str(abi),
                "implementation": platform.python_implementation().lower(),
                "version": platform.python_version(),
            },
            (sysconfig.get_platform(),),
        )
    process = subprocess.run(  # noqa: S603 - explicit Python interpreter selected by the caller
        (str(python_executable), "-c", _PROBE_SCRIPT),
        check=True,
        capture_output=True,
        text=True,
    )
    document = json.loads(process.stdout)
    if not isinstance(document, dict):
        raise TypeError("Python identity probe must return an object")
    return _probe_string_mapping(document, "python"), _probe_string_tuple(document, "platform_tags")


def _probe_distributions(document: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    value = document.get("distributions")
    if not isinstance(value, list):
        raise TypeError("environment probe distributions must be a list")
    result: list[tuple[str, str]] = []
    for row in value:
        if not isinstance(row, list) or len(row) != 2 or not all(isinstance(item, str) and item for item in row):
            raise TypeError("environment probe distribution must be a name/version pair")
        result.append((row[0], row[1]))
    return tuple(result)


def _probe_string_mapping(document: dict[str, Any], key: str) -> dict[str, str]:
    value = document.get(key)
    if not isinstance(value, dict) or any(
        not isinstance(item_key, str) or not isinstance(item_value, str) or not item_value
        for item_key, item_value in value.items()
    ):
        raise TypeError(f"environment probe field {key!r} must be a string mapping")
    return value


def _probe_string_tuple(document: dict[str, Any], key: str) -> tuple[str, ...]:
    value = document.get(key)
    if not isinstance(value, list) or not value or not all(isinstance(item, str) and item for item in value):
        raise TypeError(f"environment probe field {key!r} must be a non-empty string list")
    return tuple(value)


def _venv_python(root: Path) -> Path:
    bindir = "Scripts" if os.name == "nt" else "bin"
    executable = "python.exe" if os.name == "nt" else "python"
    return root / bindir / executable


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ValueError(f"cannot hash environment input {path}: {exc}") from exc
    return digest.hexdigest()


def _digest(payload: object) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()
