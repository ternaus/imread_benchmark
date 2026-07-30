from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import sys
import sysconfig
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ENVIRONMENT_SCHEMA_VERSION = "2.0"
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")


@dataclass(frozen=True, slots=True)
class EnvironmentDescriptor:
    environment_id: str
    dependency_group: str
    lock_sha256: str
    project_sha256: str
    runner_revision: str
    python: dict[str, str]
    platform_tags: tuple[str, ...]
    distributions: tuple[tuple[str, str], ...]
    native_backends: dict[str, str]

    @classmethod
    def build(  # noqa: PLR0913 - every provenance input is an independent identity dimension
        cls,
        *,
        dependency_group: str,
        lock_sha256: str,
        project_sha256: str,
        runner_revision: str,
        python: dict[str, str],
        platform_tags: tuple[str, ...],
        distributions: tuple[tuple[str, str], ...],
        native_backends: dict[str, str],
    ) -> EnvironmentDescriptor:
        if not dependency_group:
            raise ValueError("dependency_group must not be empty")
        _validate_digest(lock_sha256, "lock_sha256")
        _validate_digest(project_sha256, "project_sha256")
        if _REVISION.fullmatch(runner_revision) is None:
            raise ValueError("runner_revision must be a 40- or 64-character hexadecimal revision")
        normalized_python = _string_mapping(python, "python")
        if set(normalized_python) != {"abi", "implementation", "version"}:
            raise ValueError("python descriptor must contain abi, implementation, and version")
        if not platform_tags or any(not tag for tag in platform_tags):
            raise ValueError("platform_tags must not be empty")
        normalized_distributions = tuple(sorted((_normalize_name(name), version) for name, version in distributions))
        if not normalized_distributions or any(not version for _, version in normalized_distributions):
            raise ValueError("distributions must contain installed package names and versions")
        if len({name for name, _ in normalized_distributions}) != len(normalized_distributions):
            raise ValueError("distributions contain duplicate package names")
        normalized_backends = _string_mapping(native_backends, "native_backends", allow_empty=True)
        normalized_tags = tuple(sorted(set(platform_tags)))
        identity = {
            "dependency_group": dependency_group,
            "distributions": _distribution_documents(normalized_distributions),
            "lock_sha256": lock_sha256,
            "native_backends": normalized_backends,
            "platform_tags": list(normalized_tags),
            "project_sha256": project_sha256,
            "python": normalized_python,
            "runner_revision": runner_revision,
            "schema_version": ENVIRONMENT_SCHEMA_VERSION,
        }
        return cls(
            environment_id=_digest(identity),
            dependency_group=dependency_group,
            lock_sha256=lock_sha256,
            project_sha256=project_sha256,
            runner_revision=runner_revision,
            python=normalized_python,
            platform_tags=normalized_tags,
            distributions=normalized_distributions,
            native_backends=normalized_backends,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "dependency_group": self.dependency_group,
            "distributions": _distribution_documents(self.distributions),
            "environment_id": self.environment_id,
            "lock_sha256": self.lock_sha256,
            "native_backends": self.native_backends,
            "platform_tags": list(self.platform_tags),
            "project_sha256": self.project_sha256,
            "python": self.python,
            "runner_revision": self.runner_revision,
            "schema_version": ENVIRONMENT_SCHEMA_VERSION,
        }


def capture_current_environment(
    *,
    lock_path: str | Path,
    project_path: str | Path,
    dependency_group: str,
    runner_revision: str,
    native_backends: dict[str, str] | None = None,
) -> EnvironmentDescriptor:
    distributions: list[tuple[str, str]] = []
    editable: list[str] = []
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata["Name"]
        if not name:
            continue
        distributions.append((name, distribution.version))
        direct_url_text = distribution.read_text("direct_url.json")
        if direct_url_text:
            try:
                direct_url = json.loads(direct_url_text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid direct_url.json for installed distribution {name}") from exc
            directory_info = direct_url.get("dir_info") if isinstance(direct_url, dict) else None
            if isinstance(directory_info, dict) and directory_info.get("editable") is True:
                editable.append(name)
    if editable:
        raise ValueError(f"editable distributions are forbidden: {', '.join(sorted(editable))}")
    abi = sysconfig.get_config_var("SOABI") or sys.implementation.cache_tag or "unknown"
    return EnvironmentDescriptor.build(
        dependency_group=dependency_group,
        lock_sha256=_sha256_file(Path(lock_path)),
        project_sha256=_sha256_file(Path(project_path)),
        runner_revision=runner_revision,
        python={
            "abi": str(abi),
            "implementation": platform.python_implementation().lower(),
            "version": platform.python_version(),
        },
        platform_tags=(sysconfig.get_platform(),),
        distributions=tuple(distributions),
        native_backends=native_backends or {},
    )


def write_environment_descriptor(path: str | Path, descriptor: EnvironmentDescriptor) -> Path:
    return _write_immutable_json(Path(path), descriptor.to_dict())


def load_environment_descriptor(path: str | Path) -> EnvironmentDescriptor:
    document = _read_object(Path(path), "environment descriptor")
    if document.get("schema_version") != ENVIRONMENT_SCHEMA_VERSION:
        raise ValueError("unsupported environment descriptor schema")
    python = _required_object(document, "python")
    backends = _required_object(document, "native_backends")
    tags = _string_tuple(document, "platform_tags")
    raw_distributions = document.get("distributions")
    if not isinstance(raw_distributions, list):
        raise TypeError("environment distributions must be a list")
    distributions: list[tuple[str, str]] = []
    for row in raw_distributions:
        if not isinstance(row, dict):
            raise TypeError("environment distribution must be an object")
        distributions.append((_required_string(row, "name"), _required_string(row, "version")))
    descriptor = EnvironmentDescriptor.build(
        dependency_group=_required_string(document, "dependency_group"),
        lock_sha256=_required_string(document, "lock_sha256"),
        project_sha256=_required_string(document, "project_sha256"),
        runner_revision=_required_string(document, "runner_revision"),
        python={key: _required_string(python, key) for key in ("abi", "implementation", "version")},
        platform_tags=tags,
        distributions=tuple(distributions),
        native_backends={key: _required_string(backends, key) for key in backends},
    )
    if document.get("environment_id") != descriptor.environment_id:
        raise ValueError("environment_id does not match descriptor content")
    return descriptor


def _distribution_documents(distributions: tuple[tuple[str, str], ...]) -> list[dict[str, str]]:
    return [{"name": name, "version": version} for name, version in distributions]


def _normalize_name(value: str) -> str:
    if not value:
        raise ValueError("distribution name must not be empty")
    return re.sub(r"[-_.]+", "-", value).lower()


def _string_mapping(value: dict[str, str], field: str, *, allow_empty: bool = False) -> dict[str, str]:
    invalid_entry = any(not isinstance(key, str) or not key or not val for key, val in value.items())
    if (not allow_empty and not value) or invalid_entry:
        raise ValueError(f"{field} must contain non-empty string keys and values")
    return dict(sorted(value.items()))


def _validate_digest(value: str, field: str) -> None:
    if _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")


def _digest(payload: object) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ValueError(f"cannot hash environment input {path}: {exc}") from exc
    return digest.hexdigest()


def _write_immutable_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() != content:
            raise ValueError(f"immutable descriptor already exists with different content: {path}")
        return path
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False) as file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
            temporary = Path(file.name)
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_text() != content:
                raise ValueError(f"concurrent descriptor writer produced different content: {path}") from None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object")
    return value


def _required_object(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"field {key!r} must be an object")
    return value


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"field {key!r} must be a non-empty string")
    return value


def _string_tuple(payload: dict[str, Any], key: str) -> tuple[str, ...]:
    value = payload.get(key)
    if not isinstance(value, list) or not value or not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"field {key!r} must be a non-empty string list")
    return tuple(value)
