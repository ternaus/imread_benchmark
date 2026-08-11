from __future__ import annotations

import hashlib
import json
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from imread_benchmark.plans import RunConfiguration

RUN_SPEC_SCHEMA_VERSION = "2.0"
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class RunSpecError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RunIdentity:
    plan_id: str
    platform_id: str
    environment_id: str
    runner_revision: str
    workload_id: str
    support_set_id: str
    support_item_ids: tuple[str, ...]
    configuration: RunConfiguration
    repetition: int
    block_position: int

    def __post_init__(self) -> None:
        for field_name in ("plan_id", "platform_id", "environment_id", "support_set_id"):
            _validate_digest(getattr(self, field_name), field=field_name)
        if _REVISION.fullmatch(self.runner_revision) is None:
            raise RunSpecError("runner_revision must be a 40- or 64-character lowercase hexadecimal revision")
        if _SAFE_ID.fullmatch(self.workload_id) is None or self.workload_id in {".", ".."}:
            raise RunSpecError("workload_id must be a safe identifier")
        if not self.support_item_ids or len(set(self.support_item_ids)) != len(self.support_item_ids):
            raise RunSpecError("support_item_ids must be non-empty and unique")
        if any(not item_id for item_id in self.support_item_ids):
            raise RunSpecError("support_item_ids must not contain empty identifiers")
        if self.repetition < 0 or self.block_position < 0:
            raise RunSpecError("repetition and block_position must be non-negative")

    def to_dict(self) -> dict[str, object]:
        return {
            "block_position": self.block_position,
            "configuration": asdict(self.configuration),
            "environment_id": self.environment_id,
            "package_id": self.configuration.package_id,
            "plan_id": self.plan_id,
            "platform_id": self.platform_id,
            "repetition": self.repetition,
            "runner_revision": self.runner_revision,
            "schema_version": RUN_SPEC_SCHEMA_VERSION,
            "support_item_ids": list(self.support_item_ids),
            "support_set_id": self.support_set_id,
            "workload_manifest_id": self.configuration.manifest_id,
            "workload_id": self.workload_id,
        }


@dataclass(frozen=True, slots=True)
class RunSpec:
    run_key: str
    identity: RunIdentity
    package_descriptor: Path
    support_set_path: Path
    environment_descriptor: Path
    platform_descriptor: Path
    schema_version: str = RUN_SPEC_SCHEMA_VERSION

    @classmethod
    def build(
        cls,
        *,
        identity: RunIdentity,
        package_descriptor: str | Path,
        support_set_path: str | Path,
        environment_descriptor: str | Path,
        platform_descriptor: str | Path,
    ) -> RunSpec:
        return cls(
            run_key=compute_run_key(identity),
            identity=identity,
            package_descriptor=Path(package_descriptor).resolve(),
            support_set_path=Path(support_set_path).resolve(),
            environment_descriptor=Path(environment_descriptor).resolve(),
            platform_descriptor=Path(platform_descriptor).resolve(),
        )

    @classmethod
    def from_dict(cls, document: dict[str, Any]) -> RunSpec:
        if document.get("schema_version") != RUN_SPEC_SCHEMA_VERSION:
            raise RunSpecError(f"unsupported run spec schema: {document.get('schema_version')!r}")
        configuration_document = _required_object(document, "configuration")
        try:
            configuration = RunConfiguration(**configuration_document)
        except (TypeError, ValueError) as exc:
            raise RunSpecError(f"invalid run configuration: {exc}") from exc
        raw_items = document.get("support_item_ids")
        if not isinstance(raw_items, list) or not all(isinstance(item, str) for item in raw_items):
            raise RunSpecError("support_item_ids must be a string list")
        identity = RunIdentity(
            plan_id=_required_string(document, "plan_id"),
            platform_id=_required_string(document, "platform_id"),
            environment_id=_required_string(document, "environment_id"),
            runner_revision=_required_string(document, "runner_revision"),
            workload_id=_required_string(document, "workload_id"),
            support_set_id=_required_string(document, "support_set_id"),
            support_item_ids=tuple(raw_items),
            configuration=configuration,
            repetition=_required_int(document, "repetition"),
            block_position=_required_int(document, "block_position"),
        )
        spec = cls.build(
            identity=identity,
            package_descriptor=_required_string(document, "package_descriptor"),
            support_set_path=_required_string(document, "support_set_path"),
            environment_descriptor=_required_string(document, "environment_descriptor"),
            platform_descriptor=_required_string(document, "platform_descriptor"),
        )
        if document.get("run_key") != spec.run_key:
            raise RunSpecError("run_key does not match run spec identity")
        return spec

    def __post_init__(self) -> None:
        _validate_digest(self.run_key, field="run_key")
        if self.schema_version != RUN_SPEC_SCHEMA_VERSION:
            raise RunSpecError(f"unsupported run spec schema: {self.schema_version!r}")

    def to_dict(self) -> dict[str, object]:
        return {
            **self.identity.to_dict(),
            "package_descriptor": str(self.package_descriptor),
            "environment_descriptor": str(self.environment_descriptor),
            "platform_descriptor": str(self.platform_descriptor),
            "run_key": self.run_key,
            "schema_version": self.schema_version,
            "support_set_path": str(self.support_set_path),
        }


def write_run_spec(path: str | Path, spec: RunSpec) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(spec.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if destination.exists():
        if destination.read_text() != content:
            raise RunSpecError(f"existing run spec has different content: {destination}")
        return destination
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            delete=False,
        ) as file:
            file.write(content)
            temporary = Path(file.name)
        temporary.rename(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def load_run_spec(path: str | Path) -> RunSpec:
    source = Path(path)
    try:
        value = json.loads(source.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RunSpecError(f"cannot read run spec {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise RunSpecError("run spec must be a JSON object")
    return RunSpec.from_dict(value)


def compute_run_key(identity: RunIdentity) -> str:
    return _digest(identity.to_dict())


def _required_object(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise RunSpecError(f"field {key!r} must be an object")
    return value


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise RunSpecError(f"field {key!r} must be a non-empty string")
    return value


def _required_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise RunSpecError(f"field {key!r} must be an integer")
    return value


def _validate_digest(value: str, *, field: str) -> None:
    if _DIGEST.fullmatch(value) is None:
        raise RunSpecError(f"{field} must be a lowercase SHA-256 digest")


def _digest(payload: object) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()
