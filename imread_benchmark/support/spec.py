from __future__ import annotations

import hashlib
import json
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SUPPORT_AUDIT_SPEC_VERSION = "2.0"
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")


class SupportAuditSpecError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class SupportAuditIdentity:
    decoder_id: str
    requested_threads: int | None
    package_id: str
    workload_id: str
    manifest_id: str
    selection_id: str
    item_ids: tuple[str, ...]
    process_context: str
    multiprocessing_start_method: str | None
    environment_id: str
    platform_id: str
    runner_revision: str

    def __post_init__(self) -> None:
        if not self.decoder_id or not self.workload_id:
            raise SupportAuditSpecError("decoder_id and workload_id must not be empty")
        for field_name in ("package_id", "manifest_id", "selection_id", "environment_id", "platform_id"):
            if _DIGEST.fullmatch(getattr(self, field_name)) is None:
                raise SupportAuditSpecError(f"{field_name} must be a lowercase SHA-256 digest")
        if self.requested_threads is not None and (
            isinstance(self.requested_threads, bool)
            or not isinstance(self.requested_threads, int)
            or self.requested_threads <= 0
        ):
            raise SupportAuditSpecError("requested_threads must be a positive integer or default")
        if (
            not self.item_ids
            or len(set(self.item_ids)) != len(self.item_ids)
            or any(not item for item in self.item_ids)
        ):
            raise SupportAuditSpecError("item_ids must be non-empty and unique")
        if self.process_context not in {"main-process", "dataloader"}:
            raise SupportAuditSpecError("process_context must be main-process or dataloader")
        if self.process_context == "main-process" and self.multiprocessing_start_method is not None:
            raise SupportAuditSpecError("main-process support audit does not accept multiprocessing_start_method")
        if self.process_context == "dataloader" and self.multiprocessing_start_method not in {
            "fork",
            "forkserver",
            "spawn",
        }:
            raise SupportAuditSpecError("dataloader support audit requires an explicit multiprocessing_start_method")
        if _REVISION.fullmatch(self.runner_revision) is None:
            raise SupportAuditSpecError("runner_revision must be a hexadecimal source revision")


@dataclass(frozen=True, slots=True)
class SupportAuditSpec:
    audit_key: str
    identity: SupportAuditIdentity
    package_descriptor: Path
    schema_version: str = SUPPORT_AUDIT_SPEC_VERSION

    @classmethod
    def build(
        cls,
        *,
        identity: SupportAuditIdentity,
        package_descriptor: str | Path,
    ) -> SupportAuditSpec:
        return cls(
            audit_key=_digest(identity),
            identity=identity,
            package_descriptor=Path(package_descriptor).resolve(),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self.identity),
            "audit_key": self.audit_key,
            "item_ids": list(self.identity.item_ids),
            "package_descriptor": str(self.package_descriptor),
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, document: dict[str, Any]) -> SupportAuditSpec:
        if document.get("schema_version") != SUPPORT_AUDIT_SPEC_VERSION:
            raise SupportAuditSpecError("unsupported support audit spec schema")
        item_ids = document.get("item_ids")
        if not isinstance(item_ids, list) or not all(isinstance(item, str) for item in item_ids):
            raise SupportAuditSpecError("item_ids must be a string list")
        identity = SupportAuditIdentity(
            decoder_id=_required_string(document, "decoder_id"),
            requested_threads=_optional_positive_int(document, "requested_threads"),
            package_id=_required_string(document, "package_id"),
            workload_id=_required_string(document, "workload_id"),
            manifest_id=_required_string(document, "manifest_id"),
            selection_id=_required_string(document, "selection_id"),
            item_ids=tuple(item_ids),
            process_context=_required_string(document, "process_context"),
            multiprocessing_start_method=_optional_string(document, "multiprocessing_start_method"),
            environment_id=_required_string(document, "environment_id"),
            platform_id=_required_string(document, "platform_id"),
            runner_revision=_required_string(document, "runner_revision"),
        )
        spec = cls.build(
            identity=identity,
            package_descriptor=_required_string(document, "package_descriptor"),
        )
        if document.get("audit_key") != spec.audit_key:
            raise SupportAuditSpecError("audit_key does not match support audit identity")
        return spec


def write_support_audit_spec(path: str | Path, spec: SupportAuditSpec) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(spec.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if destination.exists():
        if destination.read_text() != content:
            raise SupportAuditSpecError(f"existing audit spec has different content: {destination}")
        return destination
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=destination.parent, prefix=".audit-spec.", delete=False) as file:
            file.write(content)
            temporary = Path(file.name)
        temporary.rename(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def load_support_audit_spec(path: str | Path) -> SupportAuditSpec:
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SupportAuditSpecError(f"cannot read support audit spec: {exc}") from exc
    if not isinstance(value, dict):
        raise SupportAuditSpecError("support audit spec must be a JSON object")
    return SupportAuditSpec.from_dict(value)


def _digest(identity: SupportAuditIdentity) -> str:
    payload = {**asdict(identity), "schema_version": SUPPORT_AUDIT_SPEC_VERSION}
    canonical = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise SupportAuditSpecError(f"field {key!r} must be a non-empty string")
    return value


def _optional_positive_int(payload: dict[str, Any], key: str) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SupportAuditSpecError(f"field {key!r} must be a positive integer or null")
    return value


def _optional_string(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise SupportAuditSpecError(f"field {key!r} must be a non-empty string or null")
    return value
