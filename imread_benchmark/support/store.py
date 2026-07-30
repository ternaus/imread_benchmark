from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from imread_benchmark.support.audit import (
    SUPPORT_SCHEMA_VERSION,
    SupportAudit,
    SupportFailure,
    SupportSet,
    SupportSetIdentity,
    _digest,
)


def write_support_audit(root: str | Path, audit: SupportAudit) -> Path:
    path = Path(root) / "audits" / f"{audit.audit_id}.json"
    _write_content_addressed(path, audit.to_dict())
    if load_support_audit(path) != audit:
        raise ValueError("persisted support audit differs from input")
    return path


def write_support_set(root: str | Path, support_set: SupportSet) -> Path:
    path = Path(root) / "sets" / f"{support_set.support_set_id}.json"
    _write_content_addressed(path, support_set.to_dict())
    if load_support_set(path) != support_set:
        raise ValueError("persisted support set differs from input")
    return path


def commit_support_audit(root: str | Path, audit_key: str, audit: SupportAudit) -> Path:
    root_path = Path(root)
    audit_path = write_support_audit(root_path, audit)
    marker = root_path / "keys" / f"{audit_key}.json"
    _write_content_addressed(
        marker,
        {
            "audit_id": audit.audit_id,
            "audit_key": audit_key,
            "schema_version": SUPPORT_SCHEMA_VERSION,
            "status": "committed",
        },
    )
    if load_committed_support_audit(root_path, audit_key) != audit:
        raise ValueError("committed support audit differs from input")
    return audit_path


def load_committed_support_audit(root: str | Path, audit_key: str) -> SupportAudit:
    root_path = Path(root)
    marker_path = root_path / "keys" / f"{audit_key}.json"
    marker = _read_object(marker_path)
    if (
        marker.get("schema_version") != SUPPORT_SCHEMA_VERSION
        or marker.get("status") != "committed"
        or marker.get("audit_key") != audit_key
    ):
        raise ValueError("invalid support audit commit marker")
    audit_id = _required_string(marker, "audit_id")
    return load_support_audit(root_path / "audits" / f"{audit_id}.json")


def load_support_audit(path: str | Path) -> SupportAudit:
    source = Path(path)
    document = _read_object(source)
    if document.get("schema_version") != SUPPORT_SCHEMA_VERSION:
        raise ValueError("unsupported support audit schema")
    audit_id = _required_string(document, "audit_id")
    identity = {key: value for key, value in document.items() if key != "audit_id"}
    if _digest(identity) != audit_id or source.stem != audit_id:
        raise ValueError("support audit audit_id does not match its content or filename")
    raw_failures = document.get("failures")
    if not isinstance(raw_failures, list):
        raise TypeError("support audit failures must be a list")
    failures = tuple(_failure(value) for value in raw_failures)
    return SupportAudit(
        audit_id=audit_id,
        decoder_id=_required_string(document, "decoder_id"),
        requested_threads=_optional_positive_int(document, "requested_threads"),
        manifest_id=_required_string(document, "manifest_id"),
        selection_id=_required_string(document, "selection_id"),
        process_context=_required_string(document, "process_context"),
        multiprocessing_start_method=_optional_string(document, "multiprocessing_start_method"),
        output_contract=_required_string(document, "output_contract"),
        environment_id=_required_string(document, "environment_id"),
        platform_id=_required_string(document, "platform_id"),
        successful_item_ids=_string_tuple(document, "successful_item_ids", allow_empty=True),
        failures=failures,
    )


def load_support_set(path: str | Path) -> SupportSet:
    source = Path(path)
    document = _read_object(source)
    if document.get("schema_version") != SUPPORT_SCHEMA_VERSION:
        raise ValueError("unsupported support set schema")
    support_set_id = _required_string(document, "support_set_id")
    policy = _required_string(document, "policy")
    if policy not in {"common", "operational"}:
        raise ValueError(f"unsupported support set policy: {policy!r}")
    audit_ids = _string_tuple(document, "audit_ids")
    item_ids = _string_tuple(document, "item_ids")
    manifest_id = _required_string(document, "manifest_id")
    selection_id = _required_string(document, "selection_id")
    process_context = _required_string(document, "process_context")
    multiprocessing_start_method = _optional_string(document, "multiprocessing_start_method")
    identity = SupportSetIdentity(
        policy=policy,
        manifest_id=manifest_id,
        selection_id=selection_id,
        process_context=process_context,
        multiprocessing_start_method=multiprocessing_start_method,
        audit_ids=audit_ids,
        item_ids=item_ids,
    ).to_dict()
    if _digest(identity) != support_set_id or source.stem != support_set_id:
        raise ValueError("support set support_set_id does not match its content or filename")
    return SupportSet(
        support_set_id=support_set_id,
        policy=policy,
        manifest_id=manifest_id,
        selection_id=selection_id,
        process_context=process_context,
        multiprocessing_start_method=multiprocessing_start_method,
        audit_ids=audit_ids,
        item_ids=item_ids,
    )


def _write_content_addressed(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() != content:
            raise ValueError(f"content-addressed artifact differs from existing file: {path}")
        return
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
                raise ValueError(f"content-addressed artifact differs from concurrent writer: {path}") from None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read support artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TypeError("support artifact must be a JSON object")
    return value


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"support artifact field {key!r} must be a non-empty string")
    return value


def _optional_string(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"support artifact field {key!r} must be a non-empty string or null")
    return value


def _optional_positive_int(payload: dict[str, Any], key: str) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"support artifact field {key!r} must be a positive integer or null")
    return value


def _string_tuple(payload: dict[str, Any], key: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    value = payload.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"support artifact field {key!r} must be a string list")
    if not allow_empty and not value:
        raise ValueError(f"support artifact field {key!r} must not be empty")
    result = tuple(value)
    if len(set(result)) != len(result):
        raise ValueError(f"support artifact field {key!r} must contain unique values")
    return result


def _failure(value: object) -> SupportFailure:
    if not isinstance(value, dict):
        raise TypeError("support audit failure must be an object")
    return SupportFailure(
        item_id=_required_string(value, "item_id"),
        failure_type=_required_string(value, "failure_type"),
        message=_required_string(value, "message"),
    )
