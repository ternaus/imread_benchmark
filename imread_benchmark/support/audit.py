from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from imread_benchmark.contracts import OutputContract, OutputContractError, validate_output

if TYPE_CHECKING:
    from collections.abc import Sequence

    from imread_benchmark.datasets.package import ResidentItem
    from imread_benchmark.decoders import BaseDecoder

SUPPORT_SCHEMA_VERSION = "2.0"


@dataclass(frozen=True, slots=True)
class SupportFailure:
    item_id: str
    failure_type: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SupportAuditContext:
    manifest_id: str
    selection_id: str
    process_context: str
    multiprocessing_start_method: str | None
    requested_threads: int | None
    environment_id: str
    platform_id: str


@dataclass(frozen=True, slots=True)
class SupportAudit:
    audit_id: str
    decoder_id: str
    requested_threads: int | None
    manifest_id: str
    selection_id: str
    process_context: str
    multiprocessing_start_method: str | None
    output_contract: str
    environment_id: str
    platform_id: str
    successful_item_ids: tuple[str, ...]
    failures: tuple[SupportFailure, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "audit_id": self.audit_id,
            "decoder_id": self.decoder_id,
            "environment_id": self.environment_id,
            "failures": [failure.to_dict() for failure in self.failures],
            "manifest_id": self.manifest_id,
            "multiprocessing_start_method": self.multiprocessing_start_method,
            "output_contract": self.output_contract,
            "platform_id": self.platform_id,
            "process_context": self.process_context,
            "requested_threads": self.requested_threads,
            "schema_version": SUPPORT_SCHEMA_VERSION,
            "selection_id": self.selection_id,
            "successful_item_ids": list(self.successful_item_ids),
        }


@dataclass(frozen=True, slots=True)
class SupportSet:
    support_set_id: str
    policy: str
    manifest_id: str
    selection_id: str
    process_context: str
    multiprocessing_start_method: str | None
    audit_ids: tuple[str, ...]
    item_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "audit_ids": list(self.audit_ids),
            "item_ids": list(self.item_ids),
            "manifest_id": self.manifest_id,
            "multiprocessing_start_method": self.multiprocessing_start_method,
            "policy": self.policy,
            "process_context": self.process_context,
            "schema_version": SUPPORT_SCHEMA_VERSION,
            "selection_id": self.selection_id,
            "support_set_id": self.support_set_id,
        }


@dataclass(frozen=True, slots=True)
class SupportSetIdentity:
    policy: str
    manifest_id: str
    selection_id: str
    process_context: str
    multiprocessing_start_method: str | None
    audit_ids: tuple[str, ...]
    item_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "audit_ids": list(self.audit_ids),
            "item_ids": list(self.item_ids),
            "manifest_id": self.manifest_id,
            "multiprocessing_start_method": self.multiprocessing_start_method,
            "policy": self.policy,
            "process_context": self.process_context,
            "schema_version": SUPPORT_SCHEMA_VERSION,
            "selection_id": self.selection_id,
        }


def audit_decoder_support(
    decoder: BaseDecoder,
    items: Sequence[ResidentItem],
    context: SupportAuditContext,
) -> SupportAudit:
    if not items:
        raise ValueError("support audit requires at least one resident item")
    successful: list[str] = []
    failures: list[SupportFailure] = []
    contract = OutputContract.normalized_rgb()
    for item in items:
        try:
            validate_output(decoder.decode(item.data), contract)
        except Exception as exc:
            failures.append(
                SupportFailure(
                    item_id=item.item_id,
                    failure_type="output_contract_error" if isinstance(exc, OutputContractError) else "decode_error",
                    message=f"{type(exc).__name__}: {exc}",
                ),
            )
        else:
            successful.append(item.item_id)
    return build_support_audit(
        decoder_id=decoder.name,
        context=context,
        successful_item_ids=tuple(successful),
        failures=tuple(failures),
    )


def build_support_audit(
    *,
    decoder_id: str,
    context: SupportAuditContext,
    successful_item_ids: tuple[str, ...],
    failures: tuple[SupportFailure, ...],
) -> SupportAudit:
    identity = {
        "decoder_id": decoder_id,
        "environment_id": context.environment_id,
        "failures": [asdict(failure) for failure in failures],
        "manifest_id": context.manifest_id,
        "multiprocessing_start_method": context.multiprocessing_start_method,
        "output_contract": "normalized-rgb",
        "platform_id": context.platform_id,
        "process_context": context.process_context,
        "requested_threads": context.requested_threads,
        "schema_version": SUPPORT_SCHEMA_VERSION,
        "selection_id": context.selection_id,
        "successful_item_ids": list(successful_item_ids),
    }
    return SupportAudit(
        audit_id=_digest(identity),
        decoder_id=decoder_id,
        requested_threads=context.requested_threads,
        manifest_id=context.manifest_id,
        selection_id=context.selection_id,
        process_context=context.process_context,
        multiprocessing_start_method=context.multiprocessing_start_method,
        output_contract="normalized-rgb",
        environment_id=context.environment_id,
        platform_id=context.platform_id,
        successful_item_ids=successful_item_ids,
        failures=failures,
    )


def build_common_support(
    audits: Sequence[SupportAudit],
    *,
    ordered_selection: Sequence[str],
) -> SupportSet:
    if not audits:
        raise ValueError("common support requires at least one audit")
    first = audits[0]
    for audit in audits[1:]:
        if (
            audit.manifest_id != first.manifest_id
            or audit.selection_id != first.selection_id
            or audit.process_context != first.process_context
            or audit.multiprocessing_start_method != first.multiprocessing_start_method
            or audit.output_contract != first.output_contract
            or audit.platform_id != first.platform_id
            or audit.environment_id != first.environment_id
        ):
            raise ValueError("common support audits describe different workloads or execution contexts")
    common_ids = set(first.successful_item_ids)
    for audit in audits[1:]:
        common_ids.intersection_update(audit.successful_item_ids)
    item_ids = tuple(item_id for item_id in ordered_selection if item_id in common_ids)
    if not item_ids:
        raise ValueError("common support set is empty")
    audit_ids = tuple(sorted(audit.audit_id for audit in audits))
    identity = SupportSetIdentity(
        policy="common",
        manifest_id=first.manifest_id,
        selection_id=first.selection_id,
        process_context=first.process_context,
        multiprocessing_start_method=first.multiprocessing_start_method,
        audit_ids=audit_ids,
        item_ids=item_ids,
    ).to_dict()
    return SupportSet(
        support_set_id=_digest(identity),
        policy="common",
        manifest_id=first.manifest_id,
        selection_id=first.selection_id,
        process_context=first.process_context,
        multiprocessing_start_method=first.multiprocessing_start_method,
        audit_ids=audit_ids,
        item_ids=item_ids,
    )


def intersect_supported_items(
    audits: Sequence[SupportAudit],
    *,
    ordered_selection: Sequence[str],
) -> tuple[str, ...]:
    """Return one ordered success intersection across all comparison contexts."""
    if not audits:
        raise ValueError("support intersection requires at least one audit")
    first = audits[0]
    for audit in audits[1:]:
        if (
            audit.manifest_id != first.manifest_id
            or audit.selection_id != first.selection_id
            or audit.output_contract != first.output_contract
            or audit.platform_id != first.platform_id
            or audit.environment_id != first.environment_id
        ):
            raise ValueError("support audits describe different comparison populations")
    successful = set(first.successful_item_ids)
    for audit in audits[1:]:
        successful.intersection_update(audit.successful_item_ids)
    item_ids = tuple(item_id for item_id in ordered_selection if item_id in successful)
    if not item_ids:
        raise ValueError("comparison support intersection is empty")
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("ordered selection contains duplicate supported item IDs")
    return item_ids


def build_operational_support(audit: SupportAudit) -> SupportSet:
    if not audit.successful_item_ids:
        raise ValueError("operational support set is empty")
    audit_ids = (audit.audit_id,)
    identity = SupportSetIdentity(
        policy="operational",
        manifest_id=audit.manifest_id,
        selection_id=audit.selection_id,
        process_context=audit.process_context,
        multiprocessing_start_method=audit.multiprocessing_start_method,
        audit_ids=audit_ids,
        item_ids=audit.successful_item_ids,
    ).to_dict()
    return SupportSet(
        support_set_id=_digest(identity),
        policy="operational",
        manifest_id=audit.manifest_id,
        selection_id=audit.selection_id,
        process_context=audit.process_context,
        multiprocessing_start_method=audit.multiprocessing_start_method,
        audit_ids=audit_ids,
        item_ids=audit.successful_item_ids,
    )


def _digest(payload: object) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()
