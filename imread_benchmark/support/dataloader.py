from __future__ import annotations

import multiprocessing
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from imread_benchmark.contracts import OutputContract, OutputContractError, validate_output
from imread_benchmark.execution.measurement import configure_decoder
from imread_benchmark.support.audit import (
    SupportAudit,
    SupportAuditContext,
    SupportFailure,
    build_support_audit,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from imread_benchmark.datasets.package import ResidentItem


@dataclass(frozen=True, slots=True)
class DataLoaderSupportAuditResult:
    audit: SupportAudit
    worker_handshakes: tuple[dict[str, object], ...]


@dataclass(frozen=True, slots=True)
class _WorkerConfig:
    decoder_id: str
    requested_threads: int | None
    multiprocessing_start_method: str


class _AuditDataset:
    def __init__(self, items: tuple[tuple[str, bytes], ...], worker_config: _WorkerConfig) -> None:
        self.items = items
        self.worker_config = worker_config
        self._decoder: Any = None
        self._effective_threads: int | None = None

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, object]:
        item_id, data = self.items[index]
        decoder = self._get_decoder()
        failure_type: str | None = None
        message: str | None = None
        try:
            validate_output(decoder.decode(data), OutputContract.normalized_rgb())
        except Exception as exc:
            failure_type = "output_contract_error" if isinstance(exc, OutputContractError) else "decode_error"
            message = f"{type(exc).__name__}: {exc}"
        return {
            "effective_threads": self._effective_threads,
            "failure_type": failure_type,
            "item_id": item_id,
            "message": message,
            "multiprocessing_start_method": self.worker_config.multiprocessing_start_method,
            "process_id": os.getpid(),
        }

    def _get_decoder(self) -> Any:
        if self._decoder is None:
            from imread_benchmark.decoders import REGISTRY

            decoder_class = REGISTRY.get(self.worker_config.decoder_id)
            if decoder_class is None:
                raise RuntimeError(f"unknown decoder {self.worker_config.decoder_id!r}")
            self._decoder = decoder_class()
            self._effective_threads = configure_decoder(self._decoder, self.worker_config.requested_threads)
        return self._decoder


def audit_decoder_support_in_dataloader(
    decoder_id: str,
    items: Sequence[ResidentItem],
    context: SupportAuditContext,
    *,
    requested_threads: int | None,
) -> DataLoaderSupportAuditResult:
    if not items:
        raise ValueError("support audit requires at least one resident item")
    if context.process_context != "dataloader":
        raise ValueError("DataLoader support audit requires process_context='dataloader'")
    try:
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise RuntimeError("DataLoader support audit requires PyTorch") from exc
    if context.multiprocessing_start_method not in {"fork", "forkserver", "spawn"}:
        raise ValueError("DataLoader support audit requires an explicit multiprocessing start method")
    try:
        multiprocessing_context = multiprocessing.get_context(context.multiprocessing_start_method)
    except ValueError as exc:
        raise RuntimeError(
            f"multiprocessing start method {context.multiprocessing_start_method!r} is unavailable",
        ) from exc
    dataset = _AuditDataset(
        tuple((item.item_id, item.data) for item in items),
        _WorkerConfig(decoder_id, requested_threads, multiprocessing_context.get_start_method()),
    )
    loader: Any = DataLoader(
        cast("Any", dataset),
        batch_size=None,
        multiprocessing_context=multiprocessing_context,
        num_workers=1,
        persistent_workers=False,
    )
    rows = tuple(loader)
    del loader
    successful = tuple(_required_string(row, "item_id") for row in rows if row.get("failure_type") is None)
    failures = tuple(
        SupportFailure(
            item_id=_required_string(row, "item_id"),
            failure_type=_required_string(row, "failure_type"),
            message=_required_string(row, "message"),
        )
        for row in rows
        if row.get("failure_type") is not None
    )
    handshakes_by_pid: dict[int, dict[str, object]] = {}
    for row in rows:
        process_id = row.get("process_id")
        if isinstance(process_id, bool) or not isinstance(process_id, int) or process_id <= 0:
            raise RuntimeError("DataLoader support audit returned an invalid worker PID")
        handshakes_by_pid.setdefault(
            process_id,
            {
                "effective_threads": row.get("effective_threads"),
                "multiprocessing_start_method": row.get("multiprocessing_start_method"),
                "process_id": process_id,
            },
        )
    if len(handshakes_by_pid) != 1:
        raise RuntimeError("DataLoader support audit expected exactly one worker process")
    audit = build_support_audit(
        decoder_id=decoder_id,
        context=context,
        successful_item_ids=successful,
        failures=failures,
    )
    return DataLoaderSupportAuditResult(
        audit=audit,
        worker_handshakes=tuple(handshakes_by_pid.values()),
    )


def _required_string(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"DataLoader support audit field {key!r} must be a non-empty string")
    return value
