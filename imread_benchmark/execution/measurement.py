from __future__ import annotations

import datetime as dt
import gc
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from imread_benchmark.artifacts import RunSample
from imread_benchmark.contracts import OutputContract, validate_output

if TYPE_CHECKING:
    from collections.abc import Sequence

    from imread_benchmark.datasets.package import ResidentItem
    from imread_benchmark.decoders import BaseDecoder
    from imread_benchmark.plans import RunConfiguration


class MeasurementError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class MeasurementResult:
    samples: tuple[RunSample, ...]
    runtime: dict[str, object]
    summary_fields: dict[str, object]
    events: tuple[dict[str, object], ...]


def configure_decoder(decoder: BaseDecoder, requested_threads: int | None) -> int:
    from imread_benchmark.decoders import BaseDecoder

    if requested_threads is not None:
        if type(decoder).set_num_threads is BaseDecoder.set_num_threads:
            raise MeasurementError(f"decoder {decoder.name!r} does not expose thread control")
        decoder.set_num_threads(requested_threads)
    effective = decoder.get_num_threads()
    if effective <= 0:
        raise MeasurementError(f"decoder {decoder.name!r} reported invalid effective thread count {effective}")
    if requested_threads is not None and effective != requested_threads:
        raise MeasurementError(
            f"decoder {decoder.name!r} reported {effective} effective threads after requesting {requested_threads}",
        )
    return effective


def run_decode_memory_measurement(
    decoder: BaseDecoder,
    items: Sequence[ResidentItem],
    configuration: RunConfiguration,
    *,
    effective_threads: int,
) -> MeasurementResult:
    if configuration.protocol_id != "decode-memory":
        raise MeasurementError(f"decode-memory runner received protocol {configuration.protocol_id!r}")
    if not items:
        raise MeasurementError("decode-memory requires at least one resident item")
    contract = OutputContract.normalized_rgb()
    events: list[dict[str, object]] = []

    validation_started = time.perf_counter()
    decoded_pixels = 0
    for item in items:
        try:
            output = decoder.decode(item.data)
            validate_output(output, contract)
        except Exception as exc:
            raise MeasurementError(
                f"pinned support item {item.item_id!r} failed validation: {type(exc).__name__}: {exc}",
            ) from exc
        decoded_pixels += int(output.shape[0]) * int(output.shape[1])
    events.append(_phase_event("validation", validation_started, item_count=len(items)))

    warmup_started = time.perf_counter()
    for _ in range(configuration.warmup_passes):
        _decode_pass(decoder, items, configuration.logical_repeat_factor)
    events.append(_phase_event("warmup", warmup_started, passes=configuration.warmup_passes))

    logical_items = len(items) * configuration.logical_repeat_factor
    samples: list[RunSample] = []
    for sample_index in range(configuration.timed_passes_per_run):
        gc.collect()
        gc.disable()
        try:
            started_at_utc = dt.datetime.now(dt.UTC).isoformat()
            started = time.perf_counter()
            _decode_pass(decoder, items, configuration.logical_repeat_factor)
            elapsed = time.perf_counter() - started
        finally:
            gc.enable()
        if elapsed < configuration.minimum_timed_seconds:
            raise MeasurementError(
                f"timed pass {sample_index} lasted {elapsed:.9f}s, below the pinned minimum "
                f"{configuration.minimum_timed_seconds:.9f}s",
            )
        samples.append(
            RunSample(
                sample_index=sample_index,
                elapsed_seconds=elapsed,
                items_processed=logical_items,
                started_at_utc=started_at_utc,
            ),
        )
    events.append(
        {
            "event": "measurement_complete",
            "sample_count": len(samples),
        },
    )
    resident_bytes = sum(item.compressed_bytes for item in items)
    return MeasurementResult(
        samples=tuple(samples),
        runtime={
            "effective_threads": effective_threads,
            "requested_threads": configuration.requested_threads,
        },
        summary_fields={
            "compressed_bytes_per_pass": resident_bytes * configuration.logical_repeat_factor,
            "decoded_pixels_per_pass": decoded_pixels * configuration.logical_repeat_factor,
            "logical_decodes_per_pass": logical_items,
            "logical_repeat_factor": configuration.logical_repeat_factor,
            "num_unique_images": len(items),
            "resident_compressed_bytes": resident_bytes,
            "timed_input_source": "resident-tar-bytes",
        },
        events=tuple(events),
    )


def _decode_pass(decoder: BaseDecoder, items: Sequence[ResidentItem], repeats: int) -> None:
    for _ in range(repeats):
        for item in items:
            decoder.decode(item.data)


def _phase_event(event: str, started: float, **details: object) -> dict[str, object]:
    return {
        "duration_seconds": time.perf_counter() - started,
        "event": event,
        **details,
    }
