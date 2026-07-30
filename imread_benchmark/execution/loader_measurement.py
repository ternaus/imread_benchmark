from __future__ import annotations

import datetime as dt
import gc
import multiprocessing
import os
import queue
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from imread_benchmark.artifacts import RunSample
from imread_benchmark.contracts import OutputContract, validate_output
from imread_benchmark.execution.measurement import MeasurementError, MeasurementResult, configure_decoder

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np

    from imread_benchmark.datasets.package import ResidentItem
    from imread_benchmark.plans import RunConfiguration


@dataclass(frozen=True, slots=True)
class _Traversal:
    item_count: int
    decoded_pixels: int


@dataclass(frozen=True, slots=True)
class _DecoderWorkerConfig:
    decoder_id: str
    requested_threads: int | None
    logical_repeat_factor: int
    multiprocessing_start_method: str


@dataclass(frozen=True, slots=True)
class _LoaderRuntime:
    loader: Any
    handshake_queue: Any
    handshake_generation: Any
    multiprocessing_start_method: str
    expected_processes: int
    logical_items: int


@dataclass(frozen=True, slots=True)
class _Preparation:
    validation: _Traversal
    validation_handshakes: list[dict[str, object]]
    persistent_workers_reused: bool
    events: list[dict[str, object]]


@dataclass(slots=True)
class _LocalGeneration:
    value: int = 0


class ResidentLoaderDataset:
    def __init__(
        self,
        images_bytes: tuple[bytes, ...],
        *,
        worker_config: _DecoderWorkerConfig,
        handshake_queue: Any,
        handshake_generation: Any,
    ) -> None:
        self.images_bytes = images_bytes
        self.worker_config = worker_config
        self.handshake_queue = handshake_queue
        self.handshake_generation = handshake_generation
        self._decoder: Any = None
        self._effective_threads: int | None = None
        self._reported_generation = -1

    def __len__(self) -> int:
        return len(self.images_bytes) * self.worker_config.logical_repeat_factor

    def __getitem__(self, index: int) -> np.ndarray:
        decoder = self._get_decoder()
        generation = int(self.handshake_generation.value)
        if self._reported_generation != generation:
            self.handshake_queue.put(
                {
                    "effective_threads": self._effective_threads,
                    "generation": generation,
                    "multiprocessing_start_method": self.worker_config.multiprocessing_start_method,
                    "process_id": os.getpid(),
                },
            )
            self._reported_generation = generation
        output = decoder.decode(self.images_bytes[index % len(self.images_bytes)])
        validate_output(output, OutputContract.normalized_rgb())
        return output

    def _get_decoder(self) -> Any:
        if self._decoder is None:
            from imread_benchmark.decoders import REGISTRY

            decoder_class = REGISTRY.get(self.worker_config.decoder_id)
            if decoder_class is None:
                raise MeasurementError(f"unknown decoder {self.worker_config.decoder_id!r} in DataLoader worker")
            self._decoder = decoder_class()
            self._effective_threads = configure_decoder(self._decoder, self.worker_config.requested_threads)
        return self._decoder


def pass_through_collate(batch: list[np.ndarray]) -> list[np.ndarray]:
    return batch


def run_loader_supply_measurement(
    items: Sequence[ResidentItem],
    configuration: RunConfiguration,
) -> MeasurementResult:
    _validate_configuration(items, configuration)
    runtime = _build_loader_runtime(items, configuration)
    preparation = _prepare_loader(runtime, configuration)
    samples = _measure_loader(runtime, configuration)
    preparation.events.append({"event": "measurement_complete", "sample_count": len(samples)})
    resident_bytes = sum(item.compressed_bytes for item in items)
    _shutdown_loader(runtime.loader)
    close_queue = getattr(runtime.handshake_queue, "close", None)
    if callable(close_queue):
        close_queue()
    join_queue = getattr(runtime.handshake_queue, "join_thread", None)
    if callable(join_queue):
        join_queue()
    return MeasurementResult(
        samples=tuple(samples),
        runtime={
            "multiprocessing_start_method": runtime.multiprocessing_start_method,
            "persistent_workers_reused": preparation.persistent_workers_reused,
            "worker_handshakes": preparation.validation_handshakes,
        },
        summary_fields={
            "batch_size": configuration.batch_size,
            "compressed_bytes_per_pass": resident_bytes * configuration.logical_repeat_factor,
            "decoded_pixels_per_pass": preparation.validation.decoded_pixels,
            "logical_decodes_per_pass": runtime.logical_items,
            "logical_repeat_factor": configuration.logical_repeat_factor,
            "num_unique_images": len(items),
            "num_workers": configuration.num_workers,
            "resident_compressed_bytes": resident_bytes,
            "timed_transport": "in-process" if configuration.num_workers == 0 else "dataloader-process-queue",
        },
        events=tuple(preparation.events),
    )


def _validate_configuration(items: Sequence[ResidentItem], configuration: RunConfiguration) -> None:
    if configuration.protocol_id != "loader-supply":
        raise MeasurementError(f"loader-supply runner received protocol {configuration.protocol_id!r}")
    if not items:
        raise MeasurementError("loader-supply requires at least one resident item")
    if configuration.num_workers is None or configuration.batch_size is None:
        raise MeasurementError("loader-supply configuration is missing worker or batch settings")


def _build_loader_runtime(
    items: Sequence[ResidentItem],
    configuration: RunConfiguration,
) -> _LoaderRuntime:
    try:
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise MeasurementError("loader-supply requires PyTorch") from exc
    num_workers = configuration.num_workers
    if num_workers is None:
        raise MeasurementError("loader-supply configuration is missing num_workers")
    handshake_queue: Any
    handshake_generation: Any
    if num_workers == 0:
        context = None
        handshake_queue = queue.Queue()
        handshake_generation = _LocalGeneration()
        start_method = "in-process"
    else:
        try:
            context = multiprocessing.get_context(configuration.multiprocessing_start_method)
        except ValueError as exc:
            raise MeasurementError(
                f"multiprocessing start method {configuration.multiprocessing_start_method!r} is unavailable",
            ) from exc
        handshake_queue = context.Queue()
        handshake_generation = context.Value("i", 0)
        start_method = context.get_start_method()
    dataset = ResidentLoaderDataset(
        tuple(item.data for item in items),
        worker_config=_DecoderWorkerConfig(
            decoder_id=configuration.decoder_id,
            requested_threads=configuration.requested_threads,
            logical_repeat_factor=configuration.logical_repeat_factor,
            multiprocessing_start_method=start_method,
        ),
        handshake_queue=handshake_queue,
        handshake_generation=handshake_generation,
    )
    loader_kwargs: dict[str, Any] = {
        "batch_size": configuration.batch_size,
        "collate_fn": pass_through_collate,
        "dataset": dataset,
        "num_workers": num_workers,
        "persistent_workers": configuration.persistent_workers,
    }
    if num_workers > 0:
        if context is None:
            raise AssertionError("worker process context was not created")
        loader_kwargs["multiprocessing_context"] = context
        loader_kwargs["prefetch_factor"] = configuration.prefetch_factor
    return _LoaderRuntime(
        loader=DataLoader(**loader_kwargs),
        handshake_queue=handshake_queue,
        handshake_generation=handshake_generation,
        multiprocessing_start_method=start_method,
        expected_processes=max(1, num_workers),
        logical_items=len(dataset),
    )


def _prepare_loader(runtime: _LoaderRuntime, configuration: RunConfiguration) -> _Preparation:
    validation_started = time.perf_counter()
    validation = _consume_loader(runtime.loader)
    _require_item_count("validation", validation.item_count, runtime.logical_items)
    validation_handshakes = _receive_handshakes(
        runtime.handshake_queue,
        runtime.expected_processes,
        generation=0,
    )
    events = [_phase_event("validation", validation_started, item_count=validation.item_count)]

    warmup_started = time.perf_counter()
    persistent_workers_reused = configuration.num_workers == 0
    if configuration.warmup_passes > 0:
        runtime.handshake_generation.value = 1
        for _ in range(configuration.warmup_passes):
            warmup = _consume_loader(runtime.loader)
            _require_item_count("warmup", warmup.item_count, runtime.logical_items)
        warmup_handshakes = _receive_handshakes(
            runtime.handshake_queue,
            runtime.expected_processes,
            generation=1,
        )
        persistent_workers_reused = _process_ids(validation_handshakes) == _process_ids(warmup_handshakes)
    events.append(
        _phase_event(
            "warmup",
            warmup_started,
            passes=configuration.warmup_passes,
            persistent_workers_reused=persistent_workers_reused,
        ),
    )
    if configuration.num_workers and not persistent_workers_reused:
        raise MeasurementError("DataLoader worker processes were not reused between validation and warmup")
    return _Preparation(validation, validation_handshakes, persistent_workers_reused, events)


def _measure_loader(runtime: _LoaderRuntime, configuration: RunConfiguration) -> list[RunSample]:
    samples: list[RunSample] = []
    for sample_index in range(configuration.timed_passes_per_run):
        gc.collect()
        gc.disable()
        try:
            started_at_utc = dt.datetime.now(dt.UTC).isoformat()
            started = time.perf_counter()
            traversal = _consume_loader(runtime.loader)
            elapsed = time.perf_counter() - started
        finally:
            gc.enable()
        _require_item_count("timed pass", traversal.item_count, runtime.logical_items)
        if elapsed < configuration.minimum_timed_seconds:
            raise MeasurementError(
                f"timed pass {sample_index} lasted {elapsed:.9f}s, below the pinned minimum "
                f"{configuration.minimum_timed_seconds:.9f}s",
            )
        samples.append(RunSample(sample_index, elapsed, runtime.logical_items, started_at_utc))
    return samples


def _require_item_count(phase: str, observed: int, expected: int) -> None:
    if observed != expected:
        raise MeasurementError(f"{phase} yielded {observed} items, expected {expected}")


def _shutdown_loader(loader: Any) -> None:
    iterator = getattr(loader, "_iterator", None)
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown):
        shutdown()


def _consume_loader(loader: Any) -> _Traversal:
    item_count = 0
    decoded_pixels = 0
    for batch in loader:
        item_count += len(batch)
        decoded_pixels += sum(int(output.shape[0]) * int(output.shape[1]) for output in batch)
    return _Traversal(item_count=item_count, decoded_pixels=decoded_pixels)


def _receive_handshakes(handshake_queue: Any, count: int, *, generation: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    try:
        for _ in range(count):
            row = handshake_queue.get(timeout=10)
            if not isinstance(row, dict) or row.get("generation") != generation:
                raise MeasurementError("DataLoader worker returned an invalid handshake")
            rows.append(row)
    except queue.Empty as exc:
        raise MeasurementError(f"received {len(rows)} of {count} expected DataLoader worker handshakes") from exc
    if len(_process_ids(rows)) != count:
        raise MeasurementError("DataLoader handshakes do not represent the expected number of processes")
    return sorted(rows, key=_handshake_process_id)


def _process_ids(handshakes: Sequence[dict[str, object]]) -> set[int]:
    process_ids = {row.get("process_id") for row in handshakes}
    if any(isinstance(value, bool) or not isinstance(value, int) for value in process_ids):
        raise MeasurementError("DataLoader worker handshake has an invalid process ID")
    return {value for value in process_ids if isinstance(value, int)}


def _handshake_process_id(handshake: dict[str, object]) -> int:
    process_id = handshake.get("process_id")
    if isinstance(process_id, bool) or not isinstance(process_id, int):
        raise MeasurementError("DataLoader worker handshake has an invalid process ID")
    return process_id


def _phase_event(event: str, started: float, **details: object) -> dict[str, object]:
    return {
        "duration_seconds": time.perf_counter() - started,
        "event": event,
        **details,
    }
