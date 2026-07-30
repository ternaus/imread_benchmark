from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass
from typing import Any

from imread_benchmark.plans.model import ExperimentPlan, PlanError

_DIGEST = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class RunConfiguration:
    protocol_id: str
    decoder_id: str
    package_id: str
    manifest_id: str
    selection_id: str
    requested_threads: int | None
    num_workers: int | None
    batch_size: int | None
    prefetch_factor: int | None
    persistent_workers: bool
    multiprocessing_start_method: str | None
    logical_repeat_factor: int
    warmup_passes: int
    timed_passes_per_run: int
    minimum_timed_seconds: float
    output_contract: str
    support_policy: str

    def __post_init__(self) -> None:
        self._validate_core_fields()
        if self.protocol_id == "decode-memory":
            self._validate_decode_fields()
        else:
            self._validate_loader_fields()

    def _validate_core_fields(self) -> None:
        if self.protocol_id not in {"decode-memory", "loader-supply"}:
            raise ValueError(f"unsupported protocol_id: {self.protocol_id!r}")
        if not self.decoder_id:
            raise ValueError("decoder_id must not be empty")
        for field_name in ("package_id", "manifest_id", "selection_id"):
            if _DIGEST.fullmatch(getattr(self, field_name)) is None:
                raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
        if self.requested_threads is not None and (
            isinstance(self.requested_threads, bool)
            or not isinstance(self.requested_threads, int)
            or self.requested_threads <= 0
        ):
            raise ValueError("requested_threads must be a positive integer or default")
        if self.logical_repeat_factor <= 0 or self.timed_passes_per_run <= 0:
            raise ValueError("logical_repeat_factor and timed_passes_per_run must be positive")
        if self.warmup_passes < 0 or self.minimum_timed_seconds <= 0:
            raise ValueError("warmup_passes must be non-negative and minimum_timed_seconds positive")
        if self.output_contract != "normalized-rgb":
            raise ValueError("output_contract must be normalized-rgb")
        if self.support_policy not in {"common", "operational"}:
            raise ValueError("support_policy must be common or operational")

    def _validate_decode_fields(self) -> None:
        if (
            self.num_workers is not None
            or self.batch_size is not None
            or self.prefetch_factor is not None
            or self.persistent_workers
            or self.multiprocessing_start_method is not None
        ):
            raise ValueError("decode-memory does not accept DataLoader fields")

    def _validate_loader_fields(self) -> None:
        if self.num_workers is None or self.num_workers < 0 or self.batch_size is None or self.batch_size <= 0:
            raise ValueError("loader-supply requires non-negative workers and positive batch_size")
        if self.num_workers == 0:
            if (
                self.prefetch_factor is not None
                or self.persistent_workers
                or self.multiprocessing_start_method is not None
            ):
                raise ValueError(
                    "loader-supply workers=0 requires inapplicable prefetch, persistence, and "
                    "multiprocessing_start_method",
                )
        elif self.prefetch_factor is None or self.prefetch_factor <= 0 or not self.persistent_workers:
            raise ValueError("loader-supply worker processes require prefetch and persistent_workers")
        if self.num_workers > 0 and self.multiprocessing_start_method not in {"fork", "forkserver", "spawn"}:
            raise ValueError(
                "loader-supply worker processes require multiprocessing_start_method=fork, forkserver, or spawn",
            )
        if self.num_workers > 0 and self.warmup_passes == 0:
            raise ValueError("loader-supply worker processes require at least one warmup pass")

    @property
    def config_id(self) -> str:
        return _digest(asdict(self))


@dataclass(frozen=True, slots=True)
class RunTemplate:
    plan_id: str
    template_id: str
    configuration: RunConfiguration
    repetition: int
    position: int


@dataclass(frozen=True, slots=True)
class _ConfigurationSeed:
    plan: ExperimentPlan
    measurement: dict[str, Any]
    decoder_id: str
    threads: int | None


@dataclass(frozen=True, slots=True)
class _WorkerSettings:
    num_workers: int
    batch_size: int
    prefetch_factor: int | None
    persistent_workers: bool
    multiprocessing_start_method: str | None


def expand_experiment_plan(plan: ExperimentPlan) -> tuple[RunTemplate, ...]:
    configurations = _configurations(plan)
    by_id = {configuration.config_id: configuration for configuration in configurations}
    if len(by_id) != len(configurations):
        raise PlanError("experiment plan expands to duplicate configurations")
    plan_id = _plan_id(plan)
    generator = random.Random(plan.seed)  # noqa: S311 - scientific randomization is intentionally reproducible
    templates: list[RunTemplate] = []
    for repetition in range(plan.repetitions):
        block = list(by_id)
        generator.shuffle(block)
        for config_id in block:
            identity = {
                "configuration_id": config_id,
                "plan_id": plan_id,
                "repetition": repetition,
            }
            templates.append(
                RunTemplate(
                    plan_id=plan_id,
                    template_id=_digest(identity),
                    configuration=by_id[config_id],
                    repetition=repetition,
                    position=len(templates),
                ),
            )
    return tuple(templates)


def _configurations(plan: ExperimentPlan) -> tuple[RunConfiguration, ...]:
    if set(plan.matrix) != {"decoders", "protocols"}:
        raise PlanError("matrix fields must be exactly decoders and protocols")
    decoders = _decoder_profiles(plan.matrix)
    protocols = _object(plan.matrix, "protocols")
    measurement = _measurement(plan.measurement)
    configurations: list[RunConfiguration] = []
    decode_memory = protocols.get("decode-memory")
    if decode_memory is not None:
        configurations.extend(_decode_configurations(plan, measurement, decoders, decode_memory))
    loader_supply = protocols.get("loader-supply")
    if loader_supply is not None:
        configurations.extend(_loader_protocol_configurations(plan, measurement, decoders, loader_supply))
    if not configurations:
        raise PlanError("matrix.protocols must select at least one protocol")
    return tuple(configurations)


def _decode_configurations(
    plan: ExperimentPlan,
    measurement: dict[str, Any],
    decoders: tuple[tuple[str, tuple[int | None, ...]], ...],
    protocol: object,
) -> list[RunConfiguration]:
    if not isinstance(protocol, dict):
        raise PlanError("matrix.protocols.decode-memory must be an object")
    if protocol:
        raise PlanError("matrix.protocols.decode-memory does not accept fields")
    return [
        _configuration(
            _ConfigurationSeed(plan, measurement, decoder_id, threads),
            protocol_id="decode-memory",
        )
        for decoder_id, thread_profile in decoders
        for threads in thread_profile
    ]


def _loader_protocol_configurations(
    plan: ExperimentPlan,
    measurement: dict[str, Any],
    decoders: tuple[tuple[str, tuple[int | None, ...]], ...],
    protocol: object,
) -> list[RunConfiguration]:
    if not isinstance(protocol, dict):
        raise PlanError("matrix.protocols.loader-supply must be an object")
    if set(protocol) != {"worker_profiles"}:
        raise PlanError("matrix.protocols.loader-supply accepts only worker_profiles")
    configurations: list[RunConfiguration] = []
    for decoder_id, thread_profile in decoders:
        for threads in thread_profile:
            configurations.extend(
                _loader_configurations(
                    _ConfigurationSeed(plan, measurement, decoder_id, threads),
                    protocol,
                ),
            )
    return configurations


def _decoder_profiles(matrix: dict[str, Any]) -> tuple[tuple[str, tuple[int | None, ...]], ...]:
    from imread_benchmark.decoders import REGISTRY, BaseDecoder

    document = matrix.get("decoders")
    if not isinstance(document, dict) or not document:
        raise PlanError("matrix.decoders must be a non-empty object")
    profiles: list[tuple[str, tuple[int | None, ...]]] = []
    for decoder_id, raw_profile in sorted(document.items()):
        if not isinstance(decoder_id, str) or not decoder_id or not isinstance(raw_profile, dict):
            raise PlanError("each decoder profile must be a named object")
        if set(raw_profile) != {"threads"}:
            raise PlanError(f"decoder {decoder_id!r} profile accepts only threads")
        decoder = REGISTRY.get(decoder_id)
        if decoder is None:
            raise PlanError(f"unknown decoder in experiment plan: {decoder_id}")
        threads = _thread_list(raw_profile, "threads")
        if any(value is not None for value in threads) and decoder.set_num_threads is BaseDecoder.set_num_threads:
            raise PlanError(f"decoder {decoder_id!r} does not expose thread control")
        profiles.append((decoder_id, threads))
    return tuple(profiles)


def _configuration(
    seed: _ConfigurationSeed,
    *,
    protocol_id: str,
    workers: _WorkerSettings | None = None,
) -> RunConfiguration:
    return RunConfiguration(
        protocol_id=protocol_id,
        decoder_id=seed.decoder_id,
        package_id=seed.plan.dataset.package_id,
        manifest_id=seed.plan.dataset.manifest_id,
        selection_id=seed.plan.dataset.selection.selection_id,
        requested_threads=seed.threads,
        num_workers=workers.num_workers if workers is not None else None,
        batch_size=workers.batch_size if workers is not None else None,
        prefetch_factor=workers.prefetch_factor if workers is not None else None,
        persistent_workers=workers.persistent_workers if workers is not None else False,
        multiprocessing_start_method=workers.multiprocessing_start_method if workers is not None else None,
        logical_repeat_factor=seed.plan.dataset.logical_repeat_factor,
        warmup_passes=seed.measurement["warmup_passes"],
        timed_passes_per_run=seed.measurement["timed_passes_per_run"],
        minimum_timed_seconds=seed.measurement["minimum_timed_seconds"],
        output_contract=seed.measurement["output_contract"],
        support_policy=seed.measurement["support_policy"],
    )


def _loader_configurations(
    seed: _ConfigurationSeed,
    protocol: dict[str, Any],
) -> list[RunConfiguration]:
    profiles = protocol.get("worker_profiles")
    if not isinstance(profiles, list) or not profiles:
        raise PlanError("loader-supply.worker_profiles must be a non-empty list")
    configurations: list[RunConfiguration] = []
    for profile in profiles:
        if not isinstance(profile, dict):
            raise PlanError("loader-supply worker profile must be an object")
        workers_values = _non_negative_int_list(profile, "workers")
        has_zero = 0 in workers_values
        has_processes = any(value > 0 for value in workers_values)
        if has_zero and has_processes:
            raise PlanError("worker profile must not mix workers=0 with worker processes")
        batch_size = _positive_int(profile, "batch_size")
        prefetch = None if has_zero else _positive_int(profile, "prefetch_factor")
        persistent = False if has_zero else profile.get("persistent_workers") is True
        if has_processes and not persistent:
            raise PlanError("loader-supply worker processes require persistent_workers=true")
        start_method = None if has_zero else _multiprocessing_start_method(profile)
        configurations.extend(
            _configuration(
                seed,
                protocol_id="loader-supply",
                workers=_WorkerSettings(
                    num_workers=workers,
                    batch_size=batch_size,
                    prefetch_factor=prefetch,
                    persistent_workers=persistent,
                    multiprocessing_start_method=start_method,
                ),
            )
            for workers in workers_values
        )
    return configurations


def _multiprocessing_start_method(profile: dict[str, Any]) -> str:
    value = profile.get("multiprocessing_start_method")
    if value not in {"fork", "forkserver", "spawn"}:
        raise PlanError(
            "loader-supply worker process profile requires multiprocessing_start_method=fork, forkserver, or spawn",
        )
    return value


def _measurement(document: dict[str, Any]) -> dict[str, Any]:
    minimum = document.get("minimum_timed_seconds")
    if isinstance(minimum, bool) or not isinstance(minimum, (int, float)) or minimum <= 0:
        raise PlanError("measurement.minimum_timed_seconds must be positive")
    output_contract = document.get("output_contract")
    if output_contract != "normalized-rgb":
        raise PlanError("measurement.output_contract must be normalized-rgb")
    support_policy = document.get("support_policy")
    if support_policy not in {"common", "operational"}:
        raise PlanError("measurement.support_policy must be common or operational")
    warmup = document.get("warmup_passes")
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise PlanError("measurement.warmup_passes must be a non-negative integer")
    return {
        "minimum_timed_seconds": float(minimum),
        "output_contract": output_contract,
        "support_policy": support_policy,
        "timed_passes_per_run": _positive_int(document, "timed_passes_per_run"),
        "warmup_passes": warmup,
    }


def _plan_id(plan: ExperimentPlan) -> str:
    return _digest(
        {
            "dataset": {
                "logical_repeat_factor": plan.dataset.logical_repeat_factor,
                "manifest_id": plan.dataset.manifest_id,
                "package_id": plan.dataset.package_id,
                "selection_id": plan.dataset.selection.selection_id,
                "workload_id": plan.dataset.workload_id,
            },
            "execution": plan.execution,
            "experiment_name": plan.experiment_name,
            "matrix": plan.matrix,
            "measurement": plan.measurement,
            "repetitions": plan.repetitions,
            "schema_version": plan.schema_version,
            "seed": plan.seed,
        },
    )


def _thread_list(payload: dict[str, Any], key: str) -> tuple[int | None, ...]:
    values = payload.get(key)
    if not isinstance(values, list) or not values:
        raise PlanError(f"field {key!r} must be a non-empty list")
    result: list[int | None] = []
    for value in values:
        if value == "default":
            result.append(None)
        elif not isinstance(value, bool) and isinstance(value, int) and value > 0:
            result.append(value)
        else:
            raise PlanError(f"field {key!r} accepts positive integers or 'default'")
    if len(set(result)) != len(result):
        raise PlanError(f"field {key!r} must contain unique values")
    return tuple(result)


def _non_negative_int_list(payload: dict[str, Any], key: str) -> tuple[int, ...]:
    values = payload.get(key)
    if (
        not isinstance(values, list)
        or not values
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values)
    ):
        raise PlanError(f"field {key!r} must be a non-empty list of non-negative integers")
    if len(set(values)) != len(values):
        raise PlanError(f"field {key!r} must contain unique values")
    return tuple(values)


def _positive_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PlanError(f"field {key!r} must be a positive integer")
    return value


def _object(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise PlanError(f"field {key!r} must be an object")
    return value


def _digest(payload: object) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()
