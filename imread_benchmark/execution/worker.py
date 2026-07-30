from __future__ import annotations

import argparse
import os
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from imread_benchmark.artifacts import BundleData, write_run_bundle
from imread_benchmark.datasets.package import DatasetPackageError, open_dataset_package
from imread_benchmark.environments import EnvironmentDescriptor, load_environment_descriptor
from imread_benchmark.execution.measurement import MeasurementError, configure_decoder, run_decode_memory_measurement
from imread_benchmark.execution.spec import RunSpecError, load_run_spec
from imread_benchmark.platforms import PlatformDescriptor, load_platform_descriptor
from imread_benchmark.support import load_support_set

if TYPE_CHECKING:
    from imread_benchmark.datasets.package import ResidentItem
    from imread_benchmark.decoders import BaseDecoder
    from imread_benchmark.execution.measurement import MeasurementResult
    from imread_benchmark.execution.spec import RunSpec
    from imread_benchmark.plans import RunConfiguration
    from imread_benchmark.support import SupportSet


@dataclass(frozen=True, slots=True)
class _BundleContext:
    runtime: dict[str, object]
    environment: dict[str, object]
    platform: dict[str, object]


def run_worker(run_spec_path: str | Path, artifact_root: str | Path) -> Path:
    spec = load_run_spec(run_spec_path)
    configuration = spec.identity.configuration
    environment, platform_descriptor = _load_provenance(spec)
    preimport_runtime = {
        "machine": platform.machine(),
        "process_id": os.getpid(),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "system": platform.system(),
    }

    selected_items = _load_selected_items(spec)
    support_set = _load_pinned_support_set(spec)
    if configuration.protocol_id == "decode-memory":
        decoder, effective_threads = _load_decoder(configuration)
        measurement = run_decode_memory_measurement(
            decoder,
            selected_items,
            configuration,
            effective_threads=effective_threads,
        )
    elif configuration.protocol_id == "loader-supply":
        from imread_benchmark.execution.loader_measurement import run_loader_supply_measurement

        measurement = run_loader_supply_measurement(selected_items, configuration)
    else:
        raise MeasurementError(f"unsupported worker protocol {configuration.protocol_id!r}")
    runtime = {**preimport_runtime, **measurement.runtime}
    return _write_measurement_bundle(
        spec,
        support_set,
        measurement,
        _BundleContext(runtime, environment.to_dict(), platform_descriptor.to_dict()),
        artifact_root,
    )


def _load_provenance(spec: RunSpec) -> tuple[EnvironmentDescriptor, PlatformDescriptor]:
    environment = load_environment_descriptor(spec.environment_descriptor)
    platform_descriptor = load_platform_descriptor(spec.platform_descriptor)
    if environment.environment_id != spec.identity.environment_id:
        raise MeasurementError("environment descriptor does not match run identity")
    if environment.runner_revision != spec.identity.runner_revision:
        raise MeasurementError("environment descriptor runner revision does not match run identity")
    if platform_descriptor.platform_id != spec.identity.platform_id:
        raise MeasurementError("platform descriptor does not match run identity")
    return environment, platform_descriptor


def _load_decoder(configuration: RunConfiguration) -> tuple[BaseDecoder, int]:
    from imread_benchmark.decoders import REGISTRY

    decoder_class = REGISTRY.get(configuration.decoder_id)
    if decoder_class is None:
        raise MeasurementError(f"unknown decoder {configuration.decoder_id!r}")
    if not decoder_class.runs_single_here():
        raise MeasurementError(f"decoder {configuration.decoder_id!r} does not support this platform")
    decoder = decoder_class()
    return decoder, configure_decoder(decoder, configuration.requested_threads)


def _load_selected_items(spec: RunSpec) -> tuple[ResidentItem, ...]:
    configuration = spec.identity.configuration
    package = open_dataset_package(spec.package_descriptor)
    if package.descriptor.get("package_id") != configuration.package_id:
        raise MeasurementError("run spec package_id does not match dataset descriptor")
    workload = package.descriptor.get("workloads", {}).get(spec.identity.workload_id)
    if not isinstance(workload, dict) or workload.get("manifest_id") != configuration.manifest_id:
        raise MeasurementError("run spec workload manifest_id does not match dataset descriptor")
    all_items = package.read_workload_items(spec.identity.workload_id)
    by_id = {item.item_id: item for item in all_items}
    if len(by_id) != len(all_items):
        raise MeasurementError("dataset workload contains duplicate item IDs")
    missing = [item_id for item_id in spec.identity.support_item_ids if item_id not in by_id]
    if missing:
        raise MeasurementError(f"support set refers to missing dataset items: {missing[:3]}")
    return tuple(by_id[item_id] for item_id in spec.identity.support_item_ids)


def _load_pinned_support_set(spec: RunSpec) -> SupportSet:
    configuration = spec.identity.configuration
    try:
        support_set = load_support_set(spec.support_set_path)
    except (TypeError, ValueError) as exc:
        raise MeasurementError(f"invalid pinned support set: {exc}") from exc
    expected_process_context = (
        "dataloader" if configuration.protocol_id == "loader-supply" and configuration.num_workers else "main-process"
    )
    if (
        support_set.support_set_id != spec.identity.support_set_id
        or support_set.manifest_id != configuration.manifest_id
        or support_set.selection_id != configuration.selection_id
        or support_set.policy != configuration.support_policy
        or support_set.process_context != expected_process_context
        or support_set.multiprocessing_start_method != configuration.multiprocessing_start_method
        or support_set.item_ids != spec.identity.support_item_ids
    ):
        raise MeasurementError("pinned support set does not match run specification")
    return support_set


def _write_measurement_bundle(
    spec: RunSpec,
    support_set: SupportSet,
    measurement: MeasurementResult,
    context: _BundleContext,
    artifact_root: str | Path,
) -> Path:
    configuration = spec.identity.configuration
    return write_run_bundle(
        root=Path(artifact_root) / "runs",
        run_key=spec.run_key,
        data=BundleData(
            config={
                **asdict(configuration),
                "block_position": spec.identity.block_position,
                "config_id": configuration.config_id,
                "plan_id": spec.identity.plan_id,
                "repetition": spec.identity.repetition,
                "run_key": spec.run_key,
                "runner_revision": spec.identity.runner_revision,
            },
            dataset={
                "manifest_id": configuration.manifest_id,
                "ordered_item_ids": list(spec.identity.support_item_ids),
                "package_id": configuration.package_id,
                "selection_id": configuration.selection_id,
                "support_set_id": spec.identity.support_set_id,
                "support_audit_ids": list(support_set.audit_ids),
                "support_policy": support_set.policy,
                "support_process_context": support_set.process_context,
                "support_multiprocessing_start_method": support_set.multiprocessing_start_method,
                "workload_id": spec.identity.workload_id,
            },
            environment=context.environment,
            platform=context.platform,
            runtime=context.runtime,
            samples=measurement.samples,
            summary_fields=measurement.summary_fields,
            events=measurement.events,
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-spec", required=True, type=Path)
    parser.add_argument("--artifact-root", required=True, type=Path)
    args = parser.parse_args()
    try:
        run_worker(args.run_spec, args.artifact_root)
    except RunSpecError as exc:
        print(f"run-spec error: {exc}", file=sys.stderr)
        return 20
    except DatasetPackageError as exc:
        print(f"dataset error: {exc}", file=sys.stderr)
        return 21
    except MeasurementError as exc:
        print(f"measurement contract error: {exc}", file=sys.stderr)
        return 22
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
