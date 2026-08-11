from __future__ import annotations

import os
from dataclasses import asdict

from imread_benchmark.artifacts import BundleData, RunSample
from imread_benchmark.environments import EnvironmentDescriptor
from imread_benchmark.execution.spec import RunIdentity, compute_run_key
from imread_benchmark.plans import RunConfiguration
from imread_benchmark.platforms import PlatformDescriptor


def valid_bundle_data(  # noqa: PLR0913
    *,
    protocol_id: str = "decode-memory",
    decoder_id: str = "pillow",
    block_position: int = 0,
    repetition: int = 0,
    elapsed_seconds: tuple[float, float] = (2.0, 1.0),
    platform_identity: dict[str, object] | None = None,
    platform_provenance: dict[str, object] | None = None,
) -> tuple[str, BundleData]:
    is_loader = protocol_id == "loader-supply"
    configuration = RunConfiguration(
        protocol_id=protocol_id,
        decoder_id=decoder_id,
        package_id="b" * 64,
        manifest_id="c" * 64,
        selection_id="d" * 64,
        requested_threads=None,
        num_workers=0 if is_loader else None,
        batch_size=1 if is_loader else None,
        prefetch_factor=None,
        persistent_workers=False,
        multiprocessing_start_method=None,
        logical_repeat_factor=1,
        warmup_passes=1,
        timed_passes_per_run=2,
        minimum_timed_seconds=1e-9,
        output_contract="normalized-rgb",
        support_policy="operational",
    )
    environment = EnvironmentDescriptor.build(
        dependency_group="mainstream",
        lock_sha256="8" * 64,
        project_sha256="9" * 64,
        runner_revision="1" * 40,
        python={"abi": "fixture-abi", "implementation": "cpython", "version": "3.12.0"},
        platform_tags=("fixture-platform",),
        distributions=(("imread-benchmark", "0.2.0"),),
        native_backends={},
    )
    platform = PlatformDescriptor.build(
        identity=platform_identity or {"architecture": "fixture", "logical_cpu_count": 1, "machine_type": "fixture"},
        runtime={"kernel": "fixture"},
        provenance=platform_provenance,
    )
    identity = RunIdentity(
        plan_id="a" * 64,
        platform_id=platform.platform_id,
        environment_id=environment.environment_id,
        runner_revision="1" * 40,
        workload_id="fixture",
        support_set_id="2" * 64,
        support_item_ids=("item-1", "item-2"),
        configuration=configuration,
        repetition=repetition,
        block_position=block_position,
    )
    run_key = compute_run_key(identity)
    runtime: dict[str, object] = {
        "effective_threads": 1,
        "process_id": os.getpid(),
        "requested_threads": None,
    }
    if is_loader:
        runtime.update(
            {
                "multiprocessing_start_method": "in-process",
                "persistent_workers_reused": True,
                "worker_handshakes": [
                    {
                        "effective_threads": 1,
                        "generation": 0,
                        "multiprocessing_start_method": "in-process",
                        "process_id": os.getpid(),
                    },
                ],
            },
        )
    return run_key, BundleData(
        config={
            **asdict(configuration),
            "block_position": identity.block_position,
            "config_id": configuration.config_id,
            "plan_id": identity.plan_id,
            "repetition": identity.repetition,
            "run_key": run_key,
            "runner_revision": identity.runner_revision,
        },
        dataset={
            "manifest_id": configuration.manifest_id,
            "ordered_item_ids": list(identity.support_item_ids),
            "package_id": configuration.package_id,
            "selection_id": configuration.selection_id,
            "support_audit_ids": ["7" * 64],
            "support_policy": configuration.support_policy,
            "support_process_context": "main-process",
            "support_multiprocessing_start_method": configuration.multiprocessing_start_method,
            "support_set_id": identity.support_set_id,
            "workload_id": identity.workload_id,
        },
        environment=environment.to_dict(),
        platform=platform.to_dict(),
        runtime=runtime,
        samples=(
            RunSample(sample_index=0, elapsed_seconds=elapsed_seconds[0], items_processed=2),
            RunSample(sample_index=1, elapsed_seconds=elapsed_seconds[1], items_processed=2),
        ),
        events=(
            {"duration_seconds": 0.1, "event": "validation"},
            {"duration_seconds": 0.1, "event": "warmup"},
            {"event": "measurement_complete", "sample_count": 2},
        ),
        summary_fields={
            "logical_decodes_per_pass": 2,
            "logical_repeat_factor": 1,
            "num_unique_images": 2,
        },
    )
