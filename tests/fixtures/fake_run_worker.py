from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from imread_benchmark.artifacts import BundleData, RunSample, write_run_bundle
from imread_benchmark.environments import load_environment_descriptor
from imread_benchmark.execution.spec import load_run_spec
from imread_benchmark.platforms import load_platform_descriptor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-spec", required=True, type=Path)
    parser.add_argument("--artifact-root", required=True, type=Path)
    args = parser.parse_args()
    spec = load_run_spec(args.run_spec)
    configuration = spec.identity.configuration
    environment = load_environment_descriptor(spec.environment_descriptor)
    platform_descriptor = load_platform_descriptor(spec.platform_descriptor)
    if configuration.decoder_id == "hard-crash":
        os.kill(os.getpid(), signal.SIGSEGV)
    if configuration.decoder_id == "slow":
        child = subprocess.Popen(
            (sys.executable, "-c", "import time; time.sleep(60)"),
        )
        print(json.dumps({"child_process_id": child.pid}), flush=True)
        time.sleep(60)
    effective_threads = configuration.requested_threads if configuration.requested_threads is not None else 17
    item_count = len(spec.identity.support_item_ids)
    write_run_bundle(
        root=args.artifact_root / "runs",
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
                "support_audit_ids": ["7" * 64],
                "support_policy": configuration.support_policy,
                "support_process_context": "main-process",
                "support_multiprocessing_start_method": None,
                "support_set_id": spec.identity.support_set_id,
                "workload_id": spec.identity.workload_id,
            },
            environment=environment.to_dict(),
            platform=platform_descriptor.to_dict(),
            runtime={
                "effective_threads": effective_threads,
                "process_id": os.getpid(),
                "requested_threads": configuration.requested_threads,
            },
            samples=tuple(
                RunSample(sample_index=index, elapsed_seconds=0.1, items_processed=item_count)
                for index in range(configuration.timed_passes_per_run)
            ),
            summary_fields={
                "logical_decodes_per_pass": item_count * configuration.logical_repeat_factor,
                "logical_repeat_factor": configuration.logical_repeat_factor,
                "num_unique_images": item_count,
            },
            events=(
                {"duration_seconds": 0.01, "event": "validation"},
                {"duration_seconds": 0.01, "event": "warmup"},
                {"event": "measurement_complete", "sample_count": configuration.timed_passes_per_run},
            ),
        ),
    )


if __name__ == "__main__":
    main()
