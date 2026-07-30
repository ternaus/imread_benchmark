from __future__ import annotations

import json
import sys
from pathlib import Path

from imread_benchmark.datasets.materializer import LocalObjectStore
from imread_benchmark.datasets.package import build_dataset_package
from imread_benchmark.environments import EnvironmentDescriptor, write_environment_descriptor
from imread_benchmark.execution.campaign import CampaignConfig, run_campaign
from imread_benchmark.execution.coordinator import AttemptStatus, RemoteCheckpoint
from imread_benchmark.platforms import PlatformDescriptor, write_platform_descriptor


def test_campaign_runs_support_then_isolated_matrix_and_resumes_on_fresh_machine(
    tmp_path: Path,
    jpeg_dir: Path,
) -> None:
    descriptor_path = build_dataset_package(
        package_name="campaign-fixture",
        workloads={"mixed": jpeg_dir},
        output_root=tmp_path / "packages",
        provenance={"source": "pytest"},
    )
    descriptor = json.loads(descriptor_path.read_text())
    workload = descriptor["workloads"]["mixed"]
    plan_path = tmp_path / "experiment.yaml"
    plan_path.write_text(_plan_yaml(descriptor, workload))
    environment = EnvironmentDescriptor.build(
        dependency_group="mainstream",
        lock_sha256="1" * 64,
        project_sha256="2" * 64,
        runner_revision="3" * 40,
        python={"abi": "fixture", "implementation": "cpython", "version": "3.12.0"},
        platform_tags=("fixture",),
        distributions=(("imread-benchmark", "0.2.0"), ("pillow", "fixture"), ("torch", "fixture")),
        native_backends={},
    )
    platform = PlatformDescriptor.build(
        identity={"architecture": "fixture", "logical_cpu_count": 4, "machine_type": "fixture"},
        runtime={"available_multiprocessing_start_methods": ["spawn"], "memory_bytes": 1024**3},
    )
    environment_path = write_environment_descriptor(tmp_path / "environment.json", environment)
    platform_path = write_platform_descriptor(tmp_path / "platform.json", platform)
    remote = RemoteCheckpoint(LocalObjectStore(tmp_path / "object-store"))

    first = run_campaign(
        CampaignConfig(
            plan_path=plan_path,
            package_descriptor=descriptor_path,
            environment_descriptor=environment_path,
            platform_descriptor=platform_path,
            artifact_root=tmp_path / "machine-one" / "artifacts",
            attempts_root=tmp_path / "machine-one" / "attempts",
            runner_revision="3" * 40,
            worker_python=Path(sys.executable),
            remote=remote,
        ),
    )
    second = run_campaign(
        CampaignConfig(
            plan_path=plan_path,
            package_descriptor=descriptor_path,
            environment_descriptor=environment_path,
            platform_descriptor=platform_path,
            artifact_root=tmp_path / "machine-two" / "artifacts",
            attempts_root=tmp_path / "machine-two" / "attempts",
            runner_revision="3" * 40,
            worker_python=Path(sys.executable),
            remote=remote,
        ),
    )

    assert first.complete is True
    assert [result.status for result in first.run_results] == [
        AttemptStatus.COMPLETED,
        AttemptStatus.COMPLETED,
        AttemptStatus.COMPLETED,
    ]
    assert [result.status for result in second.run_results] == [
        AttemptStatus.SKIPPED,
        AttemptStatus.SKIPPED,
        AttemptStatus.SKIPPED,
    ]
    assert first.support_audit_count == 2
    assert first.support_set_ids == second.support_set_ids
    assert len(first.support_set_ids) == 2


def _plan_yaml(descriptor: dict[str, object], workload: dict[str, object]) -> str:
    return f"""\
schema_version: "2.0"
experiment_name: campaign-fixture
seed: 7
repetitions: 1
dataset:
  descriptor: unavailable/package.json
  package_id: "{descriptor["package_id"]}"
  workload_id: mixed
  manifest_id: "{workload["manifest_id"]}"
  selection:
    method: all
    expected_items: 4
  logical_repeat_factor: 1
matrix:
  decoders:
    pillow:
      threads: [default]
  protocols:
    decode-memory: {{}}
    loader-supply:
      worker_profiles:
        - workers: [0]
          batch_size: 1
        - workers: [2]
          batch_size: 1
          multiprocessing_start_method: spawn
          prefetch_factor: 1
          persistent_workers: true
measurement:
  warmup_passes: 1
  timed_passes_per_run: 2
  minimum_timed_seconds: 0.000000001
  output_contract: normalized-rgb
  support_policy: common
execution:
  per_run_subprocess: true
  run_timeout_seconds: 30
  checkpoint_each_run: true
  maximum_memory_fraction: 0.6
"""
