from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest

from imread_benchmark.datasets.materializer import LocalObjectStore
from imread_benchmark.datasets.package import build_dataset_package, open_dataset_package
from imread_benchmark.environments import EnvironmentDescriptor, write_environment_descriptor
from imread_benchmark.execution.coordinator import (
    AttemptStatus,
    CoordinatorConfig,
    RemoteCheckpoint,
    execute_run_specs,
)
from imread_benchmark.execution.spec import RunIdentity, RunSpec, RunSpecError, load_run_spec, write_run_spec
from imread_benchmark.plans import RunConfiguration
from imread_benchmark.platforms import PlatformDescriptor, write_platform_descriptor
from imread_benchmark.support import SupportAudit, build_operational_support, write_support_set


def _configuration(  # noqa: PLR0913 - compact factory keeps run-spec tests focused on the varying fields
    *,
    decoder_id: str = "fake",
    package_id: str = "b" * 64,
    manifest_id: str = "c" * 64,
    requested_threads: int | None = None,
    minimum_timed_seconds: float = 0.01,
    protocol_id: str = "decode-memory",
    num_workers: int | None = None,
) -> RunConfiguration:
    is_loader = protocol_id == "loader-supply"
    if is_loader and num_workers is None:
        raise ValueError("loader-supply test configuration requires num_workers")
    return RunConfiguration(
        protocol_id=protocol_id,
        decoder_id=decoder_id,
        package_id=package_id,
        manifest_id=manifest_id,
        selection_id="d" * 64,
        requested_threads=requested_threads,
        num_workers=num_workers if is_loader else None,
        batch_size=1 if is_loader else None,
        prefetch_factor=1 if is_loader and num_workers else None,
        persistent_workers=bool(is_loader and num_workers),
        multiprocessing_start_method="spawn" if is_loader and num_workers else None,
        logical_repeat_factor=1,
        warmup_passes=1,
        timed_passes_per_run=2,
        minimum_timed_seconds=minimum_timed_seconds,
        output_contract="normalized-rgb",
        support_policy="operational",
    )


def _provenance(tmp_path: Path) -> tuple[Path, str, Path, str]:
    environment = EnvironmentDescriptor.build(
        dependency_group="mainstream",
        lock_sha256="8" * 64,
        project_sha256="9" * 64,
        runner_revision="1" * 40,
        python={"abi": "fixture-abi", "implementation": "cpython", "version": "3.12.0"},
        platform_tags=("fixture-platform",),
        distributions=(("imread-benchmark", "0.2.0"), ("pillow", "fixture")),
        native_backends={},
    )
    platform = PlatformDescriptor.build(
        identity={"architecture": "fixture", "logical_cpu_count": 4, "machine_type": "fixture"},
        runtime={"kernel": "fixture"},
    )
    environment_path = write_environment_descriptor(tmp_path / "provenance" / "environment.json", environment)
    platform_path = write_platform_descriptor(tmp_path / "provenance" / "platform.json", platform)
    return environment_path, environment.environment_id, platform_path, platform.platform_id


def test_run_spec_pins_every_identity_input_and_round_trips(tmp_path: Path) -> None:
    environment_path, environment_id, platform_path, platform_id = _provenance(tmp_path)
    identity = RunIdentity(
        plan_id="a" * 64,
        platform_id=platform_id,
        environment_id=environment_id,
        runner_revision="1" * 40,
        workload_id="mixed",
        support_set_id="2" * 64,
        support_item_ids=("item-1", "item-2"),
        configuration=_configuration(requested_threads=4),
        repetition=3,
        block_position=7,
    )
    spec = RunSpec.build(
        identity=identity,
        package_descriptor=tmp_path / "package.json",
        support_set_path=tmp_path / "support-set.json",
        environment_descriptor=environment_path,
        platform_descriptor=platform_path,
    )

    path = write_run_spec(tmp_path / "spec.json", spec)
    loaded = load_run_spec(path)

    assert loaded == spec
    assert loaded.run_key == spec.run_key
    assert len(spec.run_key) == 64
    configuration = spec.to_dict()["configuration"]
    assert isinstance(configuration, dict)
    assert configuration["requested_threads"] == 4
    assert spec.to_dict()["support_item_ids"] == ["item-1", "item-2"]
    assert spec.to_dict()["support_set_path"] == str((tmp_path / "support-set.json").resolve())
    assert spec.to_dict()["environment_descriptor"] == str(environment_path)
    assert spec.to_dict()["platform_descriptor"] == str(platform_path)


def test_run_spec_rejects_a_forged_run_key(tmp_path: Path) -> None:
    environment_path, environment_id, platform_path, platform_id = _provenance(tmp_path)
    identity = RunIdentity(
        plan_id="a" * 64,
        platform_id=platform_id,
        environment_id=environment_id,
        runner_revision="1" * 40,
        workload_id="mixed",
        support_set_id="2" * 64,
        support_item_ids=("item-1",),
        configuration=_configuration(),
        repetition=0,
        block_position=0,
    )
    spec = RunSpec.build(
        identity=identity,
        package_descriptor=tmp_path / "package.json",
        support_set_path=tmp_path / "support-set.json",
        environment_descriptor=environment_path,
        platform_descriptor=platform_path,
    )
    document = spec.to_dict()
    document["run_key"] = "9" * 64

    with pytest.raises(RunSpecError, match="run_key"):
        RunSpec.from_dict(document)


def _run_spec(
    tmp_path: Path,
    *,
    decoder_id: str,
    position: int,
    requested_threads: int | None,
) -> RunSpec:
    environment_path, environment_id, platform_path, platform_id = _provenance(tmp_path)
    return RunSpec.build(
        identity=RunIdentity(
            plan_id="a" * 64,
            platform_id=platform_id,
            environment_id=environment_id,
            runner_revision="1" * 40,
            workload_id="mixed",
            support_set_id="2" * 64,
            support_item_ids=("item-1", "item-2"),
            configuration=_configuration(decoder_id=decoder_id, requested_threads=requested_threads),
            repetition=0,
            block_position=position,
        ),
        package_descriptor=tmp_path / "package.json",
        support_set_path=tmp_path / "support-set.json",
        environment_descriptor=environment_path,
        platform_descriptor=platform_path,
    )


def _support_set_path(
    tmp_path: Path,
    configuration: RunConfiguration,
    item_ids: tuple[str, ...],
) -> tuple[Path, str]:
    uses_worker_process = configuration.protocol_id == "loader-supply" and bool(configuration.num_workers)
    audit = SupportAudit(
        audit_id="7" * 64,
        decoder_id=configuration.decoder_id,
        manifest_id=configuration.manifest_id,
        selection_id=configuration.selection_id,
        process_context="dataloader" if uses_worker_process else "main-process",
        multiprocessing_start_method=configuration.multiprocessing_start_method,
        requested_threads=configuration.requested_threads,
        output_contract="normalized-rgb",
        environment_id="f" * 64,
        platform_id="e" * 64,
        successful_item_ids=item_ids,
        failures=(),
    )
    support_set = build_operational_support(audit)
    return write_support_set(tmp_path / "support", support_set), support_set.support_set_id


def test_coordinator_uses_fresh_processes_isolates_crash_and_resumes(tmp_path: Path) -> None:
    specs = (
        _run_spec(tmp_path, decoder_id="fake", position=0, requested_threads=4),
        _run_spec(tmp_path, decoder_id="hard-crash", position=1, requested_threads=8),
        _run_spec(tmp_path, decoder_id="fake", position=2, requested_threads=None),
    )
    fake_worker = Path(__file__).parent / "fixtures" / "fake_run_worker.py"
    artifact_root = tmp_path / "artifacts"

    results = execute_run_specs(
        specs,
        CoordinatorConfig(
            artifact_root=artifact_root,
            attempts_root=tmp_path / "attempts",
            timeout_seconds=10,
            worker_command=(sys.executable, str(fake_worker)),
        ),
    )

    assert [result.status for result in results] == [
        AttemptStatus.COMPLETED,
        AttemptStatus.FAILED,
        AttemptStatus.COMPLETED,
    ]
    runtimes = [
        json.loads((artifact_root / "runs" / spec.run_key / "runtime.json").read_text())
        for spec in (specs[0], specs[2])
    ]
    assert runtimes[0]["process_id"] != runtimes[1]["process_id"]
    assert runtimes[0]["effective_threads"] == 4
    assert runtimes[1]["effective_threads"] == 17
    failed_attempt = results[1].attempt_directory
    assert failed_attempt is not None
    assert (failed_attempt / "status.json").is_file()
    assert not (artifact_root / "runs" / specs[1].run_key).exists()

    completed_mtime = (artifact_root / "runs" / specs[0].run_key / "COMMITTED.json").stat().st_mtime_ns
    resumed = execute_run_specs(
        specs,
        CoordinatorConfig(
            artifact_root=artifact_root,
            attempts_root=tmp_path / "attempts",
            timeout_seconds=10,
            worker_command=(sys.executable, str(fake_worker)),
        ),
    )
    assert [result.status for result in resumed] == [
        AttemptStatus.SKIPPED,
        AttemptStatus.FAILED,
        AttemptStatus.SKIPPED,
    ]
    assert (artifact_root / "runs" / specs[0].run_key / "COMMITTED.json").stat().st_mtime_ns == completed_mtime


def test_coordinator_resumes_committed_runs_on_a_fresh_machine(tmp_path: Path) -> None:
    specs = tuple(
        _run_spec(tmp_path, decoder_id="fake", position=position, requested_threads=None) for position in range(3)
    )
    fake_worker = Path(__file__).parent / "fixtures" / "fake_run_worker.py"
    store = LocalObjectStore(tmp_path / "object-store")
    first_machine = tmp_path / "machine-one"
    (first_result,) = execute_run_specs(
        (specs[0],),
        CoordinatorConfig(
            artifact_root=first_machine,
            attempts_root=tmp_path / "attempts-one",
            timeout_seconds=10,
            worker_command=(sys.executable, str(fake_worker)),
            remote=RemoteCheckpoint(store),
        ),
    )
    assert first_result.status is AttemptStatus.COMPLETED
    first_bundle = first_machine / "runs" / specs[0].run_key
    first_bundle_id = json.loads((first_bundle / "COMMITTED.json").read_text())["bundle_id"]

    second_machine = tmp_path / "machine-two"
    resumed = execute_run_specs(
        specs,
        CoordinatorConfig(
            artifact_root=second_machine,
            attempts_root=tmp_path / "attempts-two",
            timeout_seconds=10,
            worker_command=(sys.executable, str(fake_worker)),
            remote=RemoteCheckpoint(store),
        ),
    )

    assert [result.status for result in resumed] == [
        AttemptStatus.SKIPPED,
        AttemptStatus.COMPLETED,
        AttemptStatus.COMPLETED,
    ]
    resumed_bundle_id = json.loads(
        (second_machine / "runs" / specs[0].run_key / "COMMITTED.json").read_text(),
    )["bundle_id"]
    assert resumed_bundle_id == first_bundle_id


def test_coordinator_timeout_terminates_the_worker_process_group(tmp_path: Path) -> None:
    spec = _run_spec(tmp_path, decoder_id="slow", position=0, requested_threads=None)
    fake_worker = Path(__file__).parent / "fixtures" / "fake_run_worker.py"

    (result,) = execute_run_specs(
        (spec,),
        CoordinatorConfig(
            artifact_root=tmp_path / "artifacts",
            attempts_root=tmp_path / "attempts",
            timeout_seconds=0.2,
            worker_command=(sys.executable, str(fake_worker)),
        ),
    )

    assert result.status is AttemptStatus.TIMED_OUT
    assert result.attempt_directory is not None
    assert (result.attempt_directory / "heartbeat.json").is_file()
    child = json.loads((result.attempt_directory / "stdout.log").read_text())
    child_pid = child["child_process_id"]
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        pytest.fail(f"child process {child_pid} survived coordinator timeout")
    assert not (tmp_path / "artifacts" / "runs" / spec.run_key).exists()


def test_production_worker_runs_decode_memory_from_resident_tar_bytes(tmp_path: Path, jpeg_dir: Path) -> None:
    descriptor_path = build_dataset_package(
        package_name="worker-fixture",
        workloads={"fixture": jpeg_dir},
        output_root=tmp_path / "packages",
        provenance={"source": "pytest"},
    )
    package = open_dataset_package(descriptor_path)
    workload = package.descriptor["workloads"]["fixture"]
    assert isinstance(workload, dict)
    package_id = package.descriptor["package_id"]
    manifest_id = workload["manifest_id"]
    assert isinstance(package_id, str)
    assert isinstance(manifest_id, str)
    item_ids = tuple(item.item_id for item in package.read_workload_items("fixture"))
    configuration = _configuration(
        decoder_id="pillow",
        package_id=package_id,
        manifest_id=manifest_id,
        requested_threads=None,
        minimum_timed_seconds=1e-9,
    )
    support_set_path, support_set_id = _support_set_path(tmp_path, configuration, item_ids)
    environment_path, environment_id, platform_path, platform_id = _provenance(tmp_path)
    spec = RunSpec.build(
        identity=RunIdentity(
            plan_id="a" * 64,
            platform_id=platform_id,
            environment_id=environment_id,
            runner_revision="1" * 40,
            workload_id="fixture",
            support_set_id=support_set_id,
            support_item_ids=item_ids,
            configuration=configuration,
            repetition=0,
            block_position=0,
        ),
        package_descriptor=descriptor_path,
        support_set_path=support_set_path,
        environment_descriptor=environment_path,
        platform_descriptor=platform_path,
    )

    (result,) = execute_run_specs(
        (spec,),
        CoordinatorConfig(
            artifact_root=tmp_path / "artifacts",
            attempts_root=tmp_path / "attempts",
            timeout_seconds=10,
        ),
    )

    assert result.status is AttemptStatus.COMPLETED
    bundle = tmp_path / "artifacts" / "runs" / spec.run_key
    runtime = json.loads((bundle / "runtime.json").read_text())
    dataset = json.loads((bundle / "dataset.json").read_text())
    samples = (bundle / "samples.jsonl").read_text().splitlines()
    assert runtime["process_id"] == result.process_id
    assert dataset["ordered_item_ids"] == list(item_ids)
    assert len(samples) == configuration.timed_passes_per_run


@pytest.mark.parametrize("num_workers", [0, 2])
def test_production_worker_exercises_real_loader_profiles(
    tmp_path: Path,
    jpeg_dir: Path,
    num_workers: int,
) -> None:
    descriptor_path = build_dataset_package(
        package_name="loader-worker-fixture",
        workloads={"fixture": jpeg_dir},
        output_root=tmp_path / "packages",
        provenance={"source": "pytest"},
    )
    package = open_dataset_package(descriptor_path)
    workload = package.descriptor["workloads"]["fixture"]
    assert isinstance(workload, dict)
    package_id = package.descriptor["package_id"]
    manifest_id = workload["manifest_id"]
    assert isinstance(package_id, str)
    assert isinstance(manifest_id, str)
    item_ids = tuple(item.item_id for item in package.read_workload_items("fixture"))
    configuration = _configuration(
        decoder_id="pillow",
        package_id=package_id,
        manifest_id=manifest_id,
        requested_threads=None,
        minimum_timed_seconds=1e-9,
        protocol_id="loader-supply",
        num_workers=num_workers,
    )
    support_set_path, support_set_id = _support_set_path(tmp_path, configuration, item_ids)
    environment_path, environment_id, platform_path, platform_id = _provenance(tmp_path)
    spec = RunSpec.build(
        identity=RunIdentity(
            plan_id="a" * 64,
            platform_id=platform_id,
            environment_id=environment_id,
            runner_revision="1" * 40,
            workload_id="fixture",
            support_set_id=support_set_id,
            support_item_ids=item_ids,
            configuration=configuration,
            repetition=0,
            block_position=0,
        ),
        package_descriptor=descriptor_path,
        support_set_path=support_set_path,
        environment_descriptor=environment_path,
        platform_descriptor=platform_path,
    )

    (result,) = execute_run_specs(
        (spec,),
        CoordinatorConfig(
            artifact_root=tmp_path / "artifacts",
            attempts_root=tmp_path / "attempts",
            timeout_seconds=30,
        ),
    )

    assert result.status is AttemptStatus.COMPLETED
    bundle = tmp_path / "artifacts" / "runs" / spec.run_key
    runtime = json.loads((bundle / "runtime.json").read_text())
    handshakes = runtime["worker_handshakes"]
    expected_processes = max(1, num_workers)
    assert len(handshakes) == expected_processes
    assert len({row["process_id"] for row in handshakes}) == expected_processes
    if num_workers == 0:
        assert handshakes[0]["process_id"] == runtime["process_id"]
    else:
        assert all(row["process_id"] != runtime["process_id"] for row in handshakes)
    assert all(row["effective_threads"] == 1 for row in handshakes)
    assert all(row["multiprocessing_start_method"] for row in handshakes)
    assert runtime["persistent_workers_reused"] is True
