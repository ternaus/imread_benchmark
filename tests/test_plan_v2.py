from __future__ import annotations

import json
from pathlib import Path

import pytest

from imread_benchmark.datasets.package import build_dataset_package
from imread_benchmark.plans import (
    PlanError,
    RunConfiguration,
    expand_experiment_plan,
    instantiate_experiment_plans,
    load_experiment_plan,
)


def test_plan_pins_exact_package_workload_manifest_and_selection(
    tmp_path: Path,
    jpeg_dir: Path,
) -> None:
    descriptor_path = build_dataset_package(
        package_name="fixture-jpegs",
        workloads={"fixture": jpeg_dir},
        output_root=tmp_path / "packages",
        provenance={"source": "pytest"},
    )
    descriptor = json.loads(descriptor_path.read_text())
    plan_path = tmp_path / "experiment.yaml"
    plan_path.write_text(_plan_yaml(descriptor_path, descriptor))

    plan = load_experiment_plan(plan_path)
    workload = descriptor["workloads"]["fixture"]

    assert plan.schema_version == "2.0"
    assert plan.dataset.package_id == descriptor["package_id"]
    assert plan.dataset.manifest_id == workload["manifest_id"]
    assert plan.dataset.selection.item_ids == tuple(item["item_id"] for item in plan.dataset.manifest["items"])
    assert plan.dataset.selection.selection_id


def test_plan_requires_one_fresh_subprocess_per_run(tmp_path: Path, jpeg_dir: Path) -> None:
    descriptor_path = build_dataset_package(
        package_name="fixture-jpegs",
        workloads={"fixture": jpeg_dir},
        output_root=tmp_path / "packages",
        provenance={"source": "pytest"},
    )
    descriptor = json.loads(descriptor_path.read_text())
    plan_path = tmp_path / "experiment.yaml"
    plan_path.write_text(
        _plan_yaml(descriptor_path, descriptor).replace("per_run_subprocess: true", "per_run_subprocess: false"),
    )

    with pytest.raises(PlanError, match="per_run_subprocess"):
        load_experiment_plan(plan_path)


def test_plan_requires_an_explicit_memory_fraction(tmp_path: Path, jpeg_dir: Path) -> None:
    descriptor_path = build_dataset_package(
        package_name="fixture-jpegs",
        workloads={"fixture": jpeg_dir},
        output_root=tmp_path / "packages",
        provenance={"source": "pytest"},
    )
    descriptor = json.loads(descriptor_path.read_text())
    plan_path = tmp_path / "experiment.yaml"
    plan_path.write_text(_plan_yaml(descriptor_path, descriptor).replace("  maximum_memory_fraction: 0.6\n", ""))

    with pytest.raises(PlanError, match="maximum_memory_fraction"):
        load_experiment_plan(plan_path)


def test_plan_rejects_requested_threads_for_a_fixed_thread_decoder(tmp_path: Path, jpeg_dir: Path) -> None:
    descriptor_path = build_dataset_package(
        package_name="fixture-jpegs",
        workloads={"fixture": jpeg_dir},
        output_root=tmp_path / "packages",
        provenance={"source": "pytest"},
    )
    descriptor = json.loads(descriptor_path.read_text())
    plan_path = tmp_path / "experiment.yaml"
    plan_path.write_text(_plan_yaml(descriptor_path, descriptor).replace("threads: [default]", "threads: [1]"))

    with pytest.raises(PlanError, match="thread control"):
        expand_experiment_plan(load_experiment_plan(plan_path))


def test_plan_accepts_a_verified_materialized_descriptor_override(tmp_path: Path, jpeg_dir: Path) -> None:
    descriptor_path = build_dataset_package(
        package_name="fixture-jpegs",
        workloads={"fixture": jpeg_dir},
        output_root=tmp_path / "packages",
        provenance={"source": "pytest"},
    )
    descriptor = json.loads(descriptor_path.read_text())
    plan_path = tmp_path / "experiment.yaml"
    plan_text = _plan_yaml(descriptor_path, descriptor).replace(str(descriptor_path), "unavailable/package.json")
    plan_path.write_text(plan_text)

    plan = load_experiment_plan(plan_path, dataset_descriptor=descriptor_path)

    assert plan.dataset.descriptor_path == descriptor_path.resolve()


def test_plan_instantiation_fills_and_validates_every_package_workload(
    tmp_path: Path,
    jpeg_dir: Path,
) -> None:
    descriptor_path = build_dataset_package(
        package_name="fixture-jpegs",
        workloads={"native": jpeg_dir, "mixed": jpeg_dir},
        output_root=tmp_path / "packages",
        provenance={"source": "pytest"},
    )
    template = Path(__file__).parents[1] / "examples" / "fodb-experiment.template.yaml"

    instantiated = instantiate_experiment_plans(
        template_path=template,
        package_descriptor=descriptor_path,
        output_dir=tmp_path / "plans",
    )

    assert tuple(item.workload_id for item in instantiated) == ("mixed", "native")
    assert all(item.run_count > 0 for item in instantiated)
    assert all(item.path.is_file() for item in instantiated)
    for item in instantiated:
        document = item.path.read_text()
        assert "PACKAGE_ID" not in document
        assert "WORKLOAD_ID" not in document
        assert "MANIFEST_ID" not in document
        assert "ITEM_COUNT" not in document
        plan = load_experiment_plan(item.path, dataset_descriptor=descriptor_path)
        assert plan.dataset.workload_id == item.workload_id
        assert expand_experiment_plan(plan)[0].plan_id == item.plan_id


def test_plan_instantiation_preserves_existing_output_when_validation_fails(
    tmp_path: Path,
    jpeg_dir: Path,
) -> None:
    descriptor_path = build_dataset_package(
        package_name="fixture-jpegs",
        workloads={"fixture": jpeg_dir},
        output_root=tmp_path / "packages",
        provenance={"source": "pytest"},
    )
    source_template = Path(__file__).parents[1] / "examples" / "fodb-experiment.template.yaml"
    invalid_template = tmp_path / "invalid.template.yaml"
    invalid_template.write_text(
        source_template.read_text().replace("per_run_subprocess: true", "per_run_subprocess: false"),
    )
    output_dir = tmp_path / "plans"
    output_dir.mkdir()
    existing = output_dir / "fixture.yaml"
    existing.write_text("previous plan\n")

    with pytest.raises(PlanError, match="per_run_subprocess"):
        instantiate_experiment_plans(
            template_path=invalid_template,
            package_descriptor=descriptor_path,
            output_dir=output_dir,
        )

    assert existing.read_text() == "previous plan\n"


@pytest.mark.parametrize(
    "template_name",
    [
        "controlled-ablation.template.yaml",
        "fodb-experiment.template.yaml",
        "fodb-tensorflow.template.yaml",
    ],
)
def test_checked_in_plan_templates_expand_against_a_real_package(
    tmp_path: Path,
    jpeg_dir: Path,
    template_name: str,
) -> None:
    descriptor_path = build_dataset_package(
        package_name="fixture-jpegs",
        workloads={"fixture": jpeg_dir},
        output_root=tmp_path / "packages",
        provenance={"source": "pytest"},
    )
    template_path = Path(__file__).parents[1] / "examples" / template_name
    instantiated = instantiate_experiment_plans(
        template_path=template_path,
        package_descriptor=descriptor_path,
        output_dir=tmp_path / "plans",
        workload_ids=("fixture",),
    )
    templates = expand_experiment_plan(
        load_experiment_plan(instantiated[0].path, dataset_descriptor=descriptor_path),
    )

    assert templates
    assert {template.configuration.package_id for template in templates} == {
        json.loads(descriptor_path.read_text())["package_id"],
    }


def test_plan_expands_worker_profiles_in_seeded_repetition_blocks(tmp_path: Path, jpeg_dir: Path) -> None:
    descriptor_path = build_dataset_package(
        package_name="fixture-jpegs",
        workloads={"fixture": jpeg_dir},
        output_root=tmp_path / "packages",
        provenance={"source": "pytest"},
    )
    descriptor = json.loads(descriptor_path.read_text())
    plan_path = tmp_path / "experiment.yaml"
    plan_path.write_text(
        _plan_yaml(descriptor_path, descriptor).replace(
            "    decode-memory: {}",
            """\
    decode-memory: {}
    loader-supply:
      worker_profiles:
        - workers: [0]
          batch_size: 1
        - workers: [2, 4]
          batch_size: 1
          multiprocessing_start_method: spawn
          prefetch_factor: 1
          persistent_workers: true""",
        ),
    )

    plan = load_experiment_plan(plan_path)
    first = expand_experiment_plan(plan)
    second = expand_experiment_plan(plan)

    assert first == second
    assert len(first) == 8  # 1 decode + 3 loader configurations, repeated twice
    assert [template.position for template in first] == list(range(8))
    for repetition in range(2):
        block = [template for template in first if template.repetition == repetition]
        assert len(block) == 4
        assert len({template.configuration.config_id for template in block}) == 4
    zero_worker = next(template.configuration for template in first if template.configuration.num_workers == 0)
    assert zero_worker.prefetch_factor is None
    assert zero_worker.persistent_workers is False
    assert zero_worker.multiprocessing_start_method is None
    process_worker = next(template.configuration for template in first if template.configuration.num_workers == 2)
    assert process_worker.multiprocessing_start_method == "spawn"


def test_run_configuration_rejects_protocol_fields_that_do_not_apply() -> None:
    with pytest.raises(ValueError, match="decode-memory"):
        RunConfiguration(
            protocol_id="decode-memory",
            decoder_id="pillow",
            package_id="a" * 64,
            manifest_id="b" * 64,
            selection_id="c" * 64,
            requested_threads=None,
            num_workers=2,
            batch_size=1,
            prefetch_factor=1,
            persistent_workers=True,
            multiprocessing_start_method="spawn",
            logical_repeat_factor=1,
            warmup_passes=1,
            timed_passes_per_run=2,
            minimum_timed_seconds=0.1,
            output_contract="normalized-rgb",
            support_policy="common",
        )


def test_loader_process_configuration_requires_explicit_start_method() -> None:
    with pytest.raises(ValueError, match="multiprocessing_start_method"):
        RunConfiguration(
            protocol_id="loader-supply",
            decoder_id="pillow",
            package_id="a" * 64,
            manifest_id="b" * 64,
            selection_id="c" * 64,
            requested_threads=None,
            num_workers=2,
            batch_size=1,
            prefetch_factor=1,
            persistent_workers=True,
            multiprocessing_start_method=None,
            logical_repeat_factor=1,
            warmup_passes=1,
            timed_passes_per_run=2,
            minimum_timed_seconds=0.1,
            output_contract="normalized-rgb",
            support_policy="common",
        )


def _plan_yaml(descriptor_path: Path, descriptor: dict[str, object]) -> str:
    workload = descriptor["workloads"]["fixture"]
    return f"""\
schema_version: "2.0"
experiment_name: fixture-study
seed: 42
repetitions: 2
dataset:
  descriptor: "{descriptor_path}"
  package_id: "{descriptor["package_id"]}"
  workload_id: fixture
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
measurement:
  warmup_passes: 1
  timed_passes_per_run: 1
  minimum_timed_seconds: 0.01
  output_contract: normalized-rgb
  support_policy: common
execution:
  per_run_subprocess: true
  run_timeout_seconds: 30
  checkpoint_each_run: true
  maximum_memory_fraction: 0.6
"""
