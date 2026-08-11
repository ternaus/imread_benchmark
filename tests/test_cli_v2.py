from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
from typer.testing import CliRunner

from imread_benchmark.cli import app
from imread_benchmark.datasets.package import build_dataset_package, open_dataset_package

runner = CliRunner()


def test_cli_exposes_only_schema_two_workflows() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "campaign" in result.stdout
    assert "dataset" in result.stdout
    assert "environment" in result.stdout
    assert "render-readme" not in result.stdout


def test_dataset_package_command_builds_a_valid_content_addressed_package(
    tmp_path: Path,
    jpeg_dir: Path,
) -> None:
    result = runner.invoke(
        app,
        [
            "dataset",
            "package",
            "--name",
            "cli-fixture",
            "--workload",
            f"mixed={jpeg_dir}",
            "--output-root",
            str(tmp_path / "packages"),
            "--provenance-source",
            "pytest",
        ],
    )

    assert result.exit_code == 0, result.stdout
    document = json.loads(result.stdout)
    package = open_dataset_package(document["descriptor"])
    assert package.descriptor["schema_version"] == "2.0"
    assert len(package.read_workload_items("mixed")) == 4


def test_controlled_package_command_builds_the_requested_factor_cell(tmp_path: Path) -> None:
    sources = tmp_path / "sources"
    sources.mkdir()
    Image.new("RGB", (64, 48), (20, 40, 60)).save(sources / "scene.png", format="PNG")

    result = runner.invoke(
        app,
        [
            "dataset",
            "controlled-package",
            "--source-dir",
            str(sources),
            "--output-root",
            str(tmp_path / "controlled"),
            "--source-name",
            "fixture-lossless",
            "--source-release",
            "test-release",
            "--source-license",
            "fixture-only",
            "--long-edge",
            "32",
            "--quality",
            "75",
            "--no-native",
            "--compressed-byte-limit",
            str(1024**2),
        ],
    )

    assert result.exit_code == 0, result.stdout
    document = json.loads(result.stdout)
    package = open_dataset_package(document["descriptor"])
    assert set(package.descriptor["workloads"]) == {"controlled-le0032-q075"}


def test_list_decoders_is_machine_readable() -> None:
    result = runner.invoke(app, ["list-decoders"])

    assert result.exit_code == 0
    rows = json.loads(result.stdout)
    assert {row["decoder_id"] for row in rows} >= {"pillow", "opencv"}
    assert all(row["schema_version"] == "2.0" for row in rows)


def test_plan_instantiate_command_generates_valid_workload_plan(tmp_path: Path, jpeg_dir: Path) -> None:
    descriptor_path = build_dataset_package(
        package_name="fixture-jpegs",
        workloads={"fixture": jpeg_dir},
        output_root=tmp_path / "packages",
        provenance={"source": "pytest"},
    )
    template = Path(__file__).parents[1] / "examples" / "fodb-experiment.template.yaml"

    result = runner.invoke(
        app,
        [
            "plan",
            "instantiate",
            str(template),
            "--package-descriptor",
            str(descriptor_path),
            "--output-dir",
            str(tmp_path / "plans"),
            "--workload",
            "fixture",
        ],
    )

    assert result.exit_code == 0, result.stdout
    document = json.loads(result.stdout)
    assert document["schema_version"] == "2.0"
    assert document["plans"][0]["workload_id"] == "fixture"
    assert Path(document["plans"][0]["path"]).is_file()
