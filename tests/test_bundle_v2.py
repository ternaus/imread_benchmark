from __future__ import annotations

import json
from pathlib import Path

import pytest

from imread_benchmark.artifacts import BundleValidationError, validate_run_bundle, write_run_bundle
from tests.factories import valid_bundle_data


def test_bundle_is_content_verified_and_summary_is_derived_from_samples(tmp_path: Path) -> None:
    run_key, data = valid_bundle_data()
    bundle = write_run_bundle(
        root=tmp_path / "runs",
        run_key=run_key,
        data=data,
    )

    expected_files = {
        "COMMITTED.json",
        "bundle_manifest.json",
        "config.json",
        "dataset.json",
        "environment.json",
        "events.jsonl",
        "failures.jsonl",
        "platform.json",
        "runtime.json",
        "samples.jsonl",
        "summary.json",
    }
    assert {path.name for path in bundle.iterdir()} == expected_files
    summary = json.loads((bundle / "summary.json").read_text())
    commit = json.loads((bundle / "COMMITTED.json").read_text())
    assert summary["statistics"]["images_per_second"]["mean"] == 1.5
    assert summary["statistics"]["images_per_second"]["sample_std"] == pytest.approx(0.7071067811865476)
    assert commit["run_key"] == run_key
    assert len(commit["bundle_id"]) == 64
    assert (bundle / "failures.jsonl").read_text() == ""
    validate_run_bundle(bundle, expected_run_key=run_key)

    samples_path = bundle / "samples.jsonl"
    samples_path.write_text(samples_path.read_text().replace('"elapsed_seconds": 2.0', '"elapsed_seconds": 3.0'))
    with pytest.raises(BundleValidationError, match="checksum mismatch"):
        validate_run_bundle(bundle, expected_run_key=run_key)


def test_bundle_recomputes_run_identity_across_config_dataset_environment_and_platform(tmp_path: Path) -> None:
    _, data = valid_bundle_data()

    with pytest.raises(BundleValidationError, match=r"run_key.*identity"):
        write_run_bundle(root=tmp_path / "runs", run_key="9" * 64, data=data)
