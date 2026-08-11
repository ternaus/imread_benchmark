from __future__ import annotations

from pathlib import Path

import pytest

from imread_benchmark.analysis import ClaimScope, assert_claim_scope, load_bundles
from imread_benchmark.artifacts import write_run_bundle
from tests.factories import valid_bundle_data


def _write_analysis_bundle(root: Path, *, protocol_id: str = "decode-memory") -> Path:
    run_key, data = valid_bundle_data(protocol_id=protocol_id)
    return write_run_bundle(
        root=root / "runs",
        run_key=run_key,
        data=data,
    )


def test_canonical_loader_keeps_raw_samples_events_runtime_and_empty_failure_log(tmp_path: Path) -> None:
    bundle = _write_analysis_bundle(tmp_path)
    incomplete = tmp_path / "runs" / ("9" * 64)
    incomplete.mkdir()
    (incomplete / "summary.json").write_text("{}")

    (record,) = load_bundles(tmp_path)

    assert record.path == bundle
    assert record.run_key == bundle.name
    assert record.config["decoder_id"] == "pillow"
    assert [sample.elapsed_seconds for sample in record.samples] == [2.0, 1.0]
    assert record.failures == ()
    assert record.events[0]["event"] == "validation"
    assert isinstance(record.runtime["process_id"], int)
    assert record.summary["statistics"]["images_per_second"]["sample_std"] == pytest.approx(
        0.7071067811865476,
    )


def test_claim_gate_rejects_training_claims_from_loader_only_evidence(tmp_path: Path) -> None:
    _write_analysis_bundle(tmp_path, protocol_id="loader-supply")
    records = load_bundles(tmp_path)

    assert_claim_scope(records, ClaimScope.LOADER_SUPPLY)
    with pytest.raises(ValueError, match="training"):
        assert_claim_scope(records, ClaimScope.TRAINING)
    with pytest.raises(ValueError, match="decode-memory"):
        assert_claim_scope(records, ClaimScope.DECODER_CAPACITY)
