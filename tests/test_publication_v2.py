from __future__ import annotations

import json
from pathlib import Path

import pytest

from imread_benchmark.analysis.publication import PublicationError, publish
from imread_benchmark.artifacts import write_run_bundle
from tests.factories import valid_bundle_data


def test_publication_is_deterministic_claim_gated_and_traceable_to_raw_bundles(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    run_keys: list[str] = []
    for position, repetition, decoder_id, elapsed in (
        (0, 0, "pillow", (2.0, 1.0)),
        (1, 0, "opencv", (1.0, 0.5)),
        (2, 1, "pillow", (1.0, 0.5)),
        (3, 1, "opencv", (0.5, 0.25)),
    ):
        run_key, data = valid_bundle_data(
            decoder_id=decoder_id,
            block_position=position,
            repetition=repetition,
            elapsed_seconds=elapsed,
        )
        run_keys.append(run_key)
        write_run_bundle(root=artifact_root / "runs", run_key=run_key, data=data)
    spec = tmp_path / "publication.yaml"
    spec.write_text(
        """\
schema_version: "2.0"
claim_scope: decoder-capacity
filters:
  config.protocol_id: decode-memory
  dataset.workload_id: fixture
statistic: images_per_second
practical_margin_percent: 5
generator_revision: "1111111111111111111111111111111111111111"
""",
    )
    output = tmp_path / "generated"

    publish(artifact_root=artifact_root, spec_path=spec, output_dir=output)

    table = json.loads((output / "results.json").read_text())
    provenance = json.loads((output / "provenance.json").read_text())
    assert [row["run_key"] for row in table["rows"]] == sorted(run_keys)
    assert table["rows"][0]["raw_samples"]
    assert len(table["groups"]) == 2
    assert all(group["n"] == 2 for group in table["groups"])
    assert all(len(group["raw_run_means"]) == 2 for group in table["groups"])
    assert all(group["sample_std"] is not None for group in table["groups"])
    assert provenance["bundle_keys"] == sorted(run_keys)
    assert provenance["claim_scope"] == "decoder-capacity"
    assert len(provenance["output_sha256"]) == 64
    publish(artifact_root=artifact_root, spec_path=spec, output_dir=output, check=True)

    (output / "results.json").write_text("{}")
    with pytest.raises(PublicationError, match="stale"):
        publish(artifact_root=artifact_root, spec_path=spec, output_dir=output, check=True)


def test_publication_rejects_training_claim_scope_for_decoder_bundles(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    run_key, data = valid_bundle_data()
    write_run_bundle(root=artifact_root / "runs", run_key=run_key, data=data)
    spec = tmp_path / "publication.yaml"
    spec.write_text(
        """\
schema_version: "2.0"
claim_scope: training
filters: {}
statistic: images_per_second
practical_margin_percent: 5
generator_revision: "1111111111111111111111111111111111111111"
""",
    )

    with pytest.raises(ValueError, match="training"):
        publish(artifact_root=artifact_root, spec_path=spec, output_dir=tmp_path / "generated")
