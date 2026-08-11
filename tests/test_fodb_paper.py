from __future__ import annotations

import pytest

from imread_benchmark.analysis.fodb_paper import (
    COMPATIBILITY_AUDIT_EMPTY_DHT_ITEM_COUNT,
    COMPATIBILITY_AUDIT_EMPTY_DHT_SUCCESSES,
    COMPATIBILITY_AUDIT_FOUR_COMPONENT_ITEM_COUNT,
    COMPATIBILITY_AUDIT_FOUR_COMPONENT_SUCCESSES,
    COMPATIBILITY_AUDIT_ITEM_COUNT,
    COMPATIBILITY_AUDIT_SUCCESSES,
    WORKER12_DECODERS,
    WORKER12_PLAN_SCOPES,
    WORKER16_DECODERS,
    WORKER16_PLAN_SCOPES,
    Aggregate,
    _compatibility_audit,
    _decoder_coverage_table,
    _linear_quantile,
    _ranks,
    _worker16_candidate_rows,
    _worker_transfer_table,
    _workload_descriptors,
)


def _package_item(*, sha256: str, provenance: str, quality: int, progressive: bool) -> dict[str, object]:
    return {
        "bits_per_pixel": 1.5,
        "jpeg": {
            "megapixels": 2.0,
            "progressive": progressive,
            "quality_estimate": quality,
            "subsampling": "4:2:0",
        },
        "jpeg_bytes": 100,
        "jpeg_sha256": sha256,
        "provenance": provenance,
    }


def test_workload_descriptors_distinguish_manifest_from_timed_common_support() -> None:
    native = _package_item(sha256="native", provenance="orig", quality=95, progressive=False)
    social = _package_item(sha256="social", provenance="whatsapp", quality=69, progressive=True)
    package = {"provenance": {"items": [native, social]}}
    manifests = {
        "fodb-native": [{"item_id": "native-id", "sha256": "native"}],
        "fodb-mixed": [
            {"item_id": "native-id", "sha256": "native"},
            {"item_id": "social-id", "sha256": "social"},
        ],
    }
    support = {"fodb-native": {"native-id"}, "fodb-mixed": {"native-id"}}

    descriptors = _workload_descriptors(package, manifests, support)

    native_row, mixed_row = descriptors
    assert native_row["items"] == native_row["manifest_items"] == 1
    assert native_row["provenance"] == {"orig": 1}
    assert mixed_row["items"] == 1
    assert mixed_row["manifest_items"] == 2
    assert mixed_row["provenance"] == {"orig": 1}
    assert mixed_row["excluded_items"] == 1
    assert mixed_row["excluded_profile"] == {
        "estimated_quality": {69.0: 1},
        "progressive_items": 1,
        "provenance": {"whatsapp": 1},
    }


def test_rank_and_quantile_helpers_are_deterministic() -> None:
    assert _ranks({"a": 3.0, "b": 2.0, "c": 2.0, "d": 1.0}) == {
        "a": 1.0,
        "b": 2.5,
        "c": 2.5,
        "d": 4.0,
    }
    assert _linear_quantile([1.0, 2.0, 3.0], 0.25) == pytest.approx(1.5)


def test_workers_12_followup_is_frozen_to_87_cells() -> None:
    assert sum(len(decoders) for decoders in WORKER12_DECODERS.values()) == 87
    assert sum(len(decoders) * 5 for decoders in WORKER12_DECODERS.values()) == 435
    plan_scopes = {
        cell: decoders for scoped_cells in WORKER12_PLAN_SCOPES.values() for cell, decoders in scoped_cells.items()
    }
    assert plan_scopes == WORKER12_DECODERS

    worker16_scopes = {
        cell: decoders for scoped_cells in WORKER16_PLAN_SCOPES.values() for cell, decoders in scoped_cells.items()
    }
    assert worker16_scopes == {cell: decoders for cell, decoders in WORKER16_DECODERS.items() if decoders}
    assert sum(len(decoders) for decoders in WORKER16_DECODERS.values()) == 58
    assert sum(len(decoders) * 5 for decoders in WORKER16_DECODERS.values()) == 290


def test_workers_16_candidates_require_material_and_consistent_workers_12_gain() -> None:
    aggregates = []
    means = {
        "pillow": {0: 50.0, 2: 70.0, 4: 80.0, 8: 90.0, 12: 95.0},
        "opencv": {0: 50.0, 2: 70.0, 4: 80.0, 8: 90.0, 12: 85.0},
        "simplejpeg": {0: 50.0, 2: 70.0, 4: 80.0, 8: 90.0, 12: 92.0},
        "kornia": {0: 50.0, 2: 70.0, 4: 80.0, 8: 90.0, 12: 96.0},
    }
    raw_run_means = {"kornia": (100.0, 100.0, 100.0, 100.0, 80.0)}
    for decoder, worker_means in means.items():
        for workers, mean in worker_means.items():
            aggregates.append(
                Aggregate(
                    workload="fodb-native",
                    machine_type="c4-standard-16",
                    protocol="loader-supply",
                    decoder=decoder,
                    requested_threads=1 if decoder == "opencv" else None,
                    workers=workers,
                    repetitions=(0, 1, 2, 3, 4),
                    raw_run_means=raw_run_means.get(decoder, (mean,) * 5) if workers == 12 else (mean,) * 5,
                    mean=mean,
                    sample_std=0.0,
                ),
            )

    candidates = _worker16_candidate_rows(
        tuple(aggregates),
        {("fodb-native", "c4-standard-16"): ("pillow", "opencv", "simplejpeg", "kornia")},
    )

    assert candidates["candidate_count"] == 1
    assert candidates["bundle_count_if_launched"] == 5
    assert candidates["minimum_mean_gain_percent"] == 5.0
    assert [row["decoder"] for row in candidates["cells"][0]["decoders"]] == ["pillow"]


def test_compatibility_audit_reports_bitstream_and_output_contract_separately() -> None:
    audit = _compatibility_audit()

    assert COMPATIBILITY_AUDIT_ITEM_COUNT == (
        COMPATIBILITY_AUDIT_EMPTY_DHT_ITEM_COUNT + COMPATIBILITY_AUDIT_FOUR_COMPONENT_ITEM_COUNT
    )
    assert set(COMPATIBILITY_AUDIT_SUCCESSES) == set(COMPATIBILITY_AUDIT_EMPTY_DHT_SUCCESSES)
    assert set(COMPATIBILITY_AUDIT_SUCCESSES) == set(COMPATIBILITY_AUDIT_FOUR_COMPONENT_SUCCESSES)
    for decoder, combined_successes in COMPATIBILITY_AUDIT_SUCCESSES.items():
        assert combined_successes == (
            COMPATIBILITY_AUDIT_EMPTY_DHT_SUCCESSES[decoder] + COMPATIBILITY_AUDIT_FOUR_COMPONENT_SUCCESSES[decoder]
        )

    assert audit["item_count"] == COMPATIBILITY_AUDIT_ITEM_COUNT
    assert audit["successes"] == COMPATIBILITY_AUDIT_SUCCESSES
    table = _decoder_coverage_table(audit)

    assert r"\texttt{ajpegli} & 0/276 & 0/1 & 0/277" in table
    assert r"\texttt{torchvision} & 276/276 & 0/1 & 276/277" in table
    assert r"\texttt{simplejpeg} & 276/276 & 1/1 & 277/277" in table


def test_worker_transfer_table_keeps_counts_readable_and_compact() -> None:
    table = _worker_transfer_table(
        [
            {
                "changed_decoders": [f"decoder-{index}" for index in range(11)],
                "mixed_peak_worker_counts": {12: 1, 16: 11},
                "native_peak_worker_counts": {0: 4, 12: 7, 16: 1},
                "platform": "Intel 8581C",
            },
        ],
    )

    assert "0: 4, 12: 7, 16: 1" in table
    assert "12: 1, 16: 11" in table
    assert "11/12" in table
    assert table.count(" & ") == 3
    assert r"\texttt" not in table
