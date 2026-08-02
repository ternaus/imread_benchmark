from __future__ import annotations

import pytest

from imread_benchmark.analysis.fodb_paper import (
    EXPECTED_MACHINES,
    ROBUSTNESS_AUDIT_EMPTY_DHT_ITEM_COUNT,
    ROBUSTNESS_AUDIT_EMPTY_DHT_SUCCESSES,
    ROBUSTNESS_AUDIT_FOUR_COMPONENT_ITEM_COUNT,
    ROBUSTNESS_AUDIT_FOUR_COMPONENT_SUCCESSES,
    ROBUSTNESS_AUDIT_ITEM_COUNT,
    ROBUSTNESS_AUDIT_SUCCESSES,
    Aggregate,
    _decoder_coverage_table,
    _linear_quantile,
    _ranks,
    _recommendation_rows,
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


def test_robustness_audit_reports_bitstream_and_output_contract_separately() -> None:
    assert ROBUSTNESS_AUDIT_ITEM_COUNT == (
        ROBUSTNESS_AUDIT_EMPTY_DHT_ITEM_COUNT + ROBUSTNESS_AUDIT_FOUR_COMPONENT_ITEM_COUNT
    )
    assert set(ROBUSTNESS_AUDIT_SUCCESSES) == set(ROBUSTNESS_AUDIT_EMPTY_DHT_SUCCESSES)
    assert set(ROBUSTNESS_AUDIT_SUCCESSES) == set(ROBUSTNESS_AUDIT_FOUR_COMPONENT_SUCCESSES)
    for decoder, combined_successes in ROBUSTNESS_AUDIT_SUCCESSES.items():
        assert combined_successes == (
            ROBUSTNESS_AUDIT_EMPTY_DHT_SUCCESSES[decoder] + ROBUSTNESS_AUDIT_FOUR_COMPONENT_SUCCESSES[decoder]
        )

    table = _decoder_coverage_table(
        {
            "robustness_audit": {
                "categories": {
                    "empty_dht_bitstream": {
                        "item_count": ROBUSTNESS_AUDIT_EMPTY_DHT_ITEM_COUNT,
                        "successes": ROBUSTNESS_AUDIT_EMPTY_DHT_SUCCESSES,
                    },
                    "four_component_rgb": {
                        "item_count": ROBUSTNESS_AUDIT_FOUR_COMPONENT_ITEM_COUNT,
                        "successes": ROBUSTNESS_AUDIT_FOUR_COMPONENT_SUCCESSES,
                    },
                },
                "item_count": ROBUSTNESS_AUDIT_ITEM_COUNT,
                "successes": ROBUSTNESS_AUDIT_SUCCESSES,
            },
        },
    )

    assert r"\texttt{ajpegli} & 0/276 & 0/1 & 0/277" in table
    assert r"\texttt{torchvision} & 276/276 & 0/1 & 276/277" in table
    assert r"\texttt{simplejpeg} & 276/276 & 1/1 & 277/277" in table


def test_recommendations_select_the_minimax_portable_decoder() -> None:
    decoders = (*[f"decoder-{index}" for index in range(8)], "simplejpeg", "imagecodecs", "opencv", "pyvips")
    aggregates = []
    for scenario_index, (machine_type, workload) in enumerate(
        (machine_type, workload) for machine_type in EXPECTED_MACHINES for workload in ("fodb-native", "fodb-mixed")
    ):
        for decoder in decoders:
            mean = 80.0
            if decoder == f"decoder-{scenario_index}":
                mean = 100.0
            elif decoder == "simplejpeg":
                mean = 97.0
            elif decoder == "imagecodecs":
                mean = 96.0
            aggregates.append(
                Aggregate(
                    workload=workload,
                    machine_type=machine_type,
                    protocol="loader-supply",
                    decoder=decoder,
                    requested_threads=1 if decoder in {"opencv", "pyvips"} else None,
                    workers=8,
                    repetitions=(0, 1, 2, 3, 4),
                    raw_run_means=(mean,) * 5,
                    mean=mean,
                    sample_std=0.0,
                ),
            )
            if decoder == "simplejpeg":
                aggregates.append(
                    Aggregate(
                        workload=workload,
                        machine_type=machine_type,
                        protocol="loader-supply",
                        decoder=decoder,
                        requested_threads=None,
                        workers=0,
                        repetitions=(0, 1, 2, 3, 4),
                        raw_run_means=(200.0,) * 5,
                        mean=200.0,
                        sample_std=0.0,
                    ),
                )

    robustness_audit_successes = dict.fromkeys(decoders, 277)
    recommendations = _recommendation_rows(tuple(aggregates), robustness_audit_successes)

    assert recommendations["portable_decoder"] == "simplejpeg"
    assert recommendations["portable_max_gap_percent"] == pytest.approx(3.0)
    assert recommendations["portable_speed_candidates"] == ["simplejpeg", "imagecodecs"]
    assert recommendations["universal_recommendations"] == ["imagecodecs", "simplejpeg"]
    assert {
        item["workers"]
        for cell in recommendations["cells"]
        for item in cell["decoders"]
        if item["decoder"] == "simplejpeg"
    } == {8}


def test_recommendations_require_complete_robustness_audit() -> None:
    decoders = (*[f"decoder-{index}" for index in range(8)], "simplejpeg", "imagecodecs", "opencv", "pyvips")
    aggregates = []
    for machine_type, workload in (
        (machine_type, workload) for machine_type in EXPECTED_MACHINES for workload in ("fodb-native", "fodb-mixed")
    ):
        for decoder in decoders:
            mean = 100.0 if decoder == "imagecodecs" else 95.0 if decoder == "simplejpeg" else 80.0
            aggregates.append(
                Aggregate(
                    workload=workload,
                    machine_type=machine_type,
                    protocol="loader-supply",
                    decoder=decoder,
                    requested_threads=1 if decoder in {"opencv", "pyvips"} else None,
                    workers=8,
                    repetitions=(0, 1, 2, 3, 4),
                    raw_run_means=(mean,) * 5,
                    mean=mean,
                    sample_std=0.0,
                ),
            )
    robustness_audit_successes = dict.fromkeys(decoders, 277)
    robustness_audit_successes["imagecodecs"] = 276

    recommendations = _recommendation_rows(tuple(aggregates), robustness_audit_successes)

    assert recommendations["portable_decoder"] == "simplejpeg"
    assert recommendations["portable_speed_candidates"] == ["imagecodecs", "simplejpeg"]
    assert recommendations["universal_recommendations"] == ["simplejpeg"]
    assert all("imagecodecs" not in cell["recommended"] for cell in recommendations["cells"])
