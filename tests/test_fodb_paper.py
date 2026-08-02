from __future__ import annotations

import pytest

from imread_benchmark.analysis.fodb_paper import _linear_quantile, _ranks, _workload_descriptors


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
