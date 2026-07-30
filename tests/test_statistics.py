from __future__ import annotations

import pytest

from imread_benchmark.analysis.statistics import summarize_benchmark, summarize_distribution


def test_distribution_uses_sample_standard_deviation() -> None:
    summary = summarize_distribution((10.0, 20.0, 30.0))

    assert summary.n == 3
    assert summary.mean == 20.0
    assert summary.median == 20.0
    assert summary.minimum == 10.0
    assert summary.maximum == 30.0
    assert summary.sample_std == 10.0
    assert summary.coefficient_of_variation == 0.5


def test_single_sample_has_no_sample_standard_deviation() -> None:
    summary = summarize_distribution((12.5,))

    assert summary.sample_std is None
    assert summary.coefficient_of_variation is None


def test_benchmark_summary_keeps_units_and_sample_statistics() -> None:
    summary = summarize_benchmark((2.0, 1.0), items_processed=10)

    assert summary == {
        "elapsed_seconds": {
            "coefficient_of_variation": pytest.approx(0.47140452079103173),
            "maximum": 2.0,
            "mean": 1.5,
            "median": 1.5,
            "minimum": 1.0,
            "n": 2,
            "sample_std": pytest.approx(0.7071067811865476),
        },
        "images_per_second": {
            "coefficient_of_variation": pytest.approx(0.47140452079103173),
            "maximum": 10.0,
            "mean": 7.5,
            "median": 7.5,
            "minimum": 5.0,
            "n": 2,
            "sample_std": pytest.approx(3.5355339059327378),
        },
        "microseconds_per_image": {
            "coefficient_of_variation": pytest.approx(0.47140452079103173),
            "maximum": 200000.0,
            "mean": 150000.0,
            "median": 150000.0,
            "minimum": 100000.0,
            "n": 2,
            "sample_std": pytest.approx(70710.67811865476),
        },
    }


@pytest.mark.parametrize("values", [(), (0.0,), (1.0, -1.0)])
def test_distribution_rejects_empty_or_non_positive_measurements(values: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="positive"):
        summarize_distribution(values)
