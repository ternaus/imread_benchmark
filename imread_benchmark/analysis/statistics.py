from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True, slots=True)
class DistributionSummary:
    n: int
    mean: float
    median: float
    minimum: float
    maximum: float
    sample_std: float | None
    coefficient_of_variation: float | None

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "coefficient_of_variation": self.coefficient_of_variation,
            "maximum": self.maximum,
            "mean": self.mean,
            "median": self.median,
            "minimum": self.minimum,
            "n": self.n,
            "sample_std": self.sample_std,
        }


def summarize_distribution(values: Sequence[float]) -> DistributionSummary:
    measurements = tuple(float(value) for value in values)
    if not measurements or any(not math.isfinite(value) or value <= 0 for value in measurements):
        raise ValueError("measurements must contain one or more finite positive values")

    mean = statistics.fmean(measurements)
    sample_std = statistics.stdev(measurements) if len(measurements) > 1 else None
    return DistributionSummary(
        n=len(measurements),
        mean=mean,
        median=statistics.median(measurements),
        minimum=min(measurements),
        maximum=max(measurements),
        sample_std=sample_std,
        coefficient_of_variation=sample_std / mean if sample_std is not None else None,
    )


def summarize_benchmark(
    elapsed_seconds: Sequence[float],
    *,
    items_processed: int,
) -> dict[str, dict[str, float | int | None]]:
    if items_processed <= 0:
        raise ValueError("items_processed must be positive")
    elapsed = tuple(float(value) for value in elapsed_seconds)
    elapsed_summary = summarize_distribution(elapsed)
    images_per_second = summarize_distribution(tuple(items_processed / value for value in elapsed))
    microseconds_per_image = summarize_distribution(
        tuple(value / items_processed * 1_000_000 for value in elapsed),
    )
    return {
        "elapsed_seconds": elapsed_summary.to_dict(),
        "images_per_second": images_per_second.to_dict(),
        "microseconds_per_image": microseconds_per_image.to_dict(),
    }
