from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

import numpy as np
from tqdm import tqdm


def run_timing_loop(
    decode_fn: Callable[[Any], np.ndarray],
    items: Sequence[Any],
    num_runs: int,
    num_warmup: int = 2,
) -> dict[str, Any]:
    """
    Time `decode_fn` called over all `items` for `num_runs` rounds.

    items is either list[bytes] (memory mode) or list[str] (disk mode).
    Returns mean/std throughput in images/sec plus raw per-run values.
    """
    for _ in range(num_warmup):
        for item in items:
            decode_fn(item)

    times: list[float] = []
    for _ in tqdm(range(num_runs), desc="Benchmarking"):
        t0 = time.perf_counter()
        for item in items:
            decode_fn(item)
        elapsed = time.perf_counter() - t0
        times.append(len(items) / elapsed)

    mean_ips = float(np.mean(times))
    std_ips = float(np.std(times))
    return {
        "images_per_second": f"{mean_ips:.2f} ± {std_ips:.2f}",
        "images_per_second_mean": mean_ips,
        "images_per_second_std": std_ips,
        "raw_times": times,
    }
