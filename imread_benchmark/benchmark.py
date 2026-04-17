from __future__ import annotations

import gc
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

import numpy as np
from tqdm import tqdm


def _summarise(times_per_run_s: list[float], n_items: int) -> dict[str, Any]:
    """
    Convert per-run wall times into summary statistics.

    Two units kept side by side:
      - images_per_second_*  — throughput (paper headlines)
      - us_per_image_*       — per-item latency (cleaner for cross-set comparison)

    Percentiles are over the per-run distribution (small N, but right-skewed
    enough that p50 is meaningfully different from mean).
    """
    times_arr = np.asarray(times_per_run_s, dtype=np.float64)
    ips_arr = n_items / times_arr
    us_arr = times_arr / n_items * 1e6

    return {
        "images_per_second_mean": float(ips_arr.mean()),
        "images_per_second_std": float(ips_arr.std()),
        "images_per_second_p50": float(np.percentile(ips_arr, 50)),
        "images_per_second_p90": float(np.percentile(ips_arr, 90)),
        "images_per_second_p99": float(np.percentile(ips_arr, 99)),
        "us_per_image_mean": float(us_arr.mean()),
        "us_per_image_p50": float(np.percentile(us_arr, 50)),
        "us_per_image_p90": float(np.percentile(us_arr, 90)),
        "us_per_image_p99": float(np.percentile(us_arr, 99)),
        "raw_times_s": times_arr.tolist(),
        "raw_throughput_ips": ips_arr.tolist(),
    }


def run_timing_loop(
    decode_fn: Callable[[Any], np.ndarray],
    items: Sequence[Any],
    num_runs: int,
    num_warmup: int = 2,
) -> dict[str, Any]:
    """
    Time `decode_fn` over all `items`, `num_runs` rounds, with `num_warmup` untimed rounds first.

    items is either list[bytes] (memory mode) or list[str] (disk mode).
    GC is disabled inside each timed pass to avoid mid-run pauses skewing tail latencies.
    """
    for _ in range(num_warmup):
        for item in items:
            decode_fn(item)

    times: list[float] = []
    for _ in tqdm(range(num_runs), desc="Benchmarking"):
        gc.collect()
        gc.disable()
        try:
            t0 = time.perf_counter()
            for item in items:
                decode_fn(item)
            elapsed = time.perf_counter() - t0
        finally:
            gc.enable()
        times.append(elapsed)

    return _summarise(times, len(items))
