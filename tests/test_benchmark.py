from __future__ import annotations

import numpy as np

from imread_benchmark.benchmark import run_timing_loop


def test_run_timing_loop_returns_expected_keys():
    items = [b"x"] * 10
    decode_fn = lambda data: np.zeros((4, 4, 3), dtype=np.uint8)  # noqa: E731

    result = run_timing_loop(decode_fn, items, num_runs=2, num_warmup=1)

    assert "images_per_second" in result
    assert "images_per_second_mean" in result
    assert "images_per_second_std" in result
    assert "raw_times" in result
    assert len(result["raw_times"]) == 2


def test_run_timing_loop_throughput_positive():
    items = [b"x"] * 50
    decode_fn = lambda data: np.zeros((8, 8, 3), dtype=np.uint8)  # noqa: E731

    result = run_timing_loop(decode_fn, items, num_runs=3, num_warmup=0)

    assert result["images_per_second_mean"] > 0
    assert all(t > 0 for t in result["raw_times"])
