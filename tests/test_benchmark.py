from __future__ import annotations

import numpy as np

from imread_benchmark.benchmark import run_timing_loop


def test_run_timing_loop_returns_expected_keys():
    items = [b"x"] * 10
    decode_fn = lambda data: np.zeros((4, 4, 3), dtype=np.uint8)  # noqa: E731

    result = run_timing_loop(decode_fn, items, num_runs=2, num_warmup=1)

    for key in (
        "images_per_second_mean",
        "images_per_second_std",
        "images_per_second_p50",
        "images_per_second_p90",
        "images_per_second_p99",
        "us_per_image_mean",
        "us_per_image_p50",
        "raw_times_s",
        "raw_throughput_ips",
    ):
        assert key in result, f"missing key: {key}"
    assert len(result["raw_times_s"]) == 2
    assert len(result["raw_throughput_ips"]) == 2


def test_run_timing_loop_throughput_positive():
    items = [b"x"] * 50
    decode_fn = lambda data: np.zeros((8, 8, 3), dtype=np.uint8)  # noqa: E731

    result = run_timing_loop(decode_fn, items, num_runs=3, num_warmup=0)

    assert result["images_per_second_mean"] > 0
    assert result["us_per_image_mean"] > 0
    assert all(t > 0 for t in result["raw_times_s"])
    assert all(ips > 0 for ips in result["raw_throughput_ips"])
    assert result["images_per_second_p50"] > 0
    assert result["images_per_second_p99"] > 0
