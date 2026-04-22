from __future__ import annotations

import numpy as np
import pytest

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
        "num_images_total",
        "num_images_decoded",
        "num_images_skipped",
        "skip_rate",
        "skip_indices",
        "skip_examples",
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


# ─── Skip-on-bad-image semantics ──────────────────────────────────────────────
# These exist to prove the partial-success path that vm_startup.sh now relies
# on. Without them, a future "tidy up the timing loop" PR could silently drop
# the discovery pass and revert us to the bug where one CMYK JPEG nukes a
# 4-hour cloud run.


def _flaky_decoder(bad_indices):
    """Raise for listed item indices; succeed for all other indices."""
    bad = set(bad_indices)
    counter = {"i": 0}

    def decode(_data):
        idx = counter["i"]
        counter["i"] += 1
        if idx in bad:
            raise OSError(f"Unsupported color conversion request (idx={idx})")
        return np.zeros((4, 4, 3), dtype=np.uint8)

    return decode


def test_skip_count_reflects_bad_items():
    items = [b"x"] * 20
    # Mark indices 3, 7, 11 as bad. _flaky_decoder uses a global counter so
    # the same indices fail in the discovery pass.
    decode_fn = _flaky_decoder([3, 7, 11])

    result = run_timing_loop(decode_fn, items, num_runs=2, num_warmup=0)

    assert result["num_images_total"] == 20
    assert result["num_images_skipped"] == 3
    assert result["num_images_decoded"] == 17
    assert result["skip_indices"] == [3, 7, 11]
    assert result["skip_rate"] == pytest.approx(3 / 20)
    # Examples are capped + tagged with the exception type for triage.
    assert len(result["skip_examples"]) == 3
    assert all("OSError" in ex for ex in result["skip_examples"])


def test_skip_examples_are_capped():
    items = [b"x"] * 50
    # Every item bad — discovery should still cap retained examples to 3.
    decode_fn = lambda _: (_ for _ in ()).throw(  # noqa: E731
        ValueError("bad image"),
    )

    with pytest.raises(RuntimeError, match="failed on all"):
        run_timing_loop(decode_fn, items, num_runs=1, num_warmup=0)


def test_zero_skips_keeps_clean_schema():
    items = [b"x"] * 10
    decode_fn = lambda _: np.zeros((4, 4, 3), dtype=np.uint8)  # noqa: E731

    result = run_timing_loop(decode_fn, items, num_runs=2, num_warmup=1)

    assert result["num_images_skipped"] == 0
    assert result["skip_rate"] == 0.0
    assert result["skip_indices"] == []
    assert result["skip_examples"] == []
    # Throughput stats should reflect ALL items (no spurious filtering).
    assert result["num_images_decoded"] == 10
