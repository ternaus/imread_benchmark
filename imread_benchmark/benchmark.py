from __future__ import annotations

import gc
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

import numpy as np
from tqdm import tqdm

# Bound on retained error strings — enough to triage in the JSON without
# bloating the file when a decoder rejects a large slice of the dataset.
_WRAPPER_BUG_EXCEPTIONS = (TypeError, AttributeError, ImportError, ModuleNotFoundError)
MAX_SKIP_EXAMPLES = 3

# If a decoder fails on every single item it's not "decoder X has a 100% skip
# rate", it's "decoder X is fundamentally broken in this venv / on this build"
# — surface as an exception so the CLI marks the run as failed, not as a
# valid result with zero throughput.
_ALL_FAILED_TEMPLATE = (
    "Decoder failed on all {n} items during discovery pass. "
    "First error: {err}. This is treated as a decoder failure, not a skip rate."
)

def _validate_output(result: Any) -> None:
    if not isinstance(result, np.ndarray):
        raise ValueError(f"expected numpy.ndarray, got {type(result).__name__}")
    if result.ndim != 3:
        raise ValueError(f"expected 3-D (H, W, 3) array, got shape {result.shape}")
    if result.shape[2] != 3:
        raise ValueError(f"expected 3 channels in shape[2], got shape {result.shape}")
    if result.dtype != np.uint8:
        raise ValueError(f"expected uint8 dtype, got {result.dtype}")

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
        "images_per_second_std": float(ips_arr.std(ddof=1)) if ips_arr.size > 1 else 0.0,
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


def _discover_skips(
    decode_fn: Callable[[Any], np.ndarray],
    items: Sequence[Any],
) -> tuple[list[int], list[str]]:
    """
    Single un-timed pass to identify items the decoder cannot handle.

    Real ImageNet val contains a handful of non-standard JPEGs (CMYK, RGBA-
    embedded, weird subsampling) that strict decoders like turbojpeg, jpeg4py,
    and kornia-rs refuse with errors like "Unsupported color conversion
    request". Tolerant decoders (Pillow, OpenCV, scikit-image) silently
    convert. Treating these as fatal would bury 99.99% of clean numbers under
    a single bad image, so we identify them up front and skip them in every
    subsequent pass — and report skip count + sample errors in the JSON so
    the paper can surface decoder robustness as a real property.
    """
    skipped: list[int] = []
    examples: list[str] = []
    for idx, item in enumerate(items):
        try:
            result = decode_fn(item)
            _validate_output(result)
        # Decoder libs raise everything from OSError to ValueError to custom
        # exception types (jpeg4py.JPEGRuntimeError). We genuinely want to
        # catch them all — anything that prevents successful decode of one
        # item is a "skip", regardless of how the library chose to express it.
        except Exception as exc:
            if isinstance(exc, _WRAPPER_BUG_EXCEPTIONS):
                raise
            skipped.append(idx)
            if len(examples) < MAX_SKIP_EXAMPLES:
                examples.append(f"idx={idx}: {type(exc).__name__}: {exc}")
    return skipped, examples


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

    Bad items (those that raise during decode — typically <0.1% of ImageNet val
    for strict decoders) are identified in a single discovery pass and excluded
    from every subsequent pass. Skip counts and sample errors are returned in
    the result so the paper can report decoder robustness honestly.
    """
    skipped_indices, skip_examples = _discover_skips(decode_fn, items)
    n_total = len(items)
    n_skipped = len(skipped_indices)
    n_good = n_total - n_skipped

    if n_good == 0:
        first_err = skip_examples[0] if skip_examples else "no example captured"
        raise RuntimeError(_ALL_FAILED_TEMPLATE.format(n=n_total, err=first_err))

    skip_set = set(skipped_indices)
    good_items = [it for i, it in enumerate(items) if i not in skip_set]

    # The discovery pass already consumed one warmup's worth of work; subtract
    # one so we don't double-pay (and don't go negative).
    remaining_warmup = max(0, num_warmup - 1)
    for _ in range(remaining_warmup):
        for item in good_items:
            decode_fn(item)

    times: list[float] = []
    for _ in tqdm(range(num_runs), desc="Benchmarking"):
        gc.collect()
        gc.disable()
        try:
            t0 = time.perf_counter()
            for item in good_items:
                decode_fn(item)
            elapsed = time.perf_counter() - t0
        finally:
            gc.enable()
        times.append(elapsed)

    summary = _summarise(times, n_good)
    summary["num_images_total"] = n_total
    summary["num_images_decoded"] = n_good
    summary["num_images_skipped"] = n_skipped
    summary["skip_rate"] = n_skipped / n_total if n_total else 0.0
    summary["skip_indices"] = skipped_indices
    summary["skip_examples"] = skip_examples
    return summary
