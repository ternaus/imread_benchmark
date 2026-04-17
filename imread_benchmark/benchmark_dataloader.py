r"""
DataLoader throughput benchmark.

Measures end-to-end throughput (images/sec) when a decoder is used inside a
PyTorch DataLoader with varying numbers of worker processes. Each worker
decodes from pre-loaded in-memory bytes, so disk I/O is excluded.

Usage:
    python -m imread_benchmark.benchmark_dataloader \
        --library opencv \
        --data-dir /path/to/imagenet/val \
        --output-dir output
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

import numpy as np
from tqdm import tqdm

from imread_benchmark.benchmark import _summarise
from imread_benchmark.decoders import REGISTRY
from imread_benchmark.utils import collect_jpeg_paths, get_package_versions, get_system_identifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_WORKERS = [0, 1, 2, 4, 8]
BATCH_SIZE = 32


class InMemoryDataset:
    """Holds raw JPEG bytes; decodes on __getitem__ using the given decode_fn."""

    def __init__(self, images_bytes: list[bytes], decode_fn: Callable[[bytes], Any]) -> None:
        self.images_bytes = images_bytes
        self.decode_fn = decode_fn

    def __len__(self) -> int:
        return len(self.images_bytes)

    def __getitem__(self, idx: int) -> np.ndarray:
        return self.decode_fn(self.images_bytes[idx])


def _collate(batch: list[np.ndarray]) -> list[np.ndarray]:
    # Images have variable resolutions — don't stack, just pass through.
    # We only care about decode throughput, not tensor batching.
    return batch


def benchmark_workers(
    dataset: InMemoryDataset,
    num_workers: int,
    num_runs: int,
    num_warmup: int = 1,
) -> dict[str, Any]:
    """
    Iterate `dataset` through a DataLoader `num_runs` times and return throughput stats.

    Warmup is critical for `num_workers > 0`: spawning worker processes (and on
    Linux fork() then re-importing libs in the worker) is wall-clock dominant
    on the first iteration and would skew the mean / p50.
    """
    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=num_workers,
        collate_fn=_collate,
        pin_memory=False,
        persistent_workers=(num_workers > 0),
    )

    for _ in range(num_warmup):
        for _ in loader:
            pass

    times_s: list[float] = []
    n_images = len(dataset)

    for _ in tqdm(range(num_runs), desc=f"workers={num_workers}"):
        gc.collect()
        gc.disable()
        try:
            t0 = time.perf_counter()
            for _ in loader:
                pass
            elapsed = time.perf_counter() - t0
        finally:
            gc.enable()
        times_s.append(elapsed)

    return {
        "num_workers": num_workers,
        "num_warmup": num_warmup,
        **_summarise(times_s, n_images),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="DataLoader throughput benchmark for JPEG decoding.")
    parser.add_argument("-l", "--library", required=True, help=f"One of: {', '.join(sorted(REGISTRY))}")
    parser.add_argument("-d", "--data-dir", required=True)
    parser.add_argument("-n", "--num-images", type=int, default=2000)
    parser.add_argument("-r", "--num-runs", type=int, default=5)
    parser.add_argument("-o", "--output-dir", type=Path, required=True)
    parser.add_argument(
        "--workers",
        nargs="+",
        type=int,
        default=DEFAULT_WORKERS,
        metavar="N",
        help=f"num_workers values to benchmark (default: {DEFAULT_WORKERS})",
    )
    args = parser.parse_args()

    library = args.library
    decoder_cls = REGISTRY.get(library)
    if decoder_cls is None:
        parser.error(f"Unknown library '{library}'. Supported: {', '.join(REGISTRY)}")

    decoder = decoder_cls()
    logger.info("DataLoader benchmark: %s", library)

    image_paths = collect_jpeg_paths(args.data_dir, args.num_images)
    if not image_paths:
        parser.error(f"No JPEG images found in {args.data_dir}")

    logger.info("Pre-loading %d images into memory…", len(image_paths))
    images_bytes = [p.read_bytes() for p in image_paths]

    dataset = InMemoryDataset(images_bytes, decoder.decode)

    system_id = get_system_identifier()
    output_dir = args.output_dir / system_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{library}_dataloader_results.json"

    results: dict[str, Any] = {
        "library": library,
        "benchmark_type": "dataloader",
        "batch_size": BATCH_SIZE,
        "num_threads": decoder.get_num_threads(),
        "system_info": get_package_versions(library),
        "worker_results": [],
        "num_images": len(image_paths),
        "num_runs": args.num_runs,
    }

    # Write after every num_workers config so a hang on a later config (e.g.
    # pyvips fork-deadlock at num_workers=8) doesn't lose the earlier results.
    # The background `gcloud storage rsync` in vm_startup.sh ships these files
    # to GCS within 30s of being written, so progress is durable.
    for n in args.workers:
        logger.info("Benchmarking with num_workers=%d", n)
        result = benchmark_workers(dataset, n, args.num_runs)
        results["worker_results"].append(result)
        with output_file.open("w") as f:
            json.dump(results, f, indent=2)
        logger.info(
            "Saved progress (%d/%d configs) to %s",
            len(results["worker_results"]),
            len(args.workers),
            output_file,
        )

    logger.info("Results saved to %s", output_file)


if __name__ == "__main__":
    main()
