r"""
DataLoader throughput benchmark.

Measures end-to-end throughput (images/sec) when a decoder is used inside a
PyTorch DataLoader with varying numbers of worker processes. Each worker
decodes from pre-loaded in-memory bytes, so disk I/O is excluded.

Usage:
    BENCHMARK_LIBRARY=opencv python imread_benchmark/benchmark_dataloader.py \
        --data-dir /path/to/imagenet/val \
        --output-dir output
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

import numpy as np
from tqdm import tqdm

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


def benchmark_workers(dataset: InMemoryDataset, num_workers: int, num_runs: int) -> dict[str, Any]:
    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=num_workers,
        collate_fn=_collate,
        pin_memory=False,
        persistent_workers=(num_workers > 0),
    )

    throughputs: list[float] = []
    n_images = len(dataset)

    for run_idx in tqdm(range(num_runs), desc=f"workers={num_workers}"):
        t0 = time.perf_counter()
        consumed = 0
        for batch in loader:
            consumed += len(batch)
        elapsed = time.perf_counter() - t0
        if run_idx == 0:
            # First run includes worker startup overhead; keep it for transparency
            pass
        throughputs.append(n_images / elapsed)

    mean_ips = float(np.mean(throughputs))
    std_ips = float(np.std(throughputs))
    return {
        "num_workers": num_workers,
        "images_per_second": f"{mean_ips:.2f} ± {std_ips:.2f}",
        "images_per_second_mean": mean_ips,
        "images_per_second_std": std_ips,
        "raw_times": throughputs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="DataLoader throughput benchmark for JPEG decoding.")
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

    library = os.environ.get("BENCHMARK_LIBRARY")
    if not library:
        parser.error("BENCHMARK_LIBRARY environment variable must be set")

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

    worker_results = []
    for n in args.workers:
        logger.info("Benchmarking with num_workers=%d", n)
        result = benchmark_workers(dataset, n, args.num_runs)
        worker_results.append(result)

    system_id = get_system_identifier()
    output_dir = args.output_dir / system_id
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "library": library,
        "benchmark_type": "dataloader",
        "batch_size": BATCH_SIZE,
        "num_threads": decoder.get_num_threads(),
        "system_info": get_package_versions(library),
        "worker_results": worker_results,
        "num_images": len(image_paths),
        "num_runs": args.num_runs,
    }

    output_file = output_dir / f"{library}_dataloader_results.json"
    with output_file.open("w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", output_file)


if __name__ == "__main__":
    main()
