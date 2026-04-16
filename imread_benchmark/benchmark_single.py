from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np

from imread_benchmark.benchmark import run_timing_loop
from imread_benchmark.decoders import REGISTRY
from imread_benchmark.utils import collect_jpeg_paths, get_package_versions, get_system_identifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark a single image-decoding library.")
    parser.add_argument("-d", "--data-dir", required=True, help="Directory containing JPEG images")
    parser.add_argument("-n", "--num-images", type=int, default=2000)
    parser.add_argument("-r", "--num-runs", type=int, default=5)
    parser.add_argument("-o", "--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=["memory", "disk"],
        default="memory",
        help="memory: decode from pre-loaded bytes (no I/O in timed loop); disk: decode from file path",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=0,
        metavar="N",
        help="threads for the library to use (0 = library default)",
    )
    args = parser.parse_args()

    library = os.environ.get("BENCHMARK_LIBRARY")
    if not library:
        parser.error("BENCHMARK_LIBRARY environment variable must be set")

    decoder_cls = REGISTRY.get(library)
    if decoder_cls is None:
        parser.error(f"Unknown library '{library}'. Supported: {', '.join(REGISTRY)}")

    decoder = decoder_cls()
    if args.num_threads > 0:
        decoder.set_num_threads(args.num_threads)
    logger.info("Benchmarking %s in %s mode, threads=%d", library, args.mode, decoder.get_num_threads())

    image_paths = collect_jpeg_paths(args.data_dir, args.num_images)
    if not image_paths:
        parser.error(f"No JPEG images found in {args.data_dir}")
    logger.info("Found %d images", len(image_paths))

    decode_fn: Callable[[Any], np.ndarray]
    items: list[Any]
    if args.mode == "memory":
        logger.info("Pre-loading images into memory…")
        items = [p.read_bytes() for p in image_paths]
        decode_fn = decoder.decode
    else:
        items = [str(p) for p in image_paths]
        decode_fn = decoder.decode_path

    benchmark_results = run_timing_loop(decode_fn, items, args.num_runs)
    num_threads = decoder.get_num_threads()

    system_id = get_system_identifier()
    output_dir = args.output_dir / system_id
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "library": library,
        "mode": args.mode,
        "num_threads": num_threads,
        "system_info": get_package_versions(library),
        "benchmark_results": benchmark_results,
        "num_images": len(image_paths),
        "num_runs": args.num_runs,
    }

    output_file = output_dir / f"{library}_{num_threads}t_results.json"
    with output_file.open("w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", output_file)


if __name__ == "__main__":
    main()
