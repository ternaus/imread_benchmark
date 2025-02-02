import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from imread_benchmark.decoders import setup_decoder
from imread_benchmark.utils import get_package_versions, get_system_identifier

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_benchmark(read_image, image_paths, num_runs):
    # Warm-up run
    logger.info("Performing warm-up run...")
    for path in tqdm(image_paths, desc="Warm-up"):
        read_image(path)

    times = []
    for _ in tqdm(range(num_runs), desc="Benchmarking"):
        start_time = time.perf_counter()
        for path in image_paths:
            read_image(path)
        end_time = time.perf_counter()

        run_time = end_time - start_time
        images_per_second = len(image_paths) / run_time
        times.append(images_per_second)

    median_ips = float(np.median(times))
    mean_ips = float(np.mean(times))
    std_ips = float(np.std(times))

    return {
        "images_per_second_median": f"{median_ips:.2f}",
        "images_per_second_mean": f"{mean_ips:.2f} ± {std_ips:.2f}",
        "raw_times": times,
        "statistics": {
            "median": median_ips,
            "mean": mean_ips,
            "std": std_ips,
            "min": float(np.min(times)),
            "max": float(np.max(times)),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data-dir", required=True, help="Path to image directory")
    parser.add_argument("-n", "--num-images", type=int, default=2000)
    parser.add_argument("-r", "--num-runs", type=int, default=5)
    parser.add_argument("-o", "--output-dir", type=Path, required=True)
    args = parser.parse_args()

    # Set up library and get read function once at startup
    library, read_image = setup_decoder(mode="disk")

    # Create output directory with detailed system info
    system_id = get_system_identifier()
    output_dir = args.output_dir / system_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define supported image extensions
    image_extensions = {".jpg", ".jpeg", ".JPEG", ".JPG"}

    # Get image paths recursively, filtering for supported extensions
    image_paths = [str(p) for p in sorted(Path(args.data_dir).rglob("*")) if p.suffix.lower() in image_extensions][
        : args.num_images
    ]

    # Run benchmark
    results = {
        "library": library,
        "system_info": get_package_versions(),
        "benchmark_results": run_benchmark(read_image, image_paths, args.num_runs),
        "num_images": args.num_images,
        "num_runs": args.num_runs,
    }

    # Save results
    output_file = output_dir / f"{library}_results.json"
    with output_file.open("w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
