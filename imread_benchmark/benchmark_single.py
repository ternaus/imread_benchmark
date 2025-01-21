import argparse
import json
import logging
import os
import platform
import sys
import time
from importlib.metadata import version
from pathlib import Path

import cpuinfo
import numpy as np
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SUPPORTED_LIBRARIES = {
    "opencv": "opencv-python-headless",
    "pil": "pillow",
    "jpeg4py": "jpeg4py",
    "skimage": "scikit-image",
    "imageio": "imageio",
    "torchvision": "torchvision",
    "tensorflow": "tensorflow",
    "kornia": "kornia-rs",
}


def get_package_versions():
    import multiprocessing

    import cpuinfo

    # Get CPU info
    try:
        cpu_info = cpuinfo.get_cpu_info()
        cpu_details = {
            "brand_raw": cpu_info.get("brand_raw", "Unknown"),
            "arch": cpu_info.get("arch", "Unknown"),
            "hz_advertised_raw": cpu_info.get("hz_advertised_raw", "Unknown"),
            "count": multiprocessing.cpu_count(),
        }
    except Exception as e:
        logger.warning(f"Failed to get CPU info: {e}")
        cpu_details = {"error": str(e)}

    versions = {
        "Python": sys.version.split()[0],
        "OS": platform.system(),
        "OS Version": platform.version(),
        "Machine": platform.machine(),
        "CPU": cpu_details,
    }

    lib_name = os.environ.get("BENCHMARK_LIBRARY")
    if lib_name:
        pkg_name = SUPPORTED_LIBRARIES.get(lib_name)
        if pkg_name:
            try:
                versions[lib_name] = version(pkg_name)
            except Exception as e:
                versions[lib_name] = f"Error getting version: {e!s}"

    return versions


def get_system_identifier() -> str:
    """Get a detailed system identifier including OS and CPU.

    Returns:
        str: A string combining OS and CPU model, formatted as 'os_cpu-model'

    """
    try:
        cpu_info = cpuinfo.get_cpu_info()
        cpu_brand = cpu_info.get("brand_raw", "Unknown")

        # Simple OS identification
        os_id = "darwin" if platform.system().lower() == "darwin" else "linux"

        # Replace spaces with hyphens but keep full names
        cpu_id = cpu_brand.replace(" ", "-")
    except Exception as e:
        logger.warning(f"Failed to get system info: {e}")
        return "unknown-system"
    else:
        return f"{os_id}_{cpu_id}"


def setup_library():
    """Set up the image reading function based on the specified library."""
    library = os.environ.get("BENCHMARK_LIBRARY")
    if not library:
        raise ValueError("BENCHMARK_LIBRARY environment variable must be set")

    if library == "opencv":
        import cv2

        def read_image(path):
            return cv2.imread(path, cv2.IMREAD_COLOR_RGB)

    elif library in {"pillow", "pillow-simd"}:
        from PIL import Image

        def read_image(path):
            img = Image.open(path)
            img = img.convert("RGB")
            return np.asarray(img)

    elif library == "jpeg4py":
        import jpeg4py

        def read_image(path):
            return jpeg4py.JPEG(path).decode()

    elif library == "skimage":
        import skimage.io

        def read_image(path):
            return skimage.io.imread(path)

    elif library == "imageio":
        import imageio.v2 as imageio

        def read_image(path):
            return imageio.imread(path)

    elif library == "torchvision":
        import torchvision

        def read_image(path):
            image = torchvision.io.read_image(path)
            return image.permute(1, 2, 0).numpy()

    elif library == "tensorflow":
        import tensorflow as tf

        def read_image(path):
            image_string = tf.io.read_file(path)
            image = tf.io.decode_image(image_string, channels=3)
            return image.numpy()

    elif library == "kornia":
        import kornia_rs as K

        def read_image(path):
            return K.read_image_jpeg(path)

    else:
        raise ValueError(f"Unsupported library: {library}")

    return library, read_image


def run_benchmark(read_image, image_paths, num_runs):
    times = []
    for _ in tqdm(range(num_runs), desc="Benchmarking"):
        start_time = time.perf_counter()
        for path in image_paths:
            read_image(path)
        end_time = time.perf_counter()

        run_time = end_time - start_time
        images_per_second = len(image_paths) / run_time
        times.append(images_per_second)

    avg_ips = np.mean(times)
    std_ips = np.std(times)

    return {
        "images_per_second": f"{avg_ips:.2f} ± {std_ips:.2f}",
        "raw_times": times,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data-dir", required=True, help="Path to image directory")
    parser.add_argument("-n", "--num-images", type=int, default=2000)
    parser.add_argument("-r", "--num-runs", type=int, default=5)
    parser.add_argument("-o", "--output-dir", type=Path, required=True)
    args = parser.parse_args()

    # Set up library and get read function once at startup
    library, read_image = setup_library()

    # Create output directory with detailed system info
    system_id = get_system_identifier()
    output_dir = args.output_dir / system_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get image paths
    image_paths = sorted(Path(args.data_dir).glob("*.*"))[: args.num_images]
    image_paths = [str(x) for x in image_paths]

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
