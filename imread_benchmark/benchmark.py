import argparse
import logging
import os
import random
import sys
import time
from abc import ABC
from collections import defaultdict
from pathlib import Path

import cv2
import imageio.v2 as imageio
import jpeg4py
import kornia_rs as K
import numpy as np
import pandas as pd
import pkg_resources
import skimage
import tensorflow as tf
import torchvision
from PIL import Image
from pytablewriter import MarkdownTableWriter
from pytablewriter.style import Style
from tqdm import tqdm

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # Attempt to disable all GPUs
    tf.config.set_visible_devices([], "GPU")
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        if device.device_type == "GPU":
            logger.warning("GPU device is still visible, disabling failed.")
except tf.errors.NotFoundError:  # Example of catching a more specific TensorFlow exception
    logger.exception("Specific TensorFlow error encountered when trying to modify GPU visibility.")
except Exception:  # Use this as a fallback if you're unsure which specific exceptions might be raised
    logger.exception("Failed to modify GPU visibility due to an unexpected error.")


package_mapping = {
    "opencv": "opencv-python-headless",  # or "opencv-python" depending on which you use
    "pil": "pillow",
    "jpeg4py": "jpeg4py",
    "skimage": "scikit-image",
    "imageio": "imageio",
    "torchvision": "torchvision",
    "tensorflow": "tensorflow",
    "kornia": "kornia-rs",
}


def get_package_versions():
    # Mapping of import names to package names as they might differ

    versions = {"Python": sys.version.split()[0]}  # Just get the major.minor.patch
    for package, dist_name in package_mapping.items():
        try:
            versions[package] = pkg_resources.get_distribution(dist_name).version
        except pkg_resources.DistributionNotFound:
            versions[package] = "Not Installed"
    return versions


class BenchmarkTest(ABC):
    def __str__(self):
        return self.__class__.__name__

    def run(self, library, image_paths: list) -> None:
        operation = getattr(self, library)
        for image in image_paths:
            operation(image)


class GetArray(BenchmarkTest):
    def pil(self, image_path: str) -> np.array:
        img = Image.open(image_path)
        img = img.convert("RGB")
        return np.asarray(img)

    def opencv(self, image_path: str) -> np.array:
        img = cv2.imread(image_path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def jpeg4py(self, image_path: str) -> np.array:
        return jpeg4py.JPEG(image_path).decode()

    def skimage(self, image_path: str) -> np.array:
        return skimage.io.imread(image_path)

    def imageio(self, image_path: str) -> np.array:
        return imageio.imread(image_path)

    def torchvision(self, image_path: str) -> np.array:
        image = torchvision.io.read_image(image_path)
        return image.permute(1, 2, 0).numpy()

    def tensorflow(self, image_path: str) -> np.array:
        # Read the image file
        image_string = tf.io.read_file(image_path)
        # Decode the image to tensor
        image = tf.io.decode_image(image_string, channels=3)
        # Convert the tensor to numpy array
        return image.numpy()

    def kornia(self, image_path: str) -> np.array:
        return K.read_image_jpeg(image_path)


class MarkdownGenerator:
    def __init__(self, df, package_versions):
        self._df = df
        self._package_versions = package_versions

    def _highlight_best_result(self, results) -> list[str]:
        # Convert all results to floats for comparison, filtering out any non-numeric values beforehand
        numeric_results = [float(r) for r in results if r.replace(".", "", 1).isdigit()]

        if not numeric_results:
            return results  # Return the original results if no numeric values were found

        best_result = max(numeric_results)

        # Highlight the best result by comparing the float representation of each result
        return [f"**{r}**" if float(r) == best_result else r for r in results]

    def _make_headers(self) -> list[str]:
        libraries = self._df.columns.to_list()
        columns = []
        for library in libraries:
            version = self._package_versions[
                (
                    library.replace("opencv", "opencv-python-headless")
                    .replace("pil", "pillow")
                    .replace("skimage", "scikit-image")
                    .replace("kornia", "kornia-rs")
                )
            ]

            columns.append(f"{library}<br><small>{version}</small>")
        return ["", *columns]

    def _make_value_matrix(self) -> list[list]:
        index = self._df.index.tolist()
        values = self._df.to_numpy().tolist()
        value_matrix = []
        for transform, results in zip(index, values, strict=False):
            row = [transform, *self._highlight_best_result(results)]
            value_matrix.append(row)
        return value_matrix

    def _make_versions_text(self) -> str:
        libraries = [
            "Python",
            "numpy",
            "pillow",
            "opencv-python-headless",
            "scikit-image",
            "scipy",
            "tensorflow",
            "kornia-rs",
        ]
        libraries_with_versions = [
            "{library} {version}".format(library=library, version=self._package_versions[library].replace("\n", ""))
            for library in libraries
        ]
        return f"Python and library versions: {', '.join(libraries_with_versions)}."

    def print(self) -> None:
        writer = MarkdownTableWriter()
        writer.headers = self._make_headers()
        writer.value_matrix = self._make_value_matrix()
        writer.styles = [Style(align="left")] + [Style(align="center") for _ in range(len(writer.headers) - 1)]
        writer.write_table()


def run_single_benchmark(benchmark, library, image_paths):
    """
    Runs a single benchmark for a given library and set of image paths.
    Returns the images per second performance.
    """
    start_time = time.perf_counter()
    benchmark.run(library, image_paths)
    end_time = time.perf_counter()

    run_time = end_time - start_time
    return len(image_paths) / run_time


def warm_up(libraries, benchmarks, image_paths, warmup_runs, shuffle_paths):
    """Performs warm-up runs for each library to ensure fair timing."""
    for library in tqdm(libraries, desc="Warming up libraries"):
        for _ in tqdm(range(warmup_runs), desc=f"Warm-up runs for {library}"):
            for benchmark in benchmarks:
                if shuffle_paths:
                    random.shuffle(image_paths)
                benchmark.run(library, image_paths)


def perform_benchmark(libraries, benchmarks, image_paths, num_runs, shuffle_paths):
    """Main benchmarking logic, performing the benchmark for each library and benchmark combination."""
    images_per_second = defaultdict(lambda: defaultdict(list))

    # for _ in range(num_runs):
    for _ in tqdm(range(num_runs), desc="Benchmarking Runs"):
        shuffled_libraries = libraries.copy()
        random.shuffle(shuffled_libraries)  # Shuffle library order for each run

        for library in tqdm(shuffled_libraries, desc="Libraries"):
            # for library in shuffled_libraries:
            for benchmark in benchmarks:
                if shuffle_paths:
                    random.shuffle(image_paths)

                ips = run_single_benchmark(benchmark, library, image_paths)
                images_per_second[library][str(benchmark)].append(ips)

    return images_per_second


def calculate_results(images_per_second):
    """Calculates the average and standard deviation of images per second for each library and benchmark."""
    final_results = defaultdict(dict)
    for library, benchmarks in images_per_second.items():
        for benchmark, times in benchmarks.items():
            avg_ips = np.mean(times)
            std_ips = np.std(times) if len(times) > 1 else 0
            final_results[library][benchmark] = f"{avg_ips:.2f} ± {std_ips:.2f}"

    return final_results


def benchmark(
    libraries: list,
    benchmarks: list,
    image_paths: list,
    num_runs: int,
    shuffle_paths: bool,
    warmup_runs: int = 1,
) -> defaultdict:
    """Orchestrates the benchmarking process, including warm-up, main benchmark, and result calculation."""
    # Warm-up phase
    warm_up(libraries, benchmarks, image_paths, warmup_runs, shuffle_paths)

    # Main benchmarking
    images_per_second = perform_benchmark(libraries, benchmarks, image_paths, num_runs, shuffle_paths)

    # Calculate and return final results
    return calculate_results(images_per_second)


def parse_args():
    parser = argparse.ArgumentParser(description="Image reading libraries performance benchmark")
    parser.add_argument("-d", "--data-dir", metavar="DIR", help="path to a directory with images")
    parser.add_argument(
        "-n",
        "--num_images",
        default=2000,
        type=int,
        metavar="N",
        help="number of images for benchmarking (default: 2000)",
    )
    parser.add_argument(
        "-r",
        "--num_runs",
        default=5,
        type=int,
        metavar="N",
        help="number of runs for each benchmark (default: 5)",
    )
    parser.add_argument(
        "--show-std",
        dest="show_std",
        action="store_true",
        help="show standard deviation for benchmark runs",
    )
    parser.add_argument("-m", "--markdown", action="store_true", help="print benchmarking results as a markdown table")
    parser.add_argument("-p", "--print-package-versions", action="store_true", help="print versions of packages")
    parser.add_argument("-s", "--shuffle", action="store_true", help="Shuffle the list of images.")
    parser.add_argument("-o", "--output_path", type=Path, help="Path to save resulting dataframe.", default="output")
    return parser.parse_args()


def get_image_paths(data_dir: str | Path, num_images: int) -> list:
    image_paths = sorted(Path(data_dir).glob("*.*"))
    return [str(x) for x in image_paths[:num_images]]


def main() -> None:
    args = parse_args()

    Path(args.output_path).mkdir(parents=True, exist_ok=True)

    package_versions = get_package_versions()

    benchmarks = [GetArray()]  # Add more benchmark classes as needed
    libraries = ["skimage", "imageio", "opencv", "pil", "jpeg4py", "torchvision", "tensorflow", "kornia"]

    image_paths = get_image_paths(args.data_dir, args.num_images)
    images_per_second = benchmark(libraries, benchmarks, image_paths, args.num_runs, args.shuffle)

    # Convert the results to a DataFrame
    results = defaultdict(list)
    for library in libraries:
        for perf in images_per_second[library].values():
            results["Library"].append(package_mapping[library])
            results["Version"].append(package_versions.get(library, "Unknown"))
            results["Performance (images/sec)"].append(perf)

    df = pd.DataFrame(results)

    if args.output_path:
        df.to_csv(args.output_path, index=False)

    if args.markdown:
        # Convert dataframe to markdown table
        print(df.to_markdown())

    return df  # Return the dataframe if needed


if __name__ == "__main__":
    df = main()
