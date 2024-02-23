import argparse
import contextlib
import math
import random
import sys
from abc import ABC
from collections import defaultdict
from pathlib import Path
from timeit import Timer
from contextlib import suppress

import cv2
import imageio.v2 as imageio
import os
from pytablewriter import MarkdownTableWriter
from pytablewriter.style import Style
import time
import jpeg4py
import numpy as np
import pandas as pd
import pkg_resources
import skimage
import torchvision
from PIL import Image
from tqdm import tqdm
import tensorflow as tf

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["export CUDA_VISIBLE_DEVICES"] = ""

try:
    # Disable all GPUS
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

def get_package_versions():
    packages = ["opencv-python", "pillow", "jpeg4py", "scikit-image", "imageio", "torchvision", "tensorflow"]
    package_versions = {"Python": sys.version}
    for package in packages:
        with suppress(pkg_resources.DistributionNotFound):
            package_versions[package] = pkg_resources.get_distribution(package).version
    return package_versions



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
        return img
        # return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

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
        image_np = image.numpy()
        return image_np


class MarkdownGenerator:
    def __init__(self, df, package_versions):
        self._df = df
        self._package_versions = package_versions

    def _highlight_best_result(self, results):
        best_result = float("-inf")
        for result in results:
            try:
                result = int(result)
            except ValueError:
                continue
            if result > best_result:
                best_result = result
        return [f"**{r}**" if r == str(best_result) else r for r in results]

    def _make_headers(self):
        libraries = self._df.columns.to_list()
        columns = []
        for library in libraries:
            version = self._package_versions[library.replace("opencv", "opencv-python").replace("pil", "pillow").replace("skimage", "scikit-image")]

            columns.append(f"{library}<br><small>{version}</small>")
        return ["", *columns]

    def _make_value_matrix(self):
        index = self._df.index.tolist()
        values = self._df.values.tolist()
        value_matrix = []
        for transform, results in zip(index, values):
            row = [transform, *self._highlight_best_result(results)]
            value_matrix.append(row)
        return value_matrix

    def _make_versions_text(self):
        libraries = ["Python", "numpy", "pillow", "opencv-python", "scikit-image", "scipy", "tensorflow"]
        libraries_with_versions = [
            "{library} {version}".format(library=library, version=self._package_versions[library].replace("\n", ""))
            for library in libraries
        ]
        return "Python and library versions: {}.".format(", ".join(libraries_with_versions))

    def print(self) -> None:
        writer = MarkdownTableWriter()
        writer.headers = self._make_headers()
        writer.value_matrix = self._make_value_matrix()
        writer.styles = [Style(align="left")] + [Style(align="center") for _ in range(len(writer.headers) - 1)]
        writer.write_table()



def benchmark(libraries: list, benchmarks: list, image_paths: list, num_runs: int, shuffle_paths: bool, warmup_runs: int = 1) -> defaultdict:
    images_per_second = defaultdict(lambda: defaultdict(list))
    num_images = len(image_paths)

    # Warm-up phase for each library
    for library in tqdm(libraries, desc='Warm-up', leave=False):
        for _ in range(warmup_runs):
            for benchmark in benchmarks:
                if shuffle_paths:
                    random.shuffle(image_paths)
                benchmark.run(library, image_paths)

    # Main benchmarking loop
    for _ in tqdm(range(num_runs), desc='Benchmark Runs'):
        shuffled_libraries = libraries.copy()
        random.shuffle(shuffled_libraries)  # Shuffle library order for each run

        for library in tqdm(shuffled_libraries, desc='Libraries', leave=False):
            for benchmark in tqdm(benchmarks, desc=f'{library} Benchmarks', leave=False):
                if shuffle_paths:
                    random.shuffle(image_paths)

                # Start timer
                start_time = time.perf_counter()

                # Run benchmark
                benchmark.run(library, image_paths)

                # End timer
                end_time = time.perf_counter()

                run_time = end_time - start_time
                images_per_second_per_run = num_images / run_time

                images_per_second[library][str(benchmark)].append(images_per_second_per_run)


    # Calculate average and std deviation
    final_results = defaultdict(dict)
    for library, benchmarks in images_per_second.items():
        for benchmark, times in benchmarks.items():
            avg_ips = np.mean(times)
            std_ips = np.std(times) if len(times) > 1 else 0
            final_results[library][benchmark] = f"{avg_ips:.2f} ± {std_ips:.2f}"


    return final_results



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
    return parser.parse_args()


def get_image_paths(data_dir: str | Path, num_images: int) -> list:
    image_paths = sorted(Path(data_dir).glob("*.*"))
    return [str(x) for x in image_paths[:num_images]]


def main() -> None:
    args = parse_args()
    package_versions = get_package_versions()

    benchmarks = [
        GetArray(),
    ]

    libraries = ["skimage", "imageio", "opencv", "pil", "jpeg4py", "torchvision", "tensorflow"]

    image_paths = get_image_paths(args.data_dir, args.num_images)

    images_per_second = benchmark(libraries, benchmarks, image_paths, args.num_runs, args.shuffle)

    pd.set_option("display.width", 1000)
    df = pd.DataFrame.from_dict(images_per_second)
    df = df[libraries]

    if args.markdown:
        makedown_generator = MarkdownGenerator(df, package_versions)
        makedown_generator.print()


if __name__ == "__main__":
    main()
