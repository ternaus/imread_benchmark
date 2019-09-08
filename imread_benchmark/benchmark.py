"""
Benchmark libraries:

cv2
skimage
PIL
jpeg4py
imageio

for cases:

jpeg images => numpy array for RGB image

The code is inspired by: https://github.com/albu/albumentations/blob/master/benchmark/benchmark.py
"""

import numpy as np
import cv2
from PIL import Image
import argparse
import pkg_resources
import sys
import math
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from timeit import Timer
from abc import ABC
from collections import defaultdict


def print_package_versions():
    packages = ["opencv-python", "pillow"]
    package_versions = {"python": sys.version}
    for package in packages:
        try:
            package_versions[package] = pkg_resources.get_distribution(package).version
        except pkg_resources.DistributionNotFound:
            pass
    print(package_versions)


def read_img_pillow(file_path: str):
    with open(file_path, "rb") as f:
        img = Image.open(f)
    return img.convert("RGB")


def read_img_cv2(file_path: str):
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def format_results(images_per_second_for_read, show_std=False):
    if images_per_second_for_read is None:
        return "-"
    result = str(math.floor(np.mean(images_per_second_for_read)))
    if show_std:
        result += " ± {}".format(math.ceil(np.std(images_per_second_for_read)))
    return result


class BenchmarkTest(ABC):
    def __str__(self):
        return self.__class__.__name__

    def run(self, library, image_paths: list):
        operation = getattr(self, library)
        for image in image_paths:
            operation(image)


class GetSize(BenchmarkTest):
    def PIL(self, image_path: str):
        width, height = Image.open(image_path).size

        return width, height

    def opencv(self, image_path: str):
        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        return width, height


class GetArray(BenchmarkTest):
    def PIL(self, image_path: str) -> np.array:
        img = Image.open(image_path)
        return img.convert("RGB")

    def opencv(self, image_path: str) -> np.array:
        img = cv2.imread(image_path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def parse_args():
    parser = argparse.ArgumentParser(description="Image reading libraries performance benchmark")
    parser.add_argument("-d", "--data-dir", required=True, metavar="DIR", help="path to a directory with images")
    parser.add_argument(
        "-i",
        "--num_images",
        default=2000,
        type=int,
        metavar="N",
        help="number of images for benchmarking (default: 2000)",
    )
    parser.add_argument(
        "-r", "--runs", default=5, type=int, metavar="N", help="number of runs for each benchmark (default: 5)"
    )
    parser.add_argument(
        "--show-std", dest="show_std", action="store_true", help="show standard deviation for benchmark runs"
    )
    parser.add_argument("-p", "--print-package-versions", action="store_true", help="print versions of packages")
    return parser.parse_args()


def benchmark(libraries: list, benchmarks: list, image_paths: list, num_runs: int):
    images_per_second = defaultdict(dict)
    num_images = len(image_paths)

    for library in libraries:
        pbar = tqdm(total=len(benchmarks))
        for benchmark in benchmarks:
            timer = Timer(lambda: benchmark.run(library, image_paths))
            run_times = timer.repeat(number=1, repeat=num_runs)
            benchmark_images_per_second = [1 / (run_time / num_images) for run_time in run_times]
        images_per_second[library][str(benchmark)] = benchmark_images_per_second
        pbar.update(1)

    pbar.close()

    return images_per_second


def get_image_paths(data_dir: (str, Path), num_images: int) -> list:
    image_paths = sorted(Path(data_dir).glob("*.*"))
    return [str(x) for x in image_paths[:num_images]]


def main():
    args = parse_args()
    if args.print_package_versions:
        print_package_versions()

    benchmarks = [GetSize(), GetArray()]

    libraries = ["opencv", "PIL"]

    image_paths = get_image_paths(args.data_dir, args.num_images)

    images_per_second = benchmark(libraries, benchmarks, image_paths, args.num_runs)

    pd.set_option("display.width", 1000)
    df = pd.DataFrame.from_dict(images_per_second)
    df = df.applymap(lambda r: format_results(r, args.show_std))
    df = df[libraries]

    print(df)


if __name__ == "__main__":
    main()
