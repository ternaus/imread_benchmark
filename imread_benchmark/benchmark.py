import argparse
import contextlib
import math
import random
import sys
from abc import ABC
from collections import defaultdict
from pathlib import Path
from timeit import Timer

import cv2
import imageio
import jpeg4py
import numpy as np
import pandas as pd
import pkg_resources
import pyvips
import skimage
import torchvision
from PIL import Image
from tqdm import tqdm


def print_package_versions() -> None:
    packages = ["opencv-python", "pillow-simd", "jpeg4py", "scikit-image", "imageio", "pyvips", "torchvision"]
    package_versions = {"python": sys.version}
    for package in packages:
        with contextlib.suppress(pkg_resources.DistributionNotFound):
            package_versions[package] = pkg_resources.get_distribution(package).version


def format_results(images_per_second_for_read, show_std=False):
    if images_per_second_for_read is None:
        return "-"
    result = str(math.floor(np.mean(images_per_second_for_read)))
    if show_std:
        result += f" ± {math.ceil(np.std(images_per_second_for_read))}"
    return result


class BenchmarkTest(ABC):
    def __str__(self):
        return self.__class__.__name__

    def run(self, library, image_paths: list) -> None:
        operation = getattr(self, library)
        for image in image_paths:
            operation(image)


class GetSize(BenchmarkTest):
    def pil(self, image_path: str) -> tuple:
        width, height = Image.open(image_path).size
        return width, height

    def opencv(self, image_path: str):
        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        return width, height

    def jpeg4py(self, image_path: str) -> np.array:
        image = jpeg4py.JPEG(image_path).decode()
        height, width = image.shape[:2]
        return width, height

    def skimage(self, image_path: str) -> np.asarray:
        image = skimage.io.imread(image_path, plugin="matplotlib")
        height, width = image.shape[:2]
        return width, height

    def imageio(self, image_path: str) -> np.array:
        image = imageio.imread(image_path)
        height, width = image.shape[:2]
        return width, height

    def pyvips(self, image_path: str) -> np.array:
        image = pyvips.Image.new_from_file(image_path, access="sequential")

        return image.width, image.height

    def torchvision(self, image_path: str) -> np.array:
        image = torchvision.io.read_image(image_path)
        return image.shape[2], image.shape[1]


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
        return skimage.io.imread(image_path, plugin="matplotlib")

    def imageio(self, image_path: str) -> np.array:
        return imageio.imread(image_path)

    def pyvips(self, image_path: str) -> np.array:
        image = pyvips.Image.new_from_file(image_path, access="sequential")

        memory_image = image.write_to_memory()
        return np.ndarray(buffer=memory_image, dtype=np.uint8, shape=[image.height, image.width, image.bands])

    def torchvision(self, image_path: str) -> np.array:
        image = torchvision.io.read_image(image_path)
        return image.permute(1, 2, 0).numpy()


def benchmark(libraries: list, benchmarks: list, image_paths: list, num_runs: int, shuffle: bool) -> defaultdict:
    images_per_second = defaultdict(dict)
    num_images = len(image_paths)

    for library in libraries:
        pbar = tqdm(total=len(benchmarks))
        for benchmark in benchmarks:
            pbar.set_description(f"Current benchmark: {library} | {benchmark}")
            if shuffle:
                random.shuffle(image_paths)
            timer = Timer(lambda: benchmark.run(library, image_paths))
            run_times = timer.repeat(number=1, repeat=num_runs)
            benchmark_images_per_second = [1 / (run_time / num_images) for run_time in run_times]
            images_per_second[library][str(benchmark)] = benchmark_images_per_second
            pbar.update(1)

        pbar.close()

    return images_per_second


def parse_args():
    parser = argparse.ArgumentParser(description="Image reading libraries performance benchmark")
    parser.add_argument("-d", "--data-dir", metavar="DIR", help="path to a directory with images")
    parser.add_argument(
        "-i",
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
    parser.add_argument("-p", "--print-package-versions", action="store_true", help="print versions of packages")
    parser.add_argument("-s", "--shuffle", action="store_true", help="Shuffle the list of images.")
    return parser.parse_args()


def get_image_paths(data_dir: str | Path, num_images: int) -> list:
    image_paths = sorted(Path(data_dir).glob("*.*"))
    return [str(x) for x in image_paths[:num_images]]


def main() -> None:
    args = parse_args()
    if args.print_package_versions:
        print_package_versions()

    benchmarks = [
        GetSize(),
        GetArray(),
    ]

    libraries = ["opencv", "pil", "jpeg4py", "skimage", "imageio", "pyvips", "torchvision"]

    image_paths = get_image_paths(args.data_dir, args.num_images)

    images_per_second = benchmark(libraries, benchmarks, image_paths, args.num_runs, args.shuffle)

    pd.set_option("display.width", 1000)
    df = pd.DataFrame.from_dict(images_per_second)
    df = df.applymap(lambda r: format_results(r, args.show_std))
    df = df[libraries]


if __name__ == "__main__":
    main()
