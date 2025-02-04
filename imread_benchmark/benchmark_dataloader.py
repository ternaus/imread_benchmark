import argparse
import json
import logging
import os
import time
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable OpenCV threading to avoid conflicts with DataLoader workers


def get_opencv_reader(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR_RGB)
    if img is None:
        return np.zeros((100, 100, 3), dtype=np.uint8)  # minimal size, will be padded later
    return img


def get_pillow_reader(path):
    try:
        from PIL import Image

        img = Image.open(path)
        img = img.convert("RGB")
        return np.array(img, dtype=np.uint8)  # use array instead of asarray to ensure writeable
    except Exception as e:
        logger.warning(f"Failed to read {path}: {e}")
        return np.zeros((100, 100, 3), dtype=np.uint8)


def get_jpeg4py_reader(path):
    try:
        import jpeg4py

        return np.array(jpeg4py.JPEG(path).decode(), dtype=np.uint8)
    except Exception as e:
        logger.warning(f"Failed to read {path}: {e}")
        return np.zeros((100, 100, 3), dtype=np.uint8)


def get_skimage_reader(path):
    try:
        import skimage.io

        return np.array(skimage.io.imread(path), dtype=np.uint8)
    except Exception as e:
        logger.warning(f"Failed to read {path}: {e}")
        return np.zeros((100, 100, 3), dtype=np.uint8)


def get_imageio_reader(path):
    try:
        import imageio.v2 as imageio

        return np.array(imageio.imread(path), dtype=np.uint8)
    except Exception as e:
        logger.warning(f"Failed to read {path}: {e}")
        return np.zeros((100, 100, 3), dtype=np.uint8)


def get_torchvision_reader(path):
    try:
        import torchvision

        image = torchvision.io.read_image(path)
        return np.array(image.permute(1, 2, 0).numpy(), dtype=np.uint8)
    except Exception as e:
        logger.warning(f"Failed to read {path}: {e}")
        return np.zeros((100, 100, 3), dtype=np.uint8)


def get_tensorflow_reader(path):
    try:
        import tensorflow as tf

        image_string = tf.io.read_file(path)
        image = tf.io.decode_image(image_string, channels=3)
        return np.array(image.numpy(), dtype=np.uint8)
    except Exception as e:
        logger.warning(f"Failed to read {path}: {e}")
        return np.zeros((100, 100, 3), dtype=np.uint8)


def get_kornia_reader(path):
    try:
        import kornia_rs as K

        return np.array(K.read_image_jpeg(path), dtype=np.uint8)
    except Exception as e:
        logger.warning(f"Failed to read {path}: {e}")
        return np.zeros((100, 100, 3), dtype=np.uint8)


def setup_library():
    """Set up the image reading function based on the specified library."""
    import os

    library = os.environ.get("BENCHMARK_LIBRARY")
    if not library:
        raise ValueError("BENCHMARK_LIBRARY environment variable must be set")

    readers = {
        "opencv": get_opencv_reader,
        "pillow": get_pillow_reader,
        "pillow-simd": get_pillow_reader,
        "jpeg4py": get_jpeg4py_reader,
        "skimage": get_skimage_reader,
        "imageio": get_imageio_reader,
        "torchvision": get_torchvision_reader,
        "tensorflow": get_tensorflow_reader,
        "kornia": get_kornia_reader,
    }

    if library not in readers:
        raise ValueError(f"Unsupported library: {library}")

    return library, readers[library]


class ImageDataset(Dataset):
    def __init__(self, image_paths, read_fn, target_size=(512, 512)):
        self.image_paths = image_paths
        self.read_fn = read_fn
        self.target_size = target_size
        self.transform = A.Compose(
            [
                A.RandomCrop(height=target_size[0], width=target_size[1], pad_if_needed=True),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                A.ToTensorV2(),
            ],
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = self.read_fn(path)

        # Handle case where image is None or invalid
        if image is None or image.size == 0:
            return np.zeros(self.target_size, dtype=np.uint8)

        # Handle both RGB and grayscale images
        if len(image.shape) == 3:
            image = image[:, :, 0]  # Take first channel of RGB

        # Apply transform and get the first channel
        transformed = self.transform(image=image)
        return transformed["image"]


def run_dataloader_benchmark(dataset, batch_size, num_workers, num_runs):
    times = []
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
    )

    for _ in tqdm(range(num_runs), desc="Benchmarking DataLoader"):
        start_time = time.perf_counter()
        for _ in dataloader:
            pass

        end_time = time.perf_counter()

        run_time = end_time - start_time
        images_per_second = len(dataset) / run_time
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
    parser.add_argument("-b", "--batch-size", type=int, default=32)
    parser.add_argument("-w", "--num-workers", type=int, default=4)
    parser.add_argument("-o", "--output-dir", type=Path, required=True)
    args = parser.parse_args()

    # Set up library and get read function
    library, read_image = setup_library()

    # Create output directory with detailed system info
    from benchmark_single import get_package_versions, get_system_identifier

    system_id = get_system_identifier()
    output_dir = (
        args.output_dir
        / system_id
        / f"batch_size_{args.batch_size}"
        / f"num_runs_{args.num_runs}"
        / f"num_workers_{args.num_workers}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define supported image extensions
    image_extensions = {".jpg", ".jpeg", ".JPEG", ".JPG"}

    # Get image paths recursively, filtering for supported extensions
    image_paths = [str(p) for p in sorted(Path(args.data_dir).rglob("*")) if p.suffix.lower() in image_extensions][
        : args.num_images
    ]

    # Create dataset
    dataset = ImageDataset(image_paths, read_image)

    # Run benchmark
    results = {
        "library": library,
        "system_info": get_package_versions(),
        "benchmark_results": run_dataloader_benchmark(
            dataset,
            args.batch_size,
            args.num_workers,
            args.num_runs,
        ),
        "num_images": args.num_images,
        "num_runs": args.num_runs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
    }

    # Save results
    output_file = output_dir / f"{library}_dataloader_results.json"
    with output_file.open("w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
