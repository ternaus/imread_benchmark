"""Module containing image decoding implementations for different libraries."""

from typing import Literal

import numpy as np


def setup_decoder(mode: Literal["disk", "memory"] = "disk"):
    """
    Set up the image decoder based on the specified library and mode.

    Args:
        mode (str): Either "disk" for file-based reading or "memory" for bytes-based decoding

    """
    import os

    library = os.environ.get("BENCHMARK_LIBRARY")
    if not library:
        raise ValueError("BENCHMARK_LIBRARY environment variable must be set")

    if library == "opencv":
        import cv2

        if mode == "disk":

            def decode(path):
                return cv2.imread(str(path), cv2.IMREAD_COLOR)
        else:

            def decode(image_bytes):
                array = np.frombuffer(image_bytes, dtype=np.uint8)
                return cv2.imdecode(array, cv2.IMREAD_COLOR)

    elif library in {"pillow", "pillow-simd"}:
        import io

        from PIL import Image

        if mode == "disk":

            def decode(path):
                img = Image.open(path)
                img = img.convert("RGB")
                return np.asarray(img)
        else:

            def decode(image_bytes):
                img = Image.open(io.BytesIO(image_bytes))
                img = img.convert("RGB")
                return np.asarray(img)

    elif library == "jpeg4py":
        import jpeg4py

        if mode == "disk":

            def decode(path):
                return jpeg4py.JPEG(path).decode()
        else:

            def decode(image_bytes):
                return jpeg4py.JPEG(image_bytes).decode()

    elif library == "skimage":
        import io

        import skimage.io
        from skimage.io import imread

        if mode == "disk":

            def decode(path):
                return skimage.io.imread(path)
        else:

            def decode(image_bytes):
                # Convert bytes to file-like object
                bytes_io = io.BytesIO(image_bytes)
                return imread(bytes_io)

    elif library == "imageio":
        import imageio.v2 as imageio

        def decode(x):
            return imageio.imread(x)

    elif library == "torchvision":
        import torch
        import torchvision

        if mode == "disk":

            def decode(path):
                image = torchvision.io.read_image(str(path))
                return image.permute(1, 2, 0).numpy()
        else:

            def decode(image_bytes):
                image = torchvision.io.decode_image(torch.frombuffer(image_bytes, dtype=torch.uint8))
                return image.permute(1, 2, 0).numpy()

    elif library == "tensorflow":
        import tensorflow as tf

        if mode == "disk":

            def decode(path):
                image_string = tf.io.read_file(str(path))
                image = tf.io.decode_image(image_string, channels=3)
                return image.numpy()
        else:

            def decode(image_bytes):
                image = tf.io.decode_image(image_bytes, channels=3)
                return image.numpy()

    elif library == "kornia":
        import kornia_rs as K

        if mode == "disk":

            def decode(path):
                return K.read_image_jpeg(str(path))
        else:
            # For memory-based decoding, we need to write bytes to a temporary file
            import os
            import tempfile

            def decode(image_bytes):
                # Create a temporary file with .jpg extension
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    tmp.write(image_bytes)
                    tmp_path = tmp.name

                try:
                    # Read the image from the temporary file
                    return K.read_image_jpeg(tmp_path)
                finally:
                    # Clean up the temporary file
                    tmp_path.unlink()

    else:
        raise ValueError(f"Unsupported library: {library}")

    return library, decode
