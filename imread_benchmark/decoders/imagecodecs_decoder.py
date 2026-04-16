from __future__ import annotations

import numpy as np

from imread_benchmark.decoders import BaseDecoder


class ImageCodecsDecoder(BaseDecoder):
    """imagecodecs — uses libjpeg-turbo 3.x internally; prebuilt wheels for macOS ARM64."""

    name = "imagecodecs"
    package_name = "imagecodecs"

    def decode(self, data: bytes) -> np.ndarray:
        import imagecodecs

        return imagecodecs.jpeg_decode(data)
