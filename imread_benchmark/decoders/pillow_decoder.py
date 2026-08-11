from __future__ import annotations

from io import BytesIO

import numpy as np

from imread_benchmark.decoders import BaseDecoder


class PillowDecoder(BaseDecoder):
    name = "pillow"
    package_name = "pillow"

    def decode(self, data: bytes) -> np.ndarray:
        from PIL import Image

        with Image.open(BytesIO(data)) as image:
            image.load()
            rgb = image.convert("RGB")
            rgb.load()
            return np.array(rgb, dtype=np.uint8, copy=True)

    def decode_path(self, path: str) -> np.ndarray:
        from PIL import Image

        with Image.open(path) as image:
            image.load()
            rgb = image.convert("RGB")
            rgb.load()
            return np.array(rgb, dtype=np.uint8, copy=True)
