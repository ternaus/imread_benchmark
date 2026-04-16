from __future__ import annotations

from io import BytesIO

import numpy as np

from imread_benchmark.decoders import BaseDecoder


class PillowDecoder(BaseDecoder):
    name = "pillow"
    package_name = "pillow"

    def decode(self, data: bytes) -> np.ndarray:
        from PIL import Image

        return np.asarray(Image.open(BytesIO(data)).convert("RGB"))

    def decode_path(self, path: str) -> np.ndarray:
        from PIL import Image

        return np.asarray(Image.open(path).convert("RGB"))
