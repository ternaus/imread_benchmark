from __future__ import annotations

import numpy as np

from imread_benchmark.decoders import BaseDecoder


class SkimageDecoder(BaseDecoder):
    name = "skimage"
    package_name = "scikit-image"

    def decode(self, data: bytes) -> np.ndarray:
        import io

        import skimage.io

        return skimage.io.imread(io.BytesIO(data))

    def decode_path(self, path: str) -> np.ndarray:
        import skimage.io

        return skimage.io.imread(path)
