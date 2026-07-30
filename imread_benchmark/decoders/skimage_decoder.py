from __future__ import annotations

import numpy as np

from imread_benchmark.decoders import BaseDecoder


class SkimageDecoder(BaseDecoder):
    name = "skimage"
    package_name = "scikit-image"

    def decode(self, data: bytes) -> np.ndarray:
        import io

        from skimage.io import imread

        return imread(io.BytesIO(data))

    def decode_path(self, path: str) -> np.ndarray:
        from skimage.io import imread

        return imread(path)
