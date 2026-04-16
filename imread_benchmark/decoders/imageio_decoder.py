from __future__ import annotations

import numpy as np

from imread_benchmark.decoders import BaseDecoder


class ImageIODecoder(BaseDecoder):
    name = "imageio"
    package_name = "imageio"

    def decode(self, data: bytes) -> np.ndarray:
        import io

        import imageio.v2 as imageio

        return imageio.imread(io.BytesIO(data))

    def decode_path(self, path: str) -> np.ndarray:
        import imageio.v2 as imageio

        return imageio.imread(path)
