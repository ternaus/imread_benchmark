from __future__ import annotations

import numpy as np

from imread_benchmark.decoders import BaseDecoder


class KorniaDecoder(BaseDecoder):
    name = "kornia"
    package_name = "kornia-rs"

    def decode(self, data: bytes) -> np.ndarray:
        import kornia_rs as K

        return K.decode_image_jpeg(data)

    def decode_path(self, path: str) -> np.ndarray:
        import kornia_rs as K

        return K.read_image_jpeg(path)
