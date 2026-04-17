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

        # Newer kornia_rs (>= 0.1.x) requires an explicit colour mode. "rgb"
        # matches what every other decoder in the suite produces.
        return K.read_image_jpeg(path, "rgb")
