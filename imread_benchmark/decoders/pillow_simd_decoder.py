from __future__ import annotations

from io import BytesIO

import numpy as np

from imread_benchmark.decoders import BaseDecoder


class PillowSIMDDecoder(BaseDecoder):
    """Pillow-SIMD drop-in — Linux x86-64 only, no macOS ARM64 wheels."""

    name = "pillow-simd"
    package_name = "pillow-simd"
    group = "pillow-simd"
    # Pillow-SIMD only ships x86 wheels.
    skip_single = (("Darwin", "*"), ("Linux", "aarch64"), ("Linux", "arm64"))
    # torchvision's transitive Pillow pin silently downgrades us to vanilla
    # Pillow inside the same venv; the measurement would be a lie.
    in_dataloader = False

    def decode(self, data: bytes) -> np.ndarray:
        from PIL import Image

        return np.asarray(Image.open(BytesIO(data)).convert("RGB"))

    def decode_path(self, path: str) -> np.ndarray:
        from PIL import Image

        return np.asarray(Image.open(path).convert("RGB"))
