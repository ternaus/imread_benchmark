from __future__ import annotations

import numpy as np

from imread_benchmark.decoders import BaseDecoder


class AJpegliDecoder(BaseDecoder):
    """ajpegli — Python bindings for Google's JPEGli decoder."""

    name = "ajpegli"
    package_name = "ajpegli"

    def decode(self, data: bytes) -> np.ndarray:
        import ajpegli

        return ajpegli.imdecode(data, mode="RGB")

    def decode_path(self, path: str) -> np.ndarray:
        import ajpegli

        return ajpegli.imread(path, mode="RGB")
