from __future__ import annotations

import numpy as np

from imread_benchmark.decoders import BaseDecoder


class SimpleJpegDecoder(BaseDecoder):
    """
    simplejpeg — CFFI binding to libjpeg-turbo.
    macOS: requires `brew install jpeg-turbo`.
    """

    name = "simplejpeg"
    package_name = "simplejpeg"

    def decode(self, data: bytes) -> np.ndarray:
        import simplejpeg

        return simplejpeg.decode_jpeg(data, colorspace="RGB")
