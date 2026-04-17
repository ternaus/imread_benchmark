from __future__ import annotations

from typing import Any

import numpy as np

from imread_benchmark.decoders import BaseDecoder


def _decode(arr_or_path: Any) -> np.ndarray:
    import jpeg4py

    return jpeg4py.JPEG(arr_or_path).decode()


class Jpeg4pyDecoder(BaseDecoder):
    """
    jpeg4py — Cython bindings for libjpeg-turbo. Linux-only (no Windows wheel,
    untested on macOS). Returns uint8 RGB arrays.
    """

    name = "jpeg4py"
    package_name = "jpeg4py"
    skip_single = (("Darwin", "*"), ("Windows", "*"))
    skip_dataloader = (("Darwin", "*"), ("Windows", "*"))

    def decode(self, data: bytes) -> np.ndarray:
        # jpeg4py.JPEG accepts either a filename or a uint8 numpy buffer.
        return _decode(np.frombuffer(data, dtype=np.uint8))

    def decode_path(self, path: str) -> np.ndarray:
        return _decode(path)
