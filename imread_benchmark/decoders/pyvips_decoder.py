from __future__ import annotations

import logging

import numpy as np

from imread_benchmark.decoders import BaseDecoder

logging.getLogger("pyvips").setLevel(logging.WARNING)


class PyVipsDecoder(BaseDecoder):
    """
    pyvips — libvips Python bindings.

    libvips comes from the `pyvips-binary` PyPI wheel (CFFI API mode), so no
    `brew install vips` / `apt install libvips-dev` is needed on supported
    platforms (Linux x86_64 + aarch64 glibc/musl, macOS x86_64 + arm64,
    Windows). Falls back to the system libvips on anything else.
    """

    name = "pyvips"
    package_name = "pyvips"

    def decode(self, data: bytes) -> np.ndarray:
        import pyvips

        img = pyvips.Image.new_from_buffer(data, "", access="sequential")
        return img.numpy()

    def decode_path(self, path: str) -> np.ndarray:
        import pyvips

        img = pyvips.Image.new_from_file(path, access="sequential")
        return img.numpy()

    def get_num_threads(self) -> int:
        import os

        env = os.environ.get("VIPS_CONCURRENCY")
        if env is not None:
            return int(env)
        # pyvips default: one thread per physical core
        return os.cpu_count() or 1

    def set_num_threads(self, n: int) -> None:
        import os

        # Must be set before pyvips initialises its thread pool (first import).
        # Works because we import pyvips lazily inside decode().
        os.environ["VIPS_CONCURRENCY"] = str(n)
