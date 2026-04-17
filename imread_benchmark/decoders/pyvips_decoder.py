from __future__ import annotations

import logging

import numpy as np

from imread_benchmark.decoders import BaseDecoder

logging.getLogger("pyvips").setLevel(logging.WARNING)


class PyVipsDecoder(BaseDecoder):
    """
    pyvips — libvips Python bindings.
    macOS: requires `brew install vips` (libvips is NOT bundled in the pip wheel).
    """

    name = "pyvips"
    package_name = "pyvips"
    # libvips spawns GLib worker threads at import time. PyTorch's default
    # `fork` start method on Linux/aarch64 copies the pthread IDs but not the
    # threads themselves → DataLoader workers hang forever waiting on threads
    # that don't exist in the child. (Linux x86 + macOS get away with it.)
    skip_dataloader = (("Linux", "aarch64"),)

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
