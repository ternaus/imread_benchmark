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
    # libvips spawns a GLib worker threadpool at module import. PyTorch
    # DataLoader with num_workers>0 forks the parent; the child inherits
    # pthread IDs but not the threads themselves → first libvips call in the
    # worker deadlocks waiting on a thread that doesn't exist.
    #
    # Originally scoped to Linux/aarch64 with a "x86 + macOS get away with
    # it" caveat — empirically wrong: c4-standard-16 (Intel, 16 vCPU) hung
    # at workers=2. More vCPUs ⇒ larger libvips threadpool ⇒ more
    # opportunities for the fork race.
    #
    # `multiprocessing_context='spawn'` would dodge it but pays full Python
    # re-import per worker, which is not how anyone runs pyvips in practice
    # — the resulting numbers would be misleading. The honest finding is
    # "pyvips is incompatible with the default fork-based DataLoader".
    in_dataloader = False

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
