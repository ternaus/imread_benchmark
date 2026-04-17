from __future__ import annotations

from typing import Any

import numpy as np

from imread_benchmark.decoders import BaseDecoder

# Module-level singleton: None is picklable, so DataLoader workers can be
# spawned safely. Each worker process initialises its own instance on first use.
_tj = None


def _get_tj() -> Any:
    global _tj
    if _tj is None:
        from turbojpeg import TurboJPEG

        _tj = TurboJPEG()
    return _tj


class TurboJPEGDecoder(BaseDecoder):
    """
    PyTurboJPEG — Python binding for libjpeg-turbo.
    macOS: requires `brew install jpeg-turbo`.
    """

    name = "turbojpeg"
    package_name = "PyTurboJPEG"

    def decode(self, data: bytes) -> np.ndarray:
        # PyTurboJPEG defaults to BGR. Pass pixel_format=TJPF_RGB so we match
        # every other decoder in the suite — otherwise the benchmark compares
        # different colour orders and the paper's numbers become meaningless.
        from turbojpeg import TJPF_RGB

        return _get_tj().decode(data, pixel_format=TJPF_RGB)
