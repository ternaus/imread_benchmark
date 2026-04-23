from __future__ import annotations

from typing import Any, cast

import numpy as np

from imread_benchmark.decoders import BaseDecoder

# Module-level singletons: None is picklable, so DataLoader workers can be
# spawned safely. Each worker process initialises on first use.
_tj: Any | None = None
_tjpf_rgb: int | None = None


def _turbo_and_rgb() -> tuple[Any, int]:
    """Lazy init: import TurboJPEG + TJPF_RGB once per worker, not per decode."""
    global _tj, _tjpf_rgb
    if _tj is None:
        from turbojpeg import TJPF_RGB, TurboJPEG

        _tj = TurboJPEG()
        _tjpf_rgb = TJPF_RGB
    return cast("Any", _tj), cast("int", _tjpf_rgb)


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
        tj, tjpf = _turbo_and_rgb()
        return tj.decode(data, pixel_format=tjpf)
