from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class BaseDecoder(ABC):
    name: str
    package_name: str

    @abstractmethod
    def decode(self, data: bytes) -> np.ndarray:
        """Decode JPEG bytes into an (H, W, 3) uint8 RGB numpy array."""

    def decode_path(self, path: str) -> np.ndarray:
        """Decode a JPEG file by path. Override for library-native path APIs."""
        return self.decode(Path(path).read_bytes())

    def get_num_threads(self) -> int:
        """Return the number of threads the library uses internally. Override per library."""
        return 1

    def set_num_threads(self, n: int) -> None:
        """Set the number of threads the library uses. No-op for single-threaded libraries."""


# Populated below — import order determines nothing; each entry is the class itself.
from imread_benchmark.decoders.imagecodecs_decoder import ImageCodecsDecoder  # noqa: E402
from imread_benchmark.decoders.imageio_decoder import ImageIODecoder  # noqa: E402
from imread_benchmark.decoders.jpeg4py_decoder import Jpeg4pyDecoder  # noqa: E402
from imread_benchmark.decoders.kornia_decoder import KorniaDecoder  # noqa: E402
from imread_benchmark.decoders.opencv_decoder import OpenCVDecoder  # noqa: E402
from imread_benchmark.decoders.pillow_decoder import PillowDecoder  # noqa: E402
from imread_benchmark.decoders.pillow_simd_decoder import PillowSIMDDecoder  # noqa: E402
from imread_benchmark.decoders.pyvips_decoder import PyVipsDecoder  # noqa: E402
from imread_benchmark.decoders.simplejpeg_decoder import SimpleJpegDecoder  # noqa: E402
from imread_benchmark.decoders.skimage_decoder import SkimageDecoder  # noqa: E402
from imread_benchmark.decoders.tensorflow_decoder import TensorFlowDecoder  # noqa: E402
from imread_benchmark.decoders.torchvision_decoder import TorchvisionDecoder  # noqa: E402
from imread_benchmark.decoders.turbojpeg_decoder import TurboJPEGDecoder  # noqa: E402

REGISTRY: dict[str, type[BaseDecoder]] = {
    "opencv": OpenCVDecoder,
    "pillow": PillowDecoder,
    "pillow-simd": PillowSIMDDecoder,
    "skimage": SkimageDecoder,
    "imageio": ImageIODecoder,
    "torchvision": TorchvisionDecoder,
    "tensorflow": TensorFlowDecoder,
    "kornia": KorniaDecoder,
    "simplejpeg": SimpleJpegDecoder,
    "turbojpeg": TurboJPEGDecoder,
    "imagecodecs": ImageCodecsDecoder,
    "pyvips": PyVipsDecoder,
    "jpeg4py": Jpeg4pyDecoder,
}

__all__ = ["REGISTRY", "BaseDecoder"]
