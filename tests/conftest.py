from __future__ import annotations

import importlib
import io
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from PIL import Image

if TYPE_CHECKING:
    from imread_benchmark.decoders import BaseDecoder

# Maps PyPI distribution name → top-level importable module. Drives test
# skipping for decoders whose third-party library isn't installed in this venv.
PKG_TO_MODULE = {
    "ajpegli": "ajpegli",
    "opencv-python-headless": "cv2",
    "pillow": "PIL",
    "scikit-image": "skimage",
    "imageio": "imageio",
    "torchvision": "torchvision",
    "tensorflow": "tensorflow",
    "kornia-rs": "kornia_rs",
    "simplejpeg": "simplejpeg",
    "PyTurboJPEG": "turbojpeg",
    "imagecodecs": "imagecodecs",
    "pyvips": "pyvips",
    "jpeg4py": "jpeg4py",
}


def is_decoder_available(decoder_cls: type[BaseDecoder]) -> bool:
    """
    Return True iff the third-party library backing this decoder can be
    imported AND initialised. Some libraries (pyvips, jpeg4py, turbojpeg)
    wrap a C library via cffi/ctypes and raise `OSError` at import time
    when the .so/.dylib is missing — not `ImportError` — so we catch
    broadly here.
    """
    try:
        importlib.import_module(PKG_TO_MODULE.get(decoder_cls.package_name, decoder_cls.package_name))
    except (ImportError, OSError):
        return False
    return True


def _make_jpeg_bytes(width: int = 64, height: int = 48, *, seed: int = 0) -> bytes:
    """Random-content RGB JPEG. Deterministic for a given seed."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _make_solid_color_jpeg(rgb: tuple[int, int, int], width: int = 64, height: int = 48) -> bytes:
    """Solid-colour JPEG. Used for channel-order checks (pure-red etc.)."""
    arr = np.full((height, width, 3), rgb, dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="JPEG", quality=95)
    return buf.getvalue()


@pytest.fixture(scope="session")
def jpeg_bytes() -> bytes:
    return _make_jpeg_bytes()


@pytest.fixture(scope="session")
def jpeg_path(tmp_path_factory) -> Path:
    data = _make_jpeg_bytes()
    p = tmp_path_factory.mktemp("images") / "test.jpg"
    p.write_bytes(data)
    return p


@pytest.fixture(scope="session")
def red_jpeg_bytes() -> bytes:
    """Pure-red JPEG. Decoders that return BGR will have low R / high B and fail the channel test."""
    return _make_solid_color_jpeg((230, 10, 10))


@pytest.fixture(scope="session")
def pillow_reference(jpeg_bytes) -> np.ndarray:
    """PIL-decoded RGB ground truth for cross-decoder agreement."""
    return np.asarray(Image.open(io.BytesIO(jpeg_bytes)).convert("RGB"))


@pytest.fixture(scope="session")
def jpeg_dir(tmp_path_factory) -> Path:
    """Build a directory with a handful of JPEGs for worker-level integration tests."""
    d = tmp_path_factory.mktemp("jpeg_dir")
    for i in range(4):
        (d / f"img_{i:02d}.jpg").write_bytes(_make_jpeg_bytes(seed=i))
    return d
