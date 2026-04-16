"""
Smoke tests for decoder classes.

Only decoders whose libraries are installed will run; others are skipped.
CI installs opencv-python-headless and pillow, so at minimum those two pass.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from imread_benchmark.decoders import REGISTRY, BaseDecoder


def _is_available(decoder_cls: type[BaseDecoder]) -> bool:
    """Return True if the library required by this decoder can be imported."""
    pkg = decoder_cls.package_name
    # Map package distribution names to importable module names
    _import_map = {
        "opencv-python-headless": "cv2",
        "pillow": "PIL",
        "pillow-simd": "PIL",
        "scikit-image": "skimage",
        "imageio": "imageio",
        "torchvision": "torchvision",
        "tensorflow": "tensorflow",
        "kornia-rs": "kornia_rs",
        "simplejpeg": "simplejpeg",
        "PyTurboJPEG": "turbojpeg",
        "imagecodecs": "imagecodecs",
        "pyvips": "pyvips",
    }
    module_name = _import_map.get(pkg, pkg)
    try:
        importlib.import_module(module_name)
    except ImportError:
        return False
    return True


available_decoders = [pytest.param(cls, id=name) for name, cls in REGISTRY.items() if _is_available(cls)]


@pytest.mark.parametrize("decoder_cls", available_decoders)
def test_decode_from_bytes_shape_and_dtype(decoder_cls, jpeg_bytes):
    decoder = decoder_cls()
    result = decoder.decode(jpeg_bytes)
    assert isinstance(result, np.ndarray), f"{decoder.name}: decode() must return ndarray"
    assert result.ndim == 3, f"{decoder.name}: expected 3-D array, got {result.ndim}-D"
    assert result.shape[2] == 3, f"{decoder.name}: expected 3 channels, got {result.shape[2]}"
    assert result.dtype == np.uint8, f"{decoder.name}: expected uint8, got {result.dtype}"


@pytest.mark.parametrize("decoder_cls", available_decoders)
def test_decode_path(decoder_cls, jpeg_path):
    decoder = decoder_cls()
    result = decoder.decode_path(str(jpeg_path))
    assert isinstance(result, np.ndarray)
    assert result.ndim == 3
    assert result.shape[2] == 3
    assert result.dtype == np.uint8


@pytest.mark.parametrize("decoder_cls", available_decoders)
def test_decode_nonzero_output(decoder_cls, jpeg_bytes):
    """Output should not be an all-zeros array (sanity check that decoding worked)."""
    decoder = decoder_cls()
    result = decoder.decode(jpeg_bytes)
    assert result.max() > 0, f"{decoder.name}: decoded image is all zeros"
