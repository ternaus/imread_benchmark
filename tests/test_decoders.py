"""
Smoke tests for decoder classes.

Only decoders whose libraries are installed will run; others are skipped.
CI installs opencv-python-headless and pillow, so at minimum those two pass.
"""

from __future__ import annotations

import platform

import numpy as np
import pytest

from imread_benchmark.contracts import OutputContract, validate_output
from imread_benchmark.decoders import REGISTRY, BaseDecoder
from imread_benchmark.decoders.pillow_decoder import PillowDecoder

from .conftest import is_decoder_available

available_decoders = [pytest.param(cls, id=name) for name, cls in REGISTRY.items() if is_decoder_available(cls)]


@pytest.mark.parametrize("decoder_cls", available_decoders)
def test_decode_from_bytes_shape_and_dtype(decoder_cls, jpeg_bytes):
    decoder = decoder_cls()
    result = decoder.decode(jpeg_bytes)
    validate_output(result, OutputContract.normalized_rgb())
    assert isinstance(result, np.ndarray), f"{decoder.name}: decode() must return ndarray"
    assert result.ndim == 3, f"{decoder.name}: expected 3-D array, got {result.ndim}-D"
    assert result.shape[2] == 3, f"{decoder.name}: expected 3 channels, got {result.shape[2]}"
    assert result.dtype == np.uint8, f"{decoder.name}: expected uint8, got {result.dtype}"


@pytest.mark.parametrize("decoder_cls", available_decoders)
def test_decode_path(decoder_cls, jpeg_path):
    decoder = decoder_cls()
    result = decoder.decode_path(str(jpeg_path))
    validate_output(result, OutputContract.normalized_rgb())
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


def test_pillow_returns_materialized_owned_rgb(jpeg_bytes):
    result = PillowDecoder().decode(jpeg_bytes)

    assert result.flags.owndata
    assert result.flags.writeable
    assert result.shape == (48, 64, 3)


# ─── Decoder metadata / contract ──────────────────────────────────────────────
# These tests don't require the third-party library — they just inspect the
# class. So we parametrize over the FULL registry, not `available_decoders`.
# Catches mistakes in new-decoder PRs (missing `name`, wrong `group`, broken
# entry-point, etc.) before CI tries to install the library.

ALLOWED_GROUPS = {"mainstream", "tensorflow"}


@pytest.mark.parametrize(("name", "cls"), list(REGISTRY.items()))
def test_decoder_class_contract(name, cls):
    assert cls.name == name, f"entry-point key '{name}' != cls.name '{cls.name}'"
    assert cls.package_name, f"{name}: package_name is empty"
    assert cls.group in ALLOWED_GROUPS, f"{name}: group '{cls.group}' not in {ALLOWED_GROUPS}"
    assert isinstance(cls.runs_single_here(), bool)
    assert isinstance(cls.runs_dataloader_here(), bool)
    assert issubclass(cls, BaseDecoder)


def test_registry_has_known_decoders():
    """Entry-point discovery in pyproject.toml shouldn't silently drop decoders."""
    expected = {
        "ajpegli",
        "opencv",
        "pillow",
        "skimage",
        "imageio",
        "torchvision",
        "tensorflow",
        "kornia",
        "simplejpeg",
        "turbojpeg",
        "imagecodecs",
        "pyvips",
        "jpeg4py",
    }
    missing = expected - set(REGISTRY)
    assert not missing, f"Registry missing decoders (broken pyproject entry-points?): {sorted(missing)}"


# ─── Platform skip logic ──────────────────────────────────────────────────────
# Pure unit tests on BaseDecoder._matches_current — these drive which decoders
# the CI matrix and the cloud benchmark actually run on each platform, so a
# regression here silently mis-counts what ran where.


def _make_decoder(skip_single=(), skip_dataloader=(), *, in_dataloader=True):
    """Synthesize a BaseDecoder subclass with the given platform attributes."""
    return type(
        "_TestDecoder",
        (BaseDecoder,),
        {
            "name": "tmp",
            "package_name": "tmp",
            "skip_single": list(skip_single),
            "skip_dataloader": list(skip_dataloader),
            "in_dataloader": in_dataloader,
            "decode": lambda self, data: np.zeros((1, 1, 3), dtype=np.uint8),
        },
    )


@pytest.mark.parametrize(
    ("system", "machine", "skips", "expected_runs"),
    [
        ("Linux", "x86_64", [], True),
        ("Linux", "x86_64", [("Linux", "x86_64")], False),
        ("Linux", "aarch64", [("Linux", "x86_64")], True),
        ("Linux", "aarch64", [("Linux", "*")], False),
        ("Darwin", "arm64", [("*", "arm64")], False),
        ("Darwin", "arm64", [("Linux", "*")], True),
        ("Darwin", "arm64", [("Darwin", "x86_64"), ("Darwin", "arm64")], False),
    ],
)
def test_platform_skip_matrix(monkeypatch, system, machine, skips, expected_runs):
    monkeypatch.setattr(platform, "system", lambda: system)
    monkeypatch.setattr(platform, "machine", lambda: machine)
    cls = _make_decoder(skip_single=skips)
    assert cls.runs_single_here() is expected_runs


def test_in_dataloader_false_overrides_everything(monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    cls = _make_decoder(in_dataloader=False)
    assert cls.runs_single_here() is True
    assert cls.runs_dataloader_here() is False
