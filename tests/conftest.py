from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def _make_jpeg_bytes(width: int = 64, height: int = 48) -> bytes:
    """Create a minimal valid JPEG in memory using Pillow."""
    arr = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
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
