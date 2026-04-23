"""
Cross-decoder correctness tests.

The whole point of the benchmark is "library X decodes JPEG Y faster than library Z".
That comparison is only meaningful if all libraries decode JPEG Y to the same pixels.
These tests guard the comparison's validity:

  - test_matches_pillow_reference  — every decoder agrees with PIL on the same JPEG
                                     (within libjpeg vs libjpeg-turbo IDCT noise)
  - test_channel_order_is_rgb      — a pure-red JPEG decodes as red, not blue
                                     (catches accidental BGR — the canonical opencv trap)

Decoders whose libraries aren't installed are silently skipped via the
`available_decoders` parametrize list.
"""

from __future__ import annotations

import numpy as np
import pytest

from imread_benchmark.decoders import REGISTRY

from .conftest import is_decoder_available

# Tolerance for cross-decoder pixel agreement.
#   - libjpeg vs libjpeg-turbo IDCT differs by a few LSBs
#   - chroma upsampling defaults differ (e.g. fancy vs simple)
# 10/255 mean abs error is generous for matching content but still ~8x smaller
# than what a BGR-vs-RGB swap would produce (~85/255 on random content).
PIXEL_AGREEMENT_MAX_MAE = 10.0


available_decoders = [
    pytest.param(cls, id=name) for name, cls in REGISTRY.items() if is_decoder_available(cls) and cls.runs_single_here()
]


@pytest.mark.parametrize("decoder_cls", available_decoders)
def test_matches_pillow_reference(decoder_cls, jpeg_bytes, pillow_reference):
    """Decoded pixels match PIL's reference within IDCT-precision tolerance."""
    out = decoder_cls().decode(jpeg_bytes)
    assert out.shape == pillow_reference.shape, (
        f"{decoder_cls.name}: shape {out.shape} != PIL ref {pillow_reference.shape}"
    )
    mae = float(np.abs(out.astype(np.int32) - pillow_reference.astype(np.int32)).mean())
    assert mae < PIXEL_AGREEMENT_MAX_MAE, (
        f"{decoder_cls.name}: mean abs diff vs PIL = {mae:.2f} (> {PIXEL_AGREEMENT_MAX_MAE}). "
        f"Likely a colour-space or channel-order bug."
    )


@pytest.mark.parametrize("decoder_cls", available_decoders)
def test_channel_order_is_rgb(decoder_cls, red_jpeg_bytes):
    """Pure-red JPEG must decode with R >> B. Catches accidental BGR output."""
    out = decoder_cls().decode(red_jpeg_bytes)
    r_mean = float(out[..., 0].mean())
    b_mean = float(out[..., 2].mean())
    assert r_mean > b_mean + 100, (
        f"{decoder_cls.name}: pure-red JPEG decoded with R={r_mean:.0f}, B={b_mean:.0f}. "
        f"Likely BGR output. Channel order is wrong."
    )
