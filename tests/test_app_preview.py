"""Phase 1 regression: the numpy fast-path ellipse blit must be pixel-identical
to the pure-Python fallback it replaces."""

from __future__ import annotations

import numpy as np
import pytest

import app
from PIL import Image


def _render(image_size, params, force_pure_python):
    img = Image.new("RGB", image_size, (10, 20, 30))
    if force_pure_python:
        # Defeat the numpy fast path.
        orig = app._load_preview_numpy
        app._load_preview_numpy = lambda: None
        try:
            app.draw_preview_ellipse_pillow(img, *params)
        finally:
            app._load_preview_numpy = orig
    else:
        app.draw_preview_ellipse_pillow(img, *params)
    return np.array(img)


@pytest.mark.parametrize(
    "params",
    [
        # x, y, w, h, rot_deg, color, scale
        (100, 80, 40, 20, 0.0, (255, 0, 0), 1.0),
        (100, 80, 40, 20, 45.0, (0, 255, 0), 1.0),
        (60, 60, 30, 30, 30.0, (0, 0, 255), 1.5),
        (20, 20, 10, 25, 123.0, (200, 100, 50), 2.0),
        (150, 100, 5, 50, 90.0, (255, 255, 0), 1.0),
    ],
)
def test_numpy_path_matches_pure_python(params):
    size = (200, 160)
    fast = _render(size, params, force_pure_python=False)
    slow = _render(size, params, force_pure_python=True)
    assert np.array_equal(fast, slow), (
        f"max abs diff {int(np.abs(fast.astype(int) - slow.astype(int)).max())}"
    )


def test_offscreen_ellipse_is_noop():
    size = (50, 50)
    params = (1000, 1000, 10, 10, 0.0, (255, 0, 0), 1.0)
    out = _render(size, params, force_pure_python=False)
    expected = np.dstack([np.full(size[::-1], c, np.uint8) for c in (10, 20, 30)])
    assert np.array_equal(out, expected)
