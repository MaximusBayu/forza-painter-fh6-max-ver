"""Phase 7: CPU differentiable vector engine (diffvg-free).

Skipped entirely when torch is not installed (optional dependency).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

import geometry_diff as gd
from geometry_json import normalize_geometry_payload, load_normalized_geometry, ShapeType


def _two_tone(h=40, w=40):
    img = np.zeros((h, w, 3), np.uint8)
    img[:, : w // 2] = (200, 40, 40)
    img[:, w // 2 :] = (40, 60, 200)
    return img


def test_optimize_reduces_loss():
    target = _two_tone()
    data, report = gd.optimize_shapes(target, num_shapes=12, iters=50, max_size=40, seed=0)
    assert report["final_loss"] < report["initial_loss"] * 0.6
    # bg + 12 ellipses
    assert len(data["shapes"]) == 13


def test_output_is_valid_geometry():
    target = _two_tone()
    data, _report = gd.optimize_shapes(target, num_shapes=8, iters=20, max_size=40, seed=1)
    norm = normalize_geometry_payload(data)
    assert int(norm["shapes"][0]["type"]) == ShapeType.RECTANGLE  # background
    for shape in norm["shapes"][1:]:
        assert int(shape["type"]) == ShapeType.ROTATED_ELLIPSE
        assert len(shape["data"]) == 5  # x, y, w, h, rot
        assert len(shape["color"]) == 4


def test_render_matches_target_better_than_blank():
    from geometry_optimize import render_geometry, mse

    target = _two_tone()
    data, _ = gd.optimize_shapes(target, num_shapes=16, iters=80, max_size=40, seed=0)
    rendered = render_geometry(normalize_geometry_payload(data), blend_alpha=True)
    # Compare at the optimization resolution (data is at opt-res here).
    assert rendered.shape[:2] == target.shape[:2]
    blank = np.zeros_like(target)
    assert mse(rendered, target) < mse(blank, target)


def test_vectorize_image_writes_json(tmp_path: Path):
    import cv2

    src = tmp_path / "in.png"
    cv2.imwrite(str(src), _two_tone()[:, :, ::-1])  # write BGR
    report = gd.vectorize_image(src, num_shapes=8, iters=15, max_size=32)
    out = Path(report["output"])
    assert out.exists()
    loaded = load_normalized_geometry(out)
    assert len(loaded["shapes"]) >= 2
    assert "final_loss" in report
