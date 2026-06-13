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


# ---------------------------------------------------------------------------
# R5 warm-start global refiner
# ---------------------------------------------------------------------------

def _warm_geometry():
    """bg + one axis rect + one rotated ellipse (exercises both convention paths)."""
    return {"shapes": [
        {"type": 1, "data": [0, 0, 48, 32], "color": [20, 30, 40, 255], "score": 0},
        {"type": 1, "data": [16, 16, 18.0, 12.0], "color": [200, 50, 60, 255], "score": 0},
        {"type": 16, "data": [34, 16, 8.0, 10.0, 30], "color": [40, 180, 90, 255], "score": 0},
    ]}


def test_soft_render_matches_render_geometry():
    # The differentiable raster MUST reproduce the importer's hard render
    # (geometry_optimize.render_geometry); a convention or compositing-order
    # bug shows up here as a large mismatch.
    from geometry_optimize import render_geometry

    torch = gd.load_torch()
    data = _warm_geometry()
    P, _bg = gd._geometry_to_params(torch, data, 1.0, "cpu")
    soft = gd._soft_render(torch, P, 32, 48, 25.0, 64, "cpu")
    soft_np = np.array(soft.detach().cpu().tolist(), dtype=np.float32)
    hard = render_geometry(data, 1.0, blend_alpha=True).astype(np.float32) / 255.0
    assert soft_np.shape == hard.shape
    mean_abs = float(np.abs(soft_np - hard).mean())
    assert mean_abs < 0.04, f"soft raster diverges from hard render: {mean_abs}"


def test_refine_geometry_reduces_loss_and_preserves_count():
    target = _two_tone(32, 32)
    warm = {"shapes": [
        {"type": 1, "data": [0, 0, 32, 32], "color": [10, 10, 10, 255], "score": 0},
        {"type": 1, "data": [10, 16, 14.0, 28.0], "color": [180, 60, 60, 255], "score": 0},
        {"type": 16, "data": [22, 16, 7.0, 13.0, 90], "color": [60, 70, 190, 255], "score": 0},
    ]}
    refined, report = gd.refine_geometry(warm, target, opt_res=32, iters=12,
                                         chunk=64, device="cpu")
    assert report["final_loss"] <= report["initial_loss"]
    assert len(refined["shapes"]) == len(warm["shapes"])  # count unchanged
    norm = normalize_geometry_payload(refined)
    assert int(norm["shapes"][0]["type"]) == ShapeType.RECTANGLE
    for s in norm["shapes"][1:]:
        assert int(s["type"]) in (int(ShapeType.RECTANGLE), int(ShapeType.ROTATED_ELLIPSE))
        assert 1 <= s["color"][3] <= 255


def test_diff_refine_image_with_warm_json(tmp_path: Path):
    import cv2

    src = tmp_path / "in.png"
    cv2.imwrite(str(src), _two_tone()[:, :, ::-1])
    warm_path = tmp_path / "warm.json"
    warm_path.write_text(json.dumps(_warm_geometry()), encoding="utf-8")
    out = tmp_path / "out.json"
    preview = tmp_path / "prev.png"
    report = gd.diff_refine_image(src, out, warm_json=str(warm_path), opt_res=48,
                                  iters=5, preview_path=str(preview))
    assert out.exists() and preview.exists()
    loaded = load_normalized_geometry(out)
    assert len(loaded["shapes"]) == 3
    assert "device" in report and "seconds" in report and report["layers"] == 2


def test_refine_with_add_shapes_grows_count():
    # A tiny warm start leaves most of the two-tone image as residual, so the
    # mid-refine seeding pass adds new shapes to cover it.
    target = _two_tone(48, 48)
    warm = {"shapes": [
        {"type": 1, "data": [0, 0, 48, 48], "color": [0, 0, 0, 255], "score": 0},
        {"type": 1, "data": [6, 6, 6.0, 6.0], "color": [200, 40, 40, 255], "score": 0},
    ]}
    refined, report = gd.refine_geometry(warm, target, opt_res=48, iters=8,
                                         add_shapes=5, chunk=64, device="cpu")
    assert len(warm["shapes"]) < len(refined["shapes"]) <= len(warm["shapes"]) + 5
    assert report["final_loss"] <= report["initial_loss"]
    normalize_geometry_payload(refined)
