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


def test_ssim_loss_zero_on_identical():
    torch = gd.load_torch()
    x = torch.rand(24, 24, 3)
    assert float(gd._ssim_loss(torch, x, x)) < 1e-4           # identical -> ~0
    y = 1.0 - x
    assert float(gd._ssim_loss(torch, x, y)) > float(gd._ssim_loss(torch, x, x))


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


def _vertical_gradient(h=48, w=48):
    ramp = np.linspace(40, 220, h).astype(np.uint8)[:, None]
    img = np.zeros((h, w, 3), np.uint8)
    img[:, :, 0] = ramp            # red rises top->bottom
    img[:, :, 2] = (255 - ramp)    # blue falls -> a smooth gradient, no hard edge
    return img


def test_seed_gradient_shapes_skips_flat_image():
    # A flat panel has no colour slope, so the smooth-gradient detector seeds
    # nothing (var(med) - var(small) ~ 0 everywhere).
    torch = gd.load_torch()
    flat = np.full((32, 32, 3), 100, np.float32) / 255.0
    assert gd._seed_gradient_shapes(torch, flat, 10, "cpu") is None


def test_seed_gradient_shapes_targets_gradient_and_is_translucent():
    torch = gd.load_torch()
    grad = _vertical_gradient().astype(np.float32) / 255.0
    seed = gd._seed_gradient_shapes(torch, grad, 16, "cpu", alpha=0.35)
    assert seed is not None
    assert 0 < seed["cx"].shape[0] <= 16
    # Seeded stamps are ellipses (isr==0) and translucent (alpha ~0.35, not solid).
    assert float(seed["isr"].abs().max()) == 0.0
    alphas = torch.sigmoid(seed["ral"])
    assert float(alphas.max()) < 0.6


def test_refine_with_gradient_shapes_grows_count():
    # A smooth gradient that flat warm stamps render as a band: gradient mode
    # seeds translucent stamps upfront and the refit reduces loss.
    target = _vertical_gradient()
    warm = {"shapes": [
        {"type": 1, "data": [0, 0, 48, 48], "color": [130, 0, 130, 255], "score": 0},
        {"type": 1, "data": [24, 24, 46.0, 46.0], "color": [130, 0, 130, 255], "score": 0},
    ]}
    refined, report = gd.refine_geometry(warm, target, opt_res=48, iters=10,
                                         gradient_shapes=20, chunk=64, device="cpu")
    assert len(warm["shapes"]) < len(refined["shapes"]) <= len(warm["shapes"]) + 20
    assert report["final_loss"] <= report["initial_loss"]
    norm = normalize_geometry_payload(refined)
    for s in norm["shapes"][1:]:
        assert 1 <= s["color"][3] <= 255


def _diagonal_line(h=64, w=64):
    import cv2
    img = np.full((h, w, 3), 235, np.uint8)
    cv2.line(img, (6, 6), (w - 6, h - 6), (20, 20, 20), 2)  # dark diagonal on light bg
    return img


def test_seed_line_shapes_finds_a_line():
    torch = gd.load_torch()
    seed = gd._seed_line_shapes(torch, _diagonal_line().astype(np.float32) / 255.0, 10, "cpu")
    assert seed is not None
    assert seed["cx"].shape[0] >= 1
    assert float(seed["isr"].min()) == 1.0                 # rotated rects, not ellipses
    assert float(torch.sigmoid(seed["ral"]).min()) > 0.6   # near-opaque hard line


def test_seed_line_shapes_none_on_flat():
    torch = gd.load_torch()
    flat = np.full((40, 40, 3), 128, np.float32) / 255.0
    assert gd._seed_line_shapes(torch, flat, 10, "cpu") is None


def test_refine_with_edge_shapes_grows_count():
    # A flat warm with no draw shapes leaves the diagonal line entirely to the
    # edge seeder; the refit then includes it and reduces loss.
    target = _diagonal_line(64, 64)
    warm = {"shapes": [{"type": 1, "data": [0, 0, 64, 64], "color": [235, 235, 235, 255], "score": 0}]}
    refined, report = gd.refine_geometry(warm, target, opt_res=64, iters=8,
                                         edge_shapes=10, chunk=64, device="cpu")
    assert len(refined["shapes"]) > len(warm["shapes"])
    assert report["final_loss"] <= report["initial_loss"]
    normalize_geometry_payload(refined)


def _spot_image(h=64, w=64):
    img = np.full((h, w, 3), 128, np.uint8)  # mid-grey field
    for (cy, cx) in [(16, 16), (16, 48), (48, 16), (48, 48)]:
        img[cy - 1:cy + 1, cx - 1:cx + 1] = (250, 250, 250)  # tiny bright specks
    return img


def test_seed_detail_shapes_finds_spots():
    torch = gd.load_torch()
    seed = gd._seed_detail_shapes(torch, _spot_image().astype(np.float32) / 255.0, 10, "cpu")
    assert seed is not None
    assert seed["cx"].shape[0] >= 1
    assert float(seed["isr"].max()) == 0.0                 # tiny ellipses, not rects
    assert float(torch.sigmoid(seed["ral"]).min()) > 0.6   # near-opaque spots
    extent = torch.nn.functional.softplus(seed["rex"]) + 0.5
    assert float(extent.max()) < 8.0                        # small stamps (spots, not regions)


def test_seed_detail_shapes_none_on_flat():
    torch = gd.load_torch()
    flat = np.full((40, 40, 3), 128, np.float32) / 255.0
    assert gd._seed_detail_shapes(torch, flat, 10, "cpu") is None


def test_refine_with_detail_shapes_grows_count():
    # A flat warm leaves the bright specks to the detail seeder; the refit then
    # includes them and does not increase loss.
    target = _spot_image(64, 64)
    warm = {"shapes": [{"type": 1, "data": [0, 0, 64, 64], "color": [128, 128, 128, 255], "score": 0}]}
    refined, report = gd.refine_geometry(warm, target, opt_res=64, iters=8,
                                         detail_shapes=10, chunk=64, device="cpu")
    assert len(refined["shapes"]) > len(warm["shapes"])
    assert report["final_loss"] <= report["initial_loss"]
    normalize_geometry_payload(refined)
