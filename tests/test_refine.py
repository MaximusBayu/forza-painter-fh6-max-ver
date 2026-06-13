from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

import refine
from refine import (
    _error_map,
    _evaluate,
    _hill_climb,
    _paint,
    _residual_pass,
    _refit_colors,
    refine_image,
)
from geometry_optimize import render_geometry
from geometry_json import normalize_geometry_payload, ROTATED_ELLIPSE, RECTANGLE


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _synthetic_rgb():
    """Gradient background + flat shapes: exercises base AND residual stages."""
    h, w = 96, 128
    rgb = np.zeros((h, w, 3), np.uint8)
    ramp = np.linspace(40, 200, w, dtype=np.uint8)
    rgb[:, :, 0] = ramp[None, :]
    rgb[:, :, 1] = 90
    rgb[:, :, 2] = ramp[::-1][None, :]
    cv2.rectangle(rgb, (10, 10), (50, 40), (220, 50, 60), -1)
    cv2.circle(rgb, (90, 60), 20, (30, 200, 90), -1)
    return rgb


def _write_png(path: Path, rgb: np.ndarray, alpha: np.ndarray | None = None):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if alpha is not None:
        bgr = np.dstack([bgr, alpha])
    ok, buf = cv2.imencode(".png", bgr)
    assert ok
    path.write_bytes(buf.tobytes())


# ---------------------------------------------------------------------------
# error map + candidate evaluation
# ---------------------------------------------------------------------------

class TestErrorMap:
    def test_zero_when_identical(self):
        rgb = _synthetic_rgb().astype(np.float32)
        opaque = np.ones(rgb.shape[:2], bool)
        assert _error_map(rgb, rgb.copy(), opaque).max() == 0.0

    def test_zero_where_transparent(self):
        rgb = _synthetic_rgb().astype(np.float32)
        opaque = np.ones(rgb.shape[:2], bool)
        opaque[:20] = False
        err = _error_map(rgb, np.zeros_like(rgb), opaque)
        assert (err[:20] == 0.0).all()
        assert err[20:].max() > 0.0


class TestEvaluate:
    def test_positive_gain_on_mismatched_region(self):
        target = np.full((40, 40, 3), 200.0, np.float32)
        canvas = np.zeros_like(target)
        opaque = np.ones((40, 40), bool)
        shape = {"type": int(RECTANGLE), "data": [20, 20, 20.0, 20.0],
                 "color": [0, 0, 0, 255], "score": 0}
        gain, color = _evaluate(target, canvas, opaque, shape)
        assert gain > 0
        assert tuple(color) == (200.0, 200.0, 200.0)

    def test_none_when_fully_transparent(self):
        target = np.full((40, 40, 3), 200.0, np.float32)
        canvas = np.zeros_like(target)
        opaque = np.zeros((40, 40), bool)
        shape = {"type": int(RECTANGLE), "data": [20, 20, 20.0, 20.0],
                 "color": [0, 0, 0, 255], "score": 0}
        assert _evaluate(target, canvas, opaque, shape) is None


class TestSliverGate:
    """Anti-artifact gates in _evaluate / _hill_climb (the streak fix)."""

    def test_rejects_subpixel_thin_shape(self):
        # A sub-pixel-thin axis is a degenerate ~zero-area fit; reject it even
        # over a perfectly-matching region (it would otherwise score well).
        target = np.full((40, 40, 3), 200.0, np.float32)
        canvas = np.zeros_like(target)
        opaque = np.ones((40, 40), bool)
        thin = {"type": int(RECTANGLE), "data": [20, 20, 30.0, 0.5],
                "color": [0, 0, 0, 255], "score": 0}
        assert _evaluate(target, canvas, opaque, thin) is None

    def test_keeps_coherent_thin_feature(self):
        # A high-aspect shape over a uniformly-matching band is a real thin
        # feature (hair/line) — coherence is high, so it must be accepted.
        target = np.full((40, 40, 3), 200.0, np.float32)
        canvas = np.zeros_like(target)
        opaque = np.ones((40, 40), bool)
        streak = {"type": int(RECTANGLE), "data": [20, 20, 38.0, 2.0],
                  "color": [0, 0, 0, 255], "score": 0}
        assert (38.0 / 2.0) > refine.MAX_ASPECT  # in the gated regime
        scored = _evaluate(target, canvas, opaque, streak)
        assert scored is not None and scored[0] > 0

    def test_rejects_incoherent_streak(self):
        # Same aspect, but the band bridges two mismatched colors: one fill
        # color cannot match both, so coherence is low -> rejected.
        target = np.zeros((40, 40, 3), np.float32)
        target[:, :20] = (250.0, 0.0, 0.0)
        target[:, 20:] = (0.0, 0.0, 250.0)
        canvas = np.zeros_like(target)
        opaque = np.ones((40, 40), bool)
        streak = {"type": int(RECTANGLE), "data": [20, 20, 38.0, 2.0],
                  "color": [0, 0, 0, 255], "score": 0}
        assert _evaluate(target, canvas, opaque, streak) is None

    def test_low_aspect_bypasses_coherence(self):
        # Only elongated shapes pay the coherence test; a compact shape over
        # the same mismatched colors is accepted normally.
        target = np.zeros((40, 40, 3), np.float32)
        target[:, :20] = (250.0, 0.0, 0.0)
        target[:, 20:] = (0.0, 0.0, 250.0)
        canvas = np.zeros_like(target)
        opaque = np.ones((40, 40), bool)
        square = {"type": int(RECTANGLE), "data": [20, 20, 36.0, 36.0],
                  "color": [0, 0, 0, 255], "score": 0}
        scored = _evaluate(target, canvas, opaque, square)
        assert scored is not None and scored[0] > 0

    def test_hill_climb_clamps_growth(self):
        # On a uniform bright target the climb would grow the shape without
        # bound; the clamp caps each axis at CLIMB_GROWTH x the seed size.
        target = np.full((48, 48, 3), 200.0, np.float32)
        canvas = np.zeros_like(target)
        opaque = np.ones((48, 48), bool)
        seed_w = seed_h = 4.0
        seed = {"type": int(RECTANGLE), "data": [24.0, 24.0, seed_w, seed_h],
                "color": [0, 0, 0, 255], "score": 0}
        start = _evaluate(target, canvas, opaque, seed)
        assert start is not None
        _hill_climb(target, canvas, opaque, seed, start[0], start[1], 1.0, 120)
        assert abs(seed["data"][2]) <= refine.CLIMB_GROWTH * seed_w + 1e-6
        assert abs(seed["data"][3]) <= refine.CLIMB_GROWTH * seed_h + 1e-6


# ---------------------------------------------------------------------------
# residual pass
# ---------------------------------------------------------------------------

class TestResidualPass:
    def test_pass_reduces_total_error(self):
        target = _synthetic_rgb().astype(np.float32)
        opaque = np.ones(target.shape[:2], bool)
        canvas = np.full_like(target, 128.0)
        before = _error_map(target, canvas, opaque).sum()
        added = _residual_pass(target, canvas, opaque, sigma=3.0, delta=16.0,
                               min_area=8, budget=200, use_rects=True)
        after = _error_map(target, canvas, opaque).sum()
        assert added, "expected the pass to place shapes"
        assert after < before

    def test_respects_budget(self):
        target = _synthetic_rgb().astype(np.float32)
        opaque = np.ones(target.shape[:2], bool)
        canvas = np.zeros_like(target)
        added = _residual_pass(target, canvas, opaque, sigma=3.0, delta=10.0,
                               min_area=4, budget=3, use_rects=True)
        assert len(added) <= 3

    def test_no_shapes_when_canvas_matches(self):
        target = _synthetic_rgb().astype(np.float32)
        opaque = np.ones(target.shape[:2], bool)
        added = _residual_pass(target, target.copy(), opaque, sigma=3.0,
                               delta=10.0, min_area=4, budget=100, use_rects=True)
        assert added == []


# ---------------------------------------------------------------------------
# color refit + dead-layer removal
# ---------------------------------------------------------------------------

class TestRefitColors:
    def test_drops_fully_occluded_shape(self):
        target = np.full((40, 40, 3), 100.0, np.float32)
        opaque = np.ones((40, 40), bool)
        background = {"type": int(RECTANGLE), "data": [0, 0, 40, 40],
                      "color": [0, 0, 0, 255], "score": 0}
        hidden = {"type": int(RECTANGLE), "data": [20, 20, 10.0, 10.0],
                  "color": [255, 0, 0, 255], "score": 0}
        cover = {"type": int(RECTANGLE), "data": [20, 20, 40.0, 40.0],
                 "color": [0, 255, 0, 255], "score": 0}
        kept = _refit_colors(background, [hidden, cover], target, opaque)
        assert kept == [cover]

    def test_refits_color_to_visible_mean(self):
        target = np.full((40, 40, 3), 0.0, np.float32)
        target[10:30, 10:30] = (50.0, 150.0, 250.0)
        opaque = np.ones((40, 40), bool)
        background = {"type": int(RECTANGLE), "data": [0, 0, 40, 40],
                      "color": [0, 0, 0, 255], "score": 0}
        shape = {"type": int(RECTANGLE), "data": [20, 20, 20.0, 20.0],
                 "color": [255, 255, 255, 255], "score": 0}
        kept = _refit_colors(background, [shape], target, opaque)
        assert kept[0]["color"] == [50, 150, 250, 255]


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------

class TestRefineImage:
    def test_end_to_end_schema_and_fidelity(self, tmp_path):
        cv2.setRNGSeed(7)
        src = tmp_path / "src.png"
        _write_png(src, _synthetic_rgb())
        out = tmp_path / "out.json"
        preview = tmp_path / "preview.png"

        report = refine_image(src, out, max_shapes=300, preview_path=preview)

        assert out.exists() and preview.exists()
        data = json.loads(out.read_text(encoding="utf-8"))
        # Importer-compatible: normalization accepts it untouched.
        normalize_geometry_payload(data)
        bg = data["shapes"][0]
        assert int(bg["type"]) == int(RECTANGLE)
        assert bg["data"][:2] == [0, 0]
        for shape in data["shapes"][1:]:
            assert int(shape["type"]) in (int(RECTANGLE), int(ROTATED_ELLIPSE))
            assert len(shape["data"]) in (4, 5)
            assert len(shape["color"]) == 4 and shape["color"][3] == 255
        assert report["layers"] == len(data["shapes"]) - 1
        assert report["layers"] <= 300
        assert report["base_layers"] <= report["layers"]

        # Fidelity: render must land far closer to the target than the
        # mean-color baseline (flat art -> should be near-exact).
        rendered = render_geometry(data, scale=1.0, blend_alpha=False)
        rgb = _synthetic_rgb()
        h, w = rgb.shape[:2]
        rendered = rendered[:h, :w]
        err = np.abs(rendered.astype(np.float32) - rgb.astype(np.float32)).mean()
        baseline = np.abs(rgb.astype(np.float32) - rgb.reshape(-1, 3).mean(0)).mean()
        assert err < baseline * 0.45, f"mean abs err {err} vs baseline {baseline}"

    def test_budget_one(self, tmp_path):
        src = tmp_path / "src.png"
        _write_png(src, _synthetic_rgb())
        out = tmp_path / "out.json"
        report = refine_image(src, out, max_shapes=1)
        assert report["layers"] <= 1

    def test_transparent_image(self, tmp_path):
        rgb = _synthetic_rgb()
        alpha = np.zeros(rgb.shape[:2], np.uint8)  # fully transparent
        src = tmp_path / "src.png"
        _write_png(src, rgb, alpha)
        out = tmp_path / "out.json"
        report = refine_image(src, out)
        data = json.loads(out.read_text(encoding="utf-8"))
        assert report["layers"] == 0
        assert data["shapes"][0]["color"][3] == 0  # transparent background

    def test_cli_smoke(self, tmp_path, capsys):
        src = tmp_path / "src.png"
        _write_png(src, _synthetic_rgb())
        out = tmp_path / "out.json"
        rc = refine._main([str(src), "-o", str(out), "--shapes", "50"])
        assert rc == 0
        assert out.exists()
        report = json.loads(capsys.readouterr().out)
        assert report["layers"] <= 50
