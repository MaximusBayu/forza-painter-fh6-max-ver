from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

import ultra
import refine
from ultra import ultra_image, _scale_shapes, _render_stack
from refine import _evaluate, _hill_climb, _residual_pass, _error_map
from geometry_optimize import render_geometry
from geometry_json import normalize_geometry_payload, ROTATED_ELLIPSE, RECTANGLE


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _synthetic_rgb():
    """Gradient + flat shapes + thin line: structure, shading, and edge detail."""
    h, w = 96, 128
    rgb = np.zeros((h, w, 3), np.uint8)
    ramp = np.linspace(40, 200, w, dtype=np.uint8)
    rgb[:, :, 0] = ramp[None, :]
    rgb[:, :, 1] = 90
    rgb[:, :, 2] = ramp[::-1][None, :]
    cv2.rectangle(rgb, (10, 10), (50, 40), (220, 50, 60), -1)
    cv2.circle(rgb, (90, 60), 20, (30, 200, 90), -1)
    cv2.line(rgb, (5, 80), (120, 70), (250, 250, 250), 2)
    return rgb


def _write_png(path: Path, rgb: np.ndarray):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    assert ok
    path.write_bytes(buf.tobytes())


# ---------------------------------------------------------------------------
# alpha-blended evaluate / paint (the polish phase primitives)
# ---------------------------------------------------------------------------

class TestAlphaEvaluate:
    def test_alpha_blend_color_is_optimal(self):
        # Canvas 100, target 160: at alpha=0.5 the optimal fill is 220
        # (0.5*100 + 0.5*220 = 160 exactly -> gain equals the full old SSE).
        target = np.full((40, 40, 3), 160.0, np.float32)
        canvas = np.full((40, 40, 3), 100.0, np.float32)
        opaque = np.ones((40, 40), bool)
        shape = {"type": int(RECTANGLE), "data": [20, 20, 20.0, 20.0],
                 "color": [0, 0, 0, 255], "score": 0}
        gain, color = _evaluate(target, canvas, opaque, shape, alpha=0.5)
        assert tuple(color) == (220.0, 220.0, 220.0)
        assert gain == pytest.approx(((160.0 - 100.0) ** 2) * 3 * 400, rel=1e-5)

    def test_alpha_color_clipped_to_byte_range(self):
        # Canvas already brighter than target: unclipped optimum would be
        # negative; the clip keeps it a valid color.
        target = np.full((40, 40, 3), 10.0, np.float32)
        canvas = np.full((40, 40, 3), 240.0, np.float32)
        opaque = np.ones((40, 40), bool)
        shape = {"type": int(RECTANGLE), "data": [20, 20, 20.0, 20.0],
                 "color": [0, 0, 0, 255], "score": 0}
        gain, color = _evaluate(target, canvas, opaque, shape, alpha=0.5)
        assert (color >= 0).all() and (color <= 255).all()


class TestHillClimb:
    def test_climb_improves_misaligned_seed(self):
        # Target: bright square at (30, 20); seed shape offset by 6px.
        target = np.zeros((64, 64, 3), np.float32)
        target[12:28, 22:38] = 200.0
        canvas = np.zeros_like(target)
        opaque = np.ones((64, 64), bool)
        seed = {"type": int(RECTANGLE), "data": [36.0, 26.0, 16.0, 16.0],
                "color": [0, 0, 0, 255], "score": 0}
        start = _evaluate(target, canvas, opaque, seed)
        assert start is not None
        gain0, color0 = start
        gain1, _color1 = _hill_climb(target, canvas, opaque, seed, gain0,
                                     color0, 1.0, max_evals=80)
        assert gain1 > gain0
        # Center moved toward the true square center (30, 20).
        assert abs(seed["data"][0] - 30.0) < abs(36.0 - 30.0)

    def test_climb_pass_reduces_more_error_than_plain(self):
        target = _synthetic_rgb().astype(np.float32)
        opaque = np.ones(target.shape[:2], bool)
        plain = np.full_like(target, 128.0)
        climbed = np.full_like(target, 128.0)
        _residual_pass(target, plain, opaque, 3.0, 16.0, 8, 100, True)
        _residual_pass(target, climbed, opaque, 3.0, 16.0, 8, 100, True, climb=48)
        assert (_error_map(target, climbed, opaque).sum()
                <= _error_map(target, plain, opaque).sum())


# ---------------------------------------------------------------------------
# ladder mechanics
# ---------------------------------------------------------------------------

class TestScaleShapes:
    def test_scales_geometry_in_place(self):
        shapes = [{"type": int(ROTATED_ELLIPSE),
                   "data": [10, 20, 5.0, 8.0, 45], "color": [1, 2, 3, 255]}]
        _scale_shapes(shapes, 2.0, 2.0)
        assert shapes[0]["data"][:4] == [20, 40, 10.0, 16.0]
        assert shapes[0]["data"][4] == 45  # rotation untouched

    def test_render_stack_matches_renderer(self):
        background = {"type": int(RECTANGLE), "data": [0, 0, 32, 24],
                      "color": [10, 20, 30, 255], "score": 0}
        shape = {"type": int(RECTANGLE), "data": [16, 12, 10.0, 8.0],
                 "color": [200, 100, 50, 255], "score": 0}
        canvas = _render_stack(background, [shape], 24, 32)
        rendered = render_geometry({"shapes": [background, shape]},
                                   scale=1.0, blend_alpha=True)
        assert np.abs(canvas - rendered.astype(np.float32)).max() <= 1.0


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------

class TestUltraImage:
    def test_end_to_end_schema_budget_fidelity(self, tmp_path):
        cv2.setRNGSeed(7)
        src = tmp_path / "src.png"
        _write_png(src, _synthetic_rgb())
        out = tmp_path / "out.json"
        preview = tmp_path / "preview.png"

        lines = []
        report = ultra_image(src, out, max_shapes=400, max_resolution=256,
                             preview_path=preview, progress=lines.append)

        assert out.exists() and preview.exists()
        assert any("level" in line for line in lines)
        data = json.loads(out.read_text(encoding="utf-8"))
        normalize_geometry_payload(data)
        bg = data["shapes"][0]
        assert bg["data"][:2] == [0, 0]
        for shape in data["shapes"][1:]:
            assert int(shape["type"]) in (int(RECTANGLE), int(ROTATED_ELLIPSE))
            assert isinstance(shape["data"][0], int)
            assert isinstance(shape["data"][1], int)
            assert 1 <= shape["color"][3] <= 255
        assert report["layers"] == len(data["shapes"]) - 1
        assert report["layers"] <= 400

        # Fidelity: must land far closer than the mean-color baseline.
        rendered = render_geometry(data, scale=1.0, blend_alpha=True)
        rgb = _synthetic_rgb()
        h, w = rendered.shape[:2]
        ref = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
        err = np.abs(rendered.astype(np.float32) - ref.astype(np.float32)).mean()
        baseline = np.abs(ref.astype(np.float32) - ref.reshape(-1, 3).mean(0)).mean()
        assert err < baseline * 0.45, f"mean abs err {err} vs baseline {baseline}"

    def test_polish_emits_blendable_output(self, tmp_path):
        cv2.setRNGSeed(7)
        src = tmp_path / "src.png"
        _write_png(src, _synthetic_rgb())
        out = tmp_path / "out.json"
        ultra_image(src, out, max_shapes=400, max_resolution=256)
        data = json.loads(out.read_text(encoding="utf-8"))
        # render must succeed in blend mode regardless of alpha mix
        render_geometry(data, scale=1.0, blend_alpha=True)

    def test_cli_smoke(self, tmp_path, capsys):
        src = tmp_path / "src.png"
        _write_png(src, _synthetic_rgb())
        out = tmp_path / "out.json"
        rc = ultra._main([str(src), "-o", str(out), "--shapes", "80",
                          "--max-res", "256"])
        assert rc == 0
        assert out.exists()
