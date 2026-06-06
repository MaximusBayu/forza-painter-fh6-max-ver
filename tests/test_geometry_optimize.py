from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from geometry_optimize import (
    render_geometry,
    ssim,
    mse,
    metric_scale,
    prune_occluded,
    prune_to_target,
    dedupe_shapes,
    optimize_geometry,
    refit_colors,
)
from geometry_json import load_normalized_geometry


def _bg(w=64, h=64, color=(10, 10, 10, 255)):
    return {"type": 1, "data": [0, 0, w, h], "color": list(color), "score": 0}


def _ellipse(x, y, w, h, rot, color):
    return {"type": 16, "data": [x, y, w, h, rot], "color": list(color), "score": 0}


def _rect(x, y, w, h, color):
    return {"type": 1, "data": [x, y, w, h], "color": list(color), "score": 0}


class TestRender:
    def test_background_fill(self):
        data = {"shapes": [_bg(8, 8, (100, 150, 200, 255))]}
        img = render_geometry(data)
        assert img.shape == (8, 8, 3)
        assert tuple(img[0, 0]) == (100, 150, 200)

    def test_transparent_background_is_black(self):
        data = {"shapes": [_bg(8, 8, (100, 150, 200, 0))]}
        img = render_geometry(data)
        assert tuple(img[0, 0]) == (0, 0, 0)

    def test_opaque_rectangle_paints(self):
        data = {"shapes": [_bg(20, 20), _rect(10, 10, 10, 10, (255, 0, 0, 255))]}
        img = render_geometry(data)
        assert tuple(img[10, 10]) == (255, 0, 0)
        assert tuple(img[0, 0]) == (10, 10, 10)  # corner untouched

    def test_alpha_blend_halfway(self):
        data = {"shapes": [_bg(20, 20, (0, 0, 0, 255)), _rect(10, 10, 20, 20, (200, 200, 200, 128))]}
        img = render_geometry(data, blend_alpha=True)
        # 200 * (128/255) ~= 100
        assert 95 <= img[10, 10][0] <= 105

    def test_alpha_ignored_when_disabled(self):
        data = {"shapes": [_bg(20, 20, (0, 0, 0, 255)), _rect(10, 10, 20, 20, (200, 200, 200, 128))]}
        img = render_geometry(data, blend_alpha=False)
        assert tuple(img[10, 10]) == (200, 200, 200)


class TestMetrics:
    def test_ssim_identical_is_one(self):
        img = (np.random.default_rng(0).random((40, 40, 3)) * 255).astype(np.uint8)
        assert ssim(img, img) == pytest.approx(1.0, abs=1e-6)

    def test_mse_identical_is_zero(self):
        img = (np.random.default_rng(1).random((40, 40, 3)) * 255).astype(np.uint8)
        assert mse(img, img) == 0.0

    def test_ssim_drops_when_different(self):
        a = np.zeros((40, 40, 3), np.uint8)
        b = np.full((40, 40, 3), 255, np.uint8)
        assert ssim(a, b) < 0.1

    def test_metric_scale_downscales_large(self):
        data = {"shapes": [_bg(1024, 512)]}
        assert metric_scale(data, 256) == pytest.approx(0.25)

    def test_metric_scale_no_upscale(self):
        data = {"shapes": [_bg(100, 100)]}
        assert metric_scale(data, 256) == 1.0


class TestPruneOccluded:
    def test_fully_occluded_shape_removed(self):
        # A small red rect fully covered by a later big opaque black rect.
        data = {
            "shapes": [
                _bg(40, 40, (0, 0, 0, 255)),
                _rect(20, 20, 6, 6, (255, 0, 0, 255)),       # hidden
                _rect(20, 20, 40, 40, (50, 50, 50, 255)),     # covers everything
            ]
        }
        new_data, removed = prune_occluded(data, scale=1.0)
        assert removed == [0]
        assert len(new_data["shapes"]) == 2  # bg + the big cover

    def test_visible_shape_kept(self):
        data = {
            "shapes": [
                _bg(40, 40, (0, 0, 0, 255)),
                _rect(10, 10, 8, 8, (255, 0, 0, 255)),
                _rect(30, 30, 8, 8, (0, 255, 0, 255)),
            ]
        }
        new_data, removed = prune_occluded(data, scale=1.0)
        assert removed == []
        assert len(new_data["shapes"]) == 3

    def test_occlusion_is_lossless_on_render(self):
        data = {
            "shapes": [
                _bg(40, 40, (0, 0, 0, 255)),
                _rect(20, 20, 6, 6, (255, 0, 0, 255)),
                _rect(20, 20, 40, 40, (50, 50, 50, 255)),
            ]
        }
        before = render_geometry(data)
        new_data, _ = prune_occluded(data, scale=1.0)
        after = render_geometry(new_data)
        assert np.array_equal(before, after)


class TestDedupe:
    def test_exact_duplicate_removed(self):
        s = _rect(10, 10, 8, 8, (255, 0, 0, 255))
        data = {"shapes": [_bg(40, 40), dict(s), dict(s), dict(s)]}
        new_data, removed = dedupe_shapes(data)
        assert removed == 2
        assert len(new_data["shapes"]) == 2


class TestPruneToTarget:
    def test_reduces_to_target(self):
        data = {
            "shapes": [_bg(60, 60, (0, 0, 0, 255))]
            + [_rect(5 + i, 5 + i, 4, 4, (i * 20 % 255, 0, 0, 255)) for i in range(6)]
        }
        new_data, dropped = prune_to_target(data, target_count=3, scale=1.0)
        assert len(new_data["shapes"]) == 4  # bg + 3
        assert len(dropped) == 3

    def test_target_above_count_is_noop(self):
        data = {"shapes": [_bg(40, 40), _rect(10, 10, 8, 8, (255, 0, 0, 255))]}
        new_data, dropped = prune_to_target(data, target_count=10, scale=1.0)
        assert dropped == []
        assert len(new_data["shapes"]) == 2


class TestOptimizeGeometry:
    def test_writes_optimized_file_and_report(self, tmp_path: Path):
        payload = [
            _bg(40, 40, (0, 0, 0, 255)),
            _rect(20, 20, 6, 6, (255, 0, 0, 255)),      # hidden
            _rect(20, 20, 40, 40, (50, 50, 50, 255)),    # cover
            _ellipse(10, 10, 5, 5, 0, (0, 0, 255, 255)),
            _ellipse(10, 10, 5, 5, 0, (0, 0, 255, 255)),  # duplicate
        ]
        in_path = tmp_path / "geom.json"
        in_path.write_text(json.dumps(payload))
        report = optimize_geometry(in_path)
        out_path = Path(report["output"])
        assert out_path.exists()
        assert out_path != in_path
        assert report["removed_duplicate"] >= 1
        assert report["removed_occluded"] >= 1
        assert report["layers_out"] < report["layers_in"]
        # Lossless passes only -> high SSIM.
        assert report["ssim"] >= 0.99
        # Output round-trips through the loader.
        reloaded = load_normalized_geometry(out_path)
        assert len(reloaded["shapes"]) >= 2

    def test_never_overwrites_input(self, tmp_path: Path):
        payload = [_bg(20, 20), _rect(10, 10, 8, 8, (255, 0, 0, 255))]
        in_path = tmp_path / "geom.json"
        original = json.dumps(payload)
        in_path.write_text(original)
        optimize_geometry(in_path)
        assert in_path.read_text() == original


class TestRefitColors:
    def test_recovers_flat_region_colors(self):
        # Two non-overlapping opaque rects with known colors.
        truth = {
            "shapes": [
                _bg(40, 40, (0, 0, 0, 255)),
                _rect(10, 20, 12, 12, (200, 30, 40, 255)),
                _rect(30, 20, 12, 12, (20, 180, 60, 255)),
            ]
        }
        source = render_geometry(truth, blend_alpha=False)
        # Corrupt the colors, then re-fit against the source.
        corrupted = {
            "shapes": [
                truth["shapes"][0],
                _rect(10, 20, 12, 12, (0, 0, 0, 255)),
                _rect(30, 20, 12, 12, (0, 0, 0, 255)),
            ]
        }
        refit = refit_colors(corrupted, source, scale=1.0)
        # Flat regions -> mean equals the original flat color exactly.
        assert tuple(refit["shapes"][1]["color"][:3]) == (200, 30, 40)
        assert tuple(refit["shapes"][2]["color"][:3]) == (20, 180, 60)

    def test_refit_improves_ssim(self):
        truth = {
            "shapes": [
                _bg(40, 40, (0, 0, 0, 255)),
                _rect(20, 20, 20, 20, (180, 90, 30, 255)),
            ]
        }
        source = render_geometry(truth, blend_alpha=False)
        corrupted = {
            "shapes": [truth["shapes"][0], _rect(20, 20, 20, 20, (0, 0, 0, 255))]
        }
        before = ssim(render_geometry(corrupted), source)
        refit = refit_colors(corrupted, source, scale=1.0)
        after = ssim(render_geometry(refit), source)
        assert after > before
        assert after >= 0.99

    def test_translucent_shape_left_unchanged(self):
        data = {
            "shapes": [
                _bg(20, 20, (0, 0, 0, 255)),
                _rect(10, 10, 20, 20, (123, 45, 67, 128)),
            ]
        }
        source = np.full((20, 20, 3), 200, np.uint8)
        refit = refit_colors(data, source, scale=1.0)
        assert tuple(refit["shapes"][1]["color"]) == (123, 45, 67, 128)
