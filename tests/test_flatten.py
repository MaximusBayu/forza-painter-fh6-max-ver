from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

import flatten
from flatten import (
    _quantize,
    _regions,
    _fit_region,
    _build_payload,
    palette_lock,
    flatten_image,
)
from geometry_optimize import render_geometry, _shape_bbox, _shape_mask
from geometry_json import normalize_geometry_payload, ROTATED_ELLIPSE, RECTANGLE


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _iou(shape, region_bool):
    h, w = region_bool.shape
    x0, y0, x1, y1 = _shape_bbox(shape, w, h, 1.0)
    full = np.zeros((h, w), bool)
    local = _shape_mask(shape, x0, y0, x1, y1, 1.0)
    if local.size:
        full[y0:y1, x0:x1] = local
    inter = np.logical_and(full, region_bool).sum()
    union = np.logical_or(full, region_bool).sum()
    return inter / union if union else 0.0


def _three_color_rgb():
    rgb = np.zeros((30, 30, 3), np.uint8)
    rgb[:10] = (200, 30, 40)
    rgb[10:20] = (30, 180, 60)
    rgb[20:] = (40, 50, 200)
    return rgb


# ---------------------------------------------------------------------------
# Task 2 — quantize
# ---------------------------------------------------------------------------

class TestQuantize:
    def test_count_caps_to_distinct_colors(self):
        rgb = _three_color_rgb()
        opaque = np.ones(rgb.shape[:2], bool)
        labels, palette = _quantize(rgb, n_colors=8, opaque_mask=opaque)
        # Only 3 distinct source colors -> palette capped at 3.
        assert palette.shape[0] == 3
        # Every opaque pixel got a label.
        assert (labels[opaque] >= 0).all()

    def test_excludes_transparent(self):
        rgb = _three_color_rgb()
        opaque = np.ones(rgb.shape[:2], bool)
        opaque[:10] = False  # hide the red band
        labels, palette = _quantize(rgb, n_colors=8, opaque_mask=opaque)
        assert palette.shape[0] == 2
        assert (labels[:10] == -1).all()


# ---------------------------------------------------------------------------
# Task 3 — disjoint regions become separate shapes
# ---------------------------------------------------------------------------

class TestDisjointRegions:
    def test_two_separated_squares_two_shapes(self):
        h, w = 40, 80
        labels = np.ones((h, w), np.int32)  # color 1 = background fill
        labels[15:25, 10:20] = 0            # square A (color 0)
        labels[15:25, 60:70] = 0            # square B (color 0), disjoint
        palette = np.array([[200, 30, 40], [10, 10, 10]], np.uint8)
        opaque = np.ones((h, w), bool)
        data = _build_payload(
            np.zeros((h, w, 3), np.uint8), labels, palette, opaque,
            had_alpha=False, use_rects=True, min_area=12,
        )
        color0 = [s for s in data["shapes"][1:] if tuple(s["color"][:3]) == (200, 30, 40)]
        assert len(color0) == 2  # disjoint same-color patches = 2 shapes


# ---------------------------------------------------------------------------
# Task 4 — shape fitting
# ---------------------------------------------------------------------------

class TestFitShape:
    def test_rect_region_one_rect_high_iou(self):
        mask = np.zeros((60, 80), np.uint8)
        mask[15:45, 20:60] = 255  # axis-aligned filled rectangle
        shapes = _fit_region(mask, use_rects=True, min_area=12)
        assert len(shapes) == 1
        s = shapes[0]
        assert int(s["type"]) == int(RECTANGLE)
        assert _iou(s, mask > 0) >= 0.95

    def test_rect_center_convention(self):
        mask = np.zeros((40, 80), np.uint8)
        mask[10:30, 30:50] = 255  # center (40, 20), full 20x20
        shapes = _fit_region(mask, use_rects=True, min_area=12)
        s = shapes[0]
        assert abs(s["data"][0] - 40) <= 1   # cx is CENTER not corner
        assert abs(s["data"][1] - 20) <= 1

    def test_circle_region_ellipse_semi_axis(self):
        mask = np.zeros((80, 80), np.uint8)
        cv2.circle(mask, (40, 40), 15, 255, -1)
        shapes = _fit_region(mask, use_rects=False, min_area=12)
        assert len(shapes) == 1
        s = shapes[0]
        assert int(s["type"]) == int(ROTATED_ELLIPSE)
        # data = [cx, cy, w_semi, h_semi, rot]; circle radius 15 -> semi ~ 15.
        assert abs(s["data"][2] - 15) <= 3
        assert abs(s["data"][3] - 15) <= 3
        assert _iou(s, mask > 0) >= 0.90


# ---------------------------------------------------------------------------
# Task 5 — payload schema
# ---------------------------------------------------------------------------

class TestPayload:
    def test_normalizes_and_bg_is_full_canvas(self):
        h, w = 30, 30
        rgb = _three_color_rgb()
        opaque = np.ones((h, w), bool)
        labels, palette = _quantize(rgb, 8, opaque)
        data = _build_payload(rgb, labels, palette, opaque,
                              had_alpha=False, use_rects=True, min_area=12)
        norm = normalize_geometry_payload(data)
        bg = norm["shapes"][0]
        assert bg["data"] == [0, 0, w, h]
        assert len(norm["shapes"]) > 1  # has drawables

    def test_transparent_source_bg_alpha_zero(self):
        h, w = 30, 30
        rgb = _three_color_rgb()
        opaque = np.ones((h, w), bool)
        opaque[:10] = False
        labels, palette = _quantize(rgb, 8, opaque)
        data = _build_payload(rgb, labels, palette, opaque,
                              had_alpha=True, use_rects=True, min_area=12)
        assert data["shapes"][0]["color"][3] == 0


# ---------------------------------------------------------------------------
# Task 6 — palette lock
# ---------------------------------------------------------------------------

class TestPaletteLock:
    def test_reduces_distinct_colors(self):
        rng = np.random.default_rng(0)
        shapes = [{"type": 1, "data": [0, 0, 64, 64], "color": [0, 0, 0, 255], "score": 0}]
        for _ in range(60):
            c = [int(v) for v in rng.integers(0, 256, 3)] + [255]
            shapes.append({"type": 1, "data": [10, 10, 4, 4], "color": c, "score": 1})
        data = {"shapes": shapes}
        palette = (rng.integers(0, 256, (8, 3))).astype(np.uint8)
        locked = palette_lock(data, palette)
        distinct = {tuple(s["color"][:3]) for s in locked["shapes"][1:]}
        assert len(distinct) <= 8
        for s in locked["shapes"][1:]:
            assert "palette" in s
        normalize_geometry_payload(locked)  # still valid


# ---------------------------------------------------------------------------
# Task 7 — end-to-end
# ---------------------------------------------------------------------------

class TestFlattenImage:
    def _write_png(self, path, rgb, alpha=None):
        if alpha is None:
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(path), bgr)
        else:
            bgra = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGRA)
            bgra[:, :, 3] = alpha
            cv2.imwrite(str(path), bgra)

    def test_end_to_end_writes_json_and_preview(self, tmp_path: Path):
        rgb = _three_color_rgb()
        img = tmp_path / "flat.png"
        self._write_png(img, rgb)
        out = tmp_path / "out.json"
        preview = tmp_path / "preview.png"
        report = flatten_image(img, out, n_colors=8, preview_path=preview)
        assert out.exists() and preview.exists()
        assert report["layers"] >= 3
        # WYSIWYG: rendered output resembles the quantized source.
        data = normalize_geometry_payload(__import__("json").loads(out.read_text()))
        rendered = render_geometry(data, blend_alpha=False)
        assert rendered.shape == (rgb.shape[0], rgb.shape[1], 3)

    def test_all_transparent_no_crash(self, tmp_path: Path):
        rgb = np.zeros((20, 20, 3), np.uint8)
        alpha = np.zeros((20, 20), np.uint8)
        img = tmp_path / "blank.png"
        self._write_png(img, rgb, alpha)
        out = tmp_path / "blank.json"
        report = flatten_image(img, out, n_colors=8)
        assert out.exists()
        assert report["layers"] == 0
        data = __import__("json").loads(out.read_text())
        assert data["shapes"][0]["color"][3] == 0  # transparent bg

    def test_unreadable_raises(self, tmp_path: Path):
        with pytest.raises(flatten.PreprocessError):
            flatten_image(tmp_path / "nope.png", tmp_path / "x.json")


# ---------------------------------------------------------------------------
# Hybrid (R2) support: ellipse-only base + stopAt parsing
# ---------------------------------------------------------------------------

class TestHybridSupport:
    def test_ellipse_only_base_has_no_rects(self):
        # use_rects=False -> every drawable must be a rotated ellipse (type 16),
        # because the generator's -resume restore drops rectangles.
        rgb = _three_color_rgb()
        opaque = np.ones(rgb.shape[:2], bool)
        labels, palette = _quantize(rgb, 8, opaque)
        data = _build_payload(rgb, labels, palette, opaque,
                              had_alpha=False, use_rects=False, min_area=12)
        for s in data["shapes"][1:]:
            assert int(s["type"]) == int(ROTATED_ELLIPSE)

    def test_generator_stop_at_parses_ini(self, tmp_path: Path):
        from generator_backend import generator_stop_at
        ini = tmp_path / "p.ini"
        ini.write_text("randomSamples = 200\nstopAt = 1234\nsaveEvery = 50\n", encoding="utf-8")
        assert generator_stop_at(ini) == 1234

    def test_generator_stop_at_defaults_when_missing(self, tmp_path: Path):
        from generator_backend import generator_stop_at
        ini = tmp_path / "p.ini"
        ini.write_text("randomSamples = 200\n", encoding="utf-8")
        assert generator_stop_at(ini, default=-1) == -1
        assert generator_stop_at(tmp_path / "nope.ini", default=7) == 7

    def test_hybrid_base_path_is_outside_output(self):
        from generator_backend import hybrid_base_path, HYBRID_BASE_DIR
        p = hybrid_base_path("C:/imgs/Custom Art.png")
        assert p.parent == HYBRID_BASE_DIR
        assert p.name == "Custom Art.flatbase.json"

    def test_write_hybrid_settings_overrides_budget(self, tmp_path: Path):
        from generator_backend import write_hybrid_settings, SettingProfile, generator_stop_at
        ini = tmp_path / "prof.ini"
        ini.write_text("description = X\nrandomSamples = 200\nstopAt = 1800\nsaveAt = 1800\n",
                       encoding="utf-8")
        setting = SettingProfile(index=0, source="bundled", path=ini, label="X")
        out = write_hybrid_settings(setting, 963)
        text = out.read_text(encoding="utf-8")
        assert generator_stop_at(out) == 963
        assert "saveAt = 963" in text
        assert "randomSamples = 200" in text  # other keys preserved


# ---------------------------------------------------------------------------
# R3 — superpixel labeling
# ---------------------------------------------------------------------------

class TestSuperpixel:
    def test_labels_cover_opaque_and_palette_sized(self):
        rgb = _three_color_rgb()
        opaque = np.ones(rgb.shape[:2], bool)
        labels, palette = flatten._superpixel(rgb, n_segments=12, opaque_mask=opaque)
        assert (labels[opaque] >= 0).all()
        assert palette.shape[0] >= 2
        # every label index is within the palette
        assert labels.max() < palette.shape[0]

    def test_excludes_transparent(self):
        rgb = _three_color_rgb()
        opaque = np.ones(rgb.shape[:2], bool)
        opaque[:10] = False
        labels, palette = flatten._superpixel(rgb, n_segments=8, opaque_mask=opaque)
        assert (labels[:10] == -1).all()

    def test_end_to_end_superpixel(self, tmp_path: Path):
        rgb = _three_color_rgb()
        img = tmp_path / "sp.png"
        cv2.imwrite(str(img), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        out = tmp_path / "sp.json"
        report = flatten_image(img, out, method="superpixel", n_segments=12)
        assert out.exists()
        assert report["layers"] >= 1
        data = normalize_geometry_payload(__import__("json").loads(out.read_text()))
        assert len(data["shapes"]) > 1
