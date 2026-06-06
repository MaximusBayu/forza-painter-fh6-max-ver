"""S1: rotated-rectangle plumbing (normalize, preview, optimizer render)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from geometry_json import normalize_geometry_payload, ShapeType
import app
from geometry_optimize import render_geometry


class TestNormalizeRectRotation:
    def test_axis_rect_stays_four_values(self):
        payload2 = [
            {"type": 1, "data": [0, 0, 100, 100], "color": [0, 0, 0, 0]},
            {"type": 1, "data": [10, 20, 30, 40], "color": [255, 0, 0, 255]},
        ]
        rect = normalize_geometry_payload(payload2)["shapes"][1]
        assert rect["data"] == [10, 20, 30, 40]

    def test_rotated_rect_keeps_rotation(self):
        payload = [
            {"type": 1, "data": [0, 0, 100, 100], "color": [0, 0, 0, 0]},
            {"type": 1, "data": [10, 20, 30, 40, 45], "color": [255, 0, 0, 255]},
        ]
        rect = normalize_geometry_payload(payload)["shapes"][1]
        assert rect["data"] == [10, 20, 30, 40, 45]
        assert int(rect["type"]) == ShapeType.RECTANGLE

    def test_rotation_360_normalized_to_zero_drops_back_to_four(self):
        payload = [
            {"type": 1, "data": [0, 0, 100, 100], "color": [0, 0, 0, 0]},
            {"type": 1, "data": [10, 20, 30, 40, 360], "color": [255, 0, 0, 255]},
        ]
        rect = normalize_geometry_payload(payload)["shapes"][1]
        assert rect["data"] == [10, 20, 30, 40]


class TestRotatedRectCorners:
    def test_zero_rotation_axis_aligned(self):
        pts = app._rotated_rect_corners(50, 50, 20, 10, 0, 1.0)
        xs = sorted({round(p[0]) for p in pts})
        ys = sorted({round(p[1]) for p in pts})
        assert xs == [40, 60]
        assert ys == [45, 55]

    def test_ninety_degree_swaps_extent(self):
        pts = app._rotated_rect_corners(50, 50, 20, 10, 90, 1.0)
        xs = sorted({round(p[0]) for p in pts})
        ys = sorted({round(p[1]) for p in pts})
        # After 90deg, width/height extents swap.
        assert xs == [45, 55]
        assert ys == [40, 60]


class TestOptimizerRendersRotatedRect:
    def test_rotated_rect_paints_pixels(self):
        data = {
            "shapes": [
                {"type": 1, "data": [0, 0, 40, 40], "color": [0, 0, 0, 255], "score": 0},
                {"type": 1, "data": [20, 20, 20, 6, 45], "color": [255, 0, 0, 255], "score": 0},
            ]
        }
        img = render_geometry(data)
        reds = int((img[:, :, 0] > 200).sum())
        assert reds > 0
        # Rotated bar should differ from the axis-aligned version.
        axis = {
            "shapes": [
                data["shapes"][0],
                {"type": 1, "data": [20, 20, 20, 6], "color": [255, 0, 0, 255], "score": 0},
            ]
        }
        img_axis = render_geometry(axis)
        assert not np.array_equal(img, img_axis)


class TestPreviewRendersRotatedRect:
    def test_pillow_preview_handles_rotated_rect(self, tmp_path, monkeypatch):
        import json

        # Avoid Tk PhotoImage (needs a root); return the raw PIL image instead.
        monkeypatch.setattr(app, "pil_to_photo", lambda img, *a, **k: img)
        payload = {
            "width": 60,
            "height": 60,
            "shapes": [
                {"type": 1, "data": [0, 0, 60, 60], "color": [0, 0, 0, 255]},
                {"type": 1, "data": [30, 30, 30, 8, 30], "color": [255, 255, 255, 255]},
            ],
        }
        path = tmp_path / "g.json"
        path.write_text(json.dumps(payload))
        image = app.render_geometry_json_pillow(path, max_size=120)
        assert image is not None
        arr = np.array(image)
        # The rotated white bar must have painted some pixels.
        assert int((arr[:, :, 0] > 200).sum()) > 0
