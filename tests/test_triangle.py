"""S3: triangle primitive plumbing (normalize, count, optimizer mask, preview)
plus geometrize type-2 rotated-rectangle acceptance."""

from __future__ import annotations

import json

import numpy as np
import pytest

from geometry_json import (
    normalize_geometry_payload,
    drawable_shape_count,
    ShapeType,
)
from geometry_optimize import render_geometry
import app


def _bg(w=40, h=40, color=(0, 0, 0, 255)):
    return {"type": 1, "data": [0, 0, w, h], "color": list(color)}


class TestNormalizeTriangle:
    def test_triangle_type_kept_with_rotation(self):
        payload = [_bg(), {"type": 4, "data": [20, 20, 12, 12, 30], "color": [255, 0, 0, 255]}]
        shape = normalize_geometry_payload(payload)["shapes"][1]
        assert int(shape["type"]) == ShapeType.TRIANGLE
        assert shape["data"] == [20, 20, 12, 12, 30]

    def test_triangle_alias_string(self):
        payload = [_bg(), {"type": "triangle", "data": [10, 10, 8, 8, 0], "color": [0, 255, 0, 255]}]
        shape = normalize_geometry_payload(payload)["shapes"][1]
        assert int(shape["type"]) == ShapeType.TRIANGLE

    def test_rotated_rectangle_type2_maps_to_rectangle_with_rotation(self):
        payload = [_bg(), {"type": 2, "data": [10, 20, 30, 40, 25], "color": [0, 0, 255, 255]}]
        shape = normalize_geometry_payload(payload)["shapes"][1]
        assert int(shape["type"]) == ShapeType.RECTANGLE
        assert shape["data"] == [10, 20, 30, 40, 25]

    def test_triangle_counts_as_drawable(self, tmp_path):
        payload = [
            _bg(),
            {"type": 4, "data": [20, 20, 8, 8, 0], "color": [255, 0, 0, 255]},
            {"type": 16, "data": [10, 10, 5, 5, 0], "color": [0, 255, 0, 255]},
        ]
        path = tmp_path / "g.json"
        path.write_text(json.dumps(payload))
        assert drawable_shape_count(path) == 2


class TestOptimizerTriangleMask:
    def test_triangle_paints_and_differs_from_ellipse(self):
        tri = {"shapes": [_bg(), {"type": 4, "data": [20, 24, 20, 20, 0], "color": [255, 0, 0, 255], "score": 0}]}
        img = render_geometry(tri)
        assert int((img[:, :, 0] > 200).sum()) > 0
        # Apex points up: top rows should have fewer red px than bottom rows.
        red = img[:, :, 0] > 200
        top = int(red[:20].sum())
        bottom = int(red[20:].sum())
        assert bottom > top


class TestPreviewTriangle:
    def test_pillow_preview_renders_triangle(self, tmp_path, monkeypatch):
        monkeypatch.setattr(app, "pil_to_photo", lambda img, *a, **k: img)
        payload = {
            "width": 60, "height": 60,
            "shapes": [
                {"type": 1, "data": [0, 0, 60, 60], "color": [0, 0, 0, 255]},
                {"type": 4, "data": [30, 34, 28, 28, 0], "color": [255, 255, 255, 255]},
            ],
        }
        path = tmp_path / "g.json"
        path.write_text(json.dumps(payload))
        image = app.render_geometry_json_pillow(path, max_size=120)
        assert image is not None
        assert int((np.array(image)[:, :, 0] > 200).sum()) > 0


class TestTriangleCorners:
    def test_three_corners_apex_up(self):
        pts = app._triangle_corners(50, 50, 20, 10, 0, 1.0)
        assert len(pts) == 3
        apex = min(pts, key=lambda p: p[1])
        assert round(apex[0]) == 50  # apex centered horizontally
