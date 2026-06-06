"""Offline geometry post-optimizer + metric harness.

Improvement-plan Phase 0 (measurement harness) + Phase 3' (occlusion / importance
prune + same-shape dedupe). Pure-numpy, no game required, fully offline.

Design constraints (from the plan's Hard Constraints):
- Output stays the same geometry-JSON schema consumed by
  ``geometry_json.load_normalized_geometry`` (``{"shapes": [...]}`` with the
  background as ``shapes[0]``). The importer is never touched.
- Rendering matches the in-game stacking model: back-to-front painter's
  algorithm with per-layer alpha blending (one solid BGRA per layer).
- The original JSON is never modified in place; optimization writes a new file.

Note on "fewer layers" (see plan Critique): pruning the JSON does NOT reduce
in-game layer count by itself (the importer fills every template slot). The
value here is (a) letting a JSON *fit* a smaller template the user chose, and
(b) a measurement harness every later phase reports against.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from geometry_json import (
    ShapeType,
    load_normalized_geometry,
    normalize_geometry_payload,
)

RECTANGLE = int(ShapeType.RECTANGLE)
TRIANGLE = int(ShapeType.TRIANGLE)
ROTATED_ELLIPSE = int(ShapeType.ROTATED_ELLIPSE)


# ---------------------------------------------------------------------------
# Rendering (numpy, matches preview + in-game stacking semantics)
# ---------------------------------------------------------------------------

def _shape_bbox(shape, width, height, scale):
    """Return integer (x0, y0, x1, y1) clamped bbox for a normalized shape."""
    data = shape["data"]
    x = float(data[0]) * scale
    y = float(data[1]) * scale
    w = float(data[2]) * scale
    h = float(data[3]) * scale
    stype = int(shape["type"])
    if stype == ROTATED_ELLIPSE or len(data) >= 5:
        # Rotated primitive: bound by the rotated extent.
        rot = float(data[4]) if len(data) >= 5 else 0.0
        if stype == ROTATED_ELLIPSE:
            rx = max(h, 1.0)
            ry = max(w, 1.0)
            theta = np.deg2rad(-90.0 + rot)
        else:  # rotated rectangle
            rx = max(w * 0.5, 0.5)
            ry = max(h * 0.5, 0.5)
            theta = np.deg2rad(rot)
        ct, st = abs(np.cos(theta)), abs(np.sin(theta))
        ex = rx * ct + ry * st
        ey = rx * st + ry * ct
        cx, cy = x, y
    else:
        ex = w * 0.5
        ey = h * 0.5
        cx, cy = x, y
    x0 = int(np.floor(cx - ex - 1.0))
    x1 = int(np.ceil(cx + ex + 1.0))
    y0 = int(np.floor(cy - ey - 1.0))
    y1 = int(np.ceil(cy + ey + 1.0))
    x0 = max(0, min(width, x0))
    x1 = max(0, min(width, x1))
    y0 = max(0, min(height, y0))
    y1 = max(0, min(height, y1))
    return x0, y0, x1, y1


def _shape_mask(shape, x0, y0, x1, y1, scale):
    """Boolean mask of the shape inside its bbox (local coords)."""
    if x1 <= x0 or y1 <= y0:
        return np.zeros((0, 0), dtype=bool)
    data = shape["data"]
    stype = int(shape["type"])
    cx = float(data[0]) * scale
    cy = float(data[1]) * scale
    w = float(data[2]) * scale
    h = float(data[3]) * scale
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dx = xx + 0.5 - cx
    dy = yy + 0.5 - cy
    if stype == ROTATED_ELLIPSE:
        rx = max(h, 1.0)
        ry = max(w, 1.0)
        theta = np.deg2rad(-90.0 + (float(data[4]) if len(data) >= 5 else 0.0))
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        xr = dx * cos_t + dy * sin_t
        yr = -dx * sin_t + dy * cos_t
        return (xr * xr) / (rx * rx) + (yr * yr) / (ry * ry) <= 1.0
    if stype == TRIANGLE:
        # Isoceles stencil in the w x h box: apex at local yr=-h/2, base at +h/2.
        theta = np.deg2rad(float(data[4]) if len(data) >= 5 else 0.0)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        xr = dx * cos_t + dy * sin_t
        yr = -dx * sin_t + dy * cos_t
        hw = max(w * 0.5, 0.0)
        hh = max(h * 0.5, 0.0)
        if hh <= 0.0:
            return np.zeros(dx.shape, dtype=bool)
        frac = np.clip((yr + hh) / (2.0 * hh), 0.0, 1.0)  # 0 at apex, 1 at base
        return (yr >= -hh) & (yr <= hh) & (np.abs(xr) <= hw * frac)

    # Rectangle (optionally rotated when a 5th rotation value is present).
    half_w = max(w * 0.5, 0.0)
    half_h = max(h * 0.5, 0.0)
    if len(data) >= 5 and float(data[4]) % 360.0 != 0.0:
        theta = np.deg2rad(float(data[4]))
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        xr = dx * cos_t + dy * sin_t
        yr = -dx * sin_t + dy * cos_t
        return (np.abs(xr) <= half_w) & (np.abs(yr) <= half_h)
    return (np.abs(dx) <= half_w) & (np.abs(dy) <= half_h)


def render_geometry(data, scale=1.0, blend_alpha=True):
    """Render a normalized geometry dict to an (H, W, 3) uint8 RGB array.

    Back-to-front painter's algorithm. With ``blend_alpha`` the per-layer alpha
    byte blends (matches in-game); otherwise shapes are drawn opaque (matches
    the GUI preview).
    """
    shapes = data["shapes"]
    bg = shapes[0]
    img_w = max(1, int(round(float(bg["data"][2]) * scale)))
    img_h = max(1, int(round(float(bg["data"][3]) * scale)))
    bg_r, bg_g, bg_b, bg_a = (int(v) for v in bg["color"])
    canvas = np.empty((img_h, img_w, 3), dtype=np.float32)
    canvas[:, :] = (bg_r, bg_g, bg_b) if bg_a > 0 else (0, 0, 0)

    for shape in shapes[1:]:
        color = shape.get("color", [])
        if len(color) < 3:
            continue
        a = int(color[3]) if len(color) >= 4 else 255
        if a <= 0:
            continue
        x0, y0, x1, y1 = _shape_bbox(shape, img_w, img_h, scale)
        mask = _shape_mask(shape, x0, y0, x1, y1, scale)
        if mask.size == 0 or not mask.any():
            continue
        rgb = np.array([int(color[0]), int(color[1]), int(color[2])], np.float32)
        region = canvas[y0:y1, x0:x1]
        if blend_alpha and a < 255:
            af = a / 255.0
            region[mask] = region[mask] * (1.0 - af) + rgb * af
        else:
            region[mask] = rgb
    return np.clip(canvas, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Metrics (pure numpy; no skimage/cv2 dependency)
# ---------------------------------------------------------------------------

def mse(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    return float(np.mean((a - b) ** 2))


def _box_filter(img, radius):
    """Mean filter via integral image (pure numpy, separable-equivalent)."""
    pad = radius
    padded = np.pad(img, ((pad, pad), (pad, pad)), mode="edge")
    integ = np.cumsum(np.cumsum(padded, axis=0), axis=1)
    integ = np.pad(integ, ((1, 0), (1, 0)), mode="constant")
    k = 2 * radius + 1
    h, w = img.shape
    s = (
        integ[k:k + h, k:k + w]
        - integ[0:h, k:k + w]
        - integ[k:k + h, 0:w]
        + integ[0:h, 0:w]
    )
    return s / float(k * k)


def ssim(a, b, window_radius=3):
    """Mean SSIM over RGB using a uniform window. Returns a float in [-1, 1].

    Uniform (box) window instead of Gaussian to stay pure-numpy and
    deterministic; good enough for a regression gate (plan wants >= 0.99).
    """
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    scores = []
    for ch in range(a.shape[2]):
        x = a[:, :, ch].astype(np.float64)
        y = b[:, :, ch].astype(np.float64)
        mu_x = _box_filter(x, window_radius)
        mu_y = _box_filter(y, window_radius)
        mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
        sigma_x = _box_filter(x * x, window_radius) - mu_x2
        sigma_y = _box_filter(y * y, window_radius) - mu_y2
        sigma_xy = _box_filter(x * y, window_radius) - mu_xy
        num = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
        den = (mu_x2 + mu_y2 + c1) * (sigma_x + sigma_y + c2)
        scores.append(np.mean(num / den))
    return float(np.mean(scores))


def metric_scale(data, max_size=256):
    """Scale that fits the geometry's canvas inside ``max_size`` (for fast metrics)."""
    bg = data["shapes"][0]
    w = float(bg["data"][2])
    h = float(bg["data"][3])
    longest = max(w, h, 1.0)
    if longest <= max_size:
        return 1.0
    return max_size / longest


# ---------------------------------------------------------------------------
# Optimization passes
# ---------------------------------------------------------------------------

def _visible_areas(data, scale):
    """Per-drawable visible pixel count under opaque top-down occlusion.

    Returns (areas, mask_areas) aligned with ``data['shapes'][1:]``. Walks
    shapes from last (top) to first (bottom): a shape's visible area = mask
    pixels not already claimed by a later opaque (alpha==255) shape. Opaque
    shapes then claim their pixels. Exact for opaque occlusion (the dominant
    case); conservative for translucent shapes (counts their whole mask).
    """
    shapes = data["shapes"]
    bg = shapes[0]
    img_w = max(1, int(round(float(bg["data"][2]) * scale)))
    img_h = max(1, int(round(float(bg["data"][3]) * scale)))
    covered = np.zeros((img_h, img_w), dtype=bool)
    drawables = shapes[1:]
    areas = [0] * len(drawables)
    mask_areas = [0] * len(drawables)
    for idx in range(len(drawables) - 1, -1, -1):
        shape = drawables[idx]
        color = shape.get("color", [])
        a = int(color[3]) if len(color) >= 4 else 255
        if a <= 0:
            continue
        x0, y0, x1, y1 = _shape_bbox(shape, img_w, img_h, scale)
        mask = _shape_mask(shape, x0, y0, x1, y1, scale)
        if mask.size == 0 or not mask.any():
            continue
        total = int(mask.sum())
        mask_areas[idx] = total
        sub = covered[y0:y1, x0:x1]
        visible = mask & ~sub
        areas[idx] = int(visible.sum())
        if a >= 255:
            sub[mask] = True
    return areas, mask_areas


def prune_occluded(data, min_visible_fraction=0.0, scale=1.0):
    """Drop fully (or near-fully) occluded shapes.

    A shape is removed when its visible area (pixels not later repainted by an
    opaque shape) is <= ``min_visible_fraction`` of its own mask area. With the
    default 0.0 only completely-occluded shapes are dropped (lossless).
    Returns (new_data, removed_indices).
    """
    areas, mask_areas = _visible_areas(data, scale)
    drawables = data["shapes"][1:]
    kept = []
    removed = []
    for idx, shape in enumerate(drawables):
        mask_area = mask_areas[idx]
        if mask_area <= 0:
            removed.append(idx)
            continue
        frac = areas[idx] / float(mask_area)
        if frac <= min_visible_fraction:
            removed.append(idx)
        else:
            kept.append(shape)
    new_data = {"shapes": [data["shapes"][0]] + kept}
    return new_data, removed


def dedupe_shapes(data):
    """Remove exact duplicate drawables (same type/data/color). Lossless."""
    seen = set()
    kept = []
    removed = 0
    for shape in data["shapes"][1:]:
        key = (
            int(shape["type"]),
            tuple(round(float(v), 3) for v in shape["data"]),
            tuple(int(v) for v in shape["color"]),
        )
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        kept.append(shape)
    return {"shapes": [data["shapes"][0]] + kept}, removed


def prune_to_target(data, target_count, scale=1.0):
    """Reduce drawable count to ``target_count`` by dropping lowest-footprint shapes.

    Lossy: drops shapes touching the fewest *final* pixels first (smallest
    visible area). Pair with the SSIM report to judge the quality cost. This is
    the "fit a smaller template" lever from the plan critique.
    """
    drawables = data["shapes"][1:]
    if target_count is None or target_count >= len(drawables):
        return {"shapes": list(data["shapes"])}, []
    areas, _mask_areas = _visible_areas(data, scale)
    # Rank by visible area ascending; drop the smallest until at target.
    order = sorted(range(len(drawables)), key=lambda i: areas[i])
    drop = set(order[: len(drawables) - target_count])
    kept = [s for i, s in enumerate(drawables) if i not in drop]
    return {"shapes": [data["shapes"][0]] + kept}, sorted(drop)


# ---------------------------------------------------------------------------
# Global color re-fit (Phase 7-lite / Option A: the OMP upgrade over greedy)
# ---------------------------------------------------------------------------

def _ownership_map(data, scale):
    """Map each pixel to the index of the top-most opaque drawable that owns it.

    Returns an int32 (H, W) array; -1 = no opaque drawable (background shows).
    Indices are into ``data['shapes'][1:]``.
    """
    shapes = data["shapes"]
    bg = shapes[0]
    img_w = max(1, int(round(float(bg["data"][2]) * scale)))
    img_h = max(1, int(round(float(bg["data"][3]) * scale)))
    owner = np.full((img_h, img_w), -1, dtype=np.int32)
    drawables = shapes[1:]
    for idx in range(len(drawables) - 1, -1, -1):
        shape = drawables[idx]
        color = shape.get("color", [])
        a = int(color[3]) if len(color) >= 4 else 255
        if a < 255:  # only opaque shapes deterministically own pixels
            continue
        x0, y0, x1, y1 = _shape_bbox(shape, img_w, img_h, scale)
        mask = _shape_mask(shape, x0, y0, x1, y1, scale)
        if mask.size == 0 or not mask.any():
            continue
        sub = owner[y0:y1, x0:x1]
        claim = mask & (sub == -1)
        sub[claim] = idx
    return owner


def refit_colors(data, source_rgb, scale=1.0, min_pixels=4):
    """Re-solve each opaque shape's color as the mean of the source pixels it owns.

    This is the Orthogonal-Matching-Pursuit color upgrade over plain greedy
    matching pursuit (plan: Algorithm Option A / Phase 7-lite). Exact
    coordinate-descent step for opaque painter's stacking. Translucent shapes
    are left unchanged (their contribution is non-local).

    ``source_rgb`` is an (H, W, 3) uint8/float array; it is resized by nearest
    sampling to the render canvas if needed. Returns a new data dict.
    """
    owner = _ownership_map(data, scale)
    h, w = owner.shape
    src = np.asarray(source_rgb)
    if src.shape[0] != h or src.shape[1] != w:
        ys = (np.linspace(0, src.shape[0] - 1, h)).astype(np.int64)
        xs = (np.linspace(0, src.shape[1] - 1, w)).astype(np.int64)
        src = src[ys][:, xs]
    src = src[:, :, :3].astype(np.float64)

    drawables = data["shapes"][1:]
    new_drawables = []
    for idx, shape in enumerate(drawables):
        new_shape = dict(shape)
        new_shape["data"] = list(shape["data"])
        color = list(shape.get("color", []))
        a = int(color[3]) if len(color) >= 4 else 255
        owned = owner == idx
        if a >= 255 and int(owned.sum()) >= min_pixels:
            mean = src[owned].mean(axis=0)
            color[0] = int(round(float(mean[0])))
            color[1] = int(round(float(mean[1])))
            color[2] = int(round(float(mean[2])))
            new_shape["color"] = color
        new_drawables.append(new_shape)
    return {"shapes": [data["shapes"][0]] + new_drawables}


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def _load_source_rgb(path):
    """Load an image as an (H, W, 3) RGB array via cv2. None on failure."""
    try:
        import cv2  # noqa: PLC0415
    except Exception:
        return None
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return bgr[:, :, ::-1].copy()


def optimize_geometry(
    input_path,
    output_path=None,
    target_count=None,
    occlusion=True,
    min_visible_fraction=0.0,
    dedupe=True,
    metric_max_size=256,
    refit_source=None,
):
    """Optimize a geometry JSON and write a new file. Returns a report dict.

    Never overwrites the input. ``output_path`` defaults to
    ``<stem>.optimized.json`` beside the input.
    """
    input_path = Path(input_path)
    data = load_normalized_geometry(input_path)
    layers_in = len(data["shapes"]) - 1

    scale = metric_scale(data, metric_max_size)
    before_render = render_geometry(data, scale=scale, blend_alpha=True)

    removed_occluded = 0
    removed_dupe = 0
    removed_target = 0
    if dedupe:
        data, removed_dupe = dedupe_shapes(data)
    if occlusion:
        data, removed = prune_occluded(data, min_visible_fraction, scale=scale)
        removed_occluded = len(removed)
    if target_count is not None:
        data, dropped = prune_to_target(data, target_count, scale=scale)
        removed_target = len(dropped)

    refit_applied = False
    if refit_source is not None:
        source_rgb = _load_source_rgb(refit_source)
        if source_rgb is not None:
            data = refit_colors(data, source_rgb, scale=scale)
            refit_applied = True

    after_render = render_geometry(data, scale=scale, blend_alpha=True)
    quality = ssim(before_render, after_render)
    error = mse(before_render, after_render)
    layers_out = len(data["shapes"]) - 1

    if output_path is None:
        output_path = input_path.with_name(f"{input_path.stem}.optimized.json")
    output_path = Path(output_path)
    output_path.write_text(json.dumps(data), encoding="utf-8")

    return {
        "input": str(input_path),
        "output": str(output_path),
        "layers_in": layers_in,
        "layers_out": layers_out,
        "removed_duplicate": removed_dupe,
        "removed_occluded": removed_occluded,
        "removed_to_target": removed_target,
        "refit_colors": refit_applied,
        "ssim": round(quality, 5),
        "mse": round(error, 5),
        "metric_scale": round(scale, 4),
    }


def _main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Optimize / measure a geometry JSON.")
    parser.add_argument("input", help="Input geometry .json")
    parser.add_argument("-o", "--output", default=None, help="Output path")
    parser.add_argument("--target", type=int, default=None, help="Target drawable count")
    parser.add_argument("--no-occlusion", action="store_true", help="Disable occlusion prune")
    parser.add_argument("--no-dedupe", action="store_true", help="Disable exact-duplicate prune")
    parser.add_argument(
        "--min-visible",
        type=float,
        default=0.0,
        help="Drop shapes whose visible fraction <= this (0=lossless occlusion only)",
    )
    parser.add_argument(
        "--refit",
        default=None,
        help="Source image to re-solve opaque shape colors against (Option A / Phase 7-lite)",
    )
    args = parser.parse_args(argv)
    report = optimize_geometry(
        args.input,
        args.output,
        target_count=args.target,
        occlusion=not args.no_occlusion,
        min_visible_fraction=args.min_visible,
        dedupe=not args.no_dedupe,
        refit_source=args.refit,
    )
    for key, value in report.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
