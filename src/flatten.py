"""Flat / Poster generate mode (R1) + palette lock (R4).

CPU-only flattener: quantize source colors -> segment connected regions per
color -> fit minimal rect/ellipse per region -> emit standard geometry-JSON.
Designed for flat/vector art (anime, logos, decals) where geometrize's greedy
alpha-ellipse stacking wastes dozens of layers on a single flat panel.

Emits the exact schema the FH6 importer expects (see ``main.py`` 213-303):
  background : {"type": 1,  "data": [0, 0, W, H],            "color": [r,g,b,a]}
  ellipse    : {"type": 16, "data": [cx, cy, w_semi, h_semi, rot], "color": ...}
  rectangle  : {"type": 1,  "data": [cx, cy, w_full, h_full, (rot)], "color": ...}

The preview PNG and IoU scoring reuse ``geometry_optimize`` so the output is
WYSIWYG: what renders here is what imports in-game.
"""
# SOURCE: src/preprocess/luma.py:1-11 (module header conventions)
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

from utils import PreprocessError
from geometry_optimize import render_geometry, _shape_bbox, _shape_mask
from geometry_json import ROTATED_ELLIPSE, RECTANGLE

DEFAULT_COLORS = 24
DEFAULT_SEGMENTS = 400
DEFAULT_COMPACTNESS = 10.0
DEFAULT_MAX_RESOLUTION = 384
DEFAULT_MIN_AREA = 12
IOU_THRESHOLD = 0.82
SPLIT_MAX_DEPTH = 4
SPLIT_MIN_AREA = 256  # regions smaller than this take their single best shape (no split)
AXIS_SNAP_DEG = 1.0


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def _load_rgb(image_path: Path, max_resolution: int):
    """Load an image as (rgb uint8 HxWx3, opaque_mask bool HxW, had_alpha bool).

    Resizes the longest edge down to ``max_resolution`` (mirrors
    ``imageutil.resizeToMax``). cv2 decodes BGR(A); we convert to RGB.
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise PreprocessError(f"could not read image: {image_path}")
    if img.ndim == 2:  # grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    had_alpha = img.shape[2] == 4 if img.ndim == 3 else False
    if had_alpha:
        bgr = img[:, :, :3]
        alpha = img[:, :, 3]
    else:
        bgr = img[:, :, :3]
        alpha = np.full(bgr.shape[:2], 255, np.uint8)

    h, w = bgr.shape[:2]
    max_dim = max(h, w)
    if max_resolution > 0 and max_dim > max_resolution:
        scale = max_resolution / float(max_dim)
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        bgr = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        alpha = cv2.resize(alpha, (nw, nh), interpolation=cv2.INTER_NEAREST)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    opaque_mask = alpha > 0
    return rgb, opaque_mask, had_alpha


# ---------------------------------------------------------------------------
# Task 2 — quantization
# ---------------------------------------------------------------------------

def _quantize(rgb: np.ndarray, n_colors: int, opaque_mask: np.ndarray):
    """k-means color quantization over opaque pixels only.

    Returns ``(labels HxW int32, palette Kx3 uint8)``. Transparent pixels get
    label -1 and are excluded from the k-means samples.
    """
    h, w = rgb.shape[:2]
    labels = np.full((h, w), -1, np.int32)
    samples = rgb[opaque_mask].reshape(-1, 3).astype(np.float32)
    if samples.shape[0] == 0:
        return labels, np.zeros((0, 3), np.uint8)

    k = int(max(2, min(64, n_colors)))
    # Cannot ask for more clusters than distinct opaque samples.
    unique = np.unique(samples, axis=0)
    k = min(k, unique.shape[0])
    if k < 1:
        k = 1

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    if k == 1:
        center = samples.mean(axis=0, keepdims=True)
        flat_labels = np.zeros((samples.shape[0], 1), np.int32)
    else:
        _compactness, flat_labels, center = cv2.kmeans(
            samples, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS
        )
    palette = np.clip(center.round(), 0, 255).astype(np.uint8)
    labels[opaque_mask] = flat_labels.flatten()
    return labels, palette


# ---------------------------------------------------------------------------
# R3 — superpixel labeling (joint color + space k-means)
# ---------------------------------------------------------------------------

def _superpixel(rgb: np.ndarray, n_segments: int, opaque_mask: np.ndarray,
                compactness: float = 10.0):
    """SLIC-like superpixels via k-means over [L, a, b, x*w, y*w] features.

    Unlike ``_quantize`` (color only -> regions are whole same-color blobs),
    this clusters color AND position, so a smooth gradient breaks into several
    compact local segments instead of one band. One shape per segment ->
    one-pass, deterministic budget, no greedy loop. Returns
    ``(labels HxW int32, palette Kx3 uint8 = per-segment mean RGB)``.
    """
    h, w = rgb.shape[:2]
    labels = np.full((h, w), -1, np.int32)
    idx = np.flatnonzero(opaque_mask.reshape(-1))
    if idx.size == 0:
        return labels, np.zeros((0, 3), np.uint8)

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32).reshape(-1, 3)
    ys, xs = np.mgrid[0:h, 0:w]
    coords = np.stack([xs.reshape(-1), ys.reshape(-1)], axis=1).astype(np.float32)

    k = int(max(2, min(8000, n_segments)))
    k = min(k, idx.size)
    # SLIC spatial weight: S = sqrt(N/K); larger compactness -> rounder, more
    # space-driven segments; smaller -> more color-driven.
    s = max(1.0, (idx.size / k) ** 0.5)
    spatial_weight = compactness / s

    feats = np.concatenate([lab[idx], coords[idx] * spatial_weight], axis=1).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _compactness, flat_labels, _centers = cv2.kmeans(
        feats, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS
    )
    flat_labels = flat_labels.flatten()
    labels.reshape(-1)[idx] = flat_labels

    # Per-segment mean RGB (the actual color, not the Lab cluster center).
    palette = np.zeros((k, 3), np.float32)
    counts = np.zeros(k, np.float32)
    rgb_samples = rgb.reshape(-1, 3).astype(np.float32)[idx]
    np.add.at(palette, flat_labels, rgb_samples)
    np.add.at(counts, flat_labels, 1.0)
    counts[counts == 0] = 1.0
    palette = (palette / counts[:, None]).round().clip(0, 255).astype(np.uint8)
    return labels, palette


# ---------------------------------------------------------------------------
# Task 3 — per-color region extraction
# ---------------------------------------------------------------------------

def _regions(labels: np.ndarray, palette_index: int, opaque_mask: np.ndarray, min_area: int):
    """Yield filled blob masks (uint8 0/255 HxW) for one palette color.

    Each connected component of the color becomes its own mask -> proves the
    FH6 rule that disjoint same-color patches are separate shapes.
    """
    h, w = labels.shape
    color_mask = ((labels == palette_index) & opaque_mask).astype(np.uint8) * 255
    if not color_mask.any():
        return
    contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        filled = np.zeros((h, w), np.uint8)
        cv2.drawContours(filled, [contour], -1, 255, thickness=-1)
        yield filled


# ---------------------------------------------------------------------------
# Task 4 — minimal shape fit per region (rect vs ellipse, recursive split)
# ---------------------------------------------------------------------------

def _rect_candidate(contour):
    """type-1 rect shape dict (no color) from ``cv2.minAreaRect``.

    minAreaRect returns FULL (w, h); rect ``data`` is center + full size, which
    matches the importer (``main.py`` x0=cx-w/2). Axis-aligned rects emit a
    4-value data (no rotation) so ``_looks_like_background``-style consumers and
    ``_shape_mask`` take the fast unrotated path.
    """
    (cx, cy), (w, h), ang = cv2.minAreaRect(contour)
    if w <= 0 or h <= 0:
        return None
    ang = ang % 180.0
    if ang > 90.0:
        ang -= 180.0
    if abs(ang) < AXIS_SNAP_DEG:
        data = [round(cx), round(cy), float(w), float(h)]
    elif abs(abs(ang) - 90.0) < AXIS_SNAP_DEG:
        data = [round(cx), round(cy), float(h), float(w)]  # swap to axis-aligned
    else:
        data = [round(cx), round(cy), float(w), float(h), round(ang) % 360]
    return {"type": int(RECTANGLE), "data": data, "color": [0, 0, 0, 255], "score": 0}


def _ellipse_candidate(contour):
    """type-16 ellipse shape dict (no color) from ``cv2.fitEllipse``.

    fitEllipse returns FULL axes; ellipse ``data`` uses SEMI axes. Importer maps
    ``data=[cx,cy,w,h,rot]`` to ``cv2.ellipse(axes=(h,w), angle=-90+rot)`` so we
    set ``rot = fit_angle + 90`` to reproduce the fitted orientation.
    """
    if len(contour) < 5:
        return None
    (cx, cy), (d1, d2), ang = cv2.fitEllipse(contour)
    if d1 <= 0 or d2 <= 0:
        return None
    w_semi = d2 / 2.0
    h_semi = d1 / 2.0
    rot = round(ang + 90.0) % 360
    data = [round(cx), round(cy), float(w_semi), float(h_semi), rot]
    return {"type": int(ROTATED_ELLIPSE), "data": data, "color": [0, 0, 0, 255], "score": 0}


def _shape_iou(shape, region_bool: np.ndarray) -> float:
    """IoU between a shape's rendered mask and the region mask (full HxW bool)."""
    h, w = region_bool.shape
    x0, y0, x1, y1 = _shape_bbox(shape, w, h, 1.0)
    full = np.zeros((h, w), dtype=bool)
    local = _shape_mask(shape, x0, y0, x1, y1, 1.0)
    if local.size:
        full[y0:y1, x0:x1] = local
    inter = np.logical_and(full, region_bool).sum()
    union = np.logical_or(full, region_bool).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def _fit_contour(contour, region_bool, w, h, use_rects, min_area, depth):
    """Fit one contour; recursively split into quadrants on poor IoU."""
    candidates = []
    if use_rects:
        rect = _rect_candidate(contour)
        if rect is not None:
            candidates.append(rect)
    ell = _ellipse_candidate(contour)
    if ell is not None:
        candidates.append(ell)
    if not candidates:
        return []

    scored = [(_shape_iou(s, region_bool), s) for s in candidates]
    best_iou, best_shape = max(scored, key=lambda t: t[0])
    area = float(region_bool.sum())

    if best_iou >= IOU_THRESHOLD or depth >= SPLIT_MAX_DEPTH or area < SPLIT_MIN_AREA:
        best_shape["score"] = area
        return [best_shape]

    # Split the contour's bounding box into quadrants and recurse on each.
    bx, by, bw, bh = cv2.boundingRect(contour)
    mx, my = bx + bw // 2, by + bh // 2
    quads = [
        (bx, by, mx, my),
        (mx, by, bx + bw, my),
        (bx, my, mx, by + bh),
        (mx, my, bx + bw, by + bh),
    ]
    out = []
    region_u8 = region_bool.astype(np.uint8) * 255
    for qx0, qy0, qx1, qy1 in quads:
        if qx1 <= qx0 or qy1 <= qy0:
            continue
        sub = np.zeros((h, w), np.uint8)
        sub[qy0:qy1, qx0:qx1] = region_u8[qy0:qy1, qx0:qx1]
        sub_contours, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for sc in sub_contours:
            if cv2.contourArea(sc) < min_area:
                continue
            sub_bool = sub > 0
            out.extend(_fit_contour(sc, sub_bool, w, h, use_rects, min_area, depth + 1))
    # Fall back to the single best shape if splitting produced nothing usable.
    if not out:
        best_shape["score"] = area
        return [best_shape]
    return out


def _fit_region(region_mask_u8, use_rects, min_area):
    """Fit a filled blob mask -> list of shape dicts (color filled later)."""
    h, w = region_mask_u8.shape
    contours, _ = cv2.findContours(region_mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shapes = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        region_bool = region_mask_u8 > 0
        shapes.extend(_fit_contour(contour, region_bool, w, h, use_rects, min_area, 0))
    return shapes


# ---------------------------------------------------------------------------
# Task 5 — assemble geometry payload
# ---------------------------------------------------------------------------

def _build_payload(rgb, labels, palette, opaque_mask, had_alpha, use_rects, min_area):
    """Run segmentation + fit over every palette color; return geometry dict."""
    h, w = rgb.shape[:2]
    drawables = []
    for idx in range(palette.shape[0]):
        r, g, b = (int(v) for v in palette[idx])
        for region_mask in _regions(labels, idx, opaque_mask, min_area):
            for shape in _fit_region(region_mask, use_rects, min_area):
                shape["color"] = [r, g, b, 255]
                drawables.append(shape)

    # Largest area first (painter's order: big panels under small detail).
    drawables.sort(key=lambda s: float(s.get("score", 0)), reverse=True)

    if opaque_mask.any():
        mean = rgb[opaque_mask].reshape(-1, 3).mean(axis=0)
        bg_r, bg_g, bg_b = (int(round(v)) for v in mean)
    else:
        bg_r = bg_g = bg_b = 0
    bg_a = 0 if had_alpha else 255
    background = {"type": int(RECTANGLE), "data": [0, 0, w, h],
                 "color": [bg_r, bg_g, bg_b, bg_a], "score": 0}
    return {"shapes": [background] + drawables}


# ---------------------------------------------------------------------------
# Task 6 — palette lock (R4)
# ---------------------------------------------------------------------------

def _to_lab(colors_rgb: np.ndarray) -> np.ndarray:
    """Convert an Nx3 uint8 RGB array to Nx3 float Lab."""
    patch = colors_rgb.reshape(-1, 1, 3).astype(np.uint8)
    lab = cv2.cvtColor(patch, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    return lab


def palette_lock(data: dict, palette: np.ndarray) -> dict:
    """Snap every drawable color to its nearest palette entry (Lab distance).

    Tags each drawable with ``shape["palette"] = index``. Works on ANY
    generator's geometry-JSON (flat-mode output is already palette colors, so
    this is a near-identity there, but it also flattens geometrize output).
    Mutates and returns ``data``.
    """
    palette = np.asarray(palette, np.uint8).reshape(-1, 3)
    if palette.shape[0] == 0:
        return data
    pal_lab = _to_lab(palette)
    for shape in data["shapes"][1:]:
        color = shape.get("color", [])
        if len(color) < 3:
            continue
        rgb = np.array([[color[0], color[1], color[2]]], np.uint8)
        lab = _to_lab(rgb)[0]
        dist = np.sum((pal_lab - lab) ** 2, axis=1)
        idx = int(np.argmin(dist))
        r, g, b = (int(v) for v in palette[idx])
        a = int(color[3]) if len(color) >= 4 else 255
        shape["color"] = [r, g, b, a]
        shape["palette"] = idx
    return data


# ---------------------------------------------------------------------------
# Task 7 — preview + top-level entry point
# ---------------------------------------------------------------------------

def _atomic_write_bytes(path: Path, write_fn):
    """Write via a temp file + os.replace (mirror preprocess/luma.py)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    write_fn(tmp)
    os.replace(tmp, path)


def _write_preview(data: dict, preview_path: Path):
    rgb = render_geometry(data, scale=1.0, blend_alpha=False)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    # Encode by the real extension, then atomic-replace (cv2.imwrite cannot
    # infer an encoder from the temp file's ".png.tmp" suffix).
    ok, buf = cv2.imencode(Path(preview_path).suffix or ".png", bgr)
    if not ok:
        raise PreprocessError(f"failed to encode preview: {preview_path}")
    _atomic_write_bytes(preview_path, lambda p: Path(p).write_bytes(buf.tobytes()))


def flatten_image(image_path, out_json_path, n_colors: int = DEFAULT_COLORS,
                  max_resolution: int = DEFAULT_MAX_RESOLUTION, use_rects: bool = True,
                  min_area: int = DEFAULT_MIN_AREA, palette_lock_enabled: bool = True,
                  preview_path=None, method: str = "poster",
                  n_segments: int = DEFAULT_SEGMENTS, compactness: float = DEFAULT_COMPACTNESS) -> dict:
    """Convert an image to flat-mode geometry-JSON.

    ``method`` selects the labeling stage:
      - ``"poster"`` (R1): color-only k-means -> few flat color regions.
      - ``"superpixel"`` (R3): joint color+space k-means -> many compact
        segments (one shape each); handles gradients better, deterministic
        budget, still one pass.

    Returns a report dict ``{output, layers, colors, seconds}``.
    """
    started = time.perf_counter()
    image_path = Path(image_path)
    out_json_path = Path(out_json_path)

    rgb, opaque_mask, had_alpha = _load_rgb(image_path, max_resolution)
    if method == "superpixel":
        labels, palette = _superpixel(rgb, n_segments, opaque_mask, compactness)
    else:
        labels, palette = _quantize(rgb, n_colors, opaque_mask)
    data = _build_payload(rgb, labels, palette, opaque_mask, had_alpha, use_rects, min_area)
    # Palette lock only makes sense for the small poster palette; superpixel
    # colors are already per-segment means (locking would be a near no-op).
    if palette_lock_enabled and method == "poster" and palette.shape[0] > 0:
        data = palette_lock(data, palette)

    _atomic_write_bytes(
        out_json_path,
        lambda p: Path(p).write_text(json.dumps(data), encoding="utf-8"),
    )
    if preview_path is not None:
        _write_preview(data, Path(preview_path))

    return {
        "output": str(out_json_path),
        "layers": len(data["shapes"]) - 1,
        "colors": int(palette.shape[0]),
        "seconds": round(time.perf_counter() - started, 3),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Flat/Poster image flattener -> geometry-JSON.")
    parser.add_argument("image", help="source image path")
    parser.add_argument("-o", "--output", required=True, help="output geometry-JSON path")
    parser.add_argument("-n", "--colors", type=int, default=DEFAULT_COLORS, help="palette size (2-64)")
    parser.add_argument("--method", choices=["poster", "superpixel"], default="poster",
                        help="poster=color-only (R1); superpixel=color+space segments (R3)")
    parser.add_argument("--segments", type=int, default=DEFAULT_SEGMENTS,
                        help="superpixel segment count (method=superpixel)")
    parser.add_argument("--compactness", type=float, default=DEFAULT_COMPACTNESS,
                        help="superpixel space-vs-color weight (higher=rounder)")
    parser.add_argument("--max-res", type=int, default=DEFAULT_MAX_RESOLUTION, help="longest-edge resize")
    parser.add_argument("--ellipse-only", action="store_true", help="disable rect candidates")
    parser.add_argument("--min-area", type=int, default=DEFAULT_MIN_AREA, help="min region area (px)")
    parser.add_argument("--no-palette-lock", action="store_true", help="skip palette lock")
    parser.add_argument("--preview", default=None, help="optional preview PNG path")
    args = parser.parse_args(argv)

    report = flatten_image(
        args.image, args.output, n_colors=args.colors, max_resolution=args.max_res,
        use_rects=not args.ellipse_only, min_area=args.min_area,
        palette_lock_enabled=not args.no_palette_lock, preview_path=args.preview,
        method=args.method, n_segments=args.segments, compactness=args.compactness,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
