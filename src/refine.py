"""Residual Pyramid generate mode (R7) — fast CPU full-image high fidelity.

A new algorithm, distinct from both the greedy GPU geometrize loop and the
one-pass flat/superpixel flatteners:

  1. **Coarse base** — k-means posterize at low color count; fit big solid
     rect/ellipse shapes per color region (reuses the R1 fitting machinery).
     This captures global structure in a handful of large shapes.
  2. **Batched residual refinement** — repeatedly render the current shape
     stack, compute the per-pixel squared error against the target, blur the
     error map (coarse-to-fine sigma schedule), threshold it, and fit ONE
     shape per connected error blob. Every candidate is scored by the actual
     SSE reduction it would cause and is only accepted if it improves the
     image. Accepted shapes are painted onto the running canvas, so each pass
     sees the true residual of everything before it.

Why this is fast: geometrize places one shape per step via random mutation
search (hundreds of trials per shape, GPU needed). Here shape placement is
*deterministic* — the error map says exactly where the worst blob is and
``minAreaRect``/``fitEllipse`` say what shape covers it — so a whole batch of
shapes lands per render pass, with no search at all. Cost is a few dozen
full-frame numpy/cv2 operations: seconds on CPU for hundreds of shapes.

Why this is high fidelity: unlike the one-pass flatteners it iterates on the
residual, so mistakes are self-correcting — a badly fitted shape leaves error
behind, and the next (finer) pass covers it. The coarse-to-fine blur schedule
spends large shapes on large errors first and small shapes on edge detail
last, mimicking what greedy matching-pursuit converges to, at a fraction of
the work.

Emits the exact geometry-JSON schema the FH6 importer expects (see
``flatten.py`` header / ``main.py`` 213-303):
  background : {"type": 1,  "data": [0, 0, W, H],                 "color": [r,g,b,a]}
  ellipse    : {"type": 16, "data": [cx, cy, w_semi, h_semi, rot], "color": ...}
  rectangle  : {"type": 1,  "data": [cx, cy, w_full, h_full, (rot)], "color": ...}

Shape order is chronological (base big->small, then refinement batches), which
is the painter's order the renderer and the game both use.

Full method write-up: ``.claude/PRPs/notes/residual-pyramid-method.md``.
"""
# SOURCE: src/flatten.py:16-30 (module header + import conventions)
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from geometry_optimize import _shape_bbox, _shape_mask
from flatten import (
    _load_rgb,
    _quantize,
    _build_payload,
    _rect_candidate,
    _ellipse_candidate,
    _atomic_write_bytes,
    _write_preview,
)

DEFAULT_SHAPES = 1500
DEFAULT_BASE_COLORS = 12
DEFAULT_MAX_RESOLUTION = 384
BASE_BUDGET_FRACTION = 0.3  # at most this share of the budget goes to the base
BASE_MAX_SHAPES = 200       # absolute base cap; the rest is residual budget
BASE_MIN_AREA = 64          # base keeps only genuinely large regions
MIN_GAIN = 1.0              # absolute SSE improvement a shape must deliver

# Anti-artifact regularization (refine + ultra). Two streak sources, both
# gated inside ``_evaluate`` so the caps apply to the deterministic fit AND
# every hill-climb trial; see ``.claude/PRPs/notes/ultra-method.md``:
#   * sampling — a ragged/disjoint error blob fitted by one minAreaRect /
#       fitEllipse becomes a sliver spanning the gaps; the worst case is a
#       near-zero-axis degenerate shape (aspect in the millions).
#   * mutation — the hill-climb stretches a shape along one axis chasing
#       bbox-local SSE, growing it into a streak.
# A blanket aspect cap also kills *legit* thin features (hair, line art), so
# elongation is judged by COHERENCE, not ratio alone: a long shape is kept
# only if most pixels it covers match its single fill color.
MIN_THICKNESS = 1.5     # reject fits whose short axis is sub-pixel thin
MAX_ASPECT = 16.0       # above this long/short ratio a shape is a streak risk
MIN_COHERENCE = 0.55    # ...and is kept only if >= this fraction of covered
COH_DELTA = 22.0        # pixels match its fill within 3*COH_DELTA^2 SSE
CLIMB_GROWTH = 2.0      # hill-climb may not grow an axis past this x its seed

# Oversized error blobs are re-thresholded at their own error percentile and
# recursed into, so one merged blob still yields many well-placed shapes.
SPLIT_PERCENTILE = 70.0
MAX_SPLIT_DEPTH = 6
BLOB_AREA_DIVISOR = 50      # blobs bigger than imageArea/50 get split

# Residual pass schedule: (blur_sigma, per-channel delta threshold, min blob area).
# The error floor is 3 * delta^2 (sum of squared per-channel differences).
# Coarse passes (big sigma, high floor, big blobs) place large shapes; fine
# passes chase edge/detail error with small shapes.
PASS_SCHEDULE = (
    (9.0, 28.0, 64),
    (5.0, 22.0, 32),
    (3.0, 16.0, 16),
    (1.5, 12.0, 8),
)
# After the schedule, the finest pass repeats until the budget is spent or a
# pass stops finding improving shapes.
FINAL_PASS = (0.8, 4.0, 2)
MAX_FINAL_PASSES = 16


# ---------------------------------------------------------------------------
# Canvas / error helpers
# ---------------------------------------------------------------------------

def _error_map(target: np.ndarray, canvas: np.ndarray, opaque_mask: np.ndarray) -> np.ndarray:
    """Per-pixel sum of squared RGB differences (float32 HxW); 0 where transparent."""
    err = ((target - canvas) ** 2).sum(axis=2)
    err[~opaque_mask] = 0.0
    return err


def _paint(canvas: np.ndarray, shape: dict, color: np.ndarray, alpha: float = 1.0) -> None:
    """Paint a shape onto the float32 canvas (same mask the renderer uses).

    ``alpha < 1`` blends over the existing canvas exactly like the in-game
    alpha byte (and ``render_geometry(blend_alpha=True)``) does.
    """
    h, w = canvas.shape[:2]
    x0, y0, x1, y1 = _shape_bbox(shape, w, h, 1.0)
    mask = _shape_mask(shape, x0, y0, x1, y1, 1.0)
    if mask.size and mask.any():
        if alpha >= 1.0:
            canvas[y0:y1, x0:x1][mask] = color
        else:
            region = canvas[y0:y1, x0:x1]
            region[mask] = region[mask] * (1.0 - alpha) + color * alpha


def _evaluate(target: np.ndarray, canvas: np.ndarray, opaque_mask: np.ndarray,
              shape: dict, alpha: float = 1.0):
    """Score a candidate by the SSE reduction painting it would cause.

    Returns ``(gain, color)`` where ``color`` is the L2-optimal fill at the
    given ``alpha`` (solid: mean target color; translucent: the color whose
    blend over the current canvas best matches the target), or ``None`` if
    the shape covers no opaque pixels, is sub-pixel thin, or is an incoherent
    gap-spanning streak (see the anti-artifact gates). ``gain`` may be
    negative.
    """
    # Anti-artifact gate 1 (geometry, cheap — runs on every hill-climb trial):
    # a sub-pixel-thin axis is a degenerate ~zero-area fit (e.g. fitEllipse on
    # a near-collinear error ridge returns a 0.0007-px semi-axis) that wastes a
    # layer and renders as a stray 1-px line. Reject before touching pixels.
    da, db = abs(float(shape["data"][2])), abs(float(shape["data"][3]))
    short, long_ = (da, db) if da <= db else (db, da)
    if short < MIN_THICKNESS:
        return None

    h, w = target.shape[:2]
    x0, y0, x1, y1 = _shape_bbox(shape, w, h, 1.0)
    mask = _shape_mask(shape, x0, y0, x1, y1, 1.0)
    if not mask.size or not mask.any():
        return None
    covered = opaque_mask[y0:y1, x0:x1] & mask
    if not covered.any():
        return None
    t = target[y0:y1, x0:x1][covered]
    c = canvas[y0:y1, x0:x1][covered]
    sse_old = float(((t - c) ** 2).sum())
    if alpha >= 1.0:
        color = t.mean(axis=0)
        resid = ((t - color) ** 2).sum(axis=1)
    else:
        # Minimize ||t - ((1-a)c + a*color)||^2 -> color = mean((t-(1-a)c)/a).
        color = ((t - (1.0 - alpha) * c) / alpha).mean(axis=0)
        color = np.clip(color, 0.0, 255.0)
        blended = (1.0 - alpha) * c + alpha * color
        resid = ((t - blended) ** 2).sum(axis=1)
    sse_new = float(resid.sum())

    # Anti-artifact gate 2 (coherence): an elongated shape is legit only if it
    # traces a genuinely thin image feature (hair, line art, an edge), where
    # nearly every pixel it covers matches its single fill color. A streak that
    # bridges the gap between two unrelated patches covers a band of mismatched
    # pixels, so its matched fraction is low. Low-aspect shapes skip the test.
    if long_ > MAX_ASPECT * max(short, 1e-3):
        coherent = float((resid < 3.0 * COH_DELTA * COH_DELTA).mean())
        if coherent < MIN_COHERENCE:
            return None
    return sse_old - sse_new, np.round(color)


# ---------------------------------------------------------------------------
# One residual pass: error map -> blobs -> recursive fit per blob
# ---------------------------------------------------------------------------

def _hill_climb(target, canvas, opaque_mask, shape, start_gain, start_color,
                alpha, max_evals):
    """Pattern-search refinement of one shape's params (bbox-local SSE).

    The deterministic contour fit is a good seed but one-shot; this is the
    geometrize-style local search that aligns the shape to the image, except
    it starts from the seed instead of random mutations, so a few dozen
    evaluations replace geometrize's thousands. Coordinate descent over
    (cx, cy, w, h, rot) with annealed steps; every trial is scored by the
    real SSE gain at the given alpha. Mutates ``shape['data']`` in place.
    """
    best_gain, best_color = start_gain, start_color
    data = list(shape["data"])
    n = len(data)
    # Growth clamp (the "mutation" streak source): the seed already covers the
    # feature, so the climb may slide/shrink/rotate it but must not stretch an
    # axis into a streak. Cap each axis at CLIMB_GROWTH x its seed size.
    max_w = CLIMB_GROWTH * max(abs(float(data[2])), MIN_THICKNESS)
    max_h = CLIMB_GROWTH * max(abs(float(data[3])), MIN_THICKNESS)
    size = max(2.0, float(data[2]), float(data[3]))
    step = max(1.0, size * 0.15)
    rot_step = 6.0
    evals = 0
    while step >= 0.35 and evals < max_evals:
        improved = False
        for idx in range(n):
            delta_step = rot_step if idx == 4 else step
            for delta in (delta_step, -delta_step):
                if evals >= max_evals:
                    break
                trial = list(data)
                trial[idx] = trial[idx] + delta
                if idx in (2, 3) and trial[idx] < 0.5:
                    continue
                if idx == 2 and abs(trial[2]) > max_w:
                    continue
                if idx == 3 and abs(trial[3]) > max_h:
                    continue
                scored = _evaluate(target, canvas, opaque_mask,
                                   {**shape, "data": trial}, alpha)
                evals += 1
                if scored is None:
                    continue
                gain, color = scored
                if gain > best_gain:
                    best_gain, best_color, data = gain, color, trial
                    improved = True
        if not improved:
            step *= 0.5
            rot_step *= 0.5
    shape["data"] = data
    return best_gain, best_color


def _fit_blob_direct(target, canvas, opaque_mask, blob_u8, origin, use_rects,
                     alphas=(1.0,), climb=0):
    """Fit the better rect/ellipse to one blob mask (local crop + origin).

    Each candidate is tried at every opacity in ``alphas`` (solid for crisp
    edges, translucent for gradients/anti-aliased detail). Returns
    ``(gain, shape, color, alpha)`` for the best combination beating
    ``MIN_GAIN``, else ``None``.
    """
    contours, _ = cv2.findContours(
        blob_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, offset=origin
    )
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)

    candidates = []
    if use_rects:
        rect = _rect_candidate(contour)
        if rect is not None:
            candidates.append(rect)
    ell = _ellipse_candidate(contour)
    if ell is not None:
        candidates.append(ell)

    best_gain, best_shape, best_color, best_alpha = MIN_GAIN, None, None, 1.0
    for cand in candidates:
        for alpha in alphas:
            scored = _evaluate(target, canvas, opaque_mask, cand, alpha)
            if scored is None:
                continue
            gain, color = scored
            if gain > best_gain:
                best_gain, best_shape, best_color, best_alpha = gain, cand, color, alpha
    if best_shape is None:
        return None
    if climb > 0:
        best_gain, best_color = _hill_climb(
            target, canvas, opaque_mask, best_shape, best_gain, best_color,
            best_alpha, climb)
    return best_gain, best_shape, best_color, best_alpha


def _process_blob(target, canvas, opaque_mask, blob_u8, origin, min_area,
                  use_rects, budget, added, depth, alphas=(1.0,), climb=0,
                  weight=None):
    """Fit one error blob, then recurse into its high-error core if oversized.

    A merged blob (widespread error) first gets one coarse covering shape,
    then is re-thresholded at its own ``SPLIT_PERCENTILE`` error percentile
    (recomputed AFTER painting, so recursion chases the true residual) and
    each sub-blob is processed the same way. Guaranteed shrinkage: the
    percentile keeps at most ~30% of the blob's pixels per level.
    """
    if len(added) >= budget:
        return
    fit = _fit_blob_direct(target, canvas, opaque_mask, blob_u8, origin,
                           use_rects, alphas, climb)
    if fit is not None:
        gain, shape, color, alpha = fit
        r, g, b = (int(v) for v in color)
        shape["color"] = [r, g, b, max(1, min(255, int(round(alpha * 255))))]
        shape["score"] = gain
        _paint(canvas, shape, color.astype(np.float32), alpha)
        added.append(shape)

    h, w = target.shape[:2]
    area = int(cv2.countNonZero(blob_u8))
    max_blob = max(4 * min_area, (h * w) // BLOB_AREA_DIVISOR)
    if depth >= MAX_SPLIT_DEPTH or area <= max_blob or len(added) >= budget:
        return

    bx, by = origin
    bh, bw = blob_u8.shape
    crop_t = target[by:by + bh, bx:bx + bw]
    crop_c = canvas[by:by + bh, bx:bx + bw]
    crop_o = opaque_mask[by:by + bh, bx:bx + bw]
    err = ((crop_t - crop_c) ** 2).sum(axis=2)
    err[~crop_o] = 0.0
    if weight is not None:
        err = err * weight[by:by + bh, bx:bx + bw]
    inside = blob_u8 > 0
    vals = err[inside]
    if vals.size == 0:
        return
    thr = float(np.percentile(vals, SPLIT_PERCENTILE))
    if thr <= 0.0:
        return
    sub_mask = ((err > thr) & inside).astype(np.uint8)
    n_comp, labels, stats, _ = cv2.connectedComponentsWithStats(sub_mask, 8)
    if n_comp <= 1:
        return
    mass = np.bincount(labels.ravel(), weights=err.ravel(), minlength=n_comp)
    order = np.argsort(mass[1:])[::-1] + 1
    for comp in order:
        if len(added) >= budget:
            return
        if stats[comp, cv2.CC_STAT_AREA] < min_area:
            continue
        sx = stats[comp, cv2.CC_STAT_LEFT]
        sy = stats[comp, cv2.CC_STAT_TOP]
        sw = stats[comp, cv2.CC_STAT_WIDTH]
        sh = stats[comp, cv2.CC_STAT_HEIGHT]
        sub_u8 = (labels[sy:sy + sh, sx:sx + sw] == comp).astype(np.uint8) * 255
        _process_blob(target, canvas, opaque_mask, sub_u8, (bx + sx, by + sy),
                      min_area, use_rects, budget, added, depth + 1, alphas,
                      climb, weight)


def _residual_pass(target, canvas, opaque_mask, sigma, delta, min_area, budget,
                   use_rects, alphas=(1.0,), climb=0, weight=None):
    """Fit shapes over the connected blobs of the thresholded error map.

    Deterministic batch placement: no random search. Each blob is fitted with
    the better of a min-area rect / fitted ellipse, judged by real SSE gain,
    accepted only if it improves the canvas; oversized blobs recurse into
    their high-error cores. Returns the list of accepted shape dicts (also
    painted onto ``canvas`` in place).
    """
    added = []
    err = _error_map(target, canvas, opaque_mask)
    # Saliency weighting steers blob SELECTION (threshold + ordering) toward
    # perceptually loud regions (edges, eyes, line art); shape ACCEPTANCE
    # still uses the raw unweighted SSE gain.
    sel = err if weight is None else err * weight
    err_blur = cv2.GaussianBlur(sel, (0, 0), sigma) if sigma > 0 else sel
    floor = 3.0 * (delta ** 2)
    mask = (err_blur > floor).astype(np.uint8)
    if not mask.any():
        return added

    n_comp, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n_comp <= 1:
        return added
    # Process worst blobs first (total error mass, not blob size).
    mass = np.bincount(labels.ravel(), weights=sel.ravel(), minlength=n_comp)
    order = np.argsort(mass[1:])[::-1] + 1

    for comp in order:
        if len(added) >= budget:
            break
        if stats[comp, cv2.CC_STAT_AREA] < min_area:
            continue
        bx = stats[comp, cv2.CC_STAT_LEFT]
        by = stats[comp, cv2.CC_STAT_TOP]
        bw = stats[comp, cv2.CC_STAT_WIDTH]
        bh = stats[comp, cv2.CC_STAT_HEIGHT]
        sub = (labels[by:by + bh, bx:bx + bw] == comp).astype(np.uint8) * 255
        _process_blob(target, canvas, opaque_mask, sub, (bx, by), min_area,
                      use_rects, budget, added, 0, alphas, climb, weight)
    return added


# ---------------------------------------------------------------------------
# Final stage: global color refit + dead-layer removal
# ---------------------------------------------------------------------------

def _refit_colors(background, shapes, target, opaque_mask):
    """Re-derive every shape's color from the pixels it actually shows.

    Painter's order means later shapes occlude earlier ones, so a shape's
    optimal color is the mean target color over its *visible* pixels, not the
    pixels it was fitted on. Shapes with zero visible opaque pixels are dead
    layers and are dropped entirely. Mutates colors; returns the kept list.
    """
    h, w = target.shape[:2]
    owner = np.full((h, w), -1, np.int32)
    for index, shape in enumerate(shapes):
        x0, y0, x1, y1 = _shape_bbox(shape, w, h, 1.0)
        mask = _shape_mask(shape, x0, y0, x1, y1, 1.0)
        if mask.size:
            owner[y0:y1, x0:x1][mask] = index

    visible_owner = owner[opaque_mask]
    visible_rgb = target[opaque_mask]
    sums = np.zeros((len(shapes), 3), np.float64)
    counts = np.zeros(len(shapes), np.float64)
    drawn = visible_owner >= 0
    np.add.at(sums, visible_owner[drawn], visible_rgb[drawn])
    np.add.at(counts, visible_owner[drawn], 1.0)

    # Background shows wherever no drawable owns the pixel.
    if int(background["color"][3]) > 0 and (~drawn).any():
        bg = np.round(visible_rgb[~drawn].mean(axis=0)).astype(int)
        background["color"][:3] = [int(bg[0]), int(bg[1]), int(bg[2])]

    kept = []
    for index, shape in enumerate(shapes):
        if counts[index] < 1.0:
            continue  # fully occluded or transparent-only: dead layer
        r, g, b = (int(v) for v in np.round(sums[index] / counts[index]))
        shape["color"] = [r, g, b, 255]
        kept.append(shape)
    return kept


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def refine_image(image_path, out_json_path, max_shapes: int = DEFAULT_SHAPES,
                 base_colors: int = DEFAULT_BASE_COLORS,
                 max_resolution: int = DEFAULT_MAX_RESOLUTION,
                 use_rects: bool = True, preview_path=None) -> dict:
    """Convert an image to geometry-JSON via the Residual Pyramid method.

    Returns a report dict ``{output, layers, base_layers, colors, seconds}``
    (superset of the ``flatten_image`` report keys the GUI logs).
    """
    started = time.perf_counter()
    image_path = Path(image_path)
    out_json_path = Path(out_json_path)
    max_shapes = max(1, int(max_shapes))

    rgb, opaque_mask, had_alpha = _load_rgb(image_path, max_resolution)
    target = rgb.astype(np.float32)

    # Stage 1 — coarse posterized base (global structure in few big shapes).
    labels, palette = _quantize(rgb, base_colors, opaque_mask)
    base = _build_payload(rgb, labels, palette, opaque_mask, had_alpha,
                          use_rects, min_area=BASE_MIN_AREA)
    background = base["shapes"][0]
    base_cap = max(1, min(BASE_MAX_SHAPES, int(max_shapes * BASE_BUDGET_FRACTION)))
    # _build_payload sorts drawables big->small; keep the biggest, residual
    # passes will re-cover whatever the dropped small ones owned.
    drawables = base["shapes"][1:1 + base_cap]

    bg_r, bg_g, bg_b, bg_a = (int(v) for v in background["color"])
    canvas = np.empty_like(target)
    canvas[:, :] = (bg_r, bg_g, bg_b) if bg_a > 0 else (0, 0, 0)
    for shape in drawables:
        _paint(canvas, shape, np.array(shape["color"][:3], np.float32))

    # Stage 2 — coarse-to-fine batched residual refinement.
    shapes = list(drawables)
    for sigma, delta, min_area in PASS_SCHEDULE:
        budget = max_shapes - len(shapes)
        if budget <= 0:
            break
        shapes.extend(_residual_pass(target, canvas, opaque_mask, sigma, delta,
                                     min_area, budget, use_rects))
    for _ in range(MAX_FINAL_PASSES):
        budget = max_shapes - len(shapes)
        if budget <= 0:
            break
        sigma, delta, min_area = FINAL_PASS
        added = _residual_pass(target, canvas, opaque_mask, sigma, delta,
                               min_area, budget, use_rects)
        if not added:
            break
        shapes.extend(added)

    # Stage 3 — global color refit on visible pixels + dead-layer removal.
    shapes = _refit_colors(background, shapes, target, opaque_mask)

    data = {"shapes": [background] + shapes}
    _atomic_write_bytes(
        out_json_path,
        lambda p: Path(p).write_text(json.dumps(data), encoding="utf-8"),
    )
    if preview_path is not None:
        _write_preview(data, Path(preview_path))

    return {
        "output": str(out_json_path),
        "layers": len(shapes),
        "base_layers": len(drawables),
        # distinct fill colors actually emitted (not the base posterize count).
        "colors": len({tuple(int(c) for c in s["color"][:3]) for s in shapes}),
        "seconds": round(time.perf_counter() - started, 3),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Residual Pyramid refiner -> geometry-JSON (fast CPU, full image).")
    parser.add_argument("image", help="source image path")
    parser.add_argument("-o", "--output", required=True, help="output geometry-JSON path")
    parser.add_argument("--shapes", type=int, default=DEFAULT_SHAPES, help="total shape budget")
    parser.add_argument("--base-colors", type=int, default=DEFAULT_BASE_COLORS,
                        help="posterize color count for the coarse base")
    parser.add_argument("--max-res", type=int, default=DEFAULT_MAX_RESOLUTION,
                        help="longest-edge resize")
    parser.add_argument("--ellipse-only", action="store_true", help="disable rect candidates")
    parser.add_argument("--preview", default=None, help="optional preview PNG path")
    args = parser.parse_args(argv)

    report = refine_image(
        args.image, args.output, max_shapes=args.shapes, base_colors=args.base_colors,
        max_resolution=args.max_res, use_rects=not args.ellipse_only,
        preview_path=args.preview,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
