"""Ultra generate mode (R8) — multi-resolution residual ladder, beats geometrize.

Builds on the Residual Pyramid (``refine.py``) and attacks the two structural
weaknesses of the greedy GPU geometrize pipeline:

  1. **Resolution.** Geometrize fits shapes at its downscaled working
     resolution (e.g. 1400 px longest edge for a 9449 px source) and the
     coarse modes here at 384 px. Every edge in the output is positioned at
     that granularity, which is exactly the "rough / pixelated edges" seen
     in-game once the livery is scaled up. Ultra climbs a resolution ladder
     (F/4 -> F/2 -> F): global structure is fitted cheaply at low res, the
     geometry is rescaled (float precision), and finer residual passes
     re-align every edge against progressively sharper targets. The final
     level fits at full working resolution F, so edge placement error is a
     fraction of geometrize's.
  2. **No re-fitting.** Geometrize never revisits a placed shape. Ultra
     re-fits twice per level transition (the residual passes correct the
     rescaled geometry) and finishes with a global visible-pixel color refit
     plus dead-layer removal; the freed budget is then re-spent on extra
     full-resolution detail passes.

The per-pass machinery (error map -> blur -> threshold -> blob -> deterministic
rect/ellipse fit -> accept only on real SSE gain -> recursive percentile split)
is imported from ``refine.py`` — see
``.claude/PRPs/notes/residual-pyramid-method.md`` for that core and
``.claude/PRPs/notes/ultra-method.md`` for this ladder.

Emits the standard geometry-JSON schema at the final working resolution
(background ``[0, 0, W, H]``; the importer scales to game space). Centers are
emitted as ints, sizes as floats (float ellipse sizes are importer-supported).
"""
# SOURCE: src/refine.py (module header + import conventions)
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from utils import PreprocessError
from geometry_optimize import render_geometry
from flatten import _load_rgb, _quantize, _build_payload, _atomic_write_bytes
from refine import (
    BASE_MIN_AREA,
    BASE_MAX_SHAPES,
    BASE_BUDGET_FRACTION,
    DEFAULT_BASE_COLORS,
    _paint,
    _residual_pass,
    _refit_colors,
)

DEFAULT_SHAPES = 3000        # = the game's drawable-layer cap (main.py:373)
DEFAULT_MAX_RESOLUTION = 1400
# FH6 import trims to min(template_count, 3000) drawable layers (main.py:373),
# keeping the FIRST ones. Shapes are emitted in painter's order (structure ->
# detail -> polish last), so requesting more than this only changes the preview
# — the extra, finest-detail layers are dropped on import and the in-game livery
# will not match the preview.
GAME_LAYER_CAP = 3000

# Per-level pass schedules: list of (blur_sigma, delta, min_area) tuples plus a
# repeated final pass. Lower levels chase structure; the top level chases edge
# placement at full working resolution.
LEVEL_SCHEDULES = (
    # level 0 (F/4) — structure ONLY. Fine detail placed here would be
    # rescaled 4x and land blurry; it belongs to the full-res level.
    {
        "passes": ((9.0, 28.0, 64), (5.0, 22.0, 32), (3.0, 16.0, 16)),
        "final": (1.5, 10.0, 8),
        "final_repeats": 2,
        "budget_share": 0.2,
    },
    # level 1 (F/2) — correct the rescaled geometry, add mid-frequency detail
    {
        "passes": ((3.0, 16.0, 24), (1.5, 10.0, 12)),
        "final": (0.8, 6.0, 6),
        "final_repeats": 4,
        "budget_share": 0.5,
    },
    # level 2 (F) — full-resolution edge alignment and fine detail
    {
        "passes": ((1.5, 12.0, 12), (0.8, 6.0, 4)),
        "final": (0.8, 4.0, 3),
        "final_repeats": 10,
        "budget_share": 0.85,
    },
)
# Alpha polish phase (after the solid levels + color refit): translucent
# shapes blend over the solid stack, reproducing anti-aliased edges and smooth
# shading that hard-edged solids cannot express — the same trick geometrize's
# alpha stacking uses, but placed deterministically on the residual.
POLISH_PASS = (0.8, 3.0, 2)
POLISH_ALPHAS = (1.0, 0.6, 0.35)
MAX_POLISH_PASSES = 12
# Pattern-search budget per accepted shape (geometrize-style local alignment,
# seeded by the deterministic fit instead of random mutations). Saturates
# around 96 on the bocchi benchmark.
CLIMB_EVALS = 96
# Gradient-saliency weighting of blob selection was tested and REJECTED:
# SSIM 0.661 vs 0.680 unweighted on the bocchi benchmark (it starves
# flat-error regions). refine._residual_pass keeps the generic ``weight``
# hook (default off) for future experiments.


def _write_preview_blend(data: dict, preview_path: Path) -> None:
    """Preview with in-game alpha blending (polish shapes are translucent)."""
    # SOURCE: flatten._write_preview (encode-by-extension + atomic replace)
    rgb = render_geometry(data, scale=1.0, blend_alpha=True)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(Path(preview_path).suffix or ".png", bgr)
    if not ok:
        raise PreprocessError(f"failed to encode preview: {preview_path}")
    _atomic_write_bytes(preview_path, lambda p: Path(p).write_bytes(buf.tobytes()))


def _scale_shapes(shapes, fx: float, fy: float) -> None:
    """Rescale shape geometry in place (level k -> level k+1, float precision).

    The x/y factors differ by <1% (independent rounding of each level's
    dimensions); the residual passes at the new level correct any sub-pixel
    drift this introduces on rotated shapes.
    """
    for shape in shapes:
        data = list(shape["data"])
        data[0] = data[0] * fx
        data[1] = data[1] * fy
        data[2] = data[2] * fx
        data[3] = data[3] * fy
        shape["data"] = data


def _render_stack(background, shapes, height, width):
    """Paint background + shapes onto a fresh float32 canvas (solid colors)."""
    bg_r, bg_g, bg_b, bg_a = (int(v) for v in background["color"])
    canvas = np.empty((height, width, 3), np.float32)
    canvas[:, :] = (bg_r, bg_g, bg_b) if bg_a > 0 else (0, 0, 0)
    for shape in shapes:
        _paint(canvas, shape, np.array(shape["color"][:3], np.float32))
    return canvas


def _run_schedule(target, canvas, opaque_mask, shapes, schedule, budget_cap,
                  use_rects, climb=CLIMB_EVALS, weight=None):
    """Run one level's passes + repeated final pass, appending to ``shapes``."""
    for sigma, delta, min_area in schedule["passes"]:
        budget = budget_cap - len(shapes)
        if budget <= 0:
            return
        shapes.extend(_residual_pass(target, canvas, opaque_mask, sigma, delta,
                                     min_area, budget, use_rects, climb=climb,
                                     weight=weight))
    sigma, delta, min_area = schedule["final"]
    for _ in range(schedule["final_repeats"]):
        budget = budget_cap - len(shapes)
        if budget <= 0:
            return
        added = _residual_pass(target, canvas, opaque_mask, sigma, delta,
                               min_area, budget, use_rects, climb=climb,
                               weight=weight)
        if not added:
            return
        shapes.extend(added)


def ultra_image(image_path, out_json_path, max_shapes: int = DEFAULT_SHAPES,
                base_colors: int = DEFAULT_BASE_COLORS,
                max_resolution: int = DEFAULT_MAX_RESOLUTION,
                use_rects: bool = True, preview_path=None,
                progress=None) -> dict:
    """Convert an image to geometry-JSON via the multi-resolution ladder.

    ``progress`` is an optional ``callable(str)`` for per-level log lines.
    Returns ``{output, layers, base_layers, colors, seconds, resolution}``.
    """
    started = time.perf_counter()
    image_path = Path(image_path)
    out_json_path = Path(out_json_path)
    max_shapes = max(1, int(max_shapes))
    if progress is not None and max_shapes > GAME_LAYER_CAP:
        progress(
            f"Warning: FH6 imports at most {GAME_LAYER_CAP} layers and trims the "
            f"rest (main.py:373) — the trimmed layers are the finest detail, so "
            f"the in-game result will not match the preview. {max_shapes} "
            f"requested; values above {GAME_LAYER_CAP} only help the preview. "
            f"Use <= {GAME_LAYER_CAP}.")

    levels = []
    for divisor in (4, 2, 1):
        res = max(256, max_resolution // divisor)
        if not levels or res > levels[-1]:
            levels.append(res)

    # ----- level 0: coarse base + structure passes (refine stage 1+2) -----
    rgb, opaque_mask, had_alpha = _load_rgb(image_path, levels[0])
    target = rgb.astype(np.float32)
    labels, palette = _quantize(rgb, base_colors, opaque_mask)
    base = _build_payload(rgb, labels, palette, opaque_mask, had_alpha,
                          use_rects, min_area=BASE_MIN_AREA)
    background = base["shapes"][0]
    base_cap = max(1, min(BASE_MAX_SHAPES, int(max_shapes * BASE_BUDGET_FRACTION)))
    shapes = base["shapes"][1:1 + base_cap]
    base_layers = len(shapes)
    canvas = _render_stack(background, shapes, *target.shape[:2])

    for index, level in enumerate(levels):
        schedule = LEVEL_SCHEDULES[min(index, len(LEVEL_SCHEDULES) - 1)]
        if index > 0:
            prev_h, prev_w = target.shape[:2]
            rgb, opaque_mask, had_alpha = _load_rgb(image_path, level)
            target = rgb.astype(np.float32)
            new_h, new_w = target.shape[:2]
            _scale_shapes(shapes, new_w / prev_w, new_h / prev_h)
            background["data"] = [0, 0, new_w, new_h]
            canvas = _render_stack(background, shapes, new_h, new_w)
        budget_cap = max(1, int(max_shapes * schedule["budget_share"]))
        _run_schedule(target, canvas, opaque_mask, shapes, schedule,
                      budget_cap, use_rects)
        if progress is not None:
            progress(f"Ultra level {index + 1}/{len(levels)} ({level}px): "
                     f"{len(shapes)} shapes")

    # ----- color refit + dead-layer removal (valid while everything is solid),
    # then re-render and spend the remaining budget on the alpha polish phase.
    shapes = _refit_colors(background, shapes, target, opaque_mask)
    canvas = _render_stack(background, shapes, *target.shape[:2])
    sigma, delta, min_area = POLISH_PASS
    polish_layers = 0
    for _ in range(MAX_POLISH_PASSES):
        budget = max_shapes - len(shapes)
        if budget <= 0:
            break
        added = _residual_pass(target, canvas, opaque_mask, sigma, delta,
                               min_area, budget, use_rects, alphas=POLISH_ALPHAS,
                               climb=CLIMB_EVALS)
        if not added:
            break
        shapes.extend(added)
        polish_layers += len(added)
    if progress is not None:
        progress(f"Ultra polish: {polish_layers} translucent/detail shapes")

    # Emit: int centers (sub-pixel is invisible at game scale), float sizes
    # (importer-supported), int rotation.
    for shape in shapes:
        data = list(shape["data"])
        data[0] = int(round(data[0]))
        data[1] = int(round(data[1]))
        data[2] = float(data[2])
        data[3] = float(data[3])
        if len(data) >= 5:
            data[4] = int(round(data[4])) % 360
        shape["data"] = data

    data = {"shapes": [background] + shapes}
    _atomic_write_bytes(
        out_json_path,
        lambda p: Path(p).write_text(json.dumps(data), encoding="utf-8"),
    )
    if preview_path is not None:
        _write_preview_blend(data, Path(preview_path))

    return {
        "output": str(out_json_path),
        "layers": len(shapes),
        "base_layers": base_layers,
        # distinct fill colors actually emitted (NOT the base posterize count —
        # every residual/polish shape derives its own color), so the GUI log
        # reflects the livery's real palette.
        "colors": len({tuple(int(c) for c in s["color"][:3]) for s in shapes}),
        "seconds": round(time.perf_counter() - started, 3),
        "resolution": levels[-1],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Ultra multi-resolution refiner -> geometry-JSON (CPU, beats geometrize fidelity).")
    parser.add_argument("image", help="source image path")
    parser.add_argument("-o", "--output", required=True, help="output geometry-JSON path")
    parser.add_argument("--shapes", type=int, default=DEFAULT_SHAPES, help="total shape budget")
    parser.add_argument("--base-colors", type=int, default=DEFAULT_BASE_COLORS,
                        help="posterize color count for the coarse base")
    parser.add_argument("--max-res", type=int, default=DEFAULT_MAX_RESOLUTION,
                        help="final working resolution (longest edge)")
    parser.add_argument("--ellipse-only", action="store_true", help="disable rect candidates")
    parser.add_argument("--preview", default=None, help="optional preview PNG path")
    args = parser.parse_args(argv)

    report = ultra_image(
        args.image, args.output, max_shapes=args.shapes, base_colors=args.base_colors,
        max_resolution=args.max_res, use_rects=not args.ellipse_only,
        preview_path=args.preview, progress=print,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
