"""Phase 7 — differentiable vector engine (LIVE-style), diffvg-free CPU build.

The plan's Phase 7 calls for a differentiable rasterizer that back-props an image
loss into *all* shape parameters at once (no greedy, every shape co-optimized).
The canonical route is diffvg, which needs CUDA + a custom C++/CMake build and
bloats the EXE (see the plan critique). This module implements the same idea with
a from-scratch **soft rasterizer in plain PyTorch on CPU** — no diffvg, no CUDA —
so it runs anywhere torch is installed and keeps the FH6 output schema.

Each shape is a rotated ellipse with a soft (sigmoid) coverage mask; shapes are
alpha-composited back-to-front and optimized with Adam against an MSE image loss.
Output maps to the standard geometry JSON the importer already consumes.

torch is an optional dependency (not in requirements*.txt); callers must handle
``torch is None``. The torch<->numpy bridge can be broken when torch was built
against a different numpy ABI, so this module converts arrays via ``.tolist()``
rather than ``torch.from_numpy`` / ``.numpy()``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from geometry_json import ShapeType

ROTATED_ELLIPSE = int(ShapeType.ROTATED_ELLIPSE)
RECTANGLE = int(ShapeType.RECTANGLE)


def load_torch():
    """Lazy-load torch. Returns the module or None if unavailable."""
    if not hasattr(load_torch, "_cache"):
        try:
            import torch  # noqa: PLC0415

            load_torch._cache = torch
        except Exception:
            load_torch._cache = None
    return load_torch._cache


def _downscale_rgb(image_rgb, max_size):
    """Downscale an (H, W, 3) uint8 array to fit max_size. Uses cv2 if present."""
    import numpy as np  # local; numpy is required for image prep

    arr = np.asarray(image_rgb)[:, :, :3]
    h, w = arr.shape[:2]
    longest = max(h, w)
    if longest <= max_size:
        return arr.astype("float32") / 255.0, 1.0
    scale = max_size / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    try:
        import cv2

        small = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    except Exception:
        ys = (np.linspace(0, h - 1, new_h)).astype("int64")
        xs = (np.linspace(0, w - 1, new_w)).astype("int64")
        small = arr[ys][:, xs]
    return small.astype("float32") / 255.0, scale


def optimize_shapes(
    target_rgb,
    num_shapes=48,
    iters=120,
    max_size=96,
    lr=0.05,
    sharpness=12.0,
    seed=0,
):
    """Co-optimize ``num_shapes`` rotated ellipses to match ``target_rgb``.

    Returns (result_dict, report). ``result_dict`` is normalized geometry at the
    downscaled optimization resolution; ``report`` has initial/final loss.
    """
    torch = load_torch()
    if torch is None:
        raise RuntimeError("PyTorch is not installed; Phase 7 engine unavailable")
    torch.manual_seed(seed)

    small, _scale = _downscale_rgb(target_rgb, max_size)
    H, W = small.shape[:2]
    target = torch.tensor(small.tolist(), dtype=torch.float32)  # (H, W, 3), 0..1

    # Coordinate grid (pixel centers); built with torch to avoid the numpy bridge.
    ys = torch.arange(H, dtype=torch.float32).reshape(H, 1) + 0.5
    xs = torch.arange(W, dtype=torch.float32).reshape(1, W) + 0.5

    # Parameters (raw, transformed below to keep them in valid ranges).
    g = torch.Generator().manual_seed(seed)
    cx = torch.rand(num_shapes, generator=g) * W
    cy = torch.rand(num_shapes, generator=g) * H
    base_r = max(2.0, 0.25 * min(H, W))
    raw_w = torch.full((num_shapes,), math.log(math.expm1(base_r)))
    raw_h = torch.full((num_shapes,), math.log(math.expm1(base_r)))
    rot = torch.rand(num_shapes, generator=g) * 180.0
    raw_col = torch.zeros(num_shapes, 3)
    raw_a = torch.full((num_shapes,), 0.5)
    bg = target.reshape(-1, 3).mean(dim=0).clone()

    for t in (cx, cy, raw_w, raw_h, rot, raw_col, raw_a, bg):
        t.requires_grad_(True)

    params = [cx, cy, raw_w, raw_h, rot, raw_col, raw_a, bg]
    opt = torch.optim.Adam(params, lr=lr)

    def render():
        canvas = bg.reshape(1, 1, 3).expand(H, W, 3).clone()
        softplus = torch.nn.functional.softplus
        for i in range(num_shapes):
            w_i = softplus(raw_w[i]) + 1.0
            h_i = softplus(raw_h[i]) + 1.0
            theta = torch.deg2rad(-90.0 + rot[i])
            ct, st = torch.cos(theta), torch.sin(theta)
            dx = xs - cx[i]
            dy = ys - cy[i]
            xr = dx * ct + dy * st
            yr = -dx * st + dy * ct
            # rx <-> h, ry <-> w (matches geometry_json/optimize convention).
            d = (xr * xr) / (h_i * h_i) + (yr * yr) / (w_i * w_i)
            mask = torch.sigmoid((1.0 - d) * sharpness)  # (H, W)
            alpha = torch.sigmoid(raw_a[i])
            color = torch.sigmoid(raw_col[i])  # (3,)
            m = (mask * alpha).unsqueeze(-1)
            canvas = canvas * (1.0 - m) + color.reshape(1, 1, 3) * m
        return canvas

    with torch.no_grad():
        initial_loss = float(((render() - target) ** 2).mean())

    final_loss = initial_loss
    for _ in range(iters):
        opt.zero_grad()
        loss = ((render() - target) ** 2).mean()
        loss.backward()
        opt.step()
        final_loss = float(loss)

    # Extract optimized values (no numpy bridge).
    softplus = torch.nn.functional.softplus
    shapes = []
    bg_rgb = [int(round(max(0.0, min(1.0, v)) * 255)) for v in bg.detach().tolist()]
    shapes.append({"type": RECTANGLE, "data": [0, 0, W, H], "color": bg_rgb + [255], "score": 0})
    cxs, cys = cx.detach().tolist(), cy.detach().tolist()
    ws = (softplus(raw_w) + 1.0).detach().tolist()
    hs = (softplus(raw_h) + 1.0).detach().tolist()
    rots = rot.detach().tolist()
    cols = torch.sigmoid(raw_col).detach().tolist()
    alphas = torch.sigmoid(raw_a).detach().tolist()
    for i in range(num_shapes):
        r = int(round(cols[i][0] * 255))
        gg = int(round(cols[i][1] * 255))
        b = int(round(cols[i][2] * 255))
        a = int(round(alphas[i] * 255))
        shapes.append({
            "type": ROTATED_ELLIPSE,
            "data": [round(cxs[i]), round(cys[i]), max(1.0, ws[i]), max(1.0, hs[i]), round(rots[i]) % 360],
            "color": [r, gg, b, a],
            "score": 0,
        })

    report = {
        "initial_loss": round(initial_loss, 6),
        "final_loss": round(final_loss, 6),
        "num_shapes": num_shapes,
        "iters": iters,
        "opt_width": W,
        "opt_height": H,
    }
    return {"shapes": shapes}, report


def _rescale_geometry(data, factor):
    """Scale all coordinates/sizes by ``factor`` (map opt-res -> source-res)."""
    if factor == 1.0:
        return data
    out = []
    for idx, shape in enumerate(data["shapes"]):
        s = dict(shape)
        d = list(shape["data"])
        if idx == 0:  # background rect [0,0,w,h]
            d = [0, 0, round(d[2] * factor), round(d[3] * factor)]
        else:  # ellipse [x,y,w,h,rot]
            d = [round(d[0] * factor), round(d[1] * factor),
                 max(1.0, d[2] * factor), max(1.0, d[3] * factor), d[4]]
        s["data"] = d
        out.append(s)
    return {"shapes": out}


def vectorize_image(
    image_path,
    output_path=None,
    num_shapes=48,
    iters=120,
    max_size=96,
    **kwargs,
):
    """Vectorize an image into FH6 geometry JSON via the CPU differentiable engine.

    Writes the geometry JSON (same schema the importer reads) and returns a report.
    """
    import numpy as np

    image_path = Path(image_path)
    try:
        import cv2

        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        rgb = None if bgr is None else bgr[:, :, ::-1].copy()
    except Exception:
        rgb = None
    if rgb is None:
        from PIL import Image

        with Image.open(image_path) as im:
            rgb = np.asarray(im.convert("RGB"))

    orig_longest = max(rgb.shape[0], rgb.shape[1])
    data, report = optimize_shapes(rgb, num_shapes=num_shapes, iters=iters, max_size=max_size, **kwargs)
    factor = orig_longest / float(max(report["opt_width"], report["opt_height"]))
    data = _rescale_geometry(data, factor)

    if output_path is None:
        output_path = image_path.with_name(f"{image_path.stem}.diff.json")
    output_path = Path(output_path)
    output_path.write_text(json.dumps(data), encoding="utf-8")
    report["output"] = str(output_path)
    report["source_scale_factor"] = round(factor, 4)
    return report


# ===========================================================================
# R5 warm-start global refiner (GPU-accelerated; "Ultra+Diff").
#
# The from-scratch ``optimize_shapes`` above starts from random ellipses, which
# converges poorly past a few dozen shapes. This refiner instead WARM-STARTS
# from an existing solid solution (Ultra / Refine geometry-JSON) and lets Adam
# co-optimize *every* shape's position/size/rotation/color/alpha at once against
# the target — the global re-fit greedy geometrize never does. That is what
# fixes the "dominant color bleeds over the minority" artifact: shapes near a
# colour boundary can slide off the minority region and grow a translucent edge
# so the minority shows through.
#
# Two things make it actually work (both learned the hard way on CPU):
#   * Back-to-front compositing. FH6 / render_geometry paint the LAST shape on
#     top; a front-to-back raster does not match and the optimizer chases a
#     phantom. ``_soft_render`` composites back-to-front via reverse-cumprod
#     transmittance (validated against ``render_geometry`` to MSE < 2e-3).
#   * An anti-blur loss. Pure pixel-MSE smears toward mean colours and DROPS
#     SSIM; the loss here is L1 + an edge-gradient (Sobel) term so boundaries
#     stay crisp, plus a sharpness anneal (soft early for gradient flow, sharp
#     late to match the hard raster).
#
# Performance: O(N_shapes x H x W) per iteration. This is GPU territory — on a
# CUDA device a 3000-shape / 512px refine is seconds/iter; on CPU it is minutes
# (so callers should prefer Ultra alone on CPU). ``device`` auto-selects CUDA.
# ===========================================================================

# 384 keeps a 3000-shape refine within ~3-4 GB VRAM (autograd memory scales with
# shapes x opt_res^2, NOT with chunk). Raise toward 512-768 on >=8 GB cards;
# drop to 256 / fewer warm shapes on 4 GB laptops.
DIFF_DEFAULT_OPT_RES = 384
DIFF_DEFAULT_ITERS = 150
DIFF_GAME_LAYER_CAP = 3000  # FH6 import trims past this (main.py:373)


def _pick_device(torch):
    return "cuda" if torch.cuda.is_available() else "cpu"


def _to_tensor(torch, arr, device):
    """ndarray -> float32 tensor on ``device``, robust to the NumPy<->torch ABI
    break (falls back to ``.tolist()`` when ``from_numpy`` is unavailable)."""
    import numpy as np
    try:
        return torch.from_numpy(np.ascontiguousarray(arr)).to(device=device, dtype=torch.float32)
    except Exception:
        return torch.tensor(np.asarray(arr).tolist(), dtype=torch.float32, device=device)


def _geometry_to_params(torch, data, scale, device):
    """Warm-start: geometry-JSON -> optimizable params at ``scale`` (src->opt res).

    Per shape we store the rotated-frame half-extents (``ex`` along the local x
    axis ``xr``, ``ey`` along ``yr``) plus an angle that already folds in the
    ellipse's -90 deg convention, so ``_soft_render`` is a single uniform path
    for rect + ellipse. Sizes/colour/alpha are kept in unconstrained space
    (softplus / sigmoid) for stable optimisation. Returns ``(params, bg)``.
    """
    bg = data["shapes"][0]
    draw = data["shapes"][1:]
    cx, cy, ex0, ey0, ang0, isr, col0, al0 = [], [], [], [], [], [], [], []
    for s in draw:
        d = s["data"]
        rot = float(d[4]) if len(d) > 4 else 0.0
        cx.append(float(d[0]) * scale)
        cy.append(float(d[1]) * scale)
        if int(s["type"]) == RECTANGLE:
            ex0.append(abs(float(d[2])) * scale * 0.5)   # full -> half extent
            ey0.append(abs(float(d[3])) * scale * 0.5)
            ang0.append(rot)
            isr.append(1.0)
        else:                                            # rotated ellipse
            ex0.append(abs(float(d[3])) * scale)         # xr extent = rx = h (data[3])
            ey0.append(abs(float(d[2])) * scale)         # yr extent = ry = w (data[2])
            ang0.append(rot - 90.0)
            isr.append(0.0)
        c = s["color"]
        col0.append([c[0] / 255.0, c[1] / 255.0, c[2] / 255.0])
        al0.append((c[3] if len(c) > 3 else 255) / 255.0)

    def inv_softplus(v):
        v = max(0.51, float(v))
        return math.log(math.expm1(v - 0.5))

    def logit(p):
        p = min(0.999, max(0.001, float(p)))
        return math.log(p / (1.0 - p))

    def mk(xs, grad=True):
        t = torch.tensor(xs, dtype=torch.float32, device=device)
        return t.requires_grad_(True) if grad else t

    params = {
        "cx": mk(cx), "cy": mk(cy),
        "rex": mk([inv_softplus(v) for v in ex0]),
        "rey": mk([inv_softplus(v) for v in ey0]),
        "ang": mk(ang0),
        "rcol": mk([[logit(x) for x in c] for c in col0]),
        "ral": mk([logit(a) for a in al0]),
        "isr": mk(isr, grad=False),
        "bgc": mk([c / 255.0 for c in bg["color"][:3]], grad=False),
    }
    return params, bg


def _soft_render(torch, P, H, W, sharp, chunk, device, use_checkpoint=True):
    """Back-to-front chunk-vectorized soft raster -> (H, W, 3) in 0..1.

    Validated to reproduce ``geometry_optimize.render_geometry`` (the importer's
    model) at high ``sharp``. The Python loop is over ~N/chunk batches, not over
    shapes.

    ``use_checkpoint`` gradient-checkpoints each chunk: the big (chunk, H, W)
    mask tensors are recomputed in the backward pass instead of being stored, so
    PEAK memory scales with ONE chunk, not the whole shape set. Without this a
    3000-shape / 512px refine needs ~17 GB and spills to host RAM on a 4 GB GPU
    (slow as CPU); with it the same fits in a couple GB. Forward output is
    identical — only backward memory/compute change — so the render-match test
    still holds. Checkpointing is a no-op under ``torch.no_grad``.
    """
    from torch.utils.checkpoint import checkpoint

    sp = torch.nn.functional.softplus
    cx, cy = P["cx"], P["cy"]
    ex = sp(P["rex"]) + 0.5
    ey = sp(P["rey"]) + 0.5
    col = torch.sigmoid(P["rcol"])
    al = torch.sigmoid(P["ral"])
    ang, isr, bgc = P["ang"], P["isr"], P["bgc"]
    N = cx.shape[0]
    ys = torch.arange(H, dtype=torch.float32, device=device).reshape(H, 1) + 0.5
    xs = torch.arange(W, dtype=torch.float32, device=device).reshape(1, W) + 0.5
    ones = torch.ones(1, H, W, device=device)

    def _composite(Tt, out, cxs, cys, exs, eys, angs, irs, cols, als):
        axx = exs.reshape(-1, 1, 1).clamp(min=0.5)
        ayy = eys.reshape(-1, 1, 1).clamp(min=0.5)
        th = torch.deg2rad(angs).reshape(-1, 1, 1)
        ct, st = torch.cos(th), torch.sin(th)
        dx = xs.unsqueeze(0) - cxs.reshape(-1, 1, 1)
        dy = ys.unsqueeze(0) - cys.reshape(-1, 1, 1)
        xr = dx * ct + dy * st
        yr = -dx * st + dy * ct
        ell = torch.sigmoid((1.0 - ((xr / axx) ** 2 + (yr / ayy) ** 2)) * sharp)
        rect = torch.sigmoid((axx - xr.abs()) * sharp) * torch.sigmoid((ayy - yr.abs()) * sharp)
        ir = irs.reshape(-1, 1, 1)
        am = (ir * rect + (1.0 - ir) * ell) * als.reshape(-1, 1, 1)
        keep = 1.0 - am + 1e-7
        cum = torch.cumprod(torch.flip(keep, [0]), 0)
        after = torch.flip(torch.cat([ones, cum[:-1]], 0), [0])  # from later shapes
        Tafter = Tt.unsqueeze(0) * after
        out = out + ((am * Tafter).unsqueeze(-1) * cols.reshape(-1, 1, 1, 3)).sum(0)
        return out, Tt * cum[-1]

    Tt = torch.ones(H, W, device=device)
    out = torch.zeros(H, W, 3, device=device)
    do_ckpt = use_checkpoint and torch.is_grad_enabled()
    # Process chunks last->first so later shapes (drawn on top) composite first.
    for s in range(((max(N, 1) - 1) // chunk) * chunk, -1, -chunk):
        e = min(N, s + chunk)
        if e <= s:
            continue
        args = (Tt, out, cx[s:e], cy[s:e], ex[s:e], ey[s:e], ang[s:e],
                isr[s:e], col[s:e], al[s:e])
        if do_ckpt:
            out, Tt = checkpoint(_composite, *args, use_reentrant=False)
        else:
            out, Tt = _composite(*args)
    return out + bgc.reshape(1, 1, 3) * Tt.unsqueeze(-1)


def _sobel(torch, img):
    """Per-channel Sobel (gx, gy) of an (H, W, 3) image, via depthwise conv2d."""
    F = torch.nn.functional
    x = img.permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
    kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
                      device=img.device).reshape(1, 1, 3, 3).repeat(3, 1, 1, 1)
    ky = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
                      device=img.device).reshape(1, 1, 3, 3).repeat(3, 1, 1, 1)
    gx = F.conv2d(x, kx, padding=1, groups=3)
    gy = F.conv2d(x, ky, padding=1, groups=3)
    return gx, gy


def _emit_geometry(torch, P, scale, bg):
    """Optimized params -> geometry-JSON (importer schema). Centers int, sizes
    float, rotation int, per-shape alpha byte."""
    sp = torch.nn.functional.softplus
    ex = (sp(P["rex"]) + 0.5).detach().cpu().tolist()
    ey = (sp(P["rey"]) + 0.5).detach().cpu().tolist()
    cx = P["cx"].detach().cpu().tolist()
    cy = P["cy"].detach().cpu().tolist()
    ang = P["ang"].detach().cpu().tolist()
    col = torch.sigmoid(P["rcol"]).detach().cpu().tolist()
    al = torch.sigmoid(P["ral"]).detach().cpu().tolist()
    isr = P["isr"].detach().cpu().tolist()
    shapes = [bg]
    for i in range(len(cx)):
        if isr[i] > 0.5:
            w = max(1.0, ex[i] * 2.0 / scale)
            h = max(1.0, ey[i] * 2.0 / scale)
            rot = round(ang[i]) % 360
            d = [int(round(cx[i] / scale)), int(round(cy[i] / scale)), float(w), float(h)]
            if rot:
                d.append(rot)
            typ = RECTANGLE
        else:
            h = max(1.0, ex[i] / scale)
            w = max(1.0, ey[i] / scale)
            rot = round(ang[i] + 90.0) % 360
            d = [int(round(cx[i] / scale)), int(round(cy[i] / scale)), float(w), float(h), rot]
            typ = ROTATED_ELLIPSE
        c = [int(round(col[i][0] * 255)), int(round(col[i][1] * 255)),
             int(round(col[i][2] * 255)), max(1, min(255, int(round(al[i] * 255))))]
        shapes.append({"type": typ, "data": d, "color": c, "score": 0})
    return {"shapes": shapes}


def _seed_residual_shapes(torch, target_np, render_np, n_add, device, min_area=6):
    """Seed up to ``n_add`` new shapes on the worst residual blobs.

    After the warm-start shapes are refined, whatever the image still gets wrong
    — thin hair strands Ultra dropped, gradient transitions flat solids cannot
    express, edges — shows up as residual error. Connected high-residual blobs
    become small axis-aligned rects (the optimizer then rotates/resizes/recolors
    them). Returns a param-dict slice on ``device`` to concatenate, or ``None``.
    """
    import numpy as np
    import cv2

    res = np.abs(target_np - render_np).sum(2).astype(np.float32)  # HxW, 0..3
    res = cv2.GaussianBlur(res, (0, 0), 1.2)
    pos = res[res > 0]
    if n_add <= 0 or pos.size == 0:
        return None
    # Threshold on a fraction of the peak residual (robust whether the residual
    # is sparse high spots or a broad uniform region); >= so a uniform blob
    # still passes.
    thr = max(0.04, 0.25 * float(res.max()))
    mask = (res >= thr).astype(np.uint8)
    n_comp, lbl, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    if n_comp <= 1:
        return None
    mass = np.bincount(lbl.ravel(), weights=res.ravel(), minlength=n_comp)
    order = np.argsort(mass[1:])[::-1] + 1  # worst residual first

    cx, cy, ex, ey, ang, isr, col, al = [], [], [], [], [], [], [], []
    for comp in order:
        if len(cx) >= n_add:
            break
        if stats[comp, cv2.CC_STAT_AREA] < min_area:
            continue
        cx.append(float(cent[comp][0]))
        cy.append(float(cent[comp][1]))
        ex.append(max(1.0, stats[comp, cv2.CC_STAT_WIDTH] / 2.0))   # rect half-width
        ey.append(max(1.0, stats[comp, cv2.CC_STAT_HEIGHT] / 2.0))  # rect half-height
        ang.append(0.0)
        isr.append(1.0)  # axis-aligned rect
        col.append([float(c) for c in target_np[lbl == comp].mean(0)])
        al.append(0.85)
    if not cx:
        return None

    def inv_softplus(v):
        v = max(0.51, float(v))
        return math.log(math.expm1(v - 0.5))

    def logit(p):
        p = min(0.999, max(0.001, float(p)))
        return math.log(p / (1.0 - p))

    def mk(xs, grad=True):
        t = torch.tensor(xs, dtype=torch.float32, device=device)
        return t.requires_grad_(True) if grad else t

    return {
        "cx": mk(cx), "cy": mk(cy),
        "rex": mk([inv_softplus(v) for v in ex]),
        "rey": mk([inv_softplus(v) for v in ey]),
        "ang": mk(ang),
        "rcol": mk([[logit(x) for x in c] for c in col]),
        "ral": mk([logit(a) for a in al]),
        "isr": mk(isr, grad=False),
    }


def refine_geometry(warm_data, target_rgb, opt_res=DIFF_DEFAULT_OPT_RES,
                    iters=DIFF_DEFAULT_ITERS, lr=0.01, sharp_start=4.0,
                    sharp_end=12.0, edge_weight=1.0, add_shapes=0, chunk=256,
                    device=None, progress=None):
    """Globally co-optimize a warm-start geometry against ``target_rgb``.

    Returns ``(refined_geometry, report)``. The shape COUNT is unchanged (the
    importer cap still applies to ``warm_data``); every shape is re-fitted.
    """
    torch = load_torch()
    if torch is None:
        raise RuntimeError("PyTorch is not installed; the R5 refiner is unavailable")
    import numpy as np
    if device is None:
        device = _pick_device(torch)

    small, _scale = _downscale_rgb(target_rgb, opt_res)     # (h,w,3) float 0..1
    H, W = small.shape[:2]
    target = _to_tensor(torch, small, device)               # (H,W,3)
    bg = warm_data["shapes"][0]
    src_long = max(float(bg["data"][2]), float(bg["data"][3]))
    scale = max(H, W) / src_long
    P, bg = _geometry_to_params(torch, warm_data, scale, device)

    grad_keys = ("cx", "cy", "rex", "rey", "ang", "rcol", "ral")
    opt = torch.optim.Adam([P[k] for k in grad_keys], lr=lr)
    tgx, tgy = _sobel(torch, target)

    def step_loss(sharp):
        ren = _soft_render(torch, P, H, W, sharp, chunk, device)
        data_l = (ren - target).abs().mean()
        rgx, rgy = _sobel(torch, ren)
        edge_l = (rgx - tgx).abs().mean() + (rgy - tgy).abs().mean()
        return data_l + edge_weight * edge_l, ren

    with torch.no_grad():
        l0 = float(step_loss(sharp_end)[0])
    final = l0
    iters = max(1, int(iters))
    seed_at = iters // 2 if add_shapes > 0 else -1
    for it in range(iters):
        if it == seed_at:
            # Phase B: seed new shapes on the worst residual (missing hair,
            # gradient transitions, edges) and keep optimizing the larger set.
            with torch.no_grad():
                ren = _soft_render(torch, P, H, W, sharp_end, chunk, device)
            new = _seed_residual_shapes(
                torch, target.detach().cpu().numpy(), ren.detach().cpu().numpy(),
                int(add_shapes), device)
            if new is not None:
                for k in grad_keys:
                    P[k] = torch.cat([P[k].detach(), new[k].detach()]).requires_grad_(True)
                P["isr"] = torch.cat([P["isr"], new["isr"]])
                opt = torch.optim.Adam([P[k] for k in grad_keys], lr=lr)
                if progress is not None:
                    progress(f"R5 seeded {new['cx'].shape[0]} new shapes on residual "
                             f"(now {P['cx'].shape[0]} total)")
        sharp = sharp_start + (sharp_end - sharp_start) * (it / max(1, iters - 1))
        opt.zero_grad()
        loss, _ = step_loss(sharp)
        loss.backward()
        opt.step()
        final = float(loss)
        if progress is not None and (it % 25 == 0 or it == iters - 1):
            progress(f"R5 refine iter {it + 1}/{iters} (loss {final:.4f}, "
                     f"{P['cx'].shape[0]} shapes)")

    refined = _emit_geometry(torch, P, scale, bg)
    report = {
        "initial_loss": round(l0, 6),
        "final_loss": round(final, 6),
        "iters": int(iters),
        "opt_width": W,
        "opt_height": H,
        "device": device,
        "layers": len(refined["shapes"]) - 1,
    }
    return refined, report


def diff_refine_image(image_path, out_json_path, warm_json=None, warm_shapes=DIFF_GAME_LAYER_CAP,
                      opt_res=DIFF_DEFAULT_OPT_RES, iters=DIFF_DEFAULT_ITERS, add_shapes=0,
                      max_resolution=1400, preview_path=None, progress=None, **kwargs):
    """Top-level R5: warm-start (given JSON, else run Ultra) then globally refine.

    Writes refined geometry-JSON (same importer schema) + optional preview and
    returns a report ``{output, layers, initial_loss, final_loss, seconds,
    device, ...}``.
    """
    import time
    torch = load_torch()
    if torch is None:
        raise RuntimeError("PyTorch is not installed; the R5 refiner is unavailable")
    started = time.perf_counter()
    image_path = Path(image_path)
    out_json_path = Path(out_json_path)
    # Keep warm + seeded shapes within the game layer cap.
    if add_shapes > 0:
        warm_shapes = max(1, min(int(warm_shapes), DIFF_GAME_LAYER_CAP - int(add_shapes)))

    from flatten import _load_rgb
    rgb, _mask, _had = _load_rgb(image_path, max_resolution)   # (H,W,3) uint8 at <=max_res

    if warm_json is not None:
        warm = json.loads(Path(warm_json).read_text(encoding="utf-8"))
    else:
        import ultra
        warm_tmp = out_json_path.with_suffix(".warm.json")
        if progress is not None:
            progress(f"R5: building Ultra warm start ({warm_shapes} shapes)...")
        ultra.ultra_image(image_path, warm_tmp, max_shapes=warm_shapes,
                          max_resolution=max_resolution, progress=progress)
        warm = json.loads(warm_tmp.read_text(encoding="utf-8"))

    refined, report = refine_geometry(warm, rgb, opt_res=opt_res, iters=iters,
                                      add_shapes=add_shapes, progress=progress, **kwargs)
    out_json_path.write_text(json.dumps(refined), encoding="utf-8")
    if preview_path is not None:
        from ultra import _write_preview_blend
        _write_preview_blend(refined, Path(preview_path))
    report["output"] = str(out_json_path)
    report["seconds"] = round(time.perf_counter() - started, 3)
    report["colors"] = len({tuple(int(c) for c in s["color"][:3]) for s in refined["shapes"][1:]})
    return report


def _main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="R5 differentiable vectorizer/refiner -> FH6 geometry JSON. "
                    "Default warm-starts from Ultra and globally refines (GPU recommended).")
    parser.add_argument("image", help="Input image")
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("--warm", default=None,
                        help="warm-start geometry JSON; if omitted, Ultra is run first")
    parser.add_argument("--warm-shapes", type=int, default=DIFF_GAME_LAYER_CAP,
                        help="shape budget for the auto Ultra warm start (<= game cap)")
    parser.add_argument("--opt-res", type=int, default=DIFF_DEFAULT_OPT_RES,
                        help="optimization resolution (longest edge); higher = sharper + slower")
    parser.add_argument("-i", "--iters", type=int, default=DIFF_DEFAULT_ITERS)
    parser.add_argument("--edge-weight", type=float, default=1.0,
                        help="weight of the anti-blur edge-gradient loss term")
    parser.add_argument("--add-shapes", type=int, default=0,
                        help="seed N new shapes on the residual mid-refine for missing "
                             "hair/gradient detail; warm budget auto-reduced so warm+add <= game cap")
    parser.add_argument("--preview", default=None, help="optional preview PNG path")
    parser.add_argument("--from-scratch", action="store_true",
                        help="legacy random-init ellipse optimizer (no warm start)")
    parser.add_argument("-n", "--num-shapes", type=int, default=48, help="(--from-scratch only)")
    parser.add_argument("--max-size", type=int, default=96, help="(--from-scratch only)")
    args = parser.parse_args(argv)
    if load_torch() is None:
        print("PyTorch not installed; cannot run the differentiable engine.")
        return 1
    if args.from_scratch:
        report = vectorize_image(
            args.image, args.output, num_shapes=args.num_shapes,
            iters=args.iters if args.iters != DIFF_DEFAULT_ITERS else 120,
            max_size=args.max_size,
        )
    else:
        out = args.output or str(Path(args.image).with_name(Path(args.image).stem + ".r5.json"))
        report = diff_refine_image(
            args.image, out, warm_json=args.warm, warm_shapes=args.warm_shapes,
            opt_res=args.opt_res, iters=args.iters, edge_weight=args.edge_weight,
            add_shapes=args.add_shapes, preview_path=args.preview, progress=print,
        )
    for key, value in report.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
