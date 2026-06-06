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


def _main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Differentiable (CPU) image vectorizer -> FH6 geometry JSON.")
    parser.add_argument("image", help="Input image")
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("-n", "--num-shapes", type=int, default=48)
    parser.add_argument("-i", "--iters", type=int, default=120)
    parser.add_argument("--max-size", type=int, default=96)
    args = parser.parse_args(argv)
    if load_torch() is None:
        print("PyTorch not installed; cannot run the differentiable engine.")
        return 1
    report = vectorize_image(
        args.image, args.output, num_shapes=args.num_shapes, iters=args.iters, max_size=args.max_size
    )
    for key, value in report.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
