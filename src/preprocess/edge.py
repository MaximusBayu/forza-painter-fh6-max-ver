"""Edge-aware preprocessing (improvement-plan Phase 4).

Flatten low-detail texture/noise (bilateral filter) so flat regions need fewer
shapes, while sharpening structural edges (unsharp mask) so the greedy
generator spends its shape budget on real detail. Strength is a 0..1 setting.

Follows preprocess/luma.py: BGRA-safe, atomic write, raises PreprocessError.
"""

from __future__ import annotations

import os

import cv2
import numpy as np
from pathlib import Path

from utils import PreprocessError

DEFAULT_STRENGTH = 0.6


def edge_aware(image_path: str | Path, strength: float = DEFAULT_STRENGTH) -> Path:
    """Apply edge-aware preprocessing and write the result atomically.

    Returns the path to the preprocessed output file.
    """
    image_path = Path(image_path)
    bgra = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if bgra is None:
        raise PreprocessError(f"failed to read image: {image_path}")

    result = _apply_preprocess(bgra, strength)
    output_path = image_path.with_name(f"{image_path.stem}.edge_aware{image_path.suffix}")

    tmp_path = output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")
    ok = cv2.imwrite(str(tmp_path), result)
    if not ok:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise PreprocessError(f"failed to write preprocessed image: {output_path}")

    try:
        os.replace(str(tmp_path), str(output_path))
    except OSError as exc:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise PreprocessError(f"failed to finalize preprocessed image: {output_path}") from exc

    return output_path


def _apply_preprocess(bgra: np.ndarray, strength: float) -> np.ndarray:
    if bgra.ndim != 3:
        raise PreprocessError(f"expected 3D image array, got shape {bgra.shape}")

    channels = bgra.shape[2]
    if channels not in (3, 4):
        raise PreprocessError(f"expected 3 or 4 channels, got {channels}")

    strength = float(max(0.0, min(1.0, strength)))
    bgr = np.clip(bgra[..., :3], 0, 255).astype(np.uint8)
    has_alpha = channels == 4
    if has_alpha:
        alpha = np.clip(bgra[..., 3], 0, 255).astype(np.uint8)

    # 1) Edge-preserving smoothing: flatten texture/noise but keep edges.
    sigma = 30.0 + 60.0 * strength
    smoothed = cv2.bilateralFilter(bgr, d=7, sigmaColor=sigma, sigmaSpace=sigma)

    # 2) Unsharp mask: re-emphasize structural edges.
    amount = 0.8 * strength
    blur = cv2.GaussianBlur(smoothed, (0, 0), sigmaX=1.5)
    sharp = cv2.addWeighted(
        smoothed.astype(np.float32), 1.0 + amount,
        blur.astype(np.float32), -amount, 0.0,
    )
    bgr_out = np.clip(sharp, 0, 255).astype(np.uint8)

    if has_alpha:
        return np.dstack([bgr_out, alpha]).astype(np.uint8)
    return bgr_out.astype(np.uint8)
