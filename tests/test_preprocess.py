"""Phase 4 (edge_aware) + Phase 5 (luma levels setting) preprocessing tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from preprocess.edge import edge_aware
from preprocess.luma import luma_band, DEFAULT_LEVELS


def _write_image(path: Path, channels=3, size=48):
    rng = np.random.default_rng(7)
    img = (rng.random((size, size, channels)) * 255).astype(np.uint8)
    # Add a hard edge so sharpening has something to act on.
    img[:, : size // 2] = 40
    img[:, size // 2 :] = 200
    cv2.imwrite(str(path), img)
    return img


class TestEdgeAware:
    def test_writes_output_same_size(self, tmp_path: Path):
        src = tmp_path / "in.png"
        original = _write_image(src, channels=3)
        out = edge_aware(src)
        assert out.exists()
        assert out != src
        result = cv2.imread(str(out), cv2.IMREAD_UNCHANGED)
        assert result.shape[:2] == original.shape[:2]

    def test_preserves_alpha_channel(self, tmp_path: Path):
        src = tmp_path / "in.png"
        img = _write_image(src, channels=4)
        # Force a known alpha pattern.
        img[..., 3] = 123
        cv2.imwrite(str(src), img)
        out = edge_aware(src)
        result = cv2.imread(str(out), cv2.IMREAD_UNCHANGED)
        assert result.shape[2] == 4
        assert int(result[..., 3].min()) == 123

    def test_strength_zero_runs(self, tmp_path: Path):
        src = tmp_path / "in.png"
        _write_image(src, channels=3)
        out = edge_aware(src, strength=0.0)
        assert out.exists()


class TestLumaLevels:
    def test_custom_levels_changes_output(self, tmp_path: Path):
        src = tmp_path / "in.png"
        _write_image(src, channels=3)
        default_out = luma_band(src)
        default_img = cv2.imread(str(default_out), cv2.IMREAD_UNCHANGED).copy()
        # Re-run with very few levels -> stronger banding -> different pixels.
        few_out = luma_band(src, levels=4)
        few_img = cv2.imread(str(few_out), cv2.IMREAD_UNCHANGED)
        assert not np.array_equal(default_img, few_img)

    def test_default_levels_constant(self):
        assert DEFAULT_LEVELS == 24.0

    def test_levels_clamped_low(self, tmp_path: Path):
        src = tmp_path / "in.png"
        _write_image(src, channels=3)
        # levels < 2 must not crash (clamped).
        out = luma_band(src, levels=1)
        assert out.exists()


class TestDispatch:
    def test_preprocess_input_image_edge_mode(self, tmp_path: Path):
        from generator_backend import preprocess_input_image, SettingProfile

        src = tmp_path / "in.png"
        _write_image(src, channels=3)
        setting = SettingProfile(
            index=1, source="custom", path=tmp_path / "x.ini", label="x",
            values={"preprocessMode": "edge_aware", "edgeAwareStrength": "0.4"},
        )
        out = preprocess_input_image(src, setting)
        assert Path(out).exists()
        assert "edge_aware" in Path(out).name

    def test_preprocess_input_image_none_returns_original(self, tmp_path: Path):
        from generator_backend import preprocess_input_image, SettingProfile

        src = tmp_path / "in.png"
        _write_image(src, channels=3)
        setting = SettingProfile(
            index=1, source="custom", path=tmp_path / "x.ini", label="x",
            values={"preprocessMode": "none"},
        )
        out = preprocess_input_image(src, setting)
        assert Path(out) == src

    def test_preprocess_input_image_luma_levels(self, tmp_path: Path):
        from generator_backend import preprocess_input_image, SettingProfile

        src = tmp_path / "in.png"
        _write_image(src, channels=3)
        setting = SettingProfile(
            index=1, source="custom", path=tmp_path / "x.ini", label="x",
            values={"preprocessMode": "luma_band", "lumaLevels": "6"},
        )
        out = preprocess_input_image(src, setting)
        assert Path(out).exists()
        assert "luma_band" in Path(out).name
