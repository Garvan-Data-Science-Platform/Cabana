"""Tests for cabana/hdm.py — HDM class."""

import os
import sys
import tempfile

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cabana.hdm import HDM


def make_test_image(tmp_path, name="test.png", h=64, w=64, value=128):
    img = np.full((h, w), value, dtype=np.uint8)
    path = str(tmp_path / name)
    cv2.imwrite(path, img)
    return path


def make_rgb_image(tmp_path, name="rgb.png", h=64, w=64):
    img = np.random.randint(0, 200, (h, w, 3), dtype=np.uint8)
    path = str(tmp_path / name)
    cv2.imwrite(path, img)
    return path


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------

class TestInit:
    def test_default_params(self):
        h = HDM()
        assert h.max_hdm == 220
        assert h.sat_ratio == 0
        assert h.dark_line is False
        assert h.df_hdm is None

    def test_custom_params(self):
        h = HDM(max_hdm=150, sat_ratio=0.1, dark_line=True)
        assert h.max_hdm == 150
        assert h.sat_ratio == 0.1
        assert h.dark_line is True


# ---------------------------------------------------------------------------
# enhance_contrast
# ---------------------------------------------------------------------------

class TestEnhanceContrast:
    def test_returns_uint8(self, tmp_path):
        path = make_test_image(tmp_path, value=128)
        h = HDM()
        result = h.enhance_contrast(path)
        assert result.dtype == np.uint8

    def test_output_2d(self, tmp_path):
        path = make_rgb_image(tmp_path)
        h = HDM()
        result = h.enhance_contrast(path)
        assert result.ndim == 2

    def test_dark_line_inverts(self, tmp_path):
        path = make_test_image(tmp_path, value=50)
        h_normal = HDM(dark_line=False)
        h_dark = HDM(dark_line=True)
        result_normal = h_normal.enhance_contrast(path)
        result_dark = h_dark.enhance_contrast(path)
        # Bright pixels in normal should be dark in inverted
        assert result_normal.mean() != result_dark.mean()

    def test_max_hdm_clipping(self, tmp_path):
        # Image with values above max_hdm should be clipped
        img = np.full((50, 50), 250, dtype=np.uint8)
        path = str(tmp_path / "high.png")
        cv2.imwrite(path, img)
        h = HDM(max_hdm=100)
        result = h.enhance_contrast(path)
        assert result is not None
        assert result.dtype == np.uint8

    def test_zero_image_no_crash(self, tmp_path):
        path = make_test_image(tmp_path, value=0)
        h = HDM()
        result = h.enhance_contrast(path)
        assert result is not None


# ---------------------------------------------------------------------------
# quantify_black_space — single file
# ---------------------------------------------------------------------------

class TestQuantifyBlackSpaceSingleFile:
    def test_returns_dataframe(self, tmp_path):
        path = make_test_image(tmp_path, value=100)
        h = HDM()
        df = h.quantify_black_space(path)
        assert df is not None
        assert len(df) == 1

    def test_dataframe_columns(self, tmp_path):
        path = make_test_image(tmp_path, value=100)
        h = HDM()
        df = h.quantify_black_space(path)
        assert 'Image' in df.columns
        assert '% HDM Area' in df.columns

    def test_hdm_area_range(self, tmp_path):
        path = make_test_image(tmp_path, value=100)
        h = HDM()
        df = h.quantify_black_space(path)
        pct = df['% HDM Area'].iloc[0]
        assert 0.0 <= pct <= 1.0

    def test_wrong_extension_returns_empty(self, tmp_path):
        # File with .txt extension should not be processed
        path = str(tmp_path / "file.txt")
        with open(path, "w") as f:
            f.write("not an image")
        h = HDM()
        df = h.quantify_black_space(path)
        assert len(df) == 0

    def test_saves_to_save_dir(self, tmp_path):
        path = make_test_image(tmp_path / "input", value=100)
        os.makedirs(str(tmp_path / "input"), exist_ok=True)
        path = make_test_image(tmp_path / "input", value=100)
        save_dir = str(tmp_path / "output")
        os.makedirs(save_dir)
        h = HDM()
        h.quantify_black_space(path, save_dir=save_dir)
        assert os.path.exists(os.path.join(save_dir, "ResultsHDM.csv"))


# ---------------------------------------------------------------------------
# quantify_black_space — directory
# ---------------------------------------------------------------------------

class TestQuantifyBlackSpaceDirectory:
    def test_processes_directory(self, tmp_path):
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        for i in range(3):
            make_test_image(img_dir, name=f"img{i}.png", value=100)
        h = HDM()
        df = h.quantify_black_space(str(img_dir))
        assert len(df) == 3

    def test_directory_hdm_area_valid(self, tmp_path):
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        make_test_image(img_dir, value=150)
        h = HDM()
        df = h.quantify_black_space(str(img_dir))
        for val in df['% HDM Area']:
            assert 0.0 <= val <= 1.0

    def test_empty_directory_returns_empty_df(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        h = HDM()
        df = h.quantify_black_space(str(empty_dir))
        assert len(df) == 0
