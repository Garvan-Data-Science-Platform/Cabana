"""Tests for cabana/wsi.py slide readers.

The flat-image reader is tested on synthetic files. The VSI reader is
exercised against the APGI TMA scans under ``large/`` when they are present
and skipped otherwise.
"""

import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cabana.wsi import FlatReader, VsiReader, open_slide

VSI = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "large", "APGI TMA", "APGI TMA 1 PicRed.vsi")


class TestFlatReader:
    def _img(self, tmp_path, name="slide.png", h=700, w=900):
        img = np.full((h, w, 3), 240, dtype=np.uint8)
        cv2.rectangle(img, (100, 100), (300, 250), (50, 60, 200), -1)
        p = str(tmp_path / name)
        cv2.imwrite(p, img)
        return p, img

    def test_pyramid_and_shapes(self, tmp_path):
        p, img = self._img(tmp_path)
        r = open_slide(p, pixel_size_um=2.0)
        assert isinstance(r, FlatReader)
        assert r.level_shape(0) == (700, 900)
        assert r.level_count >= 2
        assert r.level_downsample(1) == pytest.approx(2.0, rel=0.05)
        assert r.channels == ("BF",) and r.pixel_size_um == 2.0

    def test_read_region_with_overhang(self, tmp_path):
        p, img = self._img(tmp_path)
        r = open_slide(p, pixel_size_um=1.0)
        reg = r.read_region(-20, -30, 200, 200)
        assert reg.shape == (200, 200, 3)
        assert tuple(reg[0, 0]) == (255, 255, 255)              # fill outside the slide
        assert np.array_equal(reg[30:, 20:], img[:170, :180])

    def test_best_level_for_pixel_size(self, tmp_path):
        p, _ = self._img(tmp_path, h=2000, w=2000)
        r = open_slide(p, pixel_size_um=1.0)
        lv = r.best_level_for_pixel_size(4.0)
        assert r.pixel_size_um * r.level_downsample(lv) <= 4.0
        assert lv == r.level_count - 1 or r.pixel_size_um * r.level_downsample(lv + 1) > 4.0

    def test_tiff_resolution_tag(self, tmp_path):
        import tifffile
        img = np.full((64, 64, 3), 200, dtype=np.uint8)
        p = str(tmp_path / "res.tif")
        tifffile.imwrite(p, img, resolution=(10000 / 0.5, 10000 / 0.5), resolutionunit="CENTIMETER")
        r = open_slide(p)
        assert r.pixel_size_um == pytest.approx(0.5, rel=1e-3)

    def test_dark_slide_is_pol(self, tmp_path):
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        p = str(tmp_path / "dark.png")
        cv2.imwrite(p, img)
        r = open_slide(p, pixel_size_um=1.0)
        assert r.channels == ("POL",)
        assert tuple(r.read_region(-5, -5, 10, 10)[0, 0]) == (0, 0, 0)


@pytest.mark.skipif(not os.path.exists(VSI), reason="APGI TMA scans not available")
class TestVsiReader:
    def test_metadata(self):
        with open_slide(VSI) as r:
            assert isinstance(r, VsiReader)
            assert r.channels == ("BF", "POL")
            assert r.pixel_size_um == pytest.approx(0.2738, abs=1e-3)
            assert r.level_count == 9
            h, w = r.level_shape(0)
            assert 58000 < h < 60000 and 82000 < w < 83000
            assert r.slide_name == "APGI TMA 1 PicRed"

    def test_region_matches_levels(self):
        with open_slide(VSI) as r:
            full = r.read_region(20480, 20480, 1024, 1024, channel="BF", level=0)
            coarse = r.read_region(10240, 10240, 512, 512, channel="BF", level=1)
            assert full.shape == (1024, 1024, 3) and coarse.shape == (512, 512, 3)
            small = cv2.resize(full, (512, 512), interpolation=cv2.INTER_AREA)
            assert np.abs(small.astype(int) - coarse.astype(int)).mean() < 12

    def test_pol_is_dark(self):
        with open_slide(VSI) as r:
            lv = r.level_count - 3
            assert r.read_level(lv, "POL").mean() < 60 < r.read_level(lv, "BF").mean()
