"""Tests for cabana/segmenter.py: tiling helpers, ROI-mask handling and the
patch-wise segmentation path. The CNN is run with very few iterations on tiny
images so the suite stays fast."""

import os
import sys
import types

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cabana.segmenter import (_feather_weights, load_roi_mask, segment_image,
                              segment_single_image, tile_windows)


def seg_args(**overrides):
    a = types.SimpleNamespace(num_channels=8, max_iter=2, min_labels=2, hue_value=0.96,
                              lr=0.1, sz_filter=5, rt=0.2, min_size=4, max_size=2048,
                              white_background=True, patch_size=0, patch_overlap=0.125,
                              roi_mask_path=None, input=None, roi_dir=None, bin_dir=None)
    for k, v in overrides.items():
        setattr(a, k, v)
    return a


def pink_square_image(h=96, w=96, box=(24, 72)):
    img = np.full((h, w, 3), 235, dtype=np.uint8)
    img[box[0]:box[1], box[0]:box[1]] = (150, 60, 220)   # BGR magenta-ish, hue ~0.96
    return img


class TestTileWindows:
    def test_fits_in_one_patch(self):
        assert tile_windows(500, 512, 64) == [0]
        assert tile_windows(512, 512, 64) == [0]

    def test_last_window_is_edge_anchored(self):
        starts = tile_windows(4000, 1024, 128)
        assert starts[0] == 0 and starts[-1] == 4000 - 1024
        assert all(s + 1024 <= 4000 for s in starts)
        assert starts == sorted(starts)

    def test_every_pixel_covered(self):
        n, p, o = 3000, 700, 90
        covered = np.zeros(n, bool)
        for s in tile_windows(n, p, o):
            covered[s:s + p] = True
        assert covered.all()

    def test_feather_weights(self):
        w = _feather_weights(10, 10, 3)
        assert w.shape == (10, 10) and w.max() == 1.0
        assert w[0, 5] < w[1, 5] < w[3, 5] == 1.0
        assert (_feather_weights(5, 5, 0) == 1).all()


class TestRoiMask:
    def test_missing_returns_none(self, tmp_path):
        assert load_roi_mask(None, (10, 10)) is None
        assert load_roi_mask(str(tmp_path / "nope.png"), (10, 10)) is None

    def test_resizes_and_binarises(self, tmp_path):
        m = np.zeros((20, 20), np.uint8)
        m[5:15, 5:15] = 200
        p = str(tmp_path / "m.png")
        cv2.imwrite(p, m)
        out = load_roi_mask(p, (40, 40))
        assert out.shape == (40, 40) and set(np.unique(out)) <= {0, 255}
        assert out[20, 20] == 255 and out[2, 2] == 0


class TestSegmentImage:
    def test_whole_image_path_finds_pink(self):
        img = pink_square_image()
        mask = segment_image(img, seg_args())
        assert mask.shape == img.shape[:2] and mask.dtype == np.uint8
        assert mask[48, 48] == 255
        assert mask[4, 4] == 0

    def test_patch_path_matches_shape_and_content(self):
        img = pink_square_image(h=120, w=150, box=(30, 90))
        mask = segment_image(img, seg_args(patch_size=64, patch_overlap=0.25), cnn_size=64)
        assert mask.shape == (120, 150)
        assert mask[60, 60] == 255 and mask[5, 5] == 0 and mask[110, 140] == 0

    def test_portrait_image_no_rotation_needed(self):
        img = pink_square_image(h=140, w=80, box=(20, 60))
        mask = segment_image(img, seg_args())
        assert mask.shape == (140, 80) and mask[40, 40] == 255


class TestSegmentSingleImage:
    def _setup(self, tmp_path, roi=None):
        img = pink_square_image()
        ip = str(tmp_path / "core.png")
        cv2.imwrite(ip, img)
        (tmp_path / "ROIs").mkdir()
        (tmp_path / "Bins").mkdir()
        a = seg_args(input=ip, roi_dir=str(tmp_path / "ROIs"), bin_dir=str(tmp_path / "Bins"))
        if roi is not None:
            mp = str(tmp_path / "roi.png")
            cv2.imwrite(mp, roi)
            a.roi_mask_path = mp
        return a

    def test_writes_roi_and_mask(self, tmp_path):
        a = self._setup(tmp_path)
        area, frac = segment_single_image(a)
        assert area > 0 and 0 < frac < 1
        assert (tmp_path / "ROIs" / "core_roi.png").exists()
        m = cv2.imread(str(tmp_path / "Bins" / "core_mask.png"), 0)
        assert m[48, 48] == 255

    def test_external_mask_is_intersected(self, tmp_path):
        roi = np.zeros((96, 96), np.uint8)
        roi[:, :48] = 255                       # keep only the left half
        a = self._setup(tmp_path, roi)
        segment_single_image(a)
        m = cv2.imread(str(tmp_path / "Bins" / "core_mask.png"), 0)
        assert m[48, 30] == 255 and m[48, 60] == 0
        r = cv2.imread(str(tmp_path / "ROIs" / "core_roi.png"))
        assert tuple(r[48, 60]) == (228, 228, 228)      # background fill outside the mask
