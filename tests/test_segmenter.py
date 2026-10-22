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

from cabana.segmenter import (_feather_weights, load_roi_mask, mask_from_score, segment_image,
                              segment_single_image, segmentation_steps, tile_windows)


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


def sparse_and_dense_image(seed=0):
    """Pale, nearly empty glass on the left; a dense magenta block on the right."""
    rng = np.random.default_rng(seed)
    img = np.clip(rng.normal(235, 3, (128, 256, 3)), 0, 255).astype(np.uint8)
    img[16:112, 144:240] = (150, 60, 220)
    return img


class TestAbsoluteThreshold:
    """The colour threshold is absolute: it is not rescaled per image or patch."""

    def test_pale_patch_next_to_dense_tissue_stays_empty(self):
        img = sparse_and_dense_image()
        mask = segment_image(img, seg_args(patch_size=64), cnn_size=64)
        assert mask[16:112, 144:240].mean() > 0.9 * 255
        # with per-patch rescaling the noise of the glass-only patches crossed the threshold
        assert (mask[:, :128] == 0).all()

    def test_patch_size_hardly_matters(self):
        img = sparse_and_dense_image()
        a = segment_image(img, seg_args(patch_size=64), cnn_size=64) > 0
        b = segment_image(img, seg_args(patch_size=128), cnn_size=64) > 0
        for m in (a, b):
            assert not m[:, :128].any() and m[24:104, 152:232].all()
        assert (a == b).mean() > 0.95         # differences only along the block's edges

    def test_image_within_one_patch_equals_patch_off(self):
        img = pink_square_image()
        assert np.array_equal(segment_image(img, seg_args(patch_size=128)), segment_image(img, seg_args()))

    def test_score_map_rethresholds_to_the_same_mask(self):
        for patch in (0, 64):
            img = sparse_and_dense_image()
            args = seg_args(patch_size=patch)
            mask, (score, min_size) = segment_image(img, args, cnn_size=64, return_score=True)
            assert np.array_equal(mask_from_score(score, min_size, args.rt, img.shape), mask)
            looser = mask_from_score(score, min_size, 0.05, img.shape)
            assert (looser >= mask).all()               # a lower threshold only adds

    def test_progress_counts_training_and_patches(self):
        img = sparse_and_dense_image()
        args = seg_args(patch_size=64, max_iter=3, min_labels=0)
        calls = []
        segment_image(img, args, iter_callback=lambda i, t: calls.append((i, t)), cnn_size=64)
        total = segmentation_steps(*img.shape[:2], args)
        assert total == 3 + len(tile_windows(128, 64, 8)) * len(tile_windows(256, 64, 8))
        assert [i for i, _ in calls] == list(range(total)) and {t for _, t in calls} == {total}


class TestGreyscale:
    """Greyscale images (SHG) keep the earlier per-image relative threshold."""

    def _shg(self):
        img = np.full((128, 256, 3), 20, np.uint8)
        img[32:96, 40:104] = 80                 # dim fibre-bright block: absolute score of 80/255 is ~0.4
        img[32:96, 168:232] = 80
        return img

    def test_dim_structure_found_whole_and_in_patches(self):
        img = self._shg()
        for patch in (0, 64):
            m = segment_image(img, seg_args(patch_size=patch), cnn_size=64) > 0
            assert m[40:88, 48:96].all() and m[40:88, 176:224].all()
            # the dark background stays (mostly) empty; this toy CNN setting forms coarse regions
            assert m[:20].mean() < 0.1 and m[:, 120:150].mean() < 0.1

    def test_neutral_glass_in_a_colour_image_is_not_greyscale(self):
        img = np.full((128, 256, 3), 235, np.uint8)          # perfectly neutral glass, no noise
        img[16:112, 144:240] = (150, 60, 220)
        m = segment_image(img, seg_args(patch_size=64), cnn_size=64)
        assert (m[:, :128] == 0).all() and m[24:104, 152:232].all()


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
