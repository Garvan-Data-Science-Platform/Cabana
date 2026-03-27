"""Tests for cabana/utils.py — pure utility functions."""

import os
import sys
import tempfile

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cabana.utils import (
    array_divide,
    contains_oversized,
    normalize,
    join_path,
    create_folder,
    get_img_paths,
    sanitize_filename,
    mean_image,
    split2batches,
    crop_img_from_center,
    cal_color_dist,
)


# ---------------------------------------------------------------------------
# array_divide
# ---------------------------------------------------------------------------

class TestArrayDivide:
    def test_basic_division(self):
        a = np.array([4.0, 6.0])
        b = np.array([2.0, 3.0])
        result = array_divide(a, b)
        np.testing.assert_array_almost_equal(result, [2.0, 2.0])

    def test_divide_by_zero_returns_zero(self):
        a = np.array([1.0, 2.0])
        b = np.array([0.0, 2.0])
        result = array_divide(a, b)
        assert result[0] == 0.0
        assert result[1] == 1.0

    def test_all_zeros_denominator(self):
        a = np.ones((3, 3))
        b = np.zeros((3, 3))
        result = array_divide(a, b)
        np.testing.assert_array_equal(result, np.zeros((3, 3)))

    def test_output_dtype_float64(self):
        a = np.array([1, 2, 3], dtype=np.int32)
        b = np.array([1, 1, 1], dtype=np.int32)
        result = array_divide(a, b)
        assert result.dtype == np.float64

    def test_2d_array(self):
        a = np.array([[6.0, 8.0], [9.0, 12.0]])
        b = np.array([[2.0, 4.0], [3.0, 6.0]])
        result = array_divide(a, b)
        np.testing.assert_array_almost_equal(result, [[3.0, 2.0], [3.0, 2.0]])


# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_output_range_clipped_0_1(self):
        x = np.random.rand(100, 100).astype(np.float32)
        result = normalize(x)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_uniform_image_no_nan(self):
        x = np.ones((50, 50), dtype=np.float32) * 128
        result = normalize(x)
        assert not np.any(np.isnan(result))

    def test_dtype_float32(self):
        x = np.arange(100, dtype=np.float64).reshape(10, 10)
        result = normalize(x)
        assert result.dtype == np.float32

    def test_monotonic_after_normalize(self):
        x = np.arange(10, dtype=np.float32)
        result = normalize(x, pmin=0, pmax=100)
        diffs = np.diff(result)
        assert np.all(diffs >= 0)

    def test_output_shape_preserved(self):
        x = np.random.rand(20, 30).astype(np.float32)
        result = normalize(x)
        assert result.shape == x.shape


# ---------------------------------------------------------------------------
# contains_oversized
# ---------------------------------------------------------------------------

class TestContainsOversized:
    def _make_png(self, tmp_dir, w, h):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        path = os.path.join(tmp_dir, f"{w}x{h}.png")
        cv2.imwrite(path, img)
        return path

    def test_small_image_not_oversized(self, tmp_path):
        p = self._make_png(str(tmp_path), 100, 100)
        assert contains_oversized([p], max_res=2048) is False

    def test_large_image_is_oversized(self, tmp_path):
        p = self._make_png(str(tmp_path), 2049, 2049)
        assert contains_oversized([p], max_res=2048) is True

    def test_exactly_at_limit_not_oversized(self, tmp_path):
        p = self._make_png(str(tmp_path), 2048, 2048)
        # 2048*2048 == max_size is NOT strictly greater than max_size
        assert contains_oversized([p], max_res=2048) is False

    def test_mixed_list_returns_true_if_any_large(self, tmp_path):
        small = self._make_png(str(tmp_path), 100, 100)
        large = self._make_png(str(tmp_path), 3000, 3000)
        assert contains_oversized([small, large], max_res=2048) is True

    def test_empty_list_returns_false(self):
        assert contains_oversized([], max_res=2048) is False


# ---------------------------------------------------------------------------
# join_path
# ---------------------------------------------------------------------------

class TestJoinPath:
    def test_basic_join(self):
        result = join_path("a", "b", "c")
        assert result == "a/b/c"

    def test_single_segment(self):
        assert join_path("only") == "only"

    def test_returns_string(self):
        assert isinstance(join_path("x", "y"), str)

    def test_no_backslash(self):
        result = join_path("a", "b")
        assert "\\" not in result


# ---------------------------------------------------------------------------
# create_folder
# ---------------------------------------------------------------------------

class TestCreateFolder:
    def test_creates_new_directory(self, tmp_path):
        new_dir = str(tmp_path / "new_folder")
        assert not os.path.exists(new_dir)
        create_folder(new_dir)
        assert os.path.isdir(new_dir)

    def test_existing_directory_overwrite(self, tmp_path):
        existing = str(tmp_path / "existing")
        os.makedirs(existing)
        # Write a file inside, then overwrite should remove it
        with open(os.path.join(existing, "test.txt"), "w") as f:
            f.write("hello")
        create_folder(existing, overwrite=True)
        assert os.path.isdir(existing)
        # File should be gone after overwrite
        assert not os.path.exists(os.path.join(existing, "test.txt"))

    def test_nested_directories(self, tmp_path):
        nested = str(tmp_path / "a" / "b" / "c")
        create_folder(nested)
        assert os.path.isdir(nested)


# ---------------------------------------------------------------------------
# get_img_paths
# ---------------------------------------------------------------------------

class TestGetImgPaths:
    def test_returns_image_files(self, tmp_path):
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        cv2.imwrite(str(tmp_path / "a.png"), img)
        (tmp_path / "c.txt").write_bytes(b"")
        paths = get_img_paths(str(tmp_path))
        fnames = [os.path.basename(p) for p in paths]
        assert "a.png" in fnames

    def test_excludes_non_image_files(self, tmp_path):
        (tmp_path / "c.txt").write_bytes(b"")
        paths = get_img_paths(str(tmp_path))
        fnames = [os.path.basename(p) for p in paths]
        assert "c.txt" not in fnames

    def test_empty_folder_returns_empty(self, tmp_path):
        paths = get_img_paths(str(tmp_path))
        assert paths == []


# ---------------------------------------------------------------------------
# sanitize_filename
# ---------------------------------------------------------------------------

class TestSanitizeFilename:
    def test_replaces_spaces(self):
        result = sanitize_filename("file name")
        assert " " not in result

    def test_replaces_colons(self):
        result = sanitize_filename("file:name")
        assert ":" not in result

    def test_replaces_slashes(self):
        result = sanitize_filename("file/name")
        assert "/" not in result

    def test_normal_name_unchanged(self):
        name = "normal_filename.png"
        result = sanitize_filename(name)
        assert result == name

    def test_returns_string(self):
        assert isinstance(sanitize_filename("test"), str)


# ---------------------------------------------------------------------------
# mean_image
# ---------------------------------------------------------------------------

class TestMeanImage:
    def test_basic_mean(self):
        # Single label region → each pixel should equal channel mean
        img = np.ones((4, 4, 3), dtype=np.uint8) * 128
        labels = np.zeros((4, 4), dtype=np.int32)
        result = mean_image(img, labels)
        assert result.shape == img.shape
        np.testing.assert_array_equal(result, img)

    def test_two_regions(self):
        img = np.zeros((4, 4, 3), dtype=np.uint8)
        img[:2, :, :] = 100   # top half
        img[2:, :, :] = 200   # bottom half
        labels = np.zeros((4, 4), dtype=np.int32)
        labels[:2, :] = 0
        labels[2:, :] = 1
        result = mean_image(img, labels)
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_output_dtype_uint8(self):
        img = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        labels = np.zeros((10, 10), dtype=np.int32)
        result = mean_image(img, labels)
        assert result.dtype == np.uint8


# ---------------------------------------------------------------------------
# split2batches  (requires real image files for EXIF reading)
# ---------------------------------------------------------------------------

class TestSplit2Batches:
    def _make_png(self, tmp_dir, name):
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        path = os.path.join(tmp_dir, name)
        cv2.imwrite(path, img)
        return path

    def test_returns_two_lists(self, tmp_path):
        paths = [self._make_png(str(tmp_path), f"img{i}.png") for i in range(3)]
        result = split2batches(paths, max_batch_size=5)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_all_images_accounted_for(self, tmp_path):
        paths = [self._make_png(str(tmp_path), f"img{i}.png") for i in range(5)]
        path_batches, _ = split2batches(paths, max_batch_size=5)
        total = sum(len(b) for b in path_batches)
        assert total == 5

    def test_batch_size_respected(self, tmp_path):
        paths = [self._make_png(str(tmp_path), f"img{i}.png") for i in range(6)]
        path_batches, _ = split2batches(paths, max_batch_size=3)
        for batch in path_batches:
            assert len(batch) <= 3

    def test_res_batches_match_path_batches(self, tmp_path):
        paths = [self._make_png(str(tmp_path), f"img{i}.png") for i in range(4)]
        path_batches, res_batches = split2batches(paths, max_batch_size=5)
        assert len(path_batches) == len(res_batches)


# ---------------------------------------------------------------------------
# crop_img_from_center
# ---------------------------------------------------------------------------

class TestCropImgFromCenter:
    # Active definition: crop_img_from_center(img, width=512)
    # height is computed proportionally to maintain aspect ratio.

    def test_2d_crop(self):
        img = np.arange(100).reshape(10, 10).astype(np.uint8)
        result = crop_img_from_center(img, width=6)
        assert result.shape[1] == 6
        assert result.ndim == 2

    def test_3d_crop(self):
        img = np.zeros((20, 20, 3), dtype=np.uint8)
        result = crop_img_from_center(img, width=10)
        assert result.shape[1] == 10
        assert result.ndim == 3

    def test_crop_too_large_raises(self):
        img = np.zeros((10, 10), dtype=np.uint8)
        with pytest.raises(AssertionError):
            crop_img_from_center(img, width=20)


# ---------------------------------------------------------------------------
# cal_color_dist
# ---------------------------------------------------------------------------

class TestCalColorDist:
    def test_returns_two_arrays(self):
        img = np.random.randint(100, 200, (50, 50, 3), dtype=np.uint8)
        abs_dist, rel_dist = cal_color_dist(img, hue=1.0)
        assert abs_dist.shape == (50, 50)
        assert rel_dist.shape == (50, 50)

    def test_relative_dist_range(self):
        img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        _, rel_dist = cal_color_dist(img, hue=0.5)
        assert rel_dist.min() >= 0.0
        assert rel_dist.max() <= 1.0 + 1e-6

    def test_grayscale_like_image(self):
        # All-zero saturation → grayscale branch
        img = np.zeros((20, 20, 3), dtype=np.uint8)
        img[:, :, 2] = 200  # only blue value
        abs_dist, rel_dist = cal_color_dist(img)
        assert abs_dist is not None
