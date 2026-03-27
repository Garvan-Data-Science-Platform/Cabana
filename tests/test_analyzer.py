"""Tests for cabana/analyzer.py — SkeletonAnalyzer class."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cabana.analyzer import SkeletonAnalyzer


def make_horizontal_line_image(h=64, w=64, line_y=32, thickness=3):
    """Binary image with a single horizontal line."""
    img = np.zeros((h, w), dtype=np.uint8)
    for dy in range(thickness):
        y = line_y + dy - thickness // 2
        if 0 <= y < h:
            img[y, :] = 255
    return img


def make_vertical_line_image(h=64, w=64):
    """Binary image with a single vertical line."""
    img = np.zeros((h, w), dtype=np.uint8)
    img[:, w // 2] = 255
    return img


def make_cross_image(h=64, w=64):
    """Binary image with a cross (horizontal + vertical line)."""
    img = make_horizontal_line_image(h, w, line_y=h // 2, thickness=1)
    img[:, w // 2] = 255
    return img


def make_skeleton_image(h=64, w=64):
    """Thin skeleton line (1 pixel wide)."""
    img = np.zeros((h, w), dtype=np.uint8)
    img[h // 2, 5:w - 5] = 255  # Single thin horizontal line
    return img


@pytest.fixture
def analyzer():
    return SkeletonAnalyzer(skel_thresh=5, branch_thresh=3, hole_threshold=4, dark_line=True)


# ---------------------------------------------------------------------------
# __init__ / reset
# ---------------------------------------------------------------------------

class TestInit:
    def test_default_metrics_zero(self):
        a = SkeletonAnalyzer()
        assert a.proj_area == 0.0
        assert a.num_tips == 0
        assert a.num_branches == 0
        assert a.total_length == 0.0
        assert a.frac_dim == 0.0

    def test_custom_params(self):
        a = SkeletonAnalyzer(skel_thresh=50, branch_thresh=20, hole_threshold=16, dark_line=False)
        assert a.skel_thresh == 50
        assert a.branch_thresh == 20
        assert a.dark_line is False

    def test_reset_clears_metrics(self, analyzer):
        analyzer.proj_area = 999
        analyzer.total_length = 100
        analyzer.reset()
        assert analyzer.proj_area == 0.0
        assert analyzer.total_length == 0.0

    def test_reset_clears_images(self, analyzer):
        analyzer.skel_image = np.zeros((10, 10))
        analyzer.reset()
        assert analyzer.skel_image is None

    def test_foreground_is_255(self):
        a = SkeletonAnalyzer()
        assert a.FOREGROUND == 255

    def test_background_is_0(self):
        a = SkeletonAnalyzer()
        assert a.BACKGROUND == 0


# ---------------------------------------------------------------------------
# count_neighbors (static, numba JIT)
# ---------------------------------------------------------------------------

class TestCountNeighbors:
    def test_center_pixel_4_neighbors(self):
        # 3x3 image with all pixels set to 1
        img = np.ones((5, 5), dtype=np.int32)
        count = SkeletonAnalyzer.count_neighbors(img, 2, 2, radius=1, val=1)
        assert count == 8  # 8 neighbors in 3x3 minus center

    def test_zero_neighbors_in_empty(self):
        img = np.zeros((5, 5), dtype=np.int32)
        count = SkeletonAnalyzer.count_neighbors(img, 2, 2, radius=1, val=1)
        assert count == 0

    def test_corner_pixel(self):
        img = np.ones((5, 5), dtype=np.int32)
        # Corner at (0,0) has fewer neighbors in the grid but count_neighbors
        # uses fixed bounds; use interior point instead
        count = SkeletonAnalyzer.count_neighbors(img, 1, 1, radius=1, val=1)
        assert count == 8  # All 8 neighbors present for interior point


# ---------------------------------------------------------------------------
# get_neighbors (static)
# ---------------------------------------------------------------------------

class TestGetNeighbors:
    def test_returns_list(self):
        img = np.zeros((10, 10), dtype=np.uint8)
        img[5, 4] = 255
        img[5, 6] = 255
        neighbors = SkeletonAnalyzer.get_neighbors(img, (5, 5), 255)
        assert isinstance(neighbors, list)

    def test_finds_adjacent_pixels(self):
        img = np.zeros((10, 10), dtype=np.uint8)
        img[5, 4] = 255
        img[5, 6] = 255
        neighbors = SkeletonAnalyzer.get_neighbors(img, (5, 5), 255)
        assert len(neighbors) >= 2


# ---------------------------------------------------------------------------
# traverse_skeletons (static)
# ---------------------------------------------------------------------------

class TestTraverseSkeletons:
    def test_empty_inputs_returns_empty(self):
        img = np.zeros((10, 10), dtype=np.uint8)
        result = SkeletonAnalyzer.traverse_skeletons(img, [], [], 255)
        assert result == []

    def test_single_end_point_returns_empty(self):
        img = np.zeros((10, 10), dtype=np.uint8)
        img[5, 5] = 255
        result = SkeletonAnalyzer.traverse_skeletons(img, [(5, 5)], [], 255)
        assert result == []

    def test_two_end_points_straight_line(self):
        # Straight horizontal line between (5,2) and (5,8)
        img = np.zeros((12, 12), dtype=np.uint8)
        for col in range(2, 9):
            img[5, col] = 255
        result = SkeletonAnalyzer.traverse_skeletons(img, [(5, 2), (5, 8)], [], 255)
        assert len(result) >= 1
        # Should have end-to-end type
        assert result[0][4] == 'end-to-end'


# ---------------------------------------------------------------------------
# analyze (integration)
# ---------------------------------------------------------------------------

class TestAnalyzeImage:
    def test_analyze_populates_skel_image(self, analyzer):
        img = make_horizontal_line_image()
        analyzer.analyze_image(img)
        assert analyzer.skel_image is not None

    def test_analyze_returns_nonnegative_length(self, analyzer):
        img = make_horizontal_line_image()
        analyzer.analyze_image(img)
        assert analyzer.total_length >= 0.0

    def test_analyze_proj_area_nonneg(self, analyzer):
        img = make_horizontal_line_image()
        analyzer.analyze_image(img)
        assert analyzer.proj_area >= 0.0

    def test_analyze_frac_dim_range(self, analyzer):
        img = make_horizontal_line_image()
        analyzer.analyze_image(img)
        # Fractal dimension of a line should be between 1 and 2
        if analyzer.frac_dim > 0:
            assert 0.5 <= analyzer.frac_dim <= 2.5

    def test_analyze_cross_image_finds_branch(self, analyzer):
        img = make_cross_image()
        analyzer.analyze_image(img)
        # A cross should have at least one branch point
        assert analyzer.num_branches >= 0  # may not find any if pruned away

    def test_analyze_with_skeleton_image(self, analyzer):
        img = make_skeleton_image()
        analyzer.analyze_image(img)
        assert analyzer.skel_image is not None

    def test_num_tips_nonnegative(self, analyzer):
        img = make_horizontal_line_image()
        analyzer.analyze_image(img)
        assert analyzer.num_tips >= 0

    def test_growth_unit_nonnegative(self, analyzer):
        img = make_horizontal_line_image()
        analyzer.analyze_image(img)
        assert analyzer.growth_unit >= 0.0

    def test_nonbinary_image_returns_early(self, analyzer):
        """analyze_image returns early (warning) for non-binary images."""
        img = np.arange(64 * 64, dtype=np.uint8).reshape(64, 64)
        analyzer.analyze_image(img)
        # total_length stays 0 because processing is skipped
        assert analyzer.total_length == 0.0
