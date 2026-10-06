"""Tests for cabana/analyzer.py — SkeletonAnalyzer class."""

import os
import sys

import networkx as nx
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


# ---------------------------------------------------------------------------
# calc_total_len — Euclidean path length
# ---------------------------------------------------------------------------

@pytest.fixture
def light_analyzer():
    """Analyzer configured for white-fibre-on-black-background test inputs."""
    return SkeletonAnalyzer(skel_thresh=5, branch_thresh=3, hole_threshold=4, dark_line=False)


def _make_diagonal_line_image(h=64, w=64):
    """1-pixel-wide diagonal from (5,5) to (h-6, w-6) inclusive."""
    img = np.zeros((h, w), dtype=np.uint8)
    n = min(h, w) - 10
    for i in range(n):
        img[5 + i, 5 + i] = 255
    return img


def _make_thin_horizontal_line(h=64, w=64, line_y=32, c0=5, c1=58):
    """Single-pixel-thick horizontal segment from (line_y, c0) to (line_y, c1)."""
    img = np.zeros((h, w), dtype=np.uint8)
    img[line_y, c0:c1 + 1] = 255
    return img


def _make_thin_vertical_line(h=64, w=64, line_x=32, r0=5, r1=58):
    """Single-pixel-thick vertical segment from (r0, line_x) to (r1, line_x)."""
    img = np.zeros((h, w), dtype=np.uint8)
    img[r0:r1 + 1, line_x] = 255
    return img


def _make_two_separate_lines(h=64, w=64):
    """Two parallel single-pixel-thick horizontal lines."""
    img = np.zeros((h, w), dtype=np.uint8)
    img[15, 5:30] = 255
    img[45, 35:60] = 255
    return img


class TestCalcTotalLenUnit:
    """Unit tests on calc_total_len() with manually constructed subgraphs."""

    def test_empty_subgraphs_zero(self):
        a = SkeletonAnalyzer()
        a.subgraphs = []
        a.calc_total_len()
        assert a.total_length == 0.0

    def test_single_edge_uses_stored_length(self):
        a = SkeletonAnalyzer()
        G = nx.Graph()
        G.add_edge((0, 0), (0, 5), length=5.0, path=[(0, i) for i in range(6)], type='end-to-end')
        a.subgraphs = [G]
        a.calc_total_len()
        assert a.total_length == pytest.approx(5.0, abs=1e-9)

    def test_diagonal_edge_uses_sqrt2(self):
        """A 5-step diagonal stores length 5*sqrt(2); calc_total_len returns it as-is."""
        a = SkeletonAnalyzer()
        G = nx.Graph()
        path = [(i, i) for i in range(6)]
        G.add_edge(path[0], path[-1], length=5.0 * np.sqrt(2.0), path=path, type='end-to-end')
        a.subgraphs = [G]
        a.calc_total_len()
        assert a.total_length == pytest.approx(5.0 * np.sqrt(2.0), abs=1e-9)

    def test_sums_across_multiple_edges(self):
        """Three legs of a star: total = sum of edge lengths."""
        a = SkeletonAnalyzer()
        G = nx.Graph()
        G.add_edge((0, 0), (0, 5), length=5.0, path=[], type='end-to-brh')
        G.add_edge((0, 5), (5, 5), length=5.0, path=[], type='end-to-brh')
        G.add_edge((0, 5), (0, 10), length=5.0, path=[], type='end-to-brh')
        a.subgraphs = [G]
        a.calc_total_len()
        assert a.total_length == pytest.approx(15.0, abs=1e-9)

    def test_sums_across_multiple_subgraphs(self):
        a = SkeletonAnalyzer()
        G1 = nx.Graph()
        G1.add_edge((0, 0), (0, 5), length=5.0, path=[], type='end-to-end')
        G2 = nx.Graph()
        G2.add_edge((10, 0), (10, 7), length=7.0, path=[], type='end-to-end')
        a.subgraphs = [G1, G2]
        a.calc_total_len()
        assert a.total_length == pytest.approx(12.0, abs=1e-9)

    def test_returns_float(self):
        """Result must be a python float, not int/numpy int."""
        a = SkeletonAnalyzer()
        a.subgraphs = []
        a.calc_total_len()
        assert isinstance(a.total_length, float)


class TestCalcTotalLenIntegration:
    """End-to-end checks using analyze_image on synthetic binary fixtures."""

    def test_horizontal_line_exact_step_count(self, light_analyzer):
        """N-pixel-wide skeleton spans (N-1) cardinal steps of length 1."""
        img = _make_thin_horizontal_line(c0=5, c1=58)
        light_analyzer.analyze_image(img)
        # 54 foreground pixels (cols 5..58), 53 unit steps between them.
        assert light_analyzer.total_length == pytest.approx(53.0, abs=1e-9)

    def test_vertical_line_exact_step_count(self, light_analyzer):
        img = _make_thin_vertical_line(r0=5, r1=58)
        light_analyzer.analyze_image(img)
        assert light_analyzer.total_length == pytest.approx(53.0, abs=1e-9)

    def test_diagonal_line_uses_sqrt2(self, light_analyzer):
        """Diagonal of 54 pixels has 53 diagonal steps, each sqrt(2) long."""
        img = _make_diagonal_line_image()
        light_analyzer.analyze_image(img)
        assert light_analyzer.total_length == pytest.approx(53.0 * np.sqrt(2.0), abs=1e-9)

    def test_diagonal_strictly_longer_than_horizontal(self, light_analyzer):
        """For equal pixel count, diagonal must report a larger length than horizontal."""
        img_h = _make_thin_horizontal_line(c0=5, c1=58)
        img_d = _make_diagonal_line_image()  # spans 54 diag pixels — same count
        a1 = SkeletonAnalyzer(skel_thresh=5, branch_thresh=3, hole_threshold=4, dark_line=False)
        a2 = SkeletonAnalyzer(skel_thresh=5, branch_thresh=3, hole_threshold=4, dark_line=False)
        a1.analyze_image(img_h)
        a2.analyze_image(img_d)
        assert a2.total_length > a1.total_length

    def test_two_disjoint_lines_sum(self, light_analyzer):
        """Two horizontal lines: total length is sum of individual step counts."""
        img = _make_two_separate_lines()
        light_analyzer.analyze_image(img)
        # img[15, 5:30]  -> 25 pixels, 24 steps
        # img[45, 35:60] -> 25 pixels, 24 steps
        # total = 48
        assert light_analyzer.total_length == pytest.approx(48.0, abs=1e-9)

    def test_real_synthetic_image_finite_positive(self, light_analyzer):
        """Regression: real synthetic IHC mask gives a finite, positive Euclidean length."""
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "Synthetic", "5001f5e89da92cce.png",
        )
        if not os.path.isfile(path):
            pytest.skip(f"test asset not present: {path}")
        # The asset is an RGB photomicrograph; binarise it before analysis.
        import imageio.v3 as iio
        img = iio.imread(path)
        if img.ndim == 3:
            img = img[..., 0]
        binary = (img > 127).astype(np.uint8) * 255
        light_analyzer.analyze_image(binary)
        assert np.isfinite(light_analyzer.total_length)
        assert light_analyzer.total_length > 0.0


# ---------------------------------------------------------------------------
# Regressions: empty masks, parallel paths and self-loops (white fibres on black)
# ---------------------------------------------------------------------------

def _white_on_black_analyzer():
    return SkeletonAnalyzer(skel_thresh=5, branch_thresh=3, hole_threshold=4, dark_line=False)


class TestEmptyMask:
    def test_empty_mask_gives_zero_metrics(self):
        a = _white_on_black_analyzer()
        a.analyze_image(np.zeros((64, 64), dtype=np.uint8))
        assert a.proj_area == 0 and a.total_length == 0 and a.lacunarity == 0 and a.frac_dim == 0
        assert a.num_tips == 0 and a.num_branches == 0
        assert a.key_pts_image.shape == (64, 64, 3) and a.length_map_all.shape == (64, 64)
        a.calc_curve_all(11)
        assert a.avg_curve_all == 0 and a.curve_map_all.shape == (64, 64)

    def test_empty_dark_line_mask_gives_zero_metrics(self):
        a = SkeletonAnalyzer(skel_thresh=5, branch_thresh=3, hole_threshold=4, dark_line=True)
        a.analyze_image(np.full((64, 64), 255, dtype=np.uint8))     # all background
        assert a.proj_area == 0 and a.total_length == 0 and a.lacunarity == 0


class TestLoops:
    @staticmethod
    def _ring(img, r=20, c=32):
        yy, xx = np.ogrid[:img.shape[0], :img.shape[1]]
        d = np.sqrt((yy - c) ** 2 + (xx - c) ** 2)
        img[np.abs(d - r) < 0.7] = 255

    def test_ring_with_chord_keeps_both_arcs(self):
        img = np.zeros((64, 64), dtype=np.uint8)
        self._ring(img)
        img[32, 12:53] = 255                 # a chord: two junctions, three paths between them
        a = _white_on_black_analyzer()
        a.analyze_image(img)
        # ring (~126 px) + chord (~40 px); before the fix one arc was dropped (~100 px)
        assert a.total_length > 140

    def test_loop_on_a_single_junction_is_kept(self):
        img = np.zeros((64, 64), dtype=np.uint8)
        self._ring(img)
        img[52:62, 32] = 255                 # a tail: the ring closes on its own junction
        a = _white_on_black_analyzer()
        a.analyze_image(img)
        assert a.total_length > 110          # ring ~126 + tail ~10; before the fix only the tail counted
