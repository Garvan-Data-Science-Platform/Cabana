"""Tests for cabana/detector.py — FibreDetector class."""

import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cabana.detector import FibreDetector


def make_gray_image_with_line(h=128, w=128, line_y=64, thickness=5):
    """White image with a dark horizontal band (dark line on light background)."""
    img = np.ones((h, w), dtype=np.uint8) * 220
    img[line_y - thickness // 2:line_y + thickness // 2 + 1, :] = 50
    return img


def make_white_image(h=64, w=64):
    return np.full((h, w), 255, dtype=np.uint8)


def make_random_image(h=64, w=64):
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (h, w), dtype=np.uint8)


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------

class TestInit:
    def test_default_line_widths_array(self):
        d = FibreDetector()
        assert isinstance(d.line_widths, np.ndarray)

    def test_custom_line_widths(self):
        d = FibreDetector(line_widths=[5, 9, 13])
        assert len(d.line_widths) == 3

    def test_scalar_line_width(self):
        d = FibreDetector(line_widths=7)
        assert d.line_widths.shape == (1,)

    def test_dark_line_flag(self):
        d = FibreDetector(dark_line=True)
        assert d.dark_line is True

    def test_sigmas_computed(self):
        d = FibreDetector(line_widths=[3, 5])
        assert len(d.sigmas) == 2
        assert np.all(d.sigmas > 0)

    def test_dark_line_adjusts_thresholds(self):
        # When dark_line=True, clow = 255 - high_contrast
        d = FibreDetector(low_contrast=50, high_contrast=150, dark_line=True)
        assert d.clow == 255 - 150
        assert d.chigh == 255 - 50

    def test_light_line_keeps_thresholds(self):
        d = FibreDetector(low_contrast=50, high_contrast=150, dark_line=False)
        assert d.clow == 50
        assert d.chigh == 150

    def test_min_len_stored(self):
        d = FibreDetector(min_len=10)
        assert d.min_len == 10

    def test_extend_line_flag(self):
        d = FibreDetector(extend_line=True)
        assert d.extend_line is True


# ---------------------------------------------------------------------------
# detect_lines + apply_filtering (integration via detect_lines)
# ---------------------------------------------------------------------------

class TestDetectLines:
    def test_detect_lines_no_crash(self):
        d = FibreDetector(line_widths=[5], dark_line=True, min_len=3)
        img = make_gray_image_with_line()
        d.detect_lines(img)  # should not raise

    def test_detect_lines_white_image_no_crash(self):
        d = FibreDetector(line_widths=[5], dark_line=True, min_len=3)
        img = make_white_image()
        d.detect_lines(img)

    def test_detect_lines_multiple_widths(self):
        d = FibreDetector(line_widths=[3, 5, 7], dark_line=True, min_len=3)
        img = make_gray_image_with_line()
        d.detect_lines(img)

    def test_detect_lines_with_estimate_width(self):
        d = FibreDetector(line_widths=[5], dark_line=True, estimate_width=True, min_len=3)
        img = make_gray_image_with_line(thickness=7)
        d.detect_lines(img)

    def test_detect_lines_no_estimate_width(self):
        d = FibreDetector(line_widths=[5], dark_line=True, estimate_width=False, min_len=3)
        img = make_gray_image_with_line()
        d.detect_lines(img)

    def test_sets_image_attribute(self):
        d = FibreDetector(line_widths=[5], dark_line=True)
        img = make_gray_image_with_line()
        d.detect_lines(img)
        assert d.image is not None

    def test_sets_gray_attribute(self):
        d = FibreDetector(line_widths=[5], dark_line=True)
        img = make_gray_image_with_line()
        d.detect_lines(img)
        assert d.gray is not None
        assert d.gray.ndim == 2

    def test_contours_attribute_exists(self):
        d = FibreDetector(line_widths=[5], dark_line=True, min_len=3)
        img = make_gray_image_with_line()
        d.detect_lines(img)
        assert hasattr(d, 'contours')


# ---------------------------------------------------------------------------
# get_results
# ---------------------------------------------------------------------------

class TestGetResults:
    def test_get_results_no_crash(self):
        d = FibreDetector(line_widths=[5], dark_line=True, min_len=3)
        img = make_gray_image_with_line()
        d.detect_lines(img)
        result = d.get_results()
        assert result is not None

    def test_get_results_returns_tuple(self):
        d = FibreDetector(line_widths=[5], dark_line=True, min_len=3)
        img = make_gray_image_with_line()
        d.detect_lines(img)
        result = d.get_results()
        assert isinstance(result, tuple)

    def test_get_results_on_white_image(self):
        d = FibreDetector(line_widths=[5], dark_line=True, min_len=3)
        img = make_white_image()
        d.detect_lines(img)
        result = d.get_results()
        assert result is not None

    def test_contour_image_is_ndarray(self):
        d = FibreDetector(line_widths=[5], dark_line=True, min_len=3)
        img = make_gray_image_with_line()
        d.detect_lines(img)
        result = d.get_results()
        # result is a tuple; at least the first element should be array-like
        assert isinstance(result[0], np.ndarray) or result[0] is not None
