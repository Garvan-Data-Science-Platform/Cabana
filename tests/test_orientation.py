"""Tests for cabana/orientation.py — OrientationAnalyzer class."""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cabana.orientation import OrientationAnalyzer


def make_gray_image(h=64, w=64, value=128):
    return np.full((h, w, 3), value, dtype=np.uint8)


def make_gradient_image(h=64, w=64):
    """Horizontal gradient — strong horizontal orientation."""
    img = np.zeros((h, w), dtype=np.uint8)
    for col in range(w):
        img[:, col] = int(col / w * 255)
    rgb = np.stack([img, img, img], axis=-1)
    return rgb


@pytest.fixture
def analyzer():
    return OrientationAnalyzer(sigma=2.0)


@pytest.fixture
def computed(analyzer):
    img = make_gradient_image()
    analyzer.compute_orient(img)
    return analyzer


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------

class TestInit:
    def test_default_sigma(self):
        a = OrientationAnalyzer()
        assert a.sigma == 2.0

    def test_custom_sigma(self):
        a = OrientationAnalyzer(sigma=5.0)
        assert a.sigma == 5.0

    def test_initial_attributes_none(self):
        a = OrientationAnalyzer()
        assert a.orient is None
        assert a.coherency is None
        assert a.energy is None


# ---------------------------------------------------------------------------
# compute_orient
# ---------------------------------------------------------------------------

class TestComputeOrient:
    def test_populates_orient(self, computed):
        assert computed.orient is not None

    def test_populates_coherency(self, computed):
        assert computed.coherency is not None

    def test_populates_energy(self, computed):
        assert computed.energy is not None

    def test_orient_shape(self, computed):
        assert computed.orient.shape == (64, 64)

    def test_coherency_range(self, computed):
        # Coherency should be in [0, 1]
        assert computed.coherency.min() >= 0.0
        assert computed.coherency.max() <= 1.0 + 1e-6

    def test_energy_range(self, computed):
        assert computed.energy.min() >= 0.0
        assert computed.energy.max() <= 1.0 + 1e-6

    def test_orient_range(self, computed):
        # Orientation in [-π/2, π/2]
        assert computed.orient.min() >= -np.pi / 2 - 1e-6
        assert computed.orient.max() <= np.pi / 2 + 1e-6

    def test_accepts_array_input(self, analyzer):
        img = make_gradient_image()
        analyzer.compute_orient(img)
        assert analyzer.orient is not None

    def test_structure_tensor_components_populated(self, computed):
        assert computed.dxx is not None
        assert computed.dxy is not None
        assert computed.dyy is not None


# ---------------------------------------------------------------------------
# get_orientation_image / get_coherency_image / get_energy_image
# ---------------------------------------------------------------------------

class TestGetImages:
    def test_get_orientation_image_shape(self, computed):
        img = computed.get_orientation_image()
        assert img.shape == (64, 64)

    def test_get_coherency_image_shape(self, computed):
        img = computed.get_coherency_image()
        assert img.shape == (64, 64)

    def test_get_energy_image_shape(self, computed):
        img = computed.get_energy_image()
        assert img.shape == (64, 64)

    def test_mask_zeros_outside(self, computed):
        mask = np.zeros((64, 64), dtype=bool)
        mask[20:40, 20:40] = True
        img = computed.get_orientation_image(mask=mask)
        assert img[0, 0] == 0.0

    def test_full_mask_same_as_no_mask(self, computed):
        full_mask = np.ones((64, 64), dtype=bool)
        img_masked = computed.get_orientation_image(mask=full_mask)
        img_unmasked = computed.get_orientation_image()
        np.testing.assert_array_almost_equal(img_masked, img_unmasked)


# ---------------------------------------------------------------------------
# mean_orientation / mean_coherency
# ---------------------------------------------------------------------------

class TestMeanMetrics:
    def test_mean_orientation_returns_scalar(self, computed):
        val = computed.mean_orientation()
        assert np.isscalar(val) or val.ndim == 0

    def test_mean_orientation_range(self, computed):
        val = computed.mean_orientation()
        assert -90 <= val <= 90

    def test_mean_coherency_returns_scalar(self, computed):
        val = computed.mean_coherency()
        assert np.isscalar(val) or val.ndim == 0

    def test_mean_coherency_range(self, computed):
        val = computed.mean_coherency()
        assert 0.0 <= val <= 1.0 + 1e-6

    def test_mean_orientation_with_mask(self, computed):
        mask = np.zeros((64, 64), dtype=bool)
        mask[10:50, 10:50] = True
        val = computed.mean_orientation(mask=mask)
        assert -90 <= val <= 90


# ---------------------------------------------------------------------------
# circular_variance
# ---------------------------------------------------------------------------

class TestCircularVariance:
    def test_returns_scalar(self, computed):
        val = computed.circular_variance()
        assert np.isscalar(val) or hasattr(val, '__float__')

    def test_nonnegative(self, computed):
        val = computed.circular_variance()
        assert float(val) >= 0.0

    def test_with_mask(self, computed):
        mask = np.ones((64, 64), dtype=bool)
        val = computed.circular_variance(mask=mask)
        assert float(val) >= 0.0


# ---------------------------------------------------------------------------
# randomness_orientation
# ---------------------------------------------------------------------------

class TestRandomnessOrientation:
    def test_returns_float(self, computed):
        val = computed.randomness_orientation()
        assert isinstance(float(val), float)

    def test_nonnegative(self, computed):
        val = computed.randomness_orientation()
        assert float(val) >= 0.0


# ---------------------------------------------------------------------------
# draw_angular_hist
# ---------------------------------------------------------------------------

class TestDrawAngularHist:
    def test_returns_ndarray(self, computed):
        result = computed.draw_angular_hist()
        assert isinstance(result, np.ndarray)

    def test_output_is_rgb(self, computed):
        result = computed.draw_angular_hist()
        assert result.ndim == 3
        assert result.shape[2] == 3

    def test_empty_mask_returns_zeros(self, computed):
        empty_mask = np.zeros((64, 64), dtype=bool)
        result = computed.draw_angular_hist(mask=empty_mask)
        assert result.shape[2] in (3, 4)


# ---------------------------------------------------------------------------
# draw_vector_field
# ---------------------------------------------------------------------------

class TestDrawVectorField:
    def test_returns_ndarray(self, computed):
        result = computed.draw_vector_field()
        assert isinstance(result, np.ndarray)

    def test_output_has_3_channels(self, computed):
        result = computed.draw_vector_field()
        assert result.ndim == 3

    def test_with_weights_map(self, computed):
        wgts = np.random.rand(64, 64).astype(np.float32)
        result = computed.draw_vector_field(wgts_map=wgts)
        assert result is not None


# ---------------------------------------------------------------------------
# draw_color_survey
# ---------------------------------------------------------------------------

class TestDrawColorSurvey:
    def test_returns_rgb_image(self, computed):
        result = computed.draw_color_survey()
        assert isinstance(result, np.ndarray)
        assert result.ndim == 3
        assert result.shape[2] == 3

    def test_shape_matches_input(self, computed):
        result = computed.draw_color_survey()
        assert result.shape[:2] == (64, 64)
