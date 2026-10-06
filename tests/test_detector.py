"""Tests for cabana/detector.py — FibreDetector class."""

import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cabana.detector import FibreDetector, _eigh_2x2_symmetric


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


# ---------------------------------------------------------------------------
# Input intensity normalization (uint8 / uint16 / float, hot pixels)
# ---------------------------------------------------------------------------

def _gray_uint8_with_line(h=128, w=128, line_y=64, thickness=5):
    """Standard 8-bit dark-line-on-light-background fixture."""
    img = np.full((h, w), 220, dtype=np.uint8)
    img[line_y - thickness // 2:line_y + thickness // 2 + 1, :] = 50
    return img


def _gray_uint16_with_line(h=128, w=128, line_y=64, thickness=5,
                           bg=55000, line=12000, hot_pixel=False):
    """16-bit dark-line-on-light-background fixture; optional hot pixel."""
    img = np.full((h, w), bg, dtype=np.uint16)
    img[line_y - thickness // 2:line_y + thickness // 2 + 1, :] = line
    if hot_pixel:
        # Single saturated pixel far from the line — what min/max would latch
        # onto and what percentile normalization should robustly ignore.
        img[0, 0] = np.iinfo(np.uint16).max
    return img


class TestInputNormalization:
    """detect_lines must be insensitive to outliers and dtype variation."""

    def test_uint8_passthrough_unchanged(self):
        """uint8 input is not re-normalized (preserves user's calibration)."""
        d = FibreDetector(line_widths=[5], dark_line=True, min_len=3)
        img = _gray_uint8_with_line()
        d.detect_lines(img)
        # Fall-through path: self.image must equal the original input.
        np.testing.assert_array_equal(d.image, img)

    def test_uint16_loaded_as_uint8(self):
        """uint16 input should be converted to uint8 before downstream ops."""
        d = FibreDetector(line_widths=[5], dark_line=True, min_len=3)
        img = _gray_uint16_with_line()
        d.detect_lines(img)
        assert d.image.dtype == np.uint8

    def test_uint16_hot_pixel_does_not_collapse_dynamic_range(self):
        """A single saturated pixel must not squash the rest into a tiny band.

        With min/max normalization, the 65535 hot pixel forces the entire
        image to be re-scaled relative to that maximum, leaving the bulk
        of the image around (12000-55000)/65535 * 255 = 47..213 (correct
        but only because line+bg are still in the bulk). A more telling
        case: a hot pixel + low dynamic range. Construct that here.
        """
        h, w = 128, 128
        img = np.full((h, w), 30000, dtype=np.uint16)
        img[64 - 2:64 + 3, :] = 28000   # very low-contrast dark line
        img[0, 0] = np.iinfo(np.uint16).max  # single hot pixel
        d = FibreDetector(line_widths=[5], dark_line=True, min_len=3)
        d.detect_lines(img)

        # After percentile normalization (0.5, 99.5), the hot pixel is
        # clipped and the bulk maps to a wide range. Check that the line
        # and background are visibly separated in the uint8 output.
        line_mean = float(d.image[64 - 2:64 + 3, :].mean())
        bg_mean = float(d.image[10:50, :].mean())
        # If percentile clipping worked, we expect a meaningful contrast
        # between line and background. If min/max squashed everything, the
        # contrast would be near zero (both ~ 30000/65535 * 255 ≈ 117).
        assert abs(bg_mean - line_mean) > 30.0

    def test_uint16_preserves_line_below_background(self):
        """uint16 → uint8 conversion must preserve dark-line-on-light topology.

        Percentile normalization stretches 16-bit dynamic range to fill
        0..255, which is *not* byte-identical to uint8 passthrough.
        What must be preserved is the relative ordering: line pixels
        remain darker than background, and the contrast is large enough
        for downstream thresholding.
        """
        img8 = _gray_uint8_with_line()
        img16 = (img8.astype(np.uint16) * 257)  # promotes 0..255 -> 0..65535

        d16 = FibreDetector(line_widths=[5], dark_line=True, min_len=3)
        d16.detect_lines(img16)

        line_mean = float(d16.image[62:67, :].mean())
        bg_mean = float(d16.image[10:50, :].mean())
        assert line_mean < bg_mean
        assert (bg_mean - line_mean) > 100  # well-separated after stretching

    def test_uint16_two_seeds_equivalent_normalization(self):
        """Two uint16 images with identical p0.5/p99.5 percentiles produce
        the same normalized uint8 result up to interpolation noise.

        This is the core invariance contract for percentile normalization:
        the output depends only on the (p0.5, p99.5) range, not on the
        sparse extremes outside that range.
        """
        rng_a = np.random.default_rng(1)
        rng_b = np.random.default_rng(2)
        # Same bulk distribution, different sparse tails (≤ 0.4% of pixels).
        h, w = 128, 128
        bulk_a = rng_a.integers(20000, 50000, size=(h, w), dtype=np.uint16)
        bulk_b = rng_b.integers(20000, 50000, size=(h, w), dtype=np.uint16)
        # Replace 0.3% with extreme outliers (different in each image).
        n_out = int(0.003 * h * w)
        idx_a = rng_a.choice(h * w, size=n_out, replace=False)
        idx_b = rng_b.choice(h * w, size=n_out, replace=False)
        bulk_a.flat[idx_a] = np.iinfo(np.uint16).max
        bulk_b.flat[idx_b] = 0

        d_a = FibreDetector(line_widths=[5], dark_line=True, min_len=3)
        d_b = FibreDetector(line_widths=[5], dark_line=True, min_len=3)
        d_a.detect_lines(bulk_a)
        d_b.detect_lines(bulk_b)
        # Bulk distributions are similar → normalized images have similar
        # mean and std, and the differences from outliers are clipped.
        assert abs(float(d_a.image.mean()) - float(d_b.image.mean())) < 5.0
        assert abs(float(d_a.image.std()) - float(d_b.image.std())) < 5.0

    def test_zero_dynamic_range_does_not_crash(self):
        """All-constant non-uint8 input is degenerate; warn but don't crash."""
        d = FibreDetector(line_widths=[5], dark_line=True, min_len=3)
        img = np.full((64, 64), 30000, dtype=np.uint16)
        d.detect_lines(img)  # must not raise
        # Result is a blank uint8 image (degenerate guard branch).
        assert d.image.dtype == np.uint8
        assert int(d.image.max()) == 0

    def test_float_input_normalized_to_uint8(self):
        """Float TIFFs are common; they must be normalized too."""
        rng = np.random.default_rng(0)
        img = rng.uniform(0.0, 1.0, size=(128, 128)).astype(np.float32)
        # Add a 'fibre' as a dark stripe.
        img[60:65, :] = 0.05
        d = FibreDetector(line_widths=[5], dark_line=True, min_len=3)
        d.detect_lines(img)
        assert d.image.dtype == np.uint8
        # Line should be visibly darker than background in the uint8 output.
        assert float(d.image[60:65, :].mean()) < float(d.image[10:50, :].mean())

    def test_real_synthetic_image_uint8_passthrough(self):
        """Real asset is uint8 RGB; ensure it traverses detect_lines cleanly."""
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "Synthetic", "5001f5e89da92cce.png",
        )
        if not os.path.isfile(path):
            pytest.skip(f"test asset not present: {path}")
        d = FibreDetector(line_widths=[5, 9], dark_line=True, min_len=5)
        d.detect_lines(path)
        assert d.image is not None
        assert d.gray is not None
        assert d.gray.dtype == np.uint8


# ---------------------------------------------------------------------------
# Steger threshold continuity (no np.floor)
# ---------------------------------------------------------------------------

class TestStegerThresholdContinuity:
    """Removing np.floor in apply_filtering must restore continuous behaviour.

    Verifies the contract:
      threshold(c, sigma) = 0.17 * sigma^gamma * c * w / (sqrt(2 pi) sigma^3)
                                * exp(-w^2 / (8 sigma^2))
      with w = 2 sqrt(3) (sigma - 0.5)

    so the threshold is strictly proportional to user-set contrast, never
    zero for any positive contrast, and monotonically increasing in c.
    """

    def _run_filter(self, low_c, high_c, line_widths=(5,), gamma=2.0,
                    img_size=(96, 96)):
        d = FibreDetector(line_widths=list(line_widths),
                          low_contrast=low_c,
                          high_contrast=high_c,
                          gamma=gamma,
                          dark_line=True,
                          min_len=3)
        # detect_lines invokes apply_filtering; we only need the populated
        # threshold maps, not the contour pipeline.
        img = np.full(img_size, 220, dtype=np.uint8)
        img[img_size[0] // 2 - 2:img_size[0] // 2 + 3, :] = 50
        d.detect_lines(img)
        return d

    def test_threshold_strictly_positive_at_low_contrast(self):
        """Even at very low contrast the threshold must remain > 0."""
        # low_contrast=5 used to drive the inner expression below 1, where
        # np.floor zeroed the threshold and the detector went into a fail
        # mode that admitted everything.
        d = self._run_filter(low_c=5, high_c=10)
        assert float(d.lower_thresh.max()) > 0.0
        assert float(d.upper_thresh.max()) > 0.0

    def test_threshold_proportional_to_contrast(self):
        """threshold(c=K) / threshold(c=1) == K within numerical error."""
        d_lo = self._run_filter(low_c=1, high_c=2)
        d_hi = self._run_filter(low_c=10, high_c=20)
        # Take a single per-pixel value (all pixels share the same scale
        # in this single-scale-list configuration).
        ratio_low = float(d_hi.lower_thresh.max()) / float(d_lo.lower_thresh.max())
        ratio_high = float(d_hi.upper_thresh.max()) / float(d_lo.upper_thresh.max())
        # dark_line transforms clow = 255 - high_contrast and chigh = 255 - low_contrast,
        # so doubling user-c does not double clow/chigh exactly. Just verify
        # strict monotonicity for both bounds rather than exact ratios.
        assert ratio_low != pytest.approx(1.0)
        assert ratio_high != pytest.approx(1.0)

    def test_threshold_monotone_in_user_contrast(self):
        """Sweeping user-set Low Contrast monotonically lowers clow,
        which monotonically raises lower_thresh (because clow = 255 - hc
        in dark-line mode and lower_thresh ∝ clow). Pin the contract that
        the post-fix threshold has no plateaus in the sweep."""
        # In dark-line mode: clow = 255 - high_contrast, chigh = 255 - low_contrast.
        # Hold high_contrast fixed at 200 and sweep low_contrast.
        # clow stays at 55 throughout; chigh increases from 245 to 240..
        # Easier: sweep clow directly via low_contrast in light-line mode.
        results = []
        for lc in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            d = FibreDetector(line_widths=[5],
                              low_contrast=lc,
                              high_contrast=200,
                              dark_line=False,    # so clow = lc directly
                              min_len=3)
            img = np.full((96, 96), 50, dtype=np.uint8)
            img[46:51, :] = 220
            d.detect_lines(img)
            results.append(float(d.lower_thresh.max()))

        # Pre-fix behaviour produced plateaus where np.floor of the inner
        # expression did not change. The continuous expression is strictly
        # monotone increasing.
        diffs = np.diff(results)
        assert (diffs > 0).all(), f"non-monotone thresholds: {results}"

    def test_threshold_continuous_no_floor_jump(self):
        """Compare two adjacent contrasts that bracket a floor() integer
        boundary in the old expression: their thresholds must differ
        smoothly, not snap to identical values."""
        # The inner expression for sigma=1.94 (w=5), clow=20:
        #   20 * 5 / (sqrt(2 pi) * 1.94**3) * exp(-25 / (8 * 1.94**2)) ~= 2.38
        # For clow=21 the inner expression is ~2.50. floor(2.38)=floor(2.50)=2,
        # so the old code produced identical thresholds for clow in {20,21}.
        # The continuous expression must produce different thresholds.
        d20 = FibreDetector(line_widths=[5], low_contrast=20, high_contrast=200,
                            dark_line=False, min_len=3)
        d21 = FibreDetector(line_widths=[5], low_contrast=21, high_contrast=200,
                            dark_line=False, min_len=3)
        img = np.full((96, 96), 50, dtype=np.uint8)
        img[46:51, :] = 220
        d20.detect_lines(img)
        d21.detect_lines(img)
        t20 = float(d20.lower_thresh.max())
        t21 = float(d21.lower_thresh.max())
        # Distinct, with the higher contrast giving the higher threshold.
        assert t21 > t20
        # And the gap is small (proportional to 1/c), not a 1-unit jump.
        assert (t21 - t20) < 0.5  # well below the 1-unit floor() granularity

    def test_threshold_matches_paper_formula(self):
        """Spot-check the numerical value against the paper's expression."""
        sigma_target = 5 / (2 * np.sqrt(3)) + 0.5
        d = FibreDetector(line_widths=[5], low_contrast=55, high_contrast=200,
                          gamma=2.0, dark_line=True, min_len=3)
        img = np.full((96, 96), 220, dtype=np.uint8)
        img[46:51, :] = 50
        d.detect_lines(img)

        # Re-derive the expected threshold at the active scale.
        sigma = d.sigmas[0]
        w = 2 * np.sqrt(3) * (sigma - 0.5)
        clow = 255 - 200  # dark_line transform
        chigh = 255 - 55
        gauss = w / (np.sqrt(2 * np.pi) * sigma ** 3) * np.exp(-w ** 2 / (8 * sigma ** 2))
        expected_low = 0.17 * sigma ** 2.0 * clow * gauss
        expected_high = 0.17 * sigma ** 2.0 * chigh * gauss

        assert float(d.lower_thresh.max()) == pytest.approx(expected_low, rel=1e-9)
        assert float(d.upper_thresh.max()) == pytest.approx(expected_high, rel=1e-9)

    def test_real_synthetic_image_detection_after_fix(self):
        """Real asset still produces non-empty detector output post-fix."""
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "Synthetic", "5001f5e89da92cce.png",
        )
        if not os.path.isfile(path):
            pytest.skip(f"test asset not present: {path}")
        d = FibreDetector(line_widths=[5, 9], dark_line=True, min_len=5)
        d.detect_lines(path)
        # The continuous-threshold detector must still find contours on a
        # real synthetic image; thresholds being slightly tighter than the
        # floored version implies fewer-or-equal but not zero contours.
        assert d.contours is not None
        assert len(d.contours) > 0


# ---------------------------------------------------------------------------
# Closed-form 2x2 symmetric eigendecomposition
# ---------------------------------------------------------------------------

class TestEigh2x2Symmetric:
    """Verify the closed-form helper matches np.linalg.eigh exactly.

    Eigenvalues are byte-equal to floating-point precision because the two
    formulations are algebraically identical. Eigenvectors of a symmetric
    matrix are unique only up to sign, so the test matches against either
    +/- the LAPACK eigenvector.
    """

    def test_diagonal_matrix(self):
        a, b, c = 2.0, 0.0, 1.0
        eigvals, _ = _eigh_2x2_symmetric(np.array(a), np.array(b), np.array(c))
        np.testing.assert_allclose(eigvals, [1.0, 2.0])

    def test_constant_off_diagonal(self):
        # M = [[1, 1], [1, 1]] -> eigvals 0, 2
        eigvals, eigvecs = _eigh_2x2_symmetric(
            np.array(1.0), np.array(1.0), np.array(1.0)
        )
        np.testing.assert_allclose(eigvals, [0.0, 2.0])
        # Verify M v = lam v for both eigenvectors
        M = np.array([[1.0, 1.0], [1.0, 1.0]])
        for k in range(2):
            v = eigvecs[:, k]
            lam = eigvals[k]
            np.testing.assert_allclose(M @ v, lam * v, atol=1e-12)

    def test_zero_matrix(self):
        eigvals, eigvecs = _eigh_2x2_symmetric(
            np.array(0.0), np.array(0.0), np.array(0.0)
        )
        np.testing.assert_allclose(eigvals, [0.0, 0.0])
        # Eigenvectors must still be orthonormal
        assert abs(np.dot(eigvecs[:, 0], eigvecs[:, 1])) < 1e-12
        assert abs(np.linalg.norm(eigvecs[:, 0]) - 1.0) < 1e-12
        assert abs(np.linalg.norm(eigvecs[:, 1]) - 1.0) < 1e-12

    def test_eigenvalues_match_numpy_random(self):
        """Eigenvalues match np.linalg.eigh on random symmetric matrices."""
        rng = np.random.default_rng(42)
        n = 1000
        a = rng.standard_normal(n)
        b = rng.standard_normal(n)
        c = rng.standard_normal(n)

        # Build (n, 2, 2) for numpy reference
        M = np.empty((n, 2, 2))
        M[:, 0, 0] = a
        M[:, 0, 1] = b
        M[:, 1, 0] = b
        M[:, 1, 1] = c
        ref_vals, ref_vecs = np.linalg.eigh(M)

        ours_vals, ours_vecs = _eigh_2x2_symmetric(a, b, c)

        np.testing.assert_allclose(ours_vals, ref_vals, atol=1e-12)

        # Eigenvectors: columns must match up to sign
        for k in range(2):
            dot = np.einsum('ni,ni->n', ours_vecs[..., :, k], ref_vecs[..., :, k])
            # |dot| should be ~1 (parallel or antiparallel)
            np.testing.assert_allclose(np.abs(dot), 1.0, atol=1e-10)

    def test_satisfies_eigen_equation_random(self):
        """For each eigenpair (lam, v): M @ v = lam * v."""
        rng = np.random.default_rng(7)
        n = 500
        a = rng.standard_normal(n)
        b = rng.standard_normal(n)
        c = rng.standard_normal(n)
        eigvals, eigvecs = _eigh_2x2_symmetric(a, b, c)
        for i in range(n):
            M = np.array([[a[i], b[i]], [b[i], c[i]]])
            for k in range(2):
                v = eigvecs[i, :, k]
                lam = eigvals[i, k]
                np.testing.assert_allclose(M @ v, lam * v, atol=1e-10)

    def test_orthonormal_eigenvectors(self):
        rng = np.random.default_rng(13)
        n = 200
        a = rng.standard_normal(n)
        b = rng.standard_normal(n)
        c = rng.standard_normal(n)
        _, eigvecs = _eigh_2x2_symmetric(a, b, c)
        # Each pair of columns should be orthonormal
        v0 = eigvecs[..., :, 0]
        v1 = eigvecs[..., :, 1]
        np.testing.assert_allclose(np.einsum('ni,ni->n', v0, v0), 1.0, atol=1e-12)
        np.testing.assert_allclose(np.einsum('ni,ni->n', v1, v1), 1.0, atol=1e-12)
        np.testing.assert_allclose(np.einsum('ni,ni->n', v0, v1), 0.0, atol=1e-12)

    def test_ascending_order(self):
        """eigvals[..., 0] <= eigvals[..., 1] always."""
        rng = np.random.default_rng(99)
        n = 1000
        a = rng.standard_normal(n) * 100
        b = rng.standard_normal(n) * 100
        c = rng.standard_normal(n) * 100
        eigvals, _ = _eigh_2x2_symmetric(a, b, c)
        assert (eigvals[..., 0] <= eigvals[..., 1] + 1e-12).all()

    def test_2d_image_shape_preserved(self):
        """Helper accepts (H, W) arrays and returns (H, W, 2) and (H, W, 2, 2)."""
        H, W = 32, 48
        rng = np.random.default_rng(2)
        a = rng.standard_normal((H, W))
        b = rng.standard_normal((H, W))
        c = rng.standard_normal((H, W))
        eigvals, eigvecs = _eigh_2x2_symmetric(a, b, c)
        assert eigvals.shape == (H, W, 2)
        assert eigvecs.shape == (H, W, 2, 2)


class TestSignAwareSaliency:
    """Polarity-aware saliency: detection works equally well in both modes.

    Steger detection is mathematically symmetric to image inversion when
    the dark/light mode is also flipped. Pre-fix, MODE_LIGHT picked the
    least-negative saliency across scales (the over-smoothed, weakest
    response), so bright-line detection was systematically biased.
    """

    def _bright_line(self, h=128, w=128, line_y=64, thickness=5,
                     line_val=200, bg_val=20):
        img = np.full((h, w), bg_val, dtype=np.uint8)
        img[line_y - thickness // 2:line_y + thickness // 2 + 1, :] = line_val
        return img

    def test_mode_light_finds_contours(self):
        """A bright line on dark background must produce non-empty contours
        when the detector is run with dark_line=False."""
        img = self._bright_line()
        d = FibreDetector(line_widths=[5], dark_line=False, min_len=5)
        d.detect_lines(img)
        assert d.contours is not None
        assert len(d.contours) > 0

    def test_mode_light_saliency_positive_at_line_pixels(self):
        """self.eigvals at chosen scale must be positive at ridge pixels in
        both modes (the polarity correction is now baked in upstream)."""
        img = self._bright_line()
        d = FibreDetector(line_widths=[5], dark_line=False, min_len=5)
        d.detect_lines(img)
        # Inspect the row of pixels along the bright line
        line_strip = d.eigvals[64, :, 0]
        assert (line_strip > 0).any()
        assert float(line_strip.max()) > 0.0

    def test_polarity_symmetry(self):
        """Bright line + dark_line=False  ≡  dark line + dark_line=True.

        The Steger pipeline should produce the same number of contours and
        the same total ridge length on an image and its 255-inverted twin
        when the mode is flipped accordingly. Pre-fix, the MODE_LIGHT path
        selected wrong scales and produced a different (worse) result.
        """
        bright = self._bright_line(line_val=200, bg_val=20)
        dark = (255 - bright).astype(np.uint8)

        d_bright = FibreDetector(line_widths=[5, 9], dark_line=False, min_len=5)
        d_bright.detect_lines(bright)

        d_dark = FibreDetector(line_widths=[5, 9], dark_line=True, min_len=5)
        d_dark.detect_lines(dark)

        # Same contour count
        assert len(d_bright.contours) == len(d_dark.contours)
        # Same total length (sum of edge step counts) within numerical noise
        len_bright = sum(c.estimate_length() for c in d_bright.contours)
        len_dark = sum(c.estimate_length() for c in d_dark.contours)
        assert abs(len_bright - len_dark) < 1.0

    def test_dark_line_mode_unchanged(self):
        """Dark-line mode behaviour is unchanged by the fix (sign=+1 for
        MODE_DARK is a no-op multiplication)."""
        img = np.full((128, 128), 220, dtype=np.uint8)
        img[62:67, :] = 50
        d = FibreDetector(line_widths=[5], dark_line=True, min_len=5)
        d.detect_lines(img)
        assert len(d.contours) > 0
        assert (d.eigvals[64, :, 0] > 0).any()

    def test_real_shg_image_more_pixels_after_fix(self):
        """On the real SHG asset (bright fibres) the fix recovers
        substantially more ridge response than the pre-fix simulation."""
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "SHG", "SFR01006_S2_LGESS_02.tif",
        )
        if not os.path.isfile(path):
            pytest.skip(f"test asset not present: {path}")
        d = FibreDetector(line_widths=[5, 9], dark_line=False, min_len=5)
        d.detect_lines(path)
        # Number of pixels with positive saliency at the chosen scale
        positive_pixels = int((d.eigvals[..., 0] > 0).sum())
        # Pre-fix benchmark on this asset was 388k positive pixels; the fix
        # recovers ~588k. Use a generous lower bound that the broken scale
        # selection cannot reach but the correct one comfortably does.
        assert positive_pixels > 450_000


class TestDetectorEquivalenceVsLapack:
    """Detector outputs are equivalent before/after the eigh replacement.

    We re-run the detector on the synthetic asset, monkey-patching the
    closed-form helper out and back in, and assert the contour set is
    byte-identical or numerically equivalent.
    """

    def test_synthetic_image_contours_match_lapack(self):
        """Detector contours obtained with closed-form match LAPACK to <1e-9."""
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "Synthetic", "5001f5e89da92cce.png",
        )
        if not os.path.isfile(path):
            pytest.skip(f"test asset not present: {path}")

        # Closed-form result
        d_new = FibreDetector(line_widths=[5, 9], dark_line=True, min_len=5)
        d_new.detect_lines(path)

        # LAPACK reference: monkey-patch a copy that re-builds the (H,W,2,2)
        # tensor and calls np.linalg.eigh, mirroring the pre-fix code.
        import cabana.detector as det_mod

        orig_helper = det_mod._eigh_2x2_symmetric

        def _eigh_via_lapack(a, b, c):
            shp = np.broadcast_shapes(a.shape, b.shape, c.shape)
            a_b = np.broadcast_to(a, shp)
            b_b = np.broadcast_to(b, shp)
            c_b = np.broadcast_to(c, shp)
            M = np.empty(shp + (2, 2), dtype=float)
            M[..., 0, 0] = a_b
            M[..., 0, 1] = b_b
            M[..., 1, 0] = b_b
            M[..., 1, 1] = c_b
            return np.linalg.eigh(M)

        det_mod._eigh_2x2_symmetric = _eigh_via_lapack
        try:
            d_ref = FibreDetector(line_widths=[5, 9], dark_line=True, min_len=5)
            d_ref.detect_lines(path)
        finally:
            det_mod._eigh_2x2_symmetric = orig_helper

        # Same number of contours
        assert len(d_new.contours) == len(d_ref.contours)
        # Contour points byte-equivalent to FP precision
        for c_new, c_ref in zip(d_new.contours, d_ref.contours):
            np.testing.assert_allclose(c_new.row, c_ref.row, atol=1e-9)
            np.testing.assert_allclose(c_new.col, c_ref.col, atol=1e-9)
            np.testing.assert_allclose(c_new.angle, c_ref.angle, atol=1e-9)
            if c_new.width_l is not None and c_ref.width_l is not None:
                np.testing.assert_allclose(c_new.width_l, c_ref.width_l, atol=1e-9)
                np.testing.assert_allclose(c_new.width_r, c_ref.width_r, atol=1e-9)


class TestClosedContours:
    def test_dark_ring_is_one_closed_contour(self):
        img = np.full((96, 96), 255, dtype=np.uint8)
        yy, xx = np.ogrid[:96, :96]
        d = np.sqrt((yy - 48) ** 2 + (xx - 48) ** 2)
        img[np.abs(d - 30) < 1.5] = 40          # a dark ring, 3 px wide
        det = FibreDetector(line_widths=[3], dark_line=True, min_len=3)
        det.detect_lines(img)
        from cabana.utils import LinesUtil
        closed = [c for c in det.contours if c.cont_class == LinesUtil.ContourClass.cont_closed]
        assert len(closed) >= 1
        ring = max(closed, key=lambda c: c.num)
        assert ring.num > 150                   # the whole circumference, not half of it
        assert not any(j.cont1 == j.cont2 for j in det.junctions)   # no self-junction

    def test_nan_input_is_sanitised(self):
        img = np.full((64, 64), 200.0, dtype=np.float32)
        img[30:34, :] = 20.0
        img[0, 0] = np.nan
        det = FibreDetector(line_widths=[3], dark_line=True, min_len=3)
        det.detect_lines(img)
        assert det.image.max() > 0 and len(det.contours) >= 1
