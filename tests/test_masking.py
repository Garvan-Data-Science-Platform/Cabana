"""End-to-end tests for external ROI masks (e.g. TMA core circles).

A synthetic "core" image has fibres drawn ONLY in the corners outside a
circular mask. With the mask applied, every fibre-related metric must be
zero, both with segmentation on and off. Without a mask, the same image
produces non-zero metrics, proving the mask is what removes the corners.
"""

import os
import sys

import cv2
import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cabana.cabana import Cabana
from cabana.batch import BatchCabana

SIZE = 240
RADIUS = 90


def _params(tmp_path, segmentation, patch_size=0):
    params = {
        "Configs": {"Segmentation": segmentation, "Quantification": True,
                    "Gap Analysis": True},
        "Segmentation": {"Number of Labels": 8, "Max Iterations": 2, "Normalized Hue Value": 0.96,
                         "Color Threshold": 0.2, "Min Size": 4, "Max Size": 2048,
                         "Patch Size": patch_size},
        "Detection": {"Dark Line": True, "Min Line Width": 3, "Max Line Width": 5,
                      "Line Width Step": 2, "Low Contrast": 50, "High Contrast": 150,
                      "Extend Line": False, "Minimum Line Length": 5, "Maximum Line Length": 0},
        "Quantification": {"Maximum Display HDM": 230, "Contrast Enhancement": 0.1,
                           "Minimum Branch Length": 5, "Minimum Curvature Window": 10,
                           "Maximum Curvature Window": 30, "Curvature Window Step": 10},
        "Gap Analysis": {"Minimum Gap Diameter": 10},
    }
    path = tmp_path / "params.yml"
    path.write_text(yaml.safe_dump(params))
    return str(path)


def corner_fibre_image():
    """White core-like image with dark magenta fibres only in the four corners."""
    img = np.full((SIZE, SIZE, 3), 240, dtype=np.uint8)
    # 40 px corner blocks; their innermost point is ~100 px from the centre,
    # comfortably outside the RADIUS=90 circle mask
    for (x0, y0) in [(8, 8), (SIZE - 48, 8), (8, SIZE - 48), (SIZE - 48, SIZE - 48)]:
        for k in range(4):
            y = y0 + 6 + k * 9
            cv2.line(img, (x0, y), (x0 + 40, y), (150, 40, 200), 3)
    return img


def circle_mask():
    m = np.zeros((SIZE, SIZE), np.uint8)
    cv2.circle(m, (SIZE // 2, SIZE // 2), RADIUS, 255, -1)
    return m


def _write_inputs(tmp_path, with_mask):
    img_dir = tmp_path / "Images"
    img_dir.mkdir()
    cv2.imwrite(str(img_dir / "core1.png"), corner_fibre_image())
    mask_dir = tmp_path / "Masks"
    if with_mask:
        mask_dir.mkdir()
        cv2.imwrite(str(mask_dir / "core1.png"), circle_mask())
    return str(img_dir), str(mask_dir) if with_mask else None


def _fibre_metrics(stats):
    row = stats.iloc[0]
    return {k: float(row[k]) for k in ("Area (WIDTH)", "% HDM Area", "Gap Circles Count (All)")
            if k in stats.columns}


# ---------------------------------------------------------------------------
# Single-image Cabana
# ---------------------------------------------------------------------------

class TestCabanaMasking:
    @pytest.mark.parametrize("segmentation", [False, True])
    def test_mask_removes_corner_fibres(self, tmp_path, segmentation):
        img_dir, mask_dir = _write_inputs(tmp_path, with_mask=True)
        out = tmp_path / "out"
        c = Cabana(_params(tmp_path, segmentation), os.path.join(img_dir, "core1.png"), str(out),
                   mask_dir=mask_dir)
        assert c.run() is not False
        # prepared mask copied next to the eligible image
        prepared = cv2.imread(str(out / "ROIMasks" / "core1.png"), 0)
        assert prepared is not None and prepared[SIZE // 2, SIZE // 2] == 255 and prepared[3, 3] == 0
        # Bins mask (analysis region) is the circle (intersected with segmentation)
        bins = cv2.imread(str(out / "Bins" / "core1_mask.png"), 0)
        assert bins[3, 3] == 0 and bins[SIZE - 4, SIZE - 4] == 0
        # ROI image background-filled outside the circle
        roi = cv2.imread(str(out / "ROIs" / "core1_roi.png"))
        assert tuple(roi[3, 3]) == (228, 228, 228)
        # corner fibres are gone from every fibre metric
        width_mask = cv2.imread(str(out / "Exports" / "core1" / "Width.png"), 0)
        assert np.all(width_mask[:48, :48] == 255)
        m = _fibre_metrics(c.stats)
        assert m["Area (WIDTH)"] == 0
        assert m["% HDM Area"] == 0
        # a fibre-free core is one big gap, and that gap must sit inside the circle
        gaps = pd.read_csv(out / "Masks" / "GapAnalysis" / "IndividualGaps_core1.csv")
        assert len(gaps) == 1
        assert gaps["Area (µm²)"].iloc[0] <= np.pi * RADIUS ** 2 * 1.05
        assert (gaps["X"].iloc[0] - SIZE // 2) ** 2 + (gaps["Y"].iloc[0] - SIZE // 2) ** 2 < RADIUS ** 2

    def test_without_mask_corners_are_analysed(self, tmp_path):
        img_dir, _ = _write_inputs(tmp_path, with_mask=False)
        out = tmp_path / "out"
        c = Cabana(_params(tmp_path, False), os.path.join(img_dir, "core1.png"), str(out))
        assert c.run() is not False
        assert not os.listdir(out / "ROIMasks")
        bins = cv2.imread(str(out / "Bins" / "core1_mask.png"), 0)
        assert bins.min() == 255                       # whole image is the analysis region
        m = _fibre_metrics(c.stats)
        assert m["Area (WIDTH)"] > 0
        # without the mask the free space reaches the image border, so the
        # largest gap is bigger than the circle
        gaps = pd.read_csv(out / "Masks" / "GapAnalysis" / "IndividualGaps_core1.csv")
        assert gaps["Area (µm²)"].max() > np.pi * RADIUS ** 2 * 1.05

    def test_hdm_percentage_relative_to_mask(self, tmp_path):
        """A dark blob inside the circle: % HDM Area = blob / circle area, not blob / image."""
        img = np.full((SIZE, SIZE, 3), 240, dtype=np.uint8)
        cv2.circle(img, (SIZE // 2, SIZE // 2), 20, (30, 30, 30), -1)
        img_dir = tmp_path / "Images"
        img_dir.mkdir()
        cv2.imwrite(str(img_dir / "core1.png"), img)
        mask_dir = tmp_path / "Masks"
        mask_dir.mkdir()
        cv2.imwrite(str(mask_dir / "core1.png"), circle_mask())
        out = tmp_path / "out"
        c = Cabana(_params(tmp_path, False), str(img_dir / "core1.png"), str(out), mask_dir=str(mask_dir))
        c.initialize_params()
        assert c.prepare_image()
        c.generate_roi()
        c.quantify_hdm()
        expected = (np.pi * 20 ** 2) / (np.pi * RADIUS ** 2)
        assert abs(float(c.stats.loc[0, "% HDM Area"]) - expected) < 0.05


# ---------------------------------------------------------------------------
# BatchCabana
# ---------------------------------------------------------------------------

class TestBatchMasking:
    @pytest.mark.parametrize("segmentation", [False, True])
    def test_batch_mask_removes_corner_fibres(self, tmp_path, segmentation):
        img_dir, mask_dir = _write_inputs(tmp_path, with_mask=True)
        # second image without a mask: analysed whole
        cv2.imwrite(os.path.join(img_dir, "core2.png"), corner_fibre_image())
        out = tmp_path / "out"
        out.mkdir()
        b = BatchCabana(_params(tmp_path, segmentation), img_dir, str(out), batch_size=5,
                        batch_idx=0, ignore_large=True, mask_dir=mask_dir)
        b.run()
        assert os.path.exists(out / "ROIMasks" / "core1.png")
        assert not os.path.exists(out / "ROIMasks" / "core2.png")
        stats = pd.read_csv(out / "QuantificationResults.csv").set_index("Image")
        assert stats.loc["core1_roi.png", "Fibre Area (WIDTH, µm²)"] == 0
        assert stats.loc["core2_roi.png", "Fibre Area (WIDTH, µm²)"] > 0
        assert stats.loc["core1_roi.png", "% HDM Area"] == 0
        gaps = pd.read_csv(out / "Masks" / "GapAnalysis" / "IndividualGaps_core1_roi.csv")
        assert len(gaps) == 1 and gaps["Area (µm²)"].iloc[0] <= np.pi * RADIUS ** 2 * 1.05
        w1 = cv2.imread(str(out / "Exports" / "core1_roi" / "Width.png"), 0)
        assert np.all(w1[:48, :48] == 255)

    def test_mask_message_counts_eligible_images(self, tmp_path, monkeypatch):
        img_dir, mask_dir = _write_inputs(tmp_path, with_mask=True)
        cv2.imwrite(os.path.join(img_dir, "core2.png"), corner_fibre_image())
        out = tmp_path / "out"
        out.mkdir()
        params = _params(tmp_path, False)
        data = yaml.safe_load(open(params))
        data["Configs"]["Quantification"] = False
        yaml.safe_dump(data, open(params, "w"))
        from cabana.log import Log
        said = []
        monkeypatch.setattr(Log.logger, "info", lambda msg, *a, **k: said.append(str(msg)))
        BatchCabana(params, img_dir, str(out), batch_size=5, batch_idx=0, ignore_large=True,
                    mask_dir=mask_dir).run()
        assert any(m.startswith("1 of 2 eligible images have an external ROI mask") for m in said)

    def test_max_size_zero_analyses_the_image_whole(self, tmp_path):
        img_dir, mask_dir = _write_inputs(tmp_path, with_mask=True)
        out = tmp_path / "out"
        out.mkdir()
        params = _params(tmp_path, False)
        data = yaml.safe_load(open(params))
        data.setdefault("Configs", {})["Max Size"] = 0         # no limit: no blocks, nothing ignored
        data["Configs"]["Quantification"] = False
        yaml.safe_dump(data, open(params, "w"))
        BatchCabana(params, img_dir, str(out), batch_size=5, batch_idx=0, ignore_large=True,
                    mask_dir=mask_dir).run()
        assert sorted(os.listdir(out / "ROIMasks")) == ["core1.png"]

    def test_oversized_image_blocks_get_cropped_masks(self, tmp_path, monkeypatch):
        import cabana.io
        monkeypatch.setattr(cabana.io, "MIN_MAX_SIZE", 100)    # small test image: lower the floor
        img_dir, mask_dir = _write_inputs(tmp_path, with_mask=True)
        out = tmp_path / "out"
        out.mkdir()
        params = _params(tmp_path, False)
        data = yaml.safe_load(open(params))
        # forces a 2x2 block split of the 240 px image; the pre-0.3.9 location
        # Segmentation: Max Size must still be honoured
        data["Segmentation"]["Max Size"] = 150
        data["Configs"]["Quantification"] = False
        yaml.safe_dump(data, open(params, "w"))
        b = BatchCabana(params, img_dir, str(out), batch_size=5, batch_idx=0, ignore_large=False,
                        mask_dir=mask_dir)
        b.run()
        blocks = sorted(os.listdir(out / "ROIMasks"))
        assert blocks == ["core1_blk_0_0.png", "core1_blk_0_1.png", "core1_blk_1_0.png", "core1_blk_1_1.png"]
        full = circle_mask()
        blk = cv2.imread(str(out / "ROIMasks" / "core1_blk_0_0.png"), 0)
        assert blk.shape == (120, 120) and np.array_equal(blk > 0, full[:120, :120] > 0)


class TestBatchStorage:
    """A finished batch keeps one copy of each output, compressed maps and no
    working copies; with visualisations off only the measurements remain."""

    def _run(self, tmp_path, save_visuals=None):
        from cabana.batch import BatchProcessor
        img_dir, mask_dir = _write_inputs(tmp_path, with_mask=True)
        out = tmp_path / "out"
        out.mkdir()
        BatchProcessor(_params(tmp_path, False), img_dir, str(out), batch_size=5, mask_dir=mask_dir,
                       save_visuals=save_visuals).run()
        return out

    def test_outputs_once_and_compressed(self, tmp_path):
        import tifffile
        out = self._run(tmp_path)
        assert not (out / "Eligible").exists() and (out / "IgnoredImages.txt").exists()
        exp, col = out / "Exports" / "core1_roi", out / "Colors" / "core1_roi"
        assert (col / "all_gaps.png").exists() and not (exp / "GapImage.png").exists()
        assert (col / "intra_gaps.png").exists() and not (exp / "GapImage_intra_gaps.png").exists()
        for name in ("Energy", "Coherency", "Orientation", "Length_Map", "Curve_Map_10"):
            with tifffile.TiffFile(exp / f"{name}.tif") as t:
                page = t.pages[0]
                assert page.dtype == np.float32 and page.compression.name in ("ADOBE_DEFLATE", "DEFLATE")
        assert not (out / "Fibres").exists()           # the fibre overlay is Colors/.../color_mask.png
        assert (col / "color_mask.png").exists()

    def test_without_visualisations_only_measurements_remain(self, tmp_path):
        out = self._run(tmp_path, save_visuals=False)
        assert not list((out / "Colors").rglob("*.png")) and not (out / "Fibres").exists()
        assert not list((out / "Masks" / "GapAnalysis").glob("*GapImage*"))
        assert (out / "Exports" / "core1_roi" / "Energy.tif").exists()
        assert (out / "QuantificationResults.csv").exists()


class TestGapFreeImages:
    """An image without gaps reports 0 for every gap metric, never NaN or a
    missing column, whatever else is in its batch."""

    def _no_gap_params(self, tmp_path):
        params = _params(tmp_path, True)          # with segmentation: ROI columns are kept
        data = yaml.safe_load(open(params))
        data["Gap Analysis"]["Minimum Gap Diameter"] = 1000      # larger than the image
        yaml.safe_dump(data, open(params, "w"))
        return params

    def test_batch_of_gap_free_images_has_zero_gap_columns(self, tmp_path):
        img_dir, _ = _write_inputs(tmp_path, with_mask=False)
        out = tmp_path / "out"
        out.mkdir()
        BatchCabana(self._no_gap_params(tmp_path), img_dir, str(out), batch_size=5, batch_idx=0,
                    ignore_large=True).run()
        stats = pd.read_csv(out / "QuantificationResults.csv")
        gap_cols = [c for c in stats.columns if "gaps" in c or "Gap Circles" in c]
        assert "Gap Circles Count (ROI)" in gap_cols and "Gap Circles Count (All)" in gap_cols
        assert (stats[gap_cols] == 0).all().all()

    def test_single_gap_free_image_has_zero_roi_gap_metrics(self, tmp_path):
        img_dir, _ = _write_inputs(tmp_path, with_mask=False)
        analyzer = Cabana(self._no_gap_params(tmp_path), os.path.join(img_dir, "core1.png"),
                          str(tmp_path / "out"))
        assert analyzer.run()
        assert analyzer.stats.loc[0, "Gap Circles Count (ROI)"] == 0
        assert analyzer.stats.loc[0, "Mean (ROI gaps area in µm²)"] == 0


def test_single_image_run_keeps_no_working_copy(tmp_path):
    img_dir, _ = _write_inputs(tmp_path, with_mask=False)
    out = tmp_path / "out"
    analyzer = Cabana(_params(tmp_path, False), os.path.join(img_dir, "core1.png"), str(out))
    assert analyzer.run()
    assert not (out / "Eligible").exists() and analyzer.eligible_dir is None
    analyzer.export_results()                     # still works after run()
    assert (out / "core1_QuantificationResults.csv").exists()


class TestColourMaps:
    """Single-image and batch runs write the same colour maps (one shared stage)."""

    def _params(self, tmp_path, visuals):
        params = _params(tmp_path, True)
        data = yaml.safe_load(open(params))
        data["Configs"]["Save Visualisations"] = visuals
        yaml.safe_dump(data, open(params, "w"))
        return params

    def test_single_image_matches_batch(self, tmp_path):
        img_dir, _ = _write_inputs(tmp_path, with_mask=False)
        params = self._params(tmp_path, True)
        single = tmp_path / "single"
        assert Cabana(params, os.path.join(img_dir, "core1.png"), str(single)).run()
        batch = tmp_path / "batch"
        batch.mkdir()
        BatchCabana(params, img_dir, str(batch), batch_size=5, batch_idx=0, ignore_large=True).run()
        single_maps = sorted(os.listdir(single / "Colors" / "core1"))
        batch_maps = sorted(os.listdir(batch / "Colors" / "core1_roi"))
        assert single_maps == batch_maps
        assert {"color_energy.png", "color_skeleton.png", "orient_color_survey.png",
                "all_gaps.png"} <= set(single_maps)
        assert not (single / "Exports" / "core1" / "GapImage.png").exists()     # moved, not copied

    def test_single_image_without_visualisations_writes_no_colour_maps(self, tmp_path):
        img_dir, _ = _write_inputs(tmp_path, with_mask=False)
        single = tmp_path / "single"
        assert Cabana(self._params(tmp_path, False), os.path.join(img_dir, "core1.png"), str(single)).run()
        assert os.listdir(single / "Colors" / "core1") == []
        assert not (single / "Exports" / "core1" / "GapImage.png").exists()


def test_fit_gap_circles_fills_gaps_without_overlap():
    from cabana.stages import fit_gap_circles
    gaps = np.zeros((200, 200), np.uint8)
    gaps[20:180, 20:180] = 255                       # one square gap
    circles = fit_gap_circles(gaps, 10)
    assert circles.shape[1] == 3 and len(circles) > 1
    assert circles[:, 2].max() == pytest.approx(80, abs=2)   # the largest fits the square
    for i in range(len(circles)):                   # circles do not overlap (placed circles are
        for j in range(i + 1, len(circles)):        # masked with whole-pixel radii: < 1 px slack)
            d = np.hypot(*(circles[i, :2] - circles[j, :2]))
            assert d >= circles[i, 2] + circles[j, 2] - 1.0
    roi = np.zeros_like(gaps)                       # outside the ROI is not a gap
    assert len(fit_gap_circles(gaps, 10, roi_mask=roi)) == 0
    assert fit_gap_circles(np.zeros_like(gaps), 10).shape == (0, 3)
