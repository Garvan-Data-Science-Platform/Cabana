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


def _params(tmp_path, segmentation, roi_masks="", patch_size=0):
    params = {
        "Configs": {"Segmentation": segmentation, "Quantification": True,
                    "Gap Analysis": True, "ROI Masks": roi_masks},
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

    def test_mask_dir_from_parameter_file(self, tmp_path):
        img_dir, mask_dir = _write_inputs(tmp_path, with_mask=True)
        out = tmp_path / "out"
        c = Cabana(_params(tmp_path, False, roi_masks=mask_dir), os.path.join(img_dir, "core1.png"), str(out))
        assert c.run() is not False
        assert c.ext_mask_dir == mask_dir
        assert _fibre_metrics(c.stats)["Area (WIDTH)"] == 0

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
        from cabana.log import Log
        Log.init_log_path(str(tmp_path / "Logs"))
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

    def test_oversized_image_blocks_get_cropped_masks(self, tmp_path):
        img_dir, mask_dir = _write_inputs(tmp_path, with_mask=True)
        out = tmp_path / "out"
        out.mkdir()
        params = _params(tmp_path, False)
        data = yaml.safe_load(open(params))
        data["Segmentation"]["Max Size"] = 150         # forces a 2x2 block split of the 240 px image
        data["Configs"]["Quantification"] = False
        yaml.safe_dump(data, open(params, "w"))
        from cabana.log import Log
        Log.init_log_path(str(tmp_path / "Logs"))
        b = BatchCabana(params, img_dir, str(out), batch_size=5, batch_idx=0, ignore_large=False,
                        mask_dir=mask_dir)
        b.run()
        blocks = sorted(os.listdir(out / "ROIMasks"))
        assert blocks == ["core1_blk_0_0.png", "core1_blk_0_1.png", "core1_blk_1_0.png", "core1_blk_1_1.png"]
        full = circle_mask()
        blk = cv2.imread(str(out / "ROIMasks" / "core1_blk_0_0.png"), 0)
        assert blk.shape == (120, 120) and np.array_equal(blk > 0, full[:120, :120] > 0)
