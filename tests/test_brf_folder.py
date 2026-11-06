"""Tests for the birefringence (BRF) Run Folder stage (stages.run_brf_folder):
colour separation of a folder of images, without the rest of the pipeline."""

import os
import sys

import cv2
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cabana import brf  # noqa: E402
from cabana.stages import (BRF_RESULTS, BRF_SETTINGS_FILE, run_brf_folder,  # noqa: E402
                           write_calibrated_tiff)

SIZE = 160
COLOURS = {"Red-Orange": (220, 60, 0), "Yellow": (230, 200, 0), "Green": (40, 210, 60)}   # RGB


def pol_image():
    """Polarised-light-like image: red-orange, yellow and green lines on black."""
    img = np.zeros((SIZE, SIZE, 3), np.uint8)
    for k, colour in enumerate(COLOURS.values()):
        for j in range(3):
            y = 20 + k * 45 + j * 12
            cv2.line(img, (10, y), (SIZE - 10, y), colour, 3)
    return img


def half_mask():
    m = np.zeros((SIZE, SIZE), np.uint8)
    m[:, : SIZE // 2] = 255
    return m


def _write_inputs(tmp_path, names=("sec1",)):
    img_dir, mask_dir = tmp_path / "Images", tmp_path / "Masks"
    img_dir.mkdir()
    mask_dir.mkdir()
    for name in names:
        cv2.imwrite(str(img_dir / f"{name}.png"), cv2.cvtColor(pol_image(), cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(mask_dir / f"{name}.png"), half_mask())
    return str(img_dir), str(mask_dir)


def _expected(region=None, pixel_size_um=1.0):
    rgb = pol_image()
    masks, gate = brf.separate(rgb, brf.settings())
    return brf.band_stats(masks, gate, rgb, region=region, pixel_size_um=pixel_size_um)


def test_folder_run_measures_every_image_inside_its_mask(tmp_path):
    from cabana.io import pixel_size_um
    img_dir, mask_dir = _write_inputs(tmp_path, names=("sec1", "sec2"))
    # a TIFF with a pixel size and no mask, a 16-bit TIFF, and an unreadable file
    write_calibrated_tiff(os.path.join(img_dir, "sec3.tif"), pol_image(), 0.5)
    import tifffile
    tifffile.imwrite(os.path.join(img_dir, "sec4.tif"), (pol_image().astype(np.uint16) * 257),
                     photometric="rgb")
    open(os.path.join(img_dir, "broken.png"), "wb").write(b"not an image")
    out = tmp_path / "out"
    seen = []
    summary = run_brf_folder(None, img_dir, str(out), mask_dir=mask_dir,
                             progress=lambda done, total, name: seen.append((done, total, name)))
    assert summary["done"] == 4 and summary["total"] == 5 and summary["skipped"] == ["broken.png"]
    # the PNGs have no pixel size; a TIFF without a unit counts as µm (Cabana's rule)
    assert not summary["cancelled"] and summary["no_pixel_size"] == 2
    results = pd.read_csv(out / "Stats" / BRF_RESULTS).set_index("Image")
    assert sorted(results.index) == ["sec1.png", "sec2.png", "sec3.tif", "sec4.tif"]
    masked, whole = _expected(region=half_mask() > 0), _expected()
    assert results.loc["sec1.png", "BRF Green Pixels"] == masked["BRF Green Pixels"]
    assert results.loc["sec1.png", "Region Pixels"] == SIZE * SIZE // 2
    assert results.loc["sec3.tif", "BRF Green Pixels"] == whole["BRF Green Pixels"]
    assert results.loc["sec3.tif", "BRF Green Area (µm²)"] == pytest.approx(whole["BRF Green Pixels"] * 0.25)
    assert results.loc["sec4.tif", "BRF Total Pixels"] > 0                 # 16-bit converted to 8-bit
    for folder, suffix in (("Total", "total"), ("Total", "original"), ("Red", "red_orange"),
                           ("Yellow", "yellow"), ("Green", "green")):
        path = out / folder / f"sec3_{suffix}.tif"
        assert path.exists() and pixel_size_um(str(path)) == pytest.approx((0.5, 0.5))
    table = pd.read_csv(out / "Stats" / "sec1_statistics.csv", comment="#")
    assert table["Band"].tolist() == ["Red-Orange", "Yellow", "Green", "Total (all bands)", "Unclassified"]
    assert table.loc[0, "Pixels"] == masked["BRF Red-Orange Pixels"] and table.loc[3, "% Signal"] == 100
    assert sorted(p.name for p in out.iterdir()) == ["BRF_Settings.yml", "Green", "Red", "Stats", "Total", "Yellow"]
    assert seen[0] == (0, 5, "broken.png") and seen[-1] == (5, 5, "")


def test_folder_run_uses_the_given_settings_and_records_them(tmp_path):
    import yaml
    img_dir, _ = _write_inputs(tmp_path)
    out = tmp_path / "out"
    run_brf_folder({"Brightness Threshold": 255}, img_dir, str(out), visuals=False)
    results = pd.read_csv(out / "Stats" / BRF_RESULTS)
    assert results.loc[0, "BRF Total Pixels"] == 0                        # nothing is that bright
    record = yaml.safe_load(open(out / BRF_SETTINGS_FILE))
    assert record["Birefringence"] == brf.settings({"Brightness Threshold": 255})
    assert record["Input Folder"] == img_dir and record["Save Images"] is False


def test_folder_run_without_images_writes_only_tables_and_can_be_cancelled(tmp_path):
    img_dir, _ = _write_inputs(tmp_path, names=("sec1", "sec2", "sec3"))
    out = tmp_path / "out"
    summary = run_brf_folder({"Sort Into Folders": False}, img_dir, str(out), visuals=False)
    assert summary["done"] == 3
    assert sorted(p.name for p in out.iterdir()) == sorted(
        [BRF_SETTINGS_FILE, BRF_RESULTS] + [f"sec{i}_statistics.csv" for i in (1, 2, 3)])
    calls = []
    summary = run_brf_folder(None, img_dir, str(tmp_path / "out2"), visuals=False,
                             cancel=lambda: len(calls) >= 1, progress=lambda *a: calls.append(a))
    assert summary["cancelled"] and summary["done"] == 1


def test_folder_run_resizes_a_mask_of_another_size(tmp_path, caplog):
    img_dir, mask_dir = _write_inputs(tmp_path)
    cv2.imwrite(os.path.join(mask_dir, "sec1.png"), np.full((10, 10), 255, np.uint8))
    run_brf_folder(None, img_dir, str(tmp_path / "out"), mask_dir=mask_dir, visuals=False)
    results = pd.read_csv(tmp_path / "out" / "Stats" / BRF_RESULTS)
    assert results.loc[0, "Region Pixels"] == SIZE * SIZE and "resizing" in caplog.text


def test_folder_run_refuses_an_output_folder_holding_the_input(tmp_path):
    img_dir, mask_dir = _write_inputs(tmp_path)
    with pytest.raises(ValueError, match="input"):
        run_brf_folder(None, img_dir, str(tmp_path))
    with pytest.raises(ValueError, match="ROI mask"):
        run_brf_folder(None, img_dir, mask_dir, mask_dir=mask_dir)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="No supported images"):
        run_brf_folder(None, str(empty), str(tmp_path / "out"))


def test_parameter_file_has_no_birefringence_entries():
    """Colour separation is on the Birefringence page only; Batch Run has no such step."""
    import yaml
    defaults = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "cabana",
                                                "default_params.yml")))
    assert "Birefringence" not in defaults and "Birefringence" not in defaults["Configs"]
