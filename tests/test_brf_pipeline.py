"""Tests for the birefringence (BRF) stage of the pipelines (stages.run_brf,
Cabana and BatchProcessor)."""

import os
import sys

import cv2
import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cabana import brf  # noqa: E402
from cabana.stages import (BRF_RESULTS, BRF_SUMMARY_COLUMNS, merge_brf_columns,  # noqa: E402
                           run_brf)

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


def _params(tmp_path, birefringence=True, sort=True, quantification=True, name="params.yml"):
    params = {
        "Configs": {"Segmentation": False, "Quantification": quantification, "Gap Analysis": False,
                    "Birefringence": birefringence},
        "Segmentation": {"Number of Labels": 8, "Max Iterations": 2, "Normalized Hue Value": 0.96,
                         "Color Threshold": 0.2, "Min Size": 4, "Patch Size": 0},
        "Detection": {"Dark Line": False, "Min Line Width": 2, "Max Line Width": 4,
                      "Line Width Step": 1, "Low Contrast": 12, "High Contrast": 100,
                      "Extend Line": False, "Minimum Line Length": 5, "Maximum Line Length": 0},
        "Quantification": {"Maximum Display HDM": 230, "Contrast Enhancement": 0.1,
                           "Minimum Branch Length": 5, "Minimum Curvature Window": 10,
                           "Maximum Curvature Window": 30, "Curvature Window Step": 10},
        "Gap Analysis": {"Minimum Gap Diameter": 10},
        "Birefringence": dict(brf.DEFAULTS, **{"Sort Into Folders": sort}),
    }
    path = tmp_path / name
    path.write_text(yaml.safe_dump(params))
    return str(path), params


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
    masks, gate = brf.separate(rgb, brf.settings({}))
    return brf.band_stats(masks, gate, rgb, region=region, pixel_size_um=pixel_size_um)


# ---------------------------------------------------------------- run_brf --

@pytest.mark.parametrize("sort", [True, False])
def test_run_brf_writes_sorted_or_flat_outputs(tmp_path, sort):
    from cabana.io import pixel_size_um
    _, args = _params(tmp_path, sort=sort)
    img_path = tmp_path / "sec1.png"
    cv2.imwrite(str(img_path), cv2.cvtColor(pol_image(), cv2.COLOR_RGB2BGR))
    mask_path = tmp_path / "mask.png"
    cv2.imwrite(str(mask_path), half_mask())
    out = tmp_path / "BRF"
    row = run_brf(args, str(img_path), "sec1", str(out), roi_mask_path=str(mask_path), pixel_size_um=0.5)

    expected = _expected(region=half_mask() > 0, pixel_size_um=0.5)
    assert row["Image"] == "sec1_roi.png"
    assert {k: row[k] for k in expected} == expected
    assert row["BRF Red-Orange Pixels"] > 0 and row["BRF Green Pixels"] > 0
    assert row["Region Pixels"] == SIZE * SIZE // 2

    where = {name: (out / folder if sort else out) for name, folder in
             (("stats", "Stats"), ("total", "Total"), ("red", "Red"), ("yellow", "Yellow"), ("green", "Green"))}
    for key, suffix in (("total", "total"), ("total", "original"), ("red", "red_orange"),
                        ("yellow", "yellow"), ("green", "green")):
        path = where[key] / f"sec1_{suffix}.tif"
        assert path.exists(), path
        assert pixel_size_um(str(path)) == pytest.approx((0.5, 0.5))
    table = pd.read_csv(where["stats"] / "sec1_statistics.csv", comment="#")
    assert table["Band"].tolist() == ["Red-Orange", "Yellow", "Green", "Total (all bands)", "Unclassified"]
    assert table.loc[0, "Pixels"] == row["BRF Red-Orange Pixels"]
    assert table.loc[3, "% Signal"] == pytest.approx(100)
    if sort:
        assert sorted(p.name for p in out.iterdir()) == ["Green", "Red", "Stats", "Total", "Yellow"]


def test_run_brf_without_visualisations_writes_only_the_table(tmp_path):
    _, args = _params(tmp_path)
    img_path = tmp_path / "sec1.png"
    cv2.imwrite(str(img_path), cv2.cvtColor(pol_image(), cv2.COLOR_RGB2BGR))
    out = tmp_path / "BRF"
    run_brf(args, str(img_path), "sec1", str(out), visuals=False)
    assert [p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()] == ["Stats/sec1_statistics.csv"]


def test_run_brf_resizes_a_mask_of_another_size(tmp_path, caplog):
    _, args = _params(tmp_path)
    img_path = tmp_path / "sec1.png"
    cv2.imwrite(str(img_path), cv2.cvtColor(pol_image(), cv2.COLOR_RGB2BGR))
    mask_path = tmp_path / "mask.png"
    cv2.imwrite(str(mask_path), np.full((10, 10), 255, np.uint8))
    row = run_brf(args, str(img_path), "sec1", str(tmp_path / "BRF"), roi_mask_path=str(mask_path),
                  visuals=False)
    assert row["Region Pixels"] == SIZE * SIZE
    assert "resizing" in caplog.text


def test_merge_brf_columns_joins_on_image():
    df = pd.DataFrame({"Image": ["a_roi.png", "b_roi.png"], "Total Length (µm)": [1.0, 2.0]})
    row = {"Image": "b_roi.png", **{c: 1.0 for c in BRF_SUMMARY_COLUMNS}, "BRF Red-Orange Pixels": 3}
    merged = merge_brf_columns(df, [row])
    assert list(merged.columns) == ["Image", "Total Length (µm)", *BRF_SUMMARY_COLUMNS]
    assert merged.loc[1, "BRF Total (% Region)"] == 1.0 and np.isnan(merged.loc[0, "BRF Total (% Region)"])
    assert merge_brf_columns(df, []) is df


# ------------------------------------------------------------- pipelines --

def test_single_image_run_adds_brf_results(tmp_path):
    from cabana.cabana import Cabana
    param_file, _ = _params(tmp_path)
    img_dir, mask_dir = _write_inputs(tmp_path)
    out = tmp_path / "out"
    c = Cabana(param_file, os.path.join(img_dir, "sec1.png"), str(out), mask_dir=mask_dir)
    assert c.run() is not False
    expected = _expected(region=half_mask() > 0)
    # Segmentation is off, which drops Cabana's ROI columns; the BRF "% Region" columns stay
    for col in BRF_SUMMARY_COLUMNS:
        assert c.stats.loc[0, col] == pytest.approx(expected[col])
    results = pd.read_csv(out / "BRF" / "Stats" / BRF_RESULTS)
    assert results.loc[0, "Image"] == "sec1_roi.png"
    assert results.loc[0, "BRF Yellow Pixels"] == expected["BRF Yellow Pixels"]
    assert (out / "BRF" / "Total" / "sec1_total.tif").exists()


def test_brf_off_writes_nothing(tmp_path):
    from cabana.cabana import Cabana
    param_file, _ = _params(tmp_path, birefringence=False)
    img_dir, mask_dir = _write_inputs(tmp_path)
    out = tmp_path / "out"
    Cabana(param_file, os.path.join(img_dir, "sec1.png"), str(out), mask_dir=mask_dir).run()
    assert not (out / "BRF").exists()


def test_brf_runs_without_quantification(tmp_path):
    from cabana.cabana import Cabana
    param_file, _ = _params(tmp_path, quantification=False)
    img_dir, mask_dir = _write_inputs(tmp_path)
    out = tmp_path / "out"
    Cabana(param_file, os.path.join(img_dir, "sec1.png"), str(out), mask_dir=mask_dir).run()
    assert (out / "BRF" / "Stats" / BRF_RESULTS).exists()


@pytest.mark.parametrize("sort", [True, False])
def test_batch_run_merges_brf_results_across_batches(tmp_path, sort):
    from cabana.batch import BatchProcessor
    param_file, _ = _params(tmp_path, sort=sort)
    img_dir, mask_dir = _write_inputs(tmp_path, names=("sec1", "sec2", "sec3"))
    out = tmp_path / "out"
    out.mkdir()
    BatchProcessor(param_file, img_dir, str(out), batch_size=2, mask_dir=mask_dir).run()

    brf_dir = out / "BRF"
    stats_dir = brf_dir / "Stats" if sort else brf_dir
    results = pd.read_csv(stats_dir / BRF_RESULTS)
    assert sorted(results["Image"]) == ["sec1_roi.png", "sec2_roi.png", "sec3_roi.png"]
    expected = _expected(region=half_mask() > 0)
    assert (results["BRF Green Pixels"] == expected["BRF Green Pixels"]).all()
    for name in ("sec1", "sec2", "sec3"):
        assert (stats_dir / f"{name}_statistics.csv").exists()
        assert ((brf_dir / "Red" if sort else brf_dir) / f"{name}_red_orange.tif").exists()
    assert not (out / "Batches").exists()

    quant = pd.read_csv(out / "QuantificationResults.csv")
    assert set(BRF_SUMMARY_COLUMNS) <= set(quant.columns)
    assert quant["BRF Total (% Region)"].round(3).tolist() == [round(expected["BRF Total (% Region)"], 3)] * 3
    compact = pd.read_csv(out / "QuantificationResults_Compact.csv")
    assert "BRF Red-Orange (% Signal)" in compact.columns and "BRF Total (% Region)" in compact.columns


def test_batch_progress_counts_the_brf_stage(tmp_path):
    from cabana.batch import BatchProcessor
    param_file, _ = _params(tmp_path)
    img_dir, mask_dir = _write_inputs(tmp_path, names=("sec1",))
    out = tmp_path / "out"
    out.mkdir()
    statuses = []
    bp = BatchProcessor(param_file, img_dir, str(out), batch_size=2, mask_dir=mask_dir)
    bp.status_callback = statuses.append
    bp.run()
    assert any("Birefringence: sec1" in s for s in statuses)


# ------------------------------------------------------------ folder run --

def test_folder_run_measures_every_image_inside_its_mask(tmp_path):
    from cabana.stages import BRF_PARAMETERS_FILE, run_brf_folder, write_calibrated_tiff
    _, args = _params(tmp_path)
    img_dir, mask_dir = _write_inputs(tmp_path, names=("sec1", "sec2"))
    # a 16-bit TIFF with a pixel size and no mask, and an unreadable file
    write_calibrated_tiff(os.path.join(img_dir, "sec3.tif"), pol_image(), 0.5)
    import tifffile
    tifffile.imwrite(os.path.join(img_dir, "sec4.tif"), (pol_image().astype(np.uint16) * 257),
                     photometric="rgb")
    open(os.path.join(img_dir, "broken.png"), "wb").write(b"not an image")
    out = tmp_path / "out"
    seen = []
    summary = run_brf_folder(args, img_dir, str(out), mask_dir=mask_dir,
                             progress=lambda done, total, name: seen.append((done, total, name)))
    assert summary["done"] == 4 and summary["total"] == 5 and summary["skipped"] == ["broken.png"]
    # the PNGs have no pixel size; a TIFF without a unit counts as µm (Cabana's rule)
    assert not summary["cancelled"] and summary["no_pixel_size"] == 2
    results = pd.read_csv(out / "Stats" / BRF_RESULTS).set_index("Image")
    assert sorted(results.index) == ["sec1.png", "sec2.png", "sec3.tif", "sec4.tif"]
    masked, whole = _expected(region=half_mask() > 0), _expected()
    assert results.loc["sec1.png", "BRF Green Pixels"] == masked["BRF Green Pixels"]
    assert results.loc["sec3.tif", "BRF Green Pixels"] == whole["BRF Green Pixels"]
    assert results.loc["sec3.tif", "BRF Green Area (µm²)"] == pytest.approx(whole["BRF Green Pixels"] * 0.25)
    assert results.loc["sec4.tif", "BRF Total Pixels"] > 0                 # 16-bit converted to 8-bit
    for folder, suffix in (("Total", "total"), ("Total", "original"), ("Red", "red_orange"),
                           ("Yellow", "yellow"), ("Green", "green")):
        assert (out / folder / f"sec1_{suffix}.tif").exists()
    assert (out / BRF_PARAMETERS_FILE).exists() and not (out / "BRF").exists()
    assert seen[0] == (0, 5, "broken.png") and seen[-1] == (5, 5, "")


def test_folder_run_without_images_writes_only_tables_and_can_be_cancelled(tmp_path):
    from cabana.stages import run_brf_folder
    _, args = _params(tmp_path, sort=False)
    img_dir, _ = _write_inputs(tmp_path, names=("sec1", "sec2", "sec3"))
    out = tmp_path / "out"
    summary = run_brf_folder(args, img_dir, str(out), visuals=False)
    assert summary["done"] == 3
    assert sorted(p.name for p in out.iterdir()) == sorted(
        ["BRF_Parameters.yml", BRF_RESULTS] + [f"sec{i}_statistics.csv" for i in (1, 2, 3)])
    calls = []
    summary = run_brf_folder(args, img_dir, str(tmp_path / "out2"), visuals=False,
                             cancel=lambda: len(calls) >= 1, progress=lambda *a: calls.append(a))
    assert summary["cancelled"] and summary["done"] == 1


def test_folder_run_refuses_an_output_folder_holding_the_input(tmp_path):
    from cabana.stages import run_brf_folder
    _, args = _params(tmp_path)
    img_dir, mask_dir = _write_inputs(tmp_path)
    with pytest.raises(ValueError, match="input"):
        run_brf_folder(args, img_dir, str(tmp_path))
    with pytest.raises(ValueError, match="ROI mask"):
        run_brf_folder(args, img_dir, mask_dir, mask_dir=mask_dir)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="No supported images"):
        run_brf_folder(args, str(empty), str(tmp_path / "out"))
