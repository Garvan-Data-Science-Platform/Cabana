"""Tests for cabana/brf.py (birefringence colour bands)."""

import os

import numpy as np
import pytest

from cabana import brf

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LARGE = os.path.join(REPO, "large", "BRF")
CORE = os.path.join(LARGE, "Image_25108125 TMA picred.vsi - 20x_POL_01B2-537162_Q2_347.tif")

# Band pixel counts measured in Fiji with the macro's thresholding steps
# (HSB Stack, setThreshold + Convert to Mask on H, S and B, AND of the three).
IMAGEJ_COUNTS = {
    "core": (97, 120, 168),
    "S01": (3220675, 470218, 27013),
    "S02": (3281001, 587894, 31565),
}


def _image(*pixels):
    return np.array([pixels], dtype=np.uint8)


def test_hsb_matches_imagej_on_random_colours():
    # Fiji's HSB Stack of a seeded random image; about 0.5 % of colours differ
    # by one hue level from a float64 conversion.
    ref = np.load(os.path.join(REPO, "data", "BRF", "imagej_hsb_random.npz"))
    img = np.random.default_rng(0).integers(0, 256, (256, 256, 3), dtype=np.uint8)
    hue, sat, bright = brf.hsb(img)
    np.testing.assert_array_equal(hue, ref["hue"])
    np.testing.assert_array_equal(sat, ref["saturation"])
    np.testing.assert_array_equal(bright, ref["brightness"])


def test_hsb_of_known_colours():
    hue, sat, bright = brf.hsb(_image((255, 0, 0), (255, 255, 0), (0, 255, 0), (128, 128, 128),
                                      (0, 0, 0), (0, 2, 5)))
    assert hue.tolist() == [[0, 42, 85, 0, 0, 152]]        # (0, 2, 5): 153 in float64
    assert sat.tolist() == [[255, 255, 255, 0, 0, 255]]
    assert bright.tolist() == [[255, 255, 255, 128, 0, 5]]


def test_hsb_works_in_chunks(monkeypatch):
    img = np.random.default_rng(1).integers(0, 256, (37, 53, 3), dtype=np.uint8)
    whole = brf.hsb(img)
    monkeypatch.setattr(brf, "_CHUNK", 100)
    for a, b in zip(whole, brf.hsb(img)):
        np.testing.assert_array_equal(a, b)


def test_hsb_rejects_non_8bit_rgb():
    with pytest.raises(ValueError):
        brf.hsb(np.zeros((4, 4), np.uint8))
    with pytest.raises(ValueError):
        brf.hsb(np.zeros((4, 4, 3), np.uint16))


def test_settings_fill_missing_keys_with_the_macro_defaults():
    params = brf.settings({"Birefringence": {"Green Max": 150}})
    assert params["Green Max"] == 150 and params["Brightness Threshold"] == 10
    assert brf.band_ranges(params) == [(0, 27), (28, 47), (48, 150)]
    assert brf.band_ranges(brf.settings({})) == [(0, 27), (28, 47), (48, 140)]


def test_band_edges_are_inclusive_and_the_gate_applies():
    hue = np.array([[27, 28, 47, 48, 140, 141, 27]], np.uint8)
    bright = np.array([[10, 10, 10, 10, 10, 10, 9]], np.uint8)
    sat = np.full(hue.shape, 200, np.uint8)
    masks, gate = brf.band_masks(hue, sat, bright, [(0, 27), (28, 47), (48, 140)])
    assert masks[0].tolist() == [[True, False, False, False, False, False, False]]
    assert masks[1].tolist() == [[False, True, True, False, False, False, False]]
    assert masks[2].tolist() == [[False, False, False, True, True, False, False]]
    assert gate.tolist() == [[True] * 6 + [False]]


def test_saturation_min_drops_grey_pixels():
    # grey has hue 0, so with Saturation Min 0 it counts as red-orange (as in the macro)
    rgb = _image((60, 60, 60), (200, 40, 10))
    masks, _ = brf.separate(rgb, brf.settings({}))
    assert masks[0].tolist() == [[True, True]]
    masks, _ = brf.separate(rgb, brf.settings({"Birefringence": {"Saturation Min": 1}}))
    assert masks[0].tolist() == [[False, True]]


def test_a_range_with_min_above_max_wraps_past_255():
    hue = np.array([[230, 255, 0, 27, 28, 229]], np.uint8)
    on = np.full(hue.shape, 255, np.uint8)
    masks, _ = brf.band_masks(hue, on, on, [(230, 27)])
    assert masks[0].tolist() == [[True, True, True, True, False, False]]


def test_band_stats_in_a_region():
    # 4 x 4 image: left half is the ROI; red, yellow, green and an out-of-band pixel inside it
    rgb = np.zeros((4, 4, 3), np.uint8)
    rgb[0, 0] = (200, 30, 0)        # red-orange
    rgb[1, 0] = (200, 160, 0)       # yellow
    rgb[2, 0] = (40, 200, 40)       # green
    rgb[3, 0] = (200, 0, 120)       # deep pink: bright, in no band
    rgb[0, 3] = (200, 30, 0)        # red-orange outside the ROI
    region = np.zeros((4, 4), bool)
    region[:, :2] = True
    masks, gate = brf.separate(rgb, brf.settings({}))
    stats = brf.band_stats(masks, gate, rgb, region=region, pixel_size_um=0.5)
    assert [stats[f"BRF {b} Pixels"] for b in brf.BANDS] == [1, 1, 1]
    assert stats["BRF Red-Orange (% Region)"] == pytest.approx(100 / 8)
    assert stats["BRF Red-Orange (% Image)"] == pytest.approx(200 / 16)   # counts the pixel outside
    assert sum(stats[f"BRF {b} (% Signal)"] for b in brf.BANDS) == pytest.approx(100)
    assert sum(stats[f"BRF {b} (% Intensity)"] for b in brf.BANDS) == pytest.approx(100)
    assert stats["BRF Red-Orange Area (µm²)"] == pytest.approx(0.25)
    assert stats["BRF Total Pixels"] == 3 and stats["BRF Total (% Region)"] == pytest.approx(300 / 8)
    assert stats["BRF Unclassified (% Region)"] == pytest.approx(100 / 8)
    assert stats["BRF Unclassified Pixels"] == 1
    assert stats["Region Pixels"] == 8 and stats["Image Pixels"] == 16


def test_band_stats_without_signal_or_region_are_zero():
    rgb = np.zeros((3, 3, 3), np.uint8)
    masks, gate = brf.separate(rgb, brf.settings({}))
    stats = brf.band_stats(masks, gate, rgb, region=np.zeros((3, 3), bool))
    assert all(v == 0 for k, v in stats.items() if k != "Image Pixels")
    whole = brf.band_stats(masks, gate, rgb)
    assert whole["Region Pixels"] == 9 and whole["BRF Total (% Region)"] == 0


def test_channel_images_keep_the_original_colour_inside_the_band():
    rgb = np.zeros((9, 9, 3), np.uint8)
    rgb[2:7, 2:7] = (200, 30, 0)
    rgb[0, 8] = (40, 200, 40)
    masks, _ = brf.separate(rgb, brf.settings({}))
    images = brf.channel_images(rgb, masks)
    assert set(images) == {"Red-Orange", "Yellow", "Green", "Total"}
    assert images["Red-Orange"][4, 4].tolist() == [200, 30, 0]
    assert not images["Yellow"].any()
    assert images["Red-Orange"][0, 8].tolist() == [0, 0, 0]
    assert images["Total"][4, 4].tolist() == [200, 30, 0]


def test_class_map_paints_bands_and_unclassified_pixels():
    rgb = _image((200, 30, 0), (200, 0, 120), (0, 0, 0))
    masks, gate = brf.separate(rgb, brf.settings({}))
    cmap = brf.class_map(masks, gate)
    assert cmap[0, 0].tolist() == list(brf.BAND_COLORS[0])
    assert cmap[0, 1].tolist() == list(brf.UNCLASSIFIED_COLOR)
    assert cmap[0, 2].tolist() == [0, 0, 0]


def test_looks_brightfield():
    assert brf.looks_brightfield(np.full((4, 4, 3), 230, np.uint8))
    assert not brf.looks_brightfield(np.full((4, 4, 3), 20, np.uint8))


# ---- Parity with the ImageJ macro on real polarised-light images (not in git) ----

def _read_rgb(path):
    import tifffile
    return tifffile.imread(path)


@pytest.mark.skipif(not os.path.exists(CORE), reason="large/BRF test images absent")
def test_macro_parity_on_a_tma_core():
    from cabana.io import pixel_size_um
    rgb = _read_rgb(CORE)
    masks, gate = brf.separate(rgb, brf.settings({}))
    stats = brf.band_stats(masks, gate, rgb)
    counts = tuple(stats[f"BRF {b} Pixels"] for b in brf.BANDS)
    assert counts == IMAGEJ_COUNTS["core"]
    # the macro's output for this core: areas (µm²) and % of birefringent signal
    px, py = pixel_size_um(CORE)
    assert [round(n * px * py, 4) for n in counts] == [7.2723, 8.9967, 12.5953]
    assert [round(stats[f"BRF {b} (% Signal)"], 4) for b in brf.BANDS] == [25.1948, 31.1688, 43.6364]


@pytest.mark.parametrize("section", ["S01", "S02"])
def test_macro_parity_on_masked_sections(section):
    image = os.path.join(LARGE, "rois", f"A1420.vsi_POL_{section}.tif")
    if not os.path.exists(image):
        pytest.skip("large/BRF test images absent")
    rgb = _read_rgb(image)
    region = _read_rgb(os.path.join(LARGE, "masks", f"A1420.vsi_POL_{section}.tif")) > 0
    masks, gate = brf.separate(rgb, brf.settings({}))
    stats = brf.band_stats(masks, gate, rgb, region=region)
    counts = tuple(stats[f"BRF {b} Pixels"] for b in brf.BANDS)
    assert counts == IMAGEJ_COUNTS[section]
    # outside the mask the export is black, so the ROI holds all the signal
    assert tuple(int(m.sum()) for m in masks) == counts
