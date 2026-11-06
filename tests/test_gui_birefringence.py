"""Tests for the Birefringence page of cabana/cabana_gui.py (offscreen Qt)."""

import os
import sys

import cv2
import numpy as np
import pytest
import yaml

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytest.importorskip("PyQt5")
from PyQt5.QtCore import QEventLoop, QTimer  # noqa: E402
from PyQt5.QtWidgets import QApplication, QLabel  # noqa: E402

from cabana import brf  # noqa: E402
from tests.test_brf_folder import pol_image  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def settle(app, ms=50):
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec_()
    app.processEvents()


@pytest.fixture
def win(app):
    from cabana.cabana_gui import MainWindow
    w = MainWindow()
    w.show()
    settle(app)
    yield w
    w.close()


def _open(win, tmp_path, rgb, name="pol.png"):
    path = tmp_path / name
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    win.load_original_image(str(path))


def _separate(win, monkeypatch):
    """Run the HSB worker synchronously and record the images shown."""
    shown = []
    monkeypatch.setattr(win.image_panel, "setImage", lambda img, preserve_view=False: shown.append(img))
    win.handle_brf_complete(brf.hsb(win.ori_img))
    return shown


def test_page_sits_between_tma_and_segmentation_and_starts_at_the_macro_defaults(win):
    titles = win._page_titles
    assert titles.index("Birefringence") == titles.index("TMA Dearrayer") + 1
    assert titles.index("Segmentation") == titles.index("Birefringence") + 1
    assert win.pages.indexOf(win.brf_tab) == titles.index("Birefringence")
    assert not win.brf_btn.isEnabled() and win.brf_view_combo.currentText() == "Grid"   # no image yet
    assert win._brf_settings == brf.DEFAULTS
    assert win.brf_brightness_slider.value() == 10 and win.brf_band_values[2].text() == "48, 140"


def test_separate_shows_the_class_map_and_a_summary(win, tmp_path, monkeypatch, app):
    _open(win, tmp_path, pol_image())
    shown = _separate(win, monkeypatch)
    masks, gate = brf.separate(win.ori_img, brf.settings({}))
    np.testing.assert_array_equal(shown[-1], brf.grid_view(win.ori_img, masks))     # default view
    win.brf_view_combo.setCurrentText("Class map")
    np.testing.assert_array_equal(shown[-1], brf.class_map(masks, gate))
    stats = brf.band_stats(masks, gate, win.ori_img)
    assert f"Red-Orange {stats['BRF Red-Orange (% Signal)']:.1f} %" in win.brf_summary_label.text()
    assert "of the signal" in win.brf_summary_label.text()

    win.brf_view_combo.setCurrentText("Green")
    np.testing.assert_array_equal(shown[-1], brf.channel_image(win.ori_img, masks[2]))
    win.brf_original_cb.setChecked(True)
    win._show_brf_view()
    assert shown[-1] is win.ori_img

    # a setting change re-thresholds the stored HSB stack (debounced), without the worker
    win.brf_original_cb.setChecked(False)
    win.brf_view_combo.setCurrentText("Class map")
    win.brf_brightness_slider.setValue(255)
    settle(app, 250)
    assert not shown[-1].any() and "0.00 % of the image" in win.brf_summary_label.text()


def test_bands_never_move_each_other(win):
    red, yellow, green = win.brf_band_sliders
    span = lambda s: (s.lower_value, s.upper_value)
    yellow.setValues(30, 45)                          # shrinking leaves gaps, the others stay
    assert span(red) == (0, 27) and span(yellow) == (30, 45) and span(green) == (48, 140)
    yellow.setValues(20, 60)                          # growing stops at the neighbours' edges
    assert span(red) == (0, 27) and span(yellow) == (28, 47) and span(green) == (48, 140)
    green.setValues(40, 150)
    assert span(yellow) == (28, 47) and span(green) == (48, 150)
    red.setValues(-20, 35)
    assert span(red) == (-20, 27) and span(yellow) == (28, 47)
    params = win._brf_settings
    assert [params[k] for k in ("Red-Orange Min", "Red-Orange Max", "Yellow Min", "Yellow Max",
                                "Green Min", "Green Max")] == [236, 27, 28, 47, 48, 150]
    assert win.brf_band_values[2].text() == "48, 150"


def test_band_rows_are_named_by_band(win):
    labels = [l.text() for l in win.brf_bands_card.findChildren(QLabel) if l.objectName() == "rowLabel"]
    assert {"Red-orange", "Yellow", "Green"} <= set(labels) and not any("hue" in l for l in labels)


def test_red_orange_wraps_below_zero(win):
    win.brf_band_sliders[0].setValues(-26, 27)
    params = win._brf_settings
    assert (params["Red-Orange Min"], params["Red-Orange Max"]) == (230, 27)
    assert win.brf_band_values[0].text() == "230, 27"
    hue = np.array([[229, 230, 255, 0, 27]], np.uint8)
    on = np.full(hue.shape, 255, np.uint8)
    masks, _ = brf.band_masks(hue, on, on, brf.band_ranges(params))
    assert masks[0].tolist() == [[False, True, True, True, True]]

def test_page_settings_stay_out_of_the_parameter_file(win, tmp_path, monkeypatch):
    import yaml
    from PyQt5.QtWidgets import QFileDialog
    win.brf_sort_cb.setChecked(False)
    win.brf_brightness_slider.setValue(30)
    assert win._brf_settings["Sort Into Folders"] is False and win._brf_settings["Brightness Threshold"] == 30
    out = tmp_path / "exported.yml"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(out), "")))
    win.export_parameters()
    exported = yaml.safe_load(open(out))
    assert "Birefringence" not in exported and "Birefringence" not in exported["Configs"]
    # importing or restoring parameters leaves the page alone
    win.restore_default_parameters()
    assert win.brf_brightness_slider.value() == 30


def test_brightfield_image_shows_the_note(win, tmp_path):
    _open(win, tmp_path, np.full((64, 64, 3), 220, np.uint8), name="bf.png")
    assert not win.brf_note.isHidden()
    _open(win, tmp_path, pol_image(), name="pol.png")
    assert win.brf_note.isHidden()


def test_opening_an_image_drops_the_previous_separation(win, tmp_path, monkeypatch):
    _open(win, tmp_path, pol_image())
    _separate(win, monkeypatch)
    assert win._brf_hsb is not None and win.brf_summary_label.text()
    _open(win, tmp_path, pol_image(), name="other.png")
    assert win._brf_hsb is None and win._brf_masks is None and win.brf_summary_label.text() == ""


def test_theme_switch_restyles_the_page(win):
    for w in (win.brf_brightness_value, win.brf_summary_label, *win.brf_band_values):
        w.setStyleSheet("")
    win.brf_btn.setStyleSheet("")
    win._reapply_widget_styles()
    assert win.brf_btn.styleSheet() == win.primary_btn_style
    for w in (win.brf_brightness_value, win.brf_summary_label, *win.brf_band_values):
        assert w.styleSheet() == win.value_label_style


def test_separate_button_runs_the_worker(win, tmp_path, app):
    _open(win, tmp_path, pol_image())
    win.brf_btn.click()
    assert win.brf_btn.text() == "Separating..." and not win.brf_btn.isEnabled()
    for _ in range(100):
        if win._brf_hsb is not None:
            break
        settle(app, 50)
    assert win._brf_hsb is not None and win.brf_btn.text() == "Separate" and win.brf_btn.isEnabled()
    assert "of the signal" in win.brf_summary_label.text()


def _wait(app, condition, steps=200):
    for _ in range(steps):
        if condition():
            return True
        settle(app, 50)
    return condition()


def test_run_folder_separates_a_folder(win, tmp_path, monkeypatch, app):
    import pandas as pd
    from cabana.stages import BRF_RESULTS
    from tests.test_brf_folder import _write_inputs
    img_dir, mask_dir = _write_inputs(tmp_path, names=("sec1", "sec2"))
    # QuPath layout: stroma/ next to stroma_mask/ is found automatically
    stroma = tmp_path / "stroma"
    os.rename(img_dir, stroma)
    os.rename(mask_dir, tmp_path / "stroma_mask")
    assert not win.brf_run_btn.isEnabled()
    win.set_brf_input_folder(str(stroma))
    assert win.brf_mask_folder == str(tmp_path / "stroma_mask") and win.brf_mask_clear_action.isVisible()
    win.set_brf_output_folder(str(tmp_path / "out"))
    assert win.brf_run_btn.isEnabled()
    messages = []
    monkeypatch.setattr(win, "_notify", lambda *a, **k: messages.append((a, k)))
    win.brf_images_cb.setChecked(False)
    win.brf_run_btn.click()
    assert win.brf_run_btn.text() == "Cancel"
    assert _wait(app, lambda: messages)
    (title, text, info), kwargs = messages[-1]
    assert text == "Separated 2 of 2 images." and not kwargs and BRF_RESULTS in info
    results = pd.read_csv(tmp_path / "out" / "Stats" / BRF_RESULTS)
    assert sorted(results["Image"]) == ["sec1.png", "sec2.png"]
    assert results["Region Pixels"].tolist() == [160 * 80, 160 * 80]                  # the masks were used
    assert not (tmp_path / "out" / "Total").exists()                                   # Save Images off
    assert win.brf_run_btn.text() == "Run Folder" and win.brf_run_btn.isEnabled()


def test_run_folder_asks_before_writing_into_a_non_empty_folder(win, tmp_path, monkeypatch):
    from tests.test_brf_folder import _write_inputs
    img_dir, _ = _write_inputs(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    (out / "keep.txt").write_text("x")
    win.set_brf_input_folder(img_dir)
    win.set_brf_output_folder(str(out))
    asked = []
    monkeypatch.setattr(win, "_ask_tma", lambda *a: asked.append(a) or False)
    win.run_brf_folder()
    assert asked and win.brf_folder_worker is None and win.brf_run_btn.text() == "Run Folder"


def test_run_folder_reports_an_output_folder_holding_the_input(win, tmp_path, monkeypatch, app):
    from tests.test_brf_folder import _write_inputs
    img_dir, _ = _write_inputs(tmp_path)
    win.set_brf_input_folder(img_dir)
    win.set_brf_output_folder(str(tmp_path))                     # the input's parent
    messages = []
    monkeypatch.setattr(win, "_ask_tma", lambda *a: True)
    monkeypatch.setattr(win, "_notify", lambda *a, **k: messages.append((a, k)))
    win.run_brf_folder()
    assert _wait(app, lambda: messages)
    assert messages[-1][1] == {"warning": True} and "output folder" in messages[-1][0][2]
    assert win.brf_run_btn.text() == "Run Folder"


def test_empty_roi_masks_means_whole_images(win, tmp_path, monkeypatch, app):
    import pandas as pd
    from cabana.stages import BRF_RESULTS
    from tests.test_brf_folder import _write_inputs
    img_dir, mask_dir = _write_inputs(tmp_path)
    win.set_brf_input_folder(img_dir)
    assert win.brf_mask_folder == mask_dir                       # Masks/ next to Images/ is picked up
    assert win.brf_mask_clear_action.isVisible()
    win.brf_mask_clear_action.trigger()                          # the clear icon in the field
    assert win.brf_mask_folder is None and win.brf_mask_path.text() == ""
    assert not win.brf_mask_clear_action.isVisible()
    win.set_brf_output_folder(str(tmp_path / "out"))
    messages = []
    monkeypatch.setattr(win, "_notify", lambda *a, **k: messages.append(a))
    win.run_brf_folder()
    assert _wait(app, lambda: messages)
    results = pd.read_csv(tmp_path / "out" / "Stats" / BRF_RESULTS)
    assert results.loc[0, "Region Pixels"] == 160 * 160                      # whole image


def test_page_has_two_titled_sections(win):
    assert [t.text() for t in win.brf_section_titles] == ["Bands", "Folder run"]
    # card titles, styled by the panel stylesheet (restyled with the theme)
    assert all(t.objectName() == "sectionTitle" for t in win.brf_section_titles)
    assert "QLabel#sectionTitle" in win.dock_contents.styleSheet()


def test_batch_run_mask_folder_is_cleared_in_the_field(win):
    win.set_mask_folder("/some/masks")
    assert win.mask_clear_action.isVisible()
    win.mask_clear_action.trigger()
    assert win.mask_folder is None and win.mask_folder_path.text() == "" and not win.mask_clear_action.isVisible()
