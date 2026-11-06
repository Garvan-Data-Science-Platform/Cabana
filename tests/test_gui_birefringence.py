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
from PyQt5.QtWidgets import QApplication  # noqa: E402

from cabana import brf  # noqa: E402
from tests.test_brf_pipeline import pol_image  # noqa: E402


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


def test_page_sits_after_segmentation_and_is_left_out_of_batch_run(win):
    titles = win._page_titles
    assert titles.index("Birefringence") == titles.index("Segmentation") + 1
    assert win.pages.indexOf(win.brf_tab) == titles.index("Birefringence")
    assert not win.toggle_brf_btn.isChecked() and not win.brf_btn.isEnabled()      # no image yet
    assert "Include in Batch Run" in win.toggle_brf_label.text() and "No" in win.toggle_brf_label.text()
    assert win.brf_brightness_slider.isEnabled() and win.brf_view_combo.currentText() == "Grid"
    assert win.yml_data["Configs"]["Birefringence"] is False
    assert win.yml_data["Birefringence"] == brf.DEFAULTS


def test_toggle_only_decides_whether_batch_run_includes_the_step(win, tmp_path):
    win.toggle_brf_btn.setChecked(True)
    assert win.yml_data["Configs"]["Birefringence"] is True and "Yes" in win.toggle_brf_label.text()
    _open(win, tmp_path, pol_image())
    assert win.brf_btn.isEnabled() and not win.brf_note.isVisible()
    win.toggle_brf_btn.setChecked(False)
    assert win.yml_data["Configs"]["Birefringence"] is False and "No" in win.toggle_brf_label.text()
    assert win.brf_btn.isEnabled() and win.brf_brightness_slider.isEnabled()


def test_separate_shows_the_class_map_and_a_summary(win, tmp_path, monkeypatch, app):
    win.toggle_brf_btn.setChecked(True)
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


def test_bands_stay_joined(win):
    win.toggle_brf_btn.setChecked(True)
    red, yellow, green = win.brf_band_sliders
    red.setValues(0, 35)
    assert (yellow.lower_value, yellow.upper_value) == (36, 47)
    yellow.setValues(20, 60)
    assert (red.lower_value, red.upper_value) == (0, 19)
    assert (green.lower_value, green.upper_value) == (61, 140)
    params = win.yml_data["Birefringence"]
    assert [params[k] for k in ("Red-Orange Max", "Yellow Min", "Yellow Max", "Green Min")] == [19, 20, 60, 61]
    assert win.brf_band_values[1].text() == "(20, 60)"


def test_red_orange_wraps_below_zero(win, tmp_path):
    win.toggle_brf_btn.setChecked(True)
    win.brf_band_sliders[0].setValues(-26, 27)
    params = win.yml_data["Birefringence"]
    assert (params["Red-Orange Min"], params["Red-Orange Max"]) == (230, 27)
    assert win.brf_band_values[0].text() == "(230, 27)"
    # a parameter file with a wrapping range sets the slider below zero
    data = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "cabana", "default_params.yml")))
    data["Birefringence"].update({"Red-Orange Min": 240, "Brightness Threshold": 20, "Saturation Min": 3,
                                  "Sort Into Folders": False})
    data["Configs"]["Birefringence"] = True
    win.yml_data = data
    win.apply_params_to_widgets()
    assert win.brf_band_sliders[0].lower_value == -16 and win.brf_band_values[0].text() == "(240, 27)"
    assert win.brf_brightness_slider.value() == 20 and win.brf_saturation_slider.value() == 3
    assert not win.brf_sort_cb.isChecked() and win.toggle_brf_btn.isChecked()
    assert win.yml_data["Birefringence"]["Red-Orange Min"] == 240


def test_old_parameter_file_without_the_section_keeps_brf_off(win):
    data = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "cabana", "default_params.yml")))
    del data["Birefringence"]
    del data["Configs"]["Birefringence"]
    win.toggle_brf_btn.setChecked(True)
    win.yml_data = data
    win.apply_params_to_widgets()
    assert not win.toggle_brf_btn.isChecked()
    assert brf.settings(win.yml_data) == brf.DEFAULTS


def test_sort_checkbox_writes_the_parameter(win):
    win.toggle_brf_btn.setChecked(True)
    win.brf_sort_cb.setChecked(False)
    assert win.yml_data["Birefringence"]["Sort Into Folders"] is False


def test_brightfield_image_shows_the_note(win, tmp_path):
    _open(win, tmp_path, np.full((64, 64, 3), 220, np.uint8), name="bf.png")
    assert not win.brf_note.isHidden()
    _open(win, tmp_path, pol_image(), name="pol.png")
    assert win.brf_note.isHidden()


def test_opening_an_image_drops_the_previous_separation(win, tmp_path, monkeypatch):
    win.toggle_brf_btn.setChecked(True)
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
    win.toggle_brf_btn.setChecked(True)
    _open(win, tmp_path, pol_image())
    win.brf_btn.click()
    assert win.brf_btn.text() == "Separating..." and not win.brf_btn.isEnabled()
    for _ in range(100):
        if win._brf_hsb is not None:
            break
        settle(app, 50)
    assert win._brf_hsb is not None and win.brf_btn.text() == "Separate" and win.brf_btn.isEnabled()
    assert "of the signal" in win.brf_summary_label.text()


def test_combine_bf_and_pol_results_from_the_file_menu(win, tmp_path, monkeypatch):
    import pandas as pd
    bf, pol, out = tmp_path / "bf.csv", tmp_path / "pol.csv", tmp_path / "combined.csv"
    pd.DataFrame({"Image": ["A.vsi_BF_S01_roi.png", "A.vsi_BF_S02_roi.png"], "X": [1, 2]}).to_csv(bf, index=False)
    pd.DataFrame({"Image": ["A.vsi_POL_S01_roi.png"], "BRF Total (% Region)": [50.0]}).to_csv(pol, index=False)
    messages = []
    monkeypatch.setattr(win, "_ask_combine_paths", lambda: (str(bf), str(pol), str(out)))
    monkeypatch.setattr(win, "_notify", lambda *a, **k: messages.append((a, k)))
    assert "Combine BF and POL" in win.combine_results_action.text()
    win.combine_results_action.trigger()
    combined = pd.read_csv(out)
    assert combined["POL: BRF Total (% Region)"].tolist()[0] == 50.0 and len(combined) == 2
    (title, text, info), kwargs = messages[-1]
    assert text.startswith("1 sample has") and "1 BF and 0 POL" in info and not kwargs

    pd.DataFrame({"Image": ["no_token_roi.png"], "X": [1]}).to_csv(bf, index=False)
    win.combine_results_action.trigger()
    assert messages[-1][1] == {"warning": True} and "no BF/POL" in messages[-1][0][2]


def _wait(app, condition, steps=200):
    for _ in range(steps):
        if condition():
            return True
        settle(app, 50)
    return condition()


def test_run_folder_separates_a_folder(win, tmp_path, monkeypatch, app):
    import pandas as pd
    from cabana.stages import BRF_RESULTS
    from tests.test_brf_pipeline import _write_inputs
    img_dir, mask_dir = _write_inputs(tmp_path, names=("sec1", "sec2"))
    # QuPath layout: stroma/ next to stroma_mask/ is found automatically
    stroma = tmp_path / "stroma"
    os.rename(img_dir, stroma)
    os.rename(mask_dir, tmp_path / "stroma_mask")
    assert not win.brf_run_btn.isEnabled()
    win.set_brf_input_folder(str(stroma))
    assert win.brf_mask_folder == str(tmp_path / "stroma_mask") and win.brf_mask_clear_btn.isEnabled()
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
    from tests.test_brf_pipeline import _write_inputs
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
    from tests.test_brf_pipeline import _write_inputs
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
