"""Tests for the Quantification parameters shown in the GUI (offscreen Qt)."""

import os
import sys

import pytest
import yaml

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytest.importorskip("PyQt5")
from PyQt5.QtCore import QEventLoop, QTimer  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402


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


def test_quantification_controls_show_the_defaults(win):
    q = win.yml_data["Quantification"]
    assert win.min_branch_slider.value() == q["Minimum Branch Length"] == 5
    assert (win.curve_window_range.lower_value, win.curve_window_range.upper_value) == (10, 30)
    assert win.curve_step_slider.value() == q["Curvature Window Step"] == 10
    assert win.hdm_saturation_slider.value() == 10 and q["Contrast Enhancement"] == 0.1


def test_quantification_controls_write_the_parameters(win):
    from cabana.stages import curve_windows_from_args
    win.min_branch_slider.setValue(12)
    win.curve_window_range.setValues(8, 40)
    win.curve_step_slider.setValue(16)
    win.hdm_saturation_slider.setValue(25)
    q = win.yml_data["Quantification"]
    assert q["Minimum Branch Length"] == 12
    assert (q["Minimum Curvature Window"], q["Maximum Curvature Window"], q["Curvature Window Step"]) == (8, 40, 16)
    assert q["Contrast Enhancement"] == 0.25
    assert win.curve_window_value.text() == "(8, 40)" and win.hdm_saturation_value.text() == "25"
    assert list(curve_windows_from_args(win.yml_data)) == [8, 24, 40]


def test_loaded_parameter_file_sets_the_controls(win, tmp_path):
    data = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "cabana", "default_params.yml")))
    data["Quantification"].update({"Minimum Branch Length": 7, "Minimum Curvature Window": 20,
                                   "Maximum Curvature Window": 60, "Curvature Window Step": 20,
                                   "Contrast Enhancement": 0.125})
    win.yml_data = data
    win.apply_params_to_widgets()
    assert win.min_branch_slider.value() == 7
    assert (win.curve_window_range.lower_value, win.curve_window_range.upper_value) == (20, 60)
    assert win.curve_step_slider.value() == 20
    assert win.hdm_saturation_slider.value() == 12
    # the slider shows whole percents; a finer value from the file is kept
    assert win.yml_data["Quantification"]["Contrast Enhancement"] == 0.125
    assert win.yml_data["Quantification"]["Minimum Curvature Window"] == 20


def test_restore_defaults_resets_the_controls(win):
    win.min_branch_slider.setValue(30)
    win.hdm_saturation_slider.setValue(40)
    win.restore_default_parameters()
    assert win.min_branch_slider.value() == 5 and win.hdm_saturation_slider.value() == 10
    assert win.yml_data["Quantification"]["Contrast Enhancement"] == 0.1


def test_theme_switch_restyles_the_new_value_labels(win):
    for label in (win.min_branch_value, win.curve_window_value, win.curve_step_value,
                  win.hdm_saturation_value):
        label.setStyleSheet("")
    win._reapply_widget_styles()
    for label in (win.min_branch_value, win.curve_window_value, win.curve_step_value,
                  win.hdm_saturation_value):
        assert label.styleSheet() == win.value_label_style
