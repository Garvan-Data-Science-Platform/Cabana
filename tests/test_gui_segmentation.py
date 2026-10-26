"""Tests for the Segmentation page of cabana/cabana_gui.py (offscreen Qt)."""

import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytest.importorskip("PyQt5")
import numpy as np  # noqa: E402
from PyQt5.QtCore import QEventLoop, QTimer  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402

from tests.test_segmenter import seg_args, sparse_and_dense_image  # noqa: E402


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


def _segment(win, monkeypatch):
    """Run the segmentation worker synchronously on a small test image."""
    import cabana.ui as ui
    from cabana.ui import SegmentationWorker
    real = ui.segment_image
    monkeypatch.setattr(ui, "segment_image", lambda img, args, **k: real(img, args, cnn_size=64, **k))
    win.ori_img = np.ascontiguousarray(sparse_and_dense_image()[:, :, ::-1])
    worker = SegmentationWorker(win.ori_img, seg_args(patch_size=64))
    worker.segmentation_complete.connect(win.handle_segmentation_complete)
    worker._run()


def test_defaults(win):
    assert win.patch_size_spinner.value() == 1024 and win.color_thresh_slider.value() == 20
    assert win.batch_max_size_spinner.value() == 5000 and win.yml_data["Configs"]["Max Size"] == 5000
    assert "Max Size" not in win.yml_data["Segmentation"]


def test_threshold_slider_rethresholds_without_segmenting(win, app, monkeypatch):
    _segment(win, monkeypatch)
    assert win._seg_score is not None
    before = win.seg_img.copy()
    import cabana.ui as ui
    monkeypatch.setattr(ui.SegmentationWorker, "start", lambda self: pytest.fail("segmented again"))
    win.color_thresh_slider.setValue(1)             # 0.01: the glass now counts as well
    settle(app, 300)
    assert not np.array_equal(win.seg_img, before)
    win.color_thresh_slider.setValue(20)
    settle(app, 300)
    assert np.array_equal(win.seg_img, before)
    # a setting that changes the training drops the stored scores
    win.num_labels_slider.setValue(win.num_labels_slider.value() + 1)
    assert win._seg_score is None


def test_batch_max_size_box_follows_the_parameter_file(win, tmp_path):
    import yaml
    new, old = tmp_path / "new.yml", tmp_path / "old.yml"
    yaml.safe_dump({"Configs": {"Max Size": 0}, "Segmentation": {}}, open(new, "w"))
    yaml.safe_dump({"Configs": {}, "Segmentation": {"Max Size": 3000}}, open(old, "w"))
    win._show_param_file_max_size(str(new))
    assert win.batch_max_size_spinner.value() == 0 and win.batch_max_size_spinner.text() == "No limit"
    win._show_param_file_max_size(str(old))
    assert win.batch_max_size_spinner.value() == 3000
    assert win.visuals_cb.isChecked()                       # default when the file does not say
    yaml.safe_dump({"Configs": {"Save Visualisations": False}}, open(new, "w"))
    win._show_param_file_max_size(str(new))
    assert not win.visuals_cb.isChecked()


def test_imported_old_file_moves_max_size_to_configs(win, tmp_path, monkeypatch):
    import yaml
    from PyQt5.QtWidgets import QFileDialog
    old = tmp_path / "old.yml"
    data = dict(win.yml_data)
    data["Configs"] = {k: v for k, v in data["Configs"].items() if k != "Max Size"}
    data["Segmentation"] = dict(data["Segmentation"], **{"Max Size": 2048})
    yaml.safe_dump(data, open(old, "w"))
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(old), ""))
    win.import_parameters()
    assert win.yml_data["Configs"]["Max Size"] == 2048 and "Max Size" not in win.yml_data["Segmentation"]


def test_max_size_box_skips_values_below_2048(app):
    from cabana.ui import MaxSizeSpinBox
    box = MaxSizeSpinBox()
    box.setSingleStep(256)
    box.setValue(0)
    box.stepBy(1)
    assert box.value() == 2048            # No limit -> smallest size
    box.stepBy(1)
    assert box.value() == 2304
    box.stepBy(-1)
    box.stepBy(-1)
    assert box.value() == 0               # 2048 -> No limit
    box.setValue(500)
    box.editingFinished.emit()            # typed value below the minimum
    assert box.value() == 2048


def test_gap_results_are_painted_live_with_a_summary(win):
    from cabana.stages import fit_gap_circles
    frb = np.full((200, 200), 255, np.uint8)        # fibres are 0, gaps 255
    frb[100, :] = 0
    win.frb_img = frb
    win.ori_img = np.full((200, 200, 3), 200, np.uint8)
    win.img_path = None
    win._seg_score = None                           # no segmentation: every gap is intra-fibre
    win.handle_gap_analysis_complete(fit_gap_circles(frb, 10))
    assert win.gap_img.shape == (200, 200, 3) and win.gap_img[100, 50, 0] == 70
    assert win.image_panel.overlay_painter == win._paint_gaps
    assert win._gap_intra.all()
    n = len(win.gap_circles)
    text = win.gap_summary_label.text()
    assert text.startswith(f"{n} gaps ({n} intra-fibre)") and "px" in text
    win.overlay_gaps_cb.setChecked(True)            # tissue underneath, circles kept
    assert win.image_panel.overlay_painter == win._paint_gaps
    win.image_panel.grab()                          # paints without errors
    win._clear_gap_results()
    assert win.gap_summary_label.text() == "" and win.gap_circles is None
