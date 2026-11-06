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
    assert win.seg_max_size_spinner.value() == 5000 and win.yml_data["Configs"]["Max Size"] == 5000
    assert win.batch_max_size_label.text() == "–"           # until a parameter file is chosen
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


def test_image_folder_can_be_chosen_before_the_parameter_file(win, tmp_path, monkeypatch):
    import cabana.cabana_gui as gui
    starts = []

    def pick(parent, title, start, *a):
        starts.append(start)
        return str(tmp_path)
    monkeypatch.setattr(gui.QFileDialog, "getExistingDirectory", pick)
    assert not hasattr(win, "param_file")
    win.input_btn.trigger()                                  # the folder icon in the Images field
    assert starts == [os.path.expanduser("~")] and win.input_folder == str(tmp_path)
    assert win.input_folder_path.text() == str(tmp_path)


def test_batch_max_size_line_follows_the_parameter_file(win, tmp_path):
    import yaml
    new, old = tmp_path / "new.yml", tmp_path / "old.yml"
    yaml.safe_dump({"Configs": {"Max Size": 0}, "Segmentation": {}}, open(new, "w"))
    yaml.safe_dump({"Configs": {}, "Segmentation": {"Max Size": 3000}}, open(old, "w"))
    win._show_param_file_max_size(str(new))
    assert win.batch_max_size_label.text() == "No limit"
    win._show_param_file_max_size(str(old))
    assert win.batch_max_size_label.text() == "3000 px"
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
    assert win.seg_max_size_spinner.value() == 2048


def test_segmentation_max_size_is_exported(win, tmp_path, monkeypatch):
    import yaml
    from PyQt5.QtWidgets import QFileDialog
    win.seg_max_size_spinner.setValue(8000)
    assert win.yml_data["Configs"]["Max Size"] == 8000
    out = tmp_path / "exported.yml"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(out), ""))
    win.export_parameters()
    assert yaml.safe_load(open(out))["Configs"]["Max Size"] == 8000
    win.restore_default_parameters()
    assert win.seg_max_size_spinner.value() == 5000


def test_size_note_explains_splitting_and_patch_size(win):
    win.seg_max_size_spinner.setValue(5000)
    win.patch_size_spinner.setValue(1024)
    win.ori_img = np.zeros((10000, 7000, 3), np.uint8)
    win._update_image_status()
    assert win.seg_size_note.text() == ("Batch runs split this image (7000 × 10000 px) into 4 blocks of "
                                        "3500 × 5000 px; the preview here segments it whole.")
    win.patch_size_spinner.setValue(8192)
    assert "Patch Size has no effect on split images: their blocks are at most 5000 px." in win.seg_size_note.text()
    win.ori_img = None
    win.patch_size_spinner.setValue(1024)
    win._update_image_status()
    assert win.seg_size_note.text() == "" and win.seg_size_note.isHidden()


def test_wheel_over_an_unclicked_box_does_not_change_it(win, app):
    from PyQt5.QtCore import QPoint, QPointF, Qt
    from PyQt5.QtGui import QWheelEvent
    from PyQt5.QtWidgets import QApplication
    box = win.seg_max_size_spinner
    box.setValue(5000)
    assert box.focusPolicy() == Qt.StrongFocus
    centre = QPointF(box.rect().center())
    event = QWheelEvent(centre, QPointF(box.mapToGlobal(QPoint(5, 5))), QPoint(0, 0), QPoint(0, 120),
                        Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False)
    QApplication.sendEvent(box, event)
    assert box.value() == 5000
    box.removeEventFilter(win._wheel_guard)          # the same wheel without the guard changes it
    QApplication.sendEvent(box, event)
    assert box.value() == 6000


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
    box.setSingleStep(1000)               # steps land on round thousands
    box.stepBy(1)
    assert box.value() == 3000
    box.setValue(9096)
    box.stepBy(-1)
    assert box.value() == 9000
    box.stepBy(1)
    assert box.value() == 10000
    box.setValue(2500)
    box.stepBy(-1)
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


def test_new_batch_run_asks_before_clearing_the_results_folder(win, tmp_path, monkeypatch):
    import cabana.cabana_gui as gui

    class _Signal:
        def connect(self, *a):
            pass

    class _Worker:                                     # records the start instead of running
        started = []

        def __init__(self, *a, **k):
            self.progress_updated = self.status_updated = _Signal()
            self.batch_complete = self.batch_cancelled = self.batch_failed = _Signal()

        def start(self):
            _Worker.started.append(1)
    monkeypatch.setattr(gui, "BatchProcessingWorker", _Worker)
    monkeypatch.setattr(win, "_check_batch_running_status", lambda: (False, 5, 0, False, {}))
    asked = []
    monkeypatch.setattr(win, "_ask_tma", lambda *a: asked.append(a) or False)
    out = tmp_path / "out"
    (out / "Logs").mkdir(parents=True)
    win.param_file, win.input_folder, win.output_folder = str(tmp_path / "p.yml"), str(tmp_path), str(out)
    win.run_batch_processing()                         # only Logs: nothing to lose, no question
    assert asked == [] and _Worker.started == [1]
    (out / "QuantificationResults.csv").write_text("x")
    win.run_batch_processing()                         # results there: asked, and Cancel keeps them
    assert len(asked) == 1 and asked[0][1] == "The results folder is not empty."
    assert _Worker.started == [1] and (out / "QuantificationResults.csv").exists()
    assert not win.process_batch_btn.isHidden() and win.cancel_batch_btn.isHidden() and win.input_btn.isEnabled()
