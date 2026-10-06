"""Tests for cabana/tma_editor.py and the TMA page's editing wiring (offscreen Qt)."""

import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytest.importorskip("PyQt5")
import cv2  # noqa: E402
from PyQt5.QtCore import QEventLoop, QPoint, Qt, QTimer  # noqa: E402
from PyQt5.QtTest import QTest  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402

from cabana.tma import EDITS_FILE, TMAPreprocessor, _transform_grid  # noqa: E402
from cabana.tma_maps import occupancy_grid  # noqa: E402
from tests.test_tma import CORE_UM, PX_UM, synthetic_slide  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def settle(app, ms=50):
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec_()
    app.processEvents()


@pytest.fixture
def fitted_gui(app, tmp_path):
    """MainWindow on the TMA page with a fitted synthetic array-1 slide."""
    from cabana.cabana_gui import MainWindow
    occ = _transform_grid(occupancy_grid(1), "90").copy()
    occ[0, 1] = False
    occ[3, 5] = False
    img, _ = synthetic_slide(occ, jitter=5)
    slide = str(tmp_path / "TMA1.png")
    cv2.imwrite(slide, img)
    win = MainWindow()
    win.resize(1400, 900)
    win.show()
    settle(app, 100)
    win.show_page(win.pages.indexOf(win.tma_tab))
    win.tma_slide = slide
    win.tma_output = str(tmp_path / "out")
    win.tma_array_combo.setCurrentIndex(win.tma_array_combo.findData(1))
    win.tma_orientation_combo.setCurrentIndex(win.tma_orientation_combo.findData("90"))
    pre = TMAPreprocessor(slide, array_number=1, slide_name="TMA1", pixel_size_um=PX_UM,
                          core_diameter_um=CORE_UM, fit_pixel_size_um=PX_UM, orientation="90")
    pre.fit()
    pre.map_to_array()
    win.handle_tma_fit_complete(pre)
    settle(app, 100)
    yield win, pre
    pre.close()
    win.close()


def _widget_point(win, pre, x0, y0):
    ds = pre.reader.level_downsample(pre._fit_level)
    wx, wy = win.image_panel.image_to_widget(x0 / ds, y0 / ds)
    return QPoint(int(wx), int(wy))


class TestOverlayAndRemap:
    def test_overlay_installed_after_fit(self, fitted_gui):
        win, pre = fitted_gui
        panel = win.image_panel
        assert panel.overlay_painter is not None and panel.edit_handler is win.tma_editor
        assert "2 map positions without a core" in win.tma_status_label.text()
        assert not panel.grab().isNull()

    def test_orientation_change_relabels_without_refit(self, fitted_gui, app):
        win, pre = fitted_gui
        win.tma_orientation_combo.setCurrentIndex(win.tma_orientation_combo.findData("270"))
        settle(app)
        assert win.tma_pre is pre and pre.matched_orientation == "270"
        assert "set manually" in win.tma_status_label.text()
        assert win.image_panel.overlay_painter is not None    # overlay survives the remap

    def test_other_page_image_clears_overlay(self, fitted_gui):
        win, pre = fitted_gui
        import numpy as np
        win.image_panel.setImage(np.zeros((10, 10, 3), dtype=np.uint8))
        assert win.image_panel.overlay_painter is None and win.image_panel.edit_handler is None


class TestEditingGestures:
    def test_drag_moves_core_into_empty_cell(self, fitted_gui, app):
        win, pre = fitted_gui
        panel = win.image_panel
        win.tma_edit_cb.setChecked(True)
        settle(app)
        assert win.tma_editor.enabled
        core = next(c for c in pre.cores if (c.row, c.col) == (0, 0))
        old_pos = pre.core_info(core).position
        tx, ty = pre.predict_centre(0, 1)
        QTest.mousePress(panel, Qt.LeftButton, Qt.NoModifier, _widget_point(win, pre, core.cx, core.cy))
        assert win.tma_editor.selected is core and win.tma_editor._drag[0] == "move"
        QTest.mouseMove(panel, _widget_point(win, pre, (core.cx + tx) / 2, (core.cy + ty) / 2))
        QTest.mouseRelease(panel, Qt.LeftButton, Qt.NoModifier, _widget_point(win, pre, tx, ty))
        settle(app)
        assert (core.row, core.col) == (0, 1) and core.manual
        assert pre.core_info(core).position != old_pos
        assert {(r, c) for _, r, c, _ in pre.missing_positions()} == {(0, 0), (3, 5)}
        assert "hand-edited" in win.tma_status_label.text()
        assert win.tma_undo_btn.isEnabled()
        win.tma_undo_btn.click()
        settle(app)
        core = next(c for c in pre.cores if pre.core_info(c).position == old_pos)
        assert (core.row, core.col) == (0, 0) and not core.manual

    def test_rim_drag_resizes_and_double_click_toggles(self, fitted_gui, app):
        win, pre = fitted_gui
        panel = win.image_panel
        win.tma_edit_cb.setChecked(True)
        settle(app)
        core = next(c for c in pre.cores if not c.excluded and pre.core_info(c).patient_id)
        r0 = core.radius
        QTest.mousePress(panel, Qt.LeftButton, Qt.NoModifier, _widget_point(win, pre, core.cx + r0, core.cy))
        assert win.tma_editor._drag == ("resize",)
        QTest.mouseRelease(panel, Qt.LeftButton, Qt.NoModifier, _widget_point(win, pre, core.cx + 0.6 * r0, core.cy))
        settle(app)
        assert abs(core.radius - 0.6 * r0) < 0.1 * r0 and core.excluded and core.reason == "diameter"
        QTest.mouseDClick(panel, Qt.LeftButton, Qt.NoModifier, _widget_point(win, pre, core.cx, core.cy))
        settle(app)
        assert not core.excluded and core.override == "include"

    def test_double_click_placeholder_adds_delete_removes(self, fitted_gui, app):
        win, pre = fitted_gui
        panel = win.image_panel
        win.tma_edit_cb.setChecked(True)
        settle(app)
        n = len(pre.cores)
        x, y = pre.predict_centre(3, 5)
        QTest.mouseDClick(panel, Qt.LeftButton, Qt.NoModifier, _widget_point(win, pre, x, y))
        settle(app)
        added = win.tma_editor.selected
        assert added is not None and added.manual and (added.row, added.col) == (3, 5)
        assert len(pre.cores) == n + 1
        QTest.keyClick(panel, Qt.Key_Delete)
        settle(app)
        assert len(pre.cores) == n and win.tma_editor.selected is None
        QTest.keyClick(panel, Qt.Key_Z, Qt.ControlModifier)
        settle(app)
        assert len(pre.cores) == n + 1

    def test_editing_disabled_lets_clicks_pan(self, fitted_gui, app):
        win, pre = fitted_gui
        panel = win.image_panel
        assert not win.tma_editor.enabled
        core = pre.cores[0]
        before = (core.cx, core.cy)
        QTest.mousePress(panel, Qt.LeftButton, Qt.NoModifier, _widget_point(win, pre, core.cx, core.cy))
        assert panel.panning and win.tma_editor.selected is None
        QTest.mouseRelease(panel, Qt.LeftButton, Qt.NoModifier, _widget_point(win, pre, core.cx + 50, core.cy))
        assert (core.cx, core.cy) == before


class TestEditPersistence:
    def test_export_saves_edits_and_load_reinstates(self, fitted_gui, app, tmp_path):
        win, pre = fitted_gui
        x, y = pre.predict_centre(0, 1)
        added = pre.add_core(x, y)
        pre.set_override(added, "include")
        os.makedirs(win.tma_output, exist_ok=True)
        pre.export(win.tma_output, channels=["BF"])
        path = os.path.join(win.tma_output, EDITS_FILE)
        assert os.path.isfile(path)
        pre2 = TMAPreprocessor(win.tma_slide, array_number=1, slide_name="TMA1", pixel_size_um=PX_UM,
                               core_diameter_um=CORE_UM, fit_pixel_size_um=PX_UM, orientation="90")
        pre2.fit()
        pre2.map_to_array()
        win.tma_pre = pre2
        win.load_tma_edits(path)
        settle(app)
        assert len(pre2.cores) == len(pre.cores) and pre2.manual_count() == 1
        assert win.tma_status_label.text().startswith("Reinstated")
        assert {(r, c) for _, r, c, _ in pre2.missing_positions()} == {(3, 5)}
        pre2.close()
