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


class TestRefit:
    def test_refit_keeps_reused_reader_open(self, fitted_gui, app, monkeypatch):
        """A second Fit Cores reuses the slide reader, so it must not close it:
        a closed VSI reader fails on the next full-resolution read (export)."""
        win, pre = fitted_gui
        reader = pre.reader
        assert win.tma_reader is reader
        closed = []
        monkeypatch.setattr(reader, "close", lambda: closed.append(True))
        win.run_tma_fit()
        win.tma_worker.wait(60000)
        settle(app, 100)
        assert not closed
        assert win.tma_pre is not pre and win.tma_pre.reader is reader


class TestMapPositionGui:
    def test_hover_hint(self, fitted_gui):
        win, pre = fitted_gui
        ed = win.tma_editor
        win.tma_edit_cb.setChecked(False)
        assert "Tick Edit cores, then right-click" in ed.tooltip(pre.cores[0])
        win.tma_edit_cb.setChecked(True)
        assert "Right-click to set this core's map position" in ed.tooltip(pre.cores[0])
        win.tma_array_combo.setCurrentIndex(win.tma_array_combo.findData(None))
        assert "map position" not in ed.tooltip(pre.cores[0])

    def test_choices_list_controls_first(self, fitted_gui):
        win, pre = fitted_gui
        choices = win.tma_editor.map_position_choices()
        n_ctrl = sum(1 for i in pre.array_map().values() if i.is_control)
        assert all("(" in lab for lab, _ in choices[:n_ctrl])
        assert choices[0][0] == "Liver (A1)"
        assert len(choices) == sum(1 for i in pre.array_map().values() if not i.empty)

    def test_set_and_undo_map_position(self, fitted_gui, app, monkeypatch):
        win, pre = fitted_gui
        ed = win.tma_editor
        win.tma_edit_cb.setChecked(True)
        core = next(c for c in pre.cores if 0 <= c.map_row < 9)
        target = (core.map_row + 1, core.map_col)
        monkeypatch.setattr(ed, "ask_map_position", lambda c: target)
        ed._set_map_position(core)
        settle(app)
        assert (core.map_row, core.map_col) == target and pre.orientation_method == "anchor"
        assert win.tma_orientation_combo.currentData() == pre.orientation == "90"
        assert "map position set by hand" in win.tma_status_label.text()
        assert win.tma_export_btn.isEnabled()
        ed.undo()
        settle(app)
        same = next(c for c in pre.cores if (c.row, c.col) == (core.row, core.col))   # undo restores copies
        assert pre.map_anchor is None and (same.map_row, same.map_col) == (target[0] - 1, target[1])

    def test_auto_in_combo_clears_map_position(self, fitted_gui, app, monkeypatch):
        win, pre = fitted_gui
        ed = win.tma_editor
        core = pre.cores[0]
        monkeypatch.setattr(ed, "ask_map_position", lambda c: (core.map_row, core.map_col))
        ed._set_map_position(core)
        win.tma_orientation_combo.setCurrentIndex(win.tma_orientation_combo.findData("auto"))
        settle(app)
        assert pre.map_anchor is None and pre.orientation == "auto"


class TestEditsFiles:
    """Load Edits… / Save Edits… buttons, the reinstate offer and the export guard."""

    def test_buttons_enabled_after_fit(self, fitted_gui):
        win, _ = fitted_gui
        assert win.tma_load_edits_btn.isEnabled() and win.tma_save_edits_btn.isEnabled()

    def test_save_then_load(self, fitted_gui, app, monkeypatch, tmp_path):
        from PyQt5.QtWidgets import QFileDialog
        win, pre = fitted_gui
        path = str(tmp_path / "keep" / "mine.json")
        core = pre.cores[0]
        pre.move_core(core, core.cx + 10, core.cy)
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (path, ""))
        win.save_tma_edits()
        assert os.path.isfile(path) and win.tma_status_label.text().startswith("Saved")
        n = len(pre.cores)
        pre.remove_core(pre.cores[-1])
        monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (path, ""))
        win.open_tma_edits()
        settle(app)
        assert len(pre.cores) == n and pre.manual_count() == 1
        assert win.tma_status_label.text().startswith("Reinstated")

    def test_load_other_slide_asks(self, fitted_gui, monkeypatch, tmp_path):
        import json
        from PyQt5.QtWidgets import QFileDialog
        win, pre = fitted_gui
        path = str(tmp_path / "other.json")
        pre.save_edits(path)
        with open(path) as f:
            data = json.load(f)
        data["slide"] = "TMA32"
        with open(path, "w") as f:
            json.dump(data, f)
        asked = []
        monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (path, ""))
        monkeypatch.setattr(win, "_ask_tma", lambda *a: asked.append(a) or False)
        loaded = []
        monkeypatch.setattr(win, "load_tma_edits", lambda p: loaded.append(p))
        win.open_tma_edits()
        assert len(asked) == 1 and "TMA32" in asked[0][1] and not loaded
        monkeypatch.setattr(win, "_ask_tma", lambda *a: True)
        win.open_tma_edits()
        assert loaded == [path]

    def test_offer_includes_map_position_only(self, fitted_gui, app, monkeypatch):
        win, pre = fitted_gui
        core = next(c for c in pre.cores if 0 <= c.map_row < 9)
        pre.set_map_position(core, core.map_row + 1, core.map_col)
        assert pre.manual_count() == 0
        os.makedirs(win.tma_output, exist_ok=True)
        pre.save_edits(os.path.join(win.tma_output, EDITS_FILE))
        anchor = dict(pre.map_anchor)
        pre.clear_map_position()
        asked = []
        monkeypatch.setattr(win, "_ask_tma", lambda *a: asked.append(a) or True)
        win._offer_saved_edits()
        settle(app)
        assert len(asked) == 1 and "map position set by hand" in asked[0][1]
        assert pre.map_anchor == anchor and pre.orientation_method == "anchor"

    def test_export_guard(self, fitted_gui, monkeypatch):
        win, pre = fitted_gui
        os.makedirs(win.tma_output, exist_ok=True)
        path = os.path.join(win.tma_output, EDITS_FILE)
        before = pre.snapshot()
        core = pre.cores[0]
        pre.move_core(core, core.cx + 10, core.cy)
        pre.save_edits(path)
        asked = []
        monkeypatch.setattr(win, "_ask_tma", lambda *a: asked.append(a) or False)
        assert win._confirm_edits_overwrite() and not asked          # screen holds the edits: no question
        pre.restore(before)                                          # e.g. a refit whose offer was declined
        assert pre.manual_count() == 0
        assert not win._confirm_edits_overwrite() and len(asked) == 1
        assert "1 hand-edited core" in asked[0][1]
        os.remove(path)
        assert win._confirm_edits_overwrite() and len(asked) == 1    # nothing to lose


class _FakeSettings:
    """Stand-in for QSettings so tests never touch the user's real settings."""
    store = {}

    def __init__(self, *a):
        pass

    def value(self, key, default=None):
        return self.store.get(key, default)

    def setValue(self, key, value):
        self.store[key] = value


class TestOutputFolder:
    @pytest.fixture
    def win(self, app, monkeypatch):
        import cabana.cabana_gui as gui
        from cabana.cabana_gui import MainWindow
        monkeypatch.setattr(gui, "QSettings", _FakeSettings)
        monkeypatch.setattr(_FakeSettings, "store", {})
        w = MainWindow()
        w.show_page(w.pages.indexOf(w.tma_tab))
        yield w
        if w.tma_worker is not None:
            w.tma_worker.wait(30000)
        w.close()

    @staticmethod
    def _slide(folder, name):
        os.makedirs(folder, exist_ok=True)
        img, _ = synthetic_slide(occupancy_grid(1)[:2, :2], jitter=0)
        path = os.path.join(folder, name)
        cv2.imwrite(path, img)
        return path

    def _open(self, win, app, path):
        win.open_tma_slide(path)
        win.tma_worker.wait(30000)
        settle(app)

    def test_output_follows_each_slide(self, win, app, tmp_path):
        a = self._slide(str(tmp_path / "scans"), "APGI TMA 2.png")
        b = self._slide(str(tmp_path / "scans"), "APGI TMA 3.png")
        self._open(win, app, a)
        assert win.tma_output == os.path.join(str(tmp_path / "scans"), "APGI_TMA_2_cores")
        self._open(win, app, b)
        assert win.tma_output == os.path.join(str(tmp_path / "scans"), "APGI_TMA_3_cores")
        assert win.tma_output_path.text() == win.tma_output

    def test_chosen_parent_is_remembered(self, win, app, tmp_path, monkeypatch):
        from PyQt5.QtWidgets import QFileDialog
        a = self._slide(str(tmp_path / "scans"), "TMA2.png")
        b = self._slide(str(tmp_path / "scans"), "TMA3.png")
        chosen = tmp_path / "analysis" / "tma2_out"
        chosen.mkdir(parents=True)
        self._open(win, app, a)
        monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **k: str(chosen))
        win.select_tma_output()
        assert win.tma_output == str(chosen)
        self._open(win, app, b)
        assert win.tma_output == os.path.join(str(tmp_path / "analysis"), "TMA3_cores")

    def test_missing_parent_falls_back_to_slide_folder(self, win, app, tmp_path):
        _FakeSettings.store[win.TMA_OUTPUT_PARENT_KEY] = str(tmp_path / "gone")
        a = self._slide(str(tmp_path / "scans"), "TMA5.png")
        self._open(win, app, a)
        assert win.tma_output == os.path.join(str(tmp_path / "scans"), "TMA5_cores")


class TestExportIntoOtherSlidesFolder:
    def test_asks_before_mixing_slides(self, fitted_gui, monkeypatch):
        import json
        win, pre = fitted_gui
        os.makedirs(win.tma_output, exist_ok=True)
        path = os.path.join(win.tma_output, EDITS_FILE)
        pre.save_edits(path)
        asked = []
        monkeypatch.setattr(win, "_ask_tma", lambda *a: asked.append(a) or False)
        assert win._confirm_output_slide() and not asked             # same slide: no question
        with open(path) as f:
            data = json.load(f)
        data["slide"] = "TMA3"
        with open(path, "w") as f:
            json.dump(data, f)
        assert not win._confirm_output_slide() and len(asked) == 1
        assert "TMA3" in asked[0][1]


class TestMapLoader:
    MAPS = "map,TMA1\n,1,2,3\nA,P1,P2,ctrl:Liver\nB,P3,?,P4\n\nmap,TMA9\n,1,2\nA,P5,P6\nB,P7,P8\nC,P9,\n"

    def _file(self, tmp_path):
        path = tmp_path / "maps.csv"
        path.write_text(self.MAPS)
        return str(path)

    def _choose_load(self, win):
        combo = win.tma_array_combo
        combo.setCurrentIndex(combo.findData(win.MAP_LOAD_ITEM))

    def test_load_item_is_last(self, fitted_gui):
        win, _ = fitted_gui
        combo = win.tma_array_combo
        assert combo.itemData(combo.count() - 1) == win.MAP_LOAD_ITEM

    def test_load_select_and_relabel(self, fitted_gui, app, tmp_path, monkeypatch):
        from PyQt5.QtWidgets import QFileDialog
        win, pre = fitted_gui
        path = self._file(tmp_path)
        seen = {}
        monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (path, ""))
        monkeypatch.setattr(win, "_preview_map_file",
                            lambda mf, pre_sel: seen.update(names=[m.name for m in mf.maps], pre=pre_sel) or pre_sel)
        self._choose_load(win)
        settle(app)
        assert seen["names"] == ["TMA1", "TMA9"] and seen["pre"].name == "TMA1"   # slide is TMA1.png
        combo = win.tma_array_combo
        assert combo.currentText() == "TMA1 (maps.csv)" and pre.tma_map.name == "TMA1"
        assert combo.findText("TMA9 (maps.csv)") >= 0
        assert combo.itemData(combo.count() - 1) == win.MAP_LOAD_ITEM
        n = combo.count()
        self._choose_load(win)                                            # same file again: replaced, not added
        settle(app)
        assert combo.count() == n

    def test_cancel_restores_previous_choice(self, fitted_gui, app, tmp_path, monkeypatch):
        from PyQt5.QtWidgets import QFileDialog
        win, pre = fitted_gui
        path = self._file(tmp_path)
        monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (path, ""))
        monkeypatch.setattr(win, "_preview_map_file", lambda mf, pre_sel: None)
        self._choose_load(win)
        settle(app)
        assert win.tma_array_combo.currentData() == 1 and pre.array_number == 1

    def test_save_templates(self, fitted_gui, tmp_path, monkeypatch):
        from PyQt5.QtWidgets import QFileDialog, QMessageBox
        win, _ = fitted_gui
        monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **k: str(tmp_path / "tpl"))
        shown = []
        monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: shown.append(a))
        win.save_tma_map_templates()
        assert os.listdir(tmp_path / "tpl") == ["tma_map_template.csv"]
        assert shown

    def test_preview_dialog(self, app, tmp_path):
        from cabana.tma_editor import MapPreviewDialog
        from cabana.tma_maps import load_map_file
        good = load_map_file(self._file(tmp_path))
        dlg = MapPreviewDialog(None, good, good.maps[1])
        assert dlg.selected_map() is good.maps[1] and dlg.table.rowCount() == 3
        assert dlg.table.item(0, 0).text() == "P5"
        dlg.map_combo.setCurrentIndex(0)
        assert dlg.table.item(0, 2).text() == "Liver" and dlg.table.item(1, 1).text() == "?"
        bad_path = tmp_path / "bad.csv"
        bad_path.write_text("row,col,patient_id\nA,1,P1\nA,1,P2\n")
        bad = load_map_file(str(bad_path))
        dlg = MapPreviewDialog(None, bad)
        assert dlg.selected_map() is None
        from PyQt5.QtWidgets import QDialogButtonBox
        assert not dlg.buttons.button(QDialogButtonBox.Ok).isEnabled()
        assert "twice" in dlg.problems.toPlainText()


def test_template_menu_item_is_named_generate(fitted_gui):
    win, _ = fitted_gui
    assert win.save_map_templates_action.text() == "Generate TMA Map &Template…"


class TestPanelScrolls:
    def test_small_window_scrolls_instead_of_squeezing(self, app):
        from cabana.cabana_gui import MainWindow
        win = MainWindow()
        win.showNormal()
        win.resize(1100, 520)
        win.show_page(win.pages.indexOf(win.tma_tab))
        settle(app, 200)
        page = win.tma_tab
        assert page.height() >= page.sizeHint().height()                 # natural height kept
        assert win.pages_scroll.verticalScrollBar().maximum() > 0          # the rest is reached by scrolling
        assert win.tma_slide_path.height() >= win.tma_slide_path.sizeHint().height()
        win.close()

    def test_short_page_has_no_scroll_range_in_a_tall_window(self, app):
        from cabana.cabana_gui import MainWindow
        win = MainWindow()
        win.showNormal()
        win.resize(1500, 980)
        win.show_page(win.pages.indexOf(win.det_tab))
        settle(app, 200)
        assert win.pages_scroll.verticalScrollBar().maximum() == 0
        win.close()


class TestLongStatusLine:
    def test_long_status_wraps_instead_of_widening_the_page(self, fitted_gui, app):
        win, _ = fitted_gui
        win.show_page(win.pages.indexOf(win.tma_tab))
        win.tma_status_label.setText("A very long fit summary that keeps going " * 12)
        settle(app, 200)
        assert win.pages_scroll.horizontalScrollBar().maximum() == 0     # nothing hidden on the right
        assert win.tma_tab.width() <= win.pages_scroll.viewport().width()
        lab = win.tma_status_label
        assert lab.height() >= lab.heightForWidth(lab.width())           # every wrapped line shown


class TestQuitConfirmation:
    def test_cancel_keeps_the_window_open(self, fitted_gui, app, monkeypatch):
        win, _ = fitted_gui
        win.confirm_on_close = True
        asked = []
        monkeypatch.setattr(win, "_ask_tma", lambda *a: asked.append(a) or False)
        win.close()
        settle(app)
        assert win.isVisible() and len(asked) == 1
        title, text, info, yes, no = asked[0]
        assert text == "Quit Cabana?" and (yes, no) == ("Confirm", "Cancel")
        monkeypatch.setattr(win, "_ask_tma", lambda *a: True)
        win.close()
        settle(app)
        assert not win.isVisible()

    def test_unsaved_edits_are_named(self, fitted_gui, app, monkeypatch, tmp_path):
        from PyQt5.QtWidgets import QFileDialog
        win, pre = fitted_gui
        assert win._work_at_risk() == []
        core = pre.cores[0]
        pre.move_core(core, core.cx + 10, core.cy)
        win._tma_edited(True)                                            # what the editor calls after an edit
        assert any("hand edits" in r for r in win._work_at_risk())
        path = str(tmp_path / "e.json")
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (path, ""))
        win.save_tma_edits()
        assert win._work_at_risk() == []
        win._tma_edited(True)
        info = []
        monkeypatch.setattr(win, "_ask_tma", lambda *a: info.append(a[2]) or True)
        assert win._confirm_quit() and "have not been exported or saved" in info[0]
