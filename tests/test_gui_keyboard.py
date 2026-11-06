"""Keyboard access to the side panel: path fields, card headers, range sliders."""
import pytest

pytest.importorskip("PyQt5")
from PyQt5.QtCore import QEvent, QEventLoop, Qt, QTimer  # noqa: E402
from PyQt5.QtGui import QKeyEvent  # noqa: E402
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


def key(widget, k, mods=Qt.NoModifier):
    QApplication.sendEvent(widget, QKeyEvent(QEvent.KeyPress, k, mods))


def test_enter_opens_a_path_field_and_delete_clears_the_mask(win, monkeypatch, tmp_path):
    import cabana.cabana_gui as gui
    picked = []
    monkeypatch.setattr(gui.QFileDialog, "getExistingDirectory",
                        lambda *a: picked.append(a[1]) or str(tmp_path))
    key(win.input_folder_path, Qt.Key_Return)
    assert picked == ["Select Input Folder"] and win.input_folder == str(tmp_path)
    win.set_mask_folder(str(tmp_path))
    key(win.mask_folder_path, Qt.Key_Delete)
    assert win.mask_folder is None and win.mask_folder_path.text() == ""
    assert win.input_folder_path.focusPolicy() == Qt.TabFocus


def test_space_folds_a_card(win):
    card = win.seg_colour_card
    assert card.header.focusPolicy() == Qt.TabFocus
    key(card.header, Qt.Key_Space)
    assert not card.is_open()
    key(card.header, Qt.Key_Return)
    assert card.is_open()


def test_range_slider_keys_move_the_active_handle(win):
    slider = win.line_width_range                      # 5, 7 on 1 to 15
    key(slider, Qt.Key_Left)
    assert (slider.lower_value, slider.upper_value) == (4, 7)
    key(slider, Qt.Key_Space)                          # now the upper handle
    key(slider, Qt.Key_Right)
    key(slider, Qt.Key_End)
    assert (slider.lower_value, slider.upper_value) == (4, 15)
    key(slider, Qt.Key_Space)
    key(slider, Qt.Key_End)                            # the lower one never passes the upper
    assert (slider.lower_value, slider.upper_value) == (15, 15)
    assert win.line_width_value.text() == "15, 15 px"


def test_buttons_take_focus_from_the_keyboard_only(win):
    for widget in (win.process_batch_btn, win.segment_btn, win.visuals_cb, win.start_open_btn):
        assert widget.focusPolicy() == Qt.TabFocus
