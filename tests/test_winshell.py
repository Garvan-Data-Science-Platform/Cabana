"""Tests for cabana/winshell.py (Windows taskbar identity) and its GUI wiring."""

import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from cabana import winshell  # noqa: E402


def test_icon_ships_with_package():
    assert winshell.ICON_PATH.is_file()


@pytest.mark.skipif(sys.platform == "win32", reason="checks the non-Windows fallback")
def test_no_op_off_windows():
    assert winshell.set_app_id() is False
    assert winshell.set_taskbar_icon(1, [sys.executable, "-m", "cabana"]) is False


@pytest.fixture(scope="module")
def app():
    pytest.importorskip("PyQt5")
    from PyQt5.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_main_window_sets_taskbar_icon_before_showing(app, monkeypatch):
    import cabana.cabana_gui as gui
    calls = []

    def fake_set_taskbar_icon(hwnd, command):
        calls.append((hwnd, command, win_holder[0].isVisible() if win_holder else None))
        return True

    win_holder = []
    real_init = gui.QMainWindow.__init__

    def init_and_hold(self, *a, **kw):
        real_init(self, *a, **kw)
        win_holder.append(self)

    with monkeypatch.context() as m:
        m.setattr(gui.QMainWindow, "__init__", init_and_hold)
        m.setattr(gui, "set_taskbar_icon", fake_set_taskbar_icon)
        m.setattr(gui, "restart_command", lambda: ["python", "-m", "cabana"])
        m.setattr(sys, "platform", "win32")
        win = gui.MainWindow()
    try:
        assert len(calls) == 1
        hwnd, command, visible = calls[0]
        assert hwnd == int(win.winId())
        assert command == ["python", "-m", "cabana"]
        assert visible is False          # set before the window is first shown
        assert win.isVisible()
    finally:
        win.close()
