"""pytest configuration: initialize CABANA logger before tests run."""

import sys
import os
import tempfile

# Ensure the project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def pytest_configure(config):
    """Initialize Log singleton so Log.logger is not None during tests."""
    from cabana.log import Log
    if Log.logger is None:
        log_dir = os.path.join(tempfile.gettempdir(), "cabana_test_logs")
        os.makedirs(log_dir, exist_ok=True)
        Log.init_log_path(log_dir)


import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _close_stray_run_logs():
    """Tests that build a BatchProcessor without calling run() leave its run
    log open; close such handlers so tests stay independent."""
    yield
    import logging
    from cabana.log import Log
    for h in list(Log.logger.handlers):
        if isinstance(h, logging.FileHandler) and h is not Log._file_handler:
            Log.logger.removeHandler(h)
            h.close()


@pytest.fixture(autouse=True)
def _no_quit_confirmation(monkeypatch):
    """Closing a MainWindow in a test must not open the quit confirmation."""
    try:
        from cabana.cabana_gui import MainWindow
    except Exception:          # PyQt5 missing: no GUI tests run anyway
        return
    monkeypatch.setattr(MainWindow, "confirm_on_close", False, raising=False)
