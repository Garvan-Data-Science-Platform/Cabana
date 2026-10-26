"""pytest configuration."""

import sys
import os

# Ensure the project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _no_quit_confirmation(monkeypatch):
    """Closing a MainWindow in a test must not open the quit confirmation."""
    try:
        from cabana.cabana_gui import MainWindow
    except Exception:          # PyQt5 missing: no GUI tests run anyway
        return
    monkeypatch.setattr(MainWindow, "confirm_on_close", False, raising=False)
