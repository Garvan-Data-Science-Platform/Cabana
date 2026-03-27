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
