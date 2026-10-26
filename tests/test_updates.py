"""Tests for cabana/updates.py and the GUI's Check for Updates wiring (offscreen Qt)."""

import io
import json
import os
import subprocess
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from cabana import updates  # noqa: E402


class TestVersions:
    @pytest.mark.parametrize("latest, current, newer", [
        ("0.3.10", "0.3.9", True),       # numeric, not string, order
        ("0.3.10", "0.3.10", False),
        ("0.3.9", "0.3.10", False),
        ("0.4", "0.3.10", True),
        ("0.3.10", "0.3.10.0", False),   # padded with zeros
        ("1.0.0", "dev", False),         # unparsable current version
        ("garbage", "0.3.10", False),
    ])
    def test_is_newer(self, latest, current, newer):
        assert updates.is_newer(latest, current) is newer


class TestLatestVersion:
    def test_reads_pypi_info_version(self, monkeypatch):
        body = json.dumps({"info": {"version": "9.9.9"}}).encode()
        seen = {}

        def fake_urlopen(request, timeout):
            seen["url"], seen["timeout"] = request.full_url, timeout
            return io.BytesIO(body)

        monkeypatch.setattr(updates.urllib.request, "urlopen", fake_urlopen)
        assert updates.latest_version(timeout=3) == "9.9.9"
        assert seen == {"url": updates.PYPI_URL, "timeout": 3}


class _FakeDist:
    def __init__(self, location, direct_url=None):
        self.location, self.direct_url = location, direct_url

    def locate_file(self, name):
        return os.path.join(self.location, name)

    def read_text(self, name):
        return self.direct_url if name == "direct_url.json" else None


class TestInstallProblem:
    @pytest.fixture
    def site(self, tmp_path, monkeypatch):
        """A fake site-packages whose cabana/ is the running package."""
        pkg = tmp_path / "site-packages" / "cabana"
        pkg.mkdir(parents=True)
        monkeypatch.setattr(updates, "__file__", str(pkg / "updates.py"))
        return pkg.parent

    def test_installed_package_can_update(self, site, monkeypatch):
        monkeypatch.setattr(updates, "distribution", lambda name: _FakeDist(str(site)))
        assert updates.install_problem() is None

    def test_not_installed(self, site, monkeypatch):
        def missing(name):
            raise updates.PackageNotFoundError(name)
        monkeypatch.setattr(updates, "distribution", missing)
        assert "not installed" in updates.install_problem()

    def test_shadowed_by_source_folder(self, site, tmp_path, monkeypatch):
        other = tmp_path / "elsewhere"
        monkeypatch.setattr(updates, "distribution", lambda name: _FakeDist(str(other)))
        assert "not from the installed package" in updates.install_problem()

    def test_git_checkout(self, site, monkeypatch):
        (site / ".git").mkdir()
        monkeypatch.setattr(updates, "distribution", lambda name: _FakeDist(str(site)))
        assert "development checkout" in updates.install_problem()

    def test_editable_install(self, site, monkeypatch):
        direct = json.dumps({"url": "file:///src", "dir_info": {"editable": True}})
        monkeypatch.setattr(updates, "distribution", lambda name: _FakeDist(str(site), direct))
        assert "editable" in updates.install_problem()


class TestInstallVersion:
    def test_runs_pip_in_this_environment(self, monkeypatch):
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

        monkeypatch.setattr(updates.subprocess, "run", fake_run)
        assert updates.install_version("9.9.9") == (True, "ok")
        assert calls == [[sys.executable, "-m", "pip", "install", "cabana==9.9.9"]]

    def test_failure_returns_output(self, monkeypatch):
        monkeypatch.setattr(updates.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(
            cmd, 1, stdout="", stderr="no matching distribution"))
        assert updates.install_version("9.9.9") == (False, "no matching distribution")

    def test_pip_missing(self, monkeypatch):
        def boom(cmd, **kw):
            raise OSError("not found")
        monkeypatch.setattr(updates.subprocess, "run", boom)
        assert updates.install_version("9.9.9") == (False, "not found")


# ------------------------------------------------------------------ GUI --
class _FakeSettings:
    """Stand-in for QSettings so tests never touch the user's real settings."""
    store = {}

    def __init__(self, *a):
        pass

    def value(self, key, default=None):
        return self.store.get(key, default)

    def setValue(self, key, value):
        self.store[key] = value


@pytest.fixture(scope="module")
def app():
    pytest.importorskip("PyQt5")
    from PyQt5.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


class TestGui:
    @pytest.fixture
    def win(self, app, monkeypatch):
        import cabana.cabana_gui as gui
        monkeypatch.setattr(gui, "QSettings", _FakeSettings)
        monkeypatch.setattr(_FakeSettings, "store", {})
        monkeypatch.setattr(gui, "__version__", "1.0.0")
        monkeypatch.setattr(gui, "install_problem", lambda: None)
        w = gui.MainWindow()
        w.asked = []
        w.answer = None
        monkeypatch.setattr(w, "_ask_update",
                            lambda title, text, info, buttons, details=None:
                            w.asked.append((text, buttons)) or w.answer)
        w.installs = []
        monkeypatch.setattr(w, "install_update", lambda v: w.installs.append(v))
        yield w
        w.close()

    def test_menu_action(self, win):
        assert win.check_updates_action.text() == "Check for &Updates…"

    def test_up_to_date(self, win):
        win._update_checked("1.0.0", quiet=False)
        assert "latest version" in win.asked[0][0]
        win._update_checked("1.0.0", quiet=True)     # quiet check: nothing shown
        assert len(win.asked) == 1

    def test_update_chosen(self, win):
        win.answer = "Update"
        win._update_checked("1.1.0", quiet=False)
        assert win.asked[0][1] == ["Update", "Later"]
        assert win.installs == ["1.1.0"]

    def test_skip_version_on_startup(self, win):
        win.answer = "Skip This Version"
        win._update_checked("1.1.0", quiet=True)
        assert win.asked[0][1] == ["Update", "Later", "Skip This Version"]
        assert not win.installs
        win._update_checked("1.1.0", quiet=True)     # skipped: not offered again on startup
        assert len(win.asked) == 1
        win._update_checked("1.1.0", quiet=False)    # but still offered on request
        win._update_checked("1.2.0", quiet=True)     # and a later version is offered
        assert len(win.asked) == 3

    def test_cannot_update_running_copy(self, win, monkeypatch):
        import cabana.cabana_gui as gui
        monkeypatch.setattr(gui, "install_problem", lambda: "Cabana is running from X.")
        win._update_checked("1.1.0", quiet=False)
        assert win.asked[0][1] == ["OK"] and not win.installs

    def test_check_failure_quiet_on_startup(self, win):
        win._update_check_failed("offline", quiet=True)
        assert not win.asked
        win._update_check_failed("offline", quiet=False)
        assert "Could not reach PyPI" in win.asked[0][0]

    def test_startup_check_once_a_day(self, win, monkeypatch):
        checks = []
        monkeypatch.setattr(win, "check_for_updates", lambda quiet=False: checks.append(quiet))
        win._startup_update_check()
        win._startup_update_check()
        assert checks == [True]

    def test_startup_check_skipped_for_source_copy(self, win, monkeypatch):
        import cabana.cabana_gui as gui
        checks = []
        monkeypatch.setattr(gui, "install_problem", lambda: "development checkout")
        monkeypatch.setattr(win, "check_for_updates", lambda quiet=False: checks.append(quiet))
        win._startup_update_check()
        assert not checks

    def test_install_refused_while_running(self, win, monkeypatch):
        import cabana.cabana_gui as gui
        monkeypatch.setattr(win, "_work_at_risk", lambda: ["a batch run is still running"])
        gui.MainWindow.install_update(win, "1.1.0")
        assert "cannot be updated" in win.asked[0][0]
        assert win.update_worker is None

    def test_install_failed(self, win):
        win._update_installed("1.1.0", False, "pip error")
        assert "could not be installed" in win.asked[0][0]
        assert win.asked[0][1] == ["OK"]
        assert win.check_updates_action.isEnabled()

    def test_installed_asks_for_manual_restart(self, win):
        win._update_installed("1.1.0", True, "Successfully installed cabana-1.1.0")
        assert "is installed" in win.asked[0][0]
        assert win.asked[0][1] == ["OK"]

    def test_installed_with_warnings_shows_details(self, win, monkeypatch):
        shown = []
        monkeypatch.setattr(win, "_ask_update",
                            lambda title, text, info, buttons, details=None: shown.append(details))
        win._update_installed("1.1.0", True, "Successfully installed")
        win._update_installed("1.1.0", True, "WARNING: Failed to remove contents of temp dir")
        assert shown == [None, "WARNING: Failed to remove contents of temp dir"]


class TestRestartCommand:
    def test_this_python(self, monkeypatch):
        monkeypatch.setattr(updates.sys, "platform", "darwin")
        assert updates.restart_command() == [sys.executable, "-m", "cabana"]

    def test_windows_prefers_pythonw(self, tmp_path, monkeypatch):
        (tmp_path / "python.exe").touch()
        monkeypatch.setattr(updates.sys, "executable", str(tmp_path / "python.exe"))
        monkeypatch.setattr(updates.sys, "platform", "win32")
        assert updates.restart_command()[0] == str(tmp_path / "python.exe")   # no pythonw.exe
        (tmp_path / "pythonw.exe").touch()
        assert updates.restart_command() == [str(tmp_path / "pythonw.exe"), "-m", "cabana"]


def test_missing_console_streams_are_replaced(monkeypatch):
    from cabana.__main__ import _ensure_std_streams
    from tqdm import tqdm
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)
    _ensure_std_streams()
    try:
        assert sys.stdout is not None and sys.stderr is not None
        for _ in tqdm(range(3)):          # raised AttributeError with stderr None
            pass
    finally:
        sys.stdout.close()
        sys.stderr.close()
