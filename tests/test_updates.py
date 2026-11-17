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
        (pkg / "__init__.py").touch()
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

    def test_duplicate_install_records_are_repairable(self, site, monkeypatch):
        monkeypatch.setattr(updates, "distribution", lambda name: _FakeDist(str(site)))
        (site / "cabana-0.3.11.dist-info").mkdir()
        (site / "other_pkg-1.0.dist-info").mkdir()
        assert not updates.needs_repair()                    # one record: fine
        (site / "cabana-0.3.7.dist-info").mkdir()
        assert updates.needs_repair()
        assert updates.install_problem() is None             # repaired by the update, not by hand
        removed = updates.remove_stale_records(site)
        assert [p.name for p in removed] == ["cabana-0.3.7.dist-info"]       # 0.3.11 is the newest
        assert sorted(p.name for p in site.glob("*.dist-info")) == [
            "cabana-0.3.11.dist-info", "other_pkg-1.0.dist-info"]

    def test_environment_this_account_cannot_change(self, site, monkeypatch):
        monkeypatch.setattr(updates, "distribution", lambda name: _FakeDist(str(site)))
        assert updates.can_modify(site)
        monkeypatch.setattr(updates, "can_modify", lambda folder: False)
        problem = updates.install_problem()
        assert "not allowed to change" in problem and str(site) in problem
        assert "administrator" in problem and "pip install --upgrade cabana" in problem

    def test_cannot_modify_without_the_package(self, tmp_path):
        assert not updates.can_modify(tmp_path)              # no cabana/__init__.py to write to
        assert not updates.can_modify(tmp_path / "missing")

    def test_per_user_copy_is_noted(self, site, monkeypatch):
        assert updates.install_note() is None
        monkeypatch.setattr(updates.site, "getusersitepackages", lambda: str(site))
        assert updates.in_user_site(site)
        note = updates.install_note()
        assert "your account only" in note and str(site) in note

    def test_problems_say_what_to_do(self, site, tmp_path, monkeypatch):
        monkeypatch.setattr(updates, "distribution", lambda name: _FakeDist(str(tmp_path / "x")))
        assert "Update that copy yourself" in updates.install_problem()

    def test_editable_install(self, site, monkeypatch):
        direct = json.dumps({"url": "file:///src", "dir_info": {"editable": True}})
        monkeypatch.setattr(updates, "distribution", lambda name: _FakeDist(str(site), direct))
        assert "editable" in updates.install_problem()


class TestInstallVersion:
    @pytest.fixture
    def site(self, tmp_path, monkeypatch):
        """A fake site-packages holding Cabana 9.9.8, and the pip calls made."""
        folder = tmp_path / "site-packages"
        (folder / "cabana").mkdir(parents=True)
        (folder / "cabana-9.9.8.dist-info").mkdir()
        monkeypatch.setattr(updates, "__file__", str(folder / "cabana" / "updates.py"))
        return folder

    @pytest.fixture
    def pip(self, site, monkeypatch):
        """pip stand-in that replaces the newest install record like an upgrade."""
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            for record in updates.install_records(site)[-1:]:
                record.rmdir()
            (site / f"cabana-{cmd[-1].split('==')[1]}.dist-info").mkdir()
            return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

        monkeypatch.setattr(updates.subprocess, "run", fake_run)
        return calls

    def test_runs_pip_in_this_environment(self, site, pip):
        assert updates.install_version("9.9.9") == (True, "ok")
        assert pip == [[sys.executable, "-m", "pip", "install", "--no-user", "cabana==9.9.9"]]

    def test_per_user_copy_is_updated_in_place(self, site, pip, monkeypatch):
        monkeypatch.setattr(updates.site, "getusersitepackages", lambda: str(site))
        assert updates.install_version("9.9.9")[0]
        assert pip == [[sys.executable, "-m", "pip", "install", "--user", "cabana==9.9.9"]]

    def test_leftover_records_are_removed_and_cabana_reinstalled(self, site, pip):
        (site / "cabana-9.9.1.dist-info").mkdir()
        ok, output = updates.install_version("9.9.9")
        assert ok and "cabana-9.9.1.dist-info" in output
        assert [p.name for p in updates.install_records(site)] == ["cabana-9.9.9.dist-info"]
        assert pip == [
            [sys.executable, "-m", "pip", "install", "--no-user", "--force-reinstall", "--no-deps", "cabana==9.9.9"],
            [sys.executable, "-m", "pip", "install", "--no-user", "cabana==9.9.9"]]

    def test_success_elsewhere_is_a_failure(self, site, monkeypatch):
        # pip succeeds but this folder still holds the old version (installed somewhere else)
        monkeypatch.setattr(updates.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(
            cmd, 0, stdout="Successfully installed", stderr=""))
        ok, output = updates.install_version("9.9.9")
        assert not ok and "holds Cabana 9.9.8, not Cabana 9.9.9 alone" in output

    def test_failure_returns_output(self, site, monkeypatch):
        monkeypatch.setattr(updates.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(
            cmd, 1, stdout="", stderr="no matching distribution"))
        assert updates.install_version("9.9.9") == (False, "no matching distribution")

    def test_pip_missing(self, site, monkeypatch):
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

    def test_leftover_records_offer_a_repair(self, win, monkeypatch):
        import cabana.cabana_gui as gui
        monkeypatch.setattr(gui, "needs_repair", lambda: True)
        win.answer = "Repair"
        win._update_checked("1.0.0", quiet=True)     # up to date, still repaired, also at startup
        assert "needs a repair" in win.asked[0][0] and win.asked[0][1] == ["Repair", "Later"]
        assert win.installs == ["1.0.0"]
        win.answer = "Update"
        win._update_checked("1.1.0", quiet=False)
        assert win.asked[1][1] == ["Update", "Later"] and win.installs == ["1.0.0", "1.1.0"]

    def test_per_user_copy_is_mentioned(self, win, monkeypatch):
        import cabana.cabana_gui as gui
        infos = []
        monkeypatch.setattr(gui, "install_note", lambda: "your account only")
        monkeypatch.setattr(win, "_ask_update",
                            lambda title, text, info, buttons, details=None: infos.append(info))
        win._update_checked("1.1.0", quiet=False)
        assert "your account only" in infos[0]

    def test_permission_failure_names_the_administrator(self, win, monkeypatch):
        infos = []
        monkeypatch.setattr(win, "_ask_update",
                            lambda title, text, info, buttons, details=None: infos.append(info))
        win._update_installed("1.1.0", False, "ERROR: [WinError 5] Access is denied")
        win._update_installed("1.1.0", False, "no matching distribution")
        assert "administrator" in infos[0] and "administrator" not in infos[1]

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
