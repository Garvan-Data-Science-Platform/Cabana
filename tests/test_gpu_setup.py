"""Tests for cabana/gpu_setup.py: choosing and installing the CUDA build of PyTorch."""

import subprocess
import sys

import pytest

from cabana import gpu_setup

PY = f"cp{sys.version_info.major}{sys.version_info.minor}"
SMI = """+-----------------------------------------------------------------------------+
| NVIDIA-SMI 551.61       Driver Version: 551.61       CUDA Version: 12.4     |
+-----------------------------------------------------------------------------+"""


def _index(wheels):
    """A stand-in for PyTorch's index: ``wheels`` maps 'cu124' to torch versions."""
    def fetch(url):
        if url == gpu_setup.INDEX_ROOT:
            return "".join(f'<a href="{n}/">{n}</a>' for n in list(wheels) + ["cu121-full", "cpu", "cuda-bindings"])
        name = url[len(gpu_setup.INDEX_ROOT):].split("/")[0]
        return "".join(f'<a href="/whl/{name}/torch-{v}%2B{name}-{PY}-{PY}-win_amd64.whl">x</a>'
                       f'<a href="/whl/{name}/torch-9.9.9%2B{name}-cp38-cp38-win_amd64.whl">x</a>'
                       for v in wheels[name])
    return fetch


@pytest.fixture
def windows(monkeypatch):
    monkeypatch.setattr(gpu_setup, "_platform_pattern", lambda: "win_amd64")


class TestDriver:
    def test_reads_the_cuda_version_of_the_driver(self, monkeypatch):
        monkeypatch.setattr(gpu_setup.shutil, "which", lambda name: "nvidia-smi")
        run = lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, stdout=SMI, stderr="")  # noqa: E731
        assert gpu_setup.driver_cuda(run) == (12, 4)

    def test_no_driver(self, monkeypatch):
        monkeypatch.setattr(gpu_setup.shutil, "which", lambda name: None)
        assert gpu_setup.driver_cuda() is None
        monkeypatch.setattr(gpu_setup.shutil, "which", lambda name: "nvidia-smi")
        run = lambda cmd, **kw: subprocess.CompletedProcess(cmd, 9, stdout="failed", stderr="")  # noqa: E731
        assert gpu_setup.driver_cuda(run) is None


class TestFindBuild:
    def test_newest_build_the_driver_supports(self, windows):
        fetch = _index({"cu118": ["2.4.0"], "cu124": ["2.5.0", "2.6.0"], "cu128": ["2.10.0"]})
        assert gpu_setup.find_build((12, 4), fetch) == (gpu_setup.INDEX_ROOT + "cu124", "2.6.0+cu124")
        assert gpu_setup.find_build((13, 0), fetch) == (gpu_setup.INDEX_ROOT + "cu128", "2.10.0+cu128")

    def test_newest_pytorch_wins_over_newest_cuda(self, windows):
        fetch = _index({"cu128": ["2.10.0"], "cu129": ["2.9.0"]})
        assert gpu_setup.find_build((12, 9), fetch)[1] == "2.10.0+cu128"

    def test_version_numbers_are_compared_as_numbers(self, windows):
        assert gpu_setup.find_build((12, 8), _index({"cu128": ["2.9.1", "2.10.0"]}))[1] == "2.10.0+cu128"

    def test_folder_without_a_wheel_for_this_python_is_skipped(self, windows):
        fetch = _index({"cu118": ["2.7.1"], "cu121": []})
        assert gpu_setup.find_build((12, 1), fetch)[1] == "2.7.1+cu118"

    def test_driver_too_old(self, windows):
        with pytest.raises(gpu_setup.SetupError, match="Update the NVIDIA driver"):
            gpu_setup.find_build((10, 2), _index({"cu118": ["2.7.1"]}))

    def test_no_cuda_builds_on_macos(self, monkeypatch):
        monkeypatch.setattr(gpu_setup, "_platform_pattern", lambda: None)
        with pytest.raises(gpu_setup.SetupError, match="operating system"):
            gpu_setup.find_build((12, 4), _index({"cu124": ["2.6.0"]}))


class TestPlan:
    @pytest.fixture
    def torch_folder(self, tmp_path, monkeypatch, windows):
        (tmp_path / "torch").mkdir()
        (tmp_path / "torch" / "__init__.py").touch()
        monkeypatch.setattr(gpu_setup, "torch_site_packages", lambda: tmp_path)
        monkeypatch.setattr(gpu_setup, "driver_cuda", lambda run=None: (12, 4))
        return tmp_path

    def test_pins_the_cuda_build_in_this_environment(self, torch_folder):
        command, version = gpu_setup.plan(fetch=_index({"cu124": ["2.6.0"]}))
        assert version == "2.6.0+cu124"
        assert command == [sys.executable, "-m", "pip", "install", "--no-user", "torch==2.6.0+cu124",
                           "--index-url", gpu_setup.INDEX_ROOT + "cu124",
                           "--extra-index-url", gpu_setup.PYPI_INDEX]

    def test_per_user_pytorch_is_replaced_in_place(self, torch_folder, monkeypatch):
        monkeypatch.setattr(gpu_setup.site, "getusersitepackages", lambda: str(torch_folder))
        assert "--user" in gpu_setup.plan(fetch=_index({"cu124": ["2.6.0"]}))[0]

    def test_folder_this_account_cannot_change(self, torch_folder, monkeypatch):
        monkeypatch.setattr(gpu_setup, "_can_modify", lambda folder: False)
        with pytest.raises(gpu_setup.SetupError, match="not allowed to change"):
            gpu_setup.plan(fetch=_index({"cu124": ["2.6.0"]}))

    def test_conda_installed_pytorch_is_left_to_conda(self, torch_folder, tmp_path, monkeypatch):
        meta = tmp_path / "env" / "conda-meta"
        meta.mkdir(parents=True)
        monkeypatch.setattr(gpu_setup.sys, "prefix", str(meta.parent))
        for other in ("pytorch-lightning-2.4.0-pyhd8ed1ab_0", "pytorch-mutex-1.0-cpu", "numpy-2.1.0-py312_0"):
            (meta / f"{other}.json").touch()                 # not PyTorch itself
        assert gpu_setup.conda_pytorch() is None
        assert gpu_setup.plan(fetch=_index({"cu124": ["2.6.0"]}))[1] == "2.6.0+cu124"
        (meta / "pytorch-2.5.1-cpu_py312h1234_0.json").touch()
        assert gpu_setup.conda_pytorch() == "pytorch-2.5.1-cpu_py312h1234_0"
        with pytest.raises(gpu_setup.SetupError, match="installed with conda"):
            gpu_setup.plan(fetch=_index({"cu124": ["2.6.0"]}))

    def test_no_driver(self, torch_folder, monkeypatch):
        monkeypatch.setattr(gpu_setup, "driver_cuda", lambda run=None: None)
        with pytest.raises(gpu_setup.SetupError, match="No NVIDIA driver"):
            gpu_setup.plan(fetch=_index({"cu124": ["2.6.0"]}))

    def test_index_unreachable(self, torch_folder):
        def offline(url):
            raise OSError("no network")
        with pytest.raises(gpu_setup.SetupError, match="Could not reach"):
            gpu_setup.plan(fetch=offline)


def test_installed_pytorch_is_checked_in_a_fresh_python():
    run = lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, stdout="2.6.0+cu124 True\n", stderr="")  # noqa: E731
    assert gpu_setup.check_installed(run) == ("2.6.0+cu124", True)
    broken = lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, stdout="", stderr="DLL load failed")  # noqa: E731
    assert gpu_setup.check_installed(broken) is None


def test_launch_runs_this_file_and_waits_for_cabana(monkeypatch):
    started = []
    monkeypatch.setattr(gpu_setup.subprocess, "Popen", lambda cmd, **kw: started.append(cmd))
    gpu_setup.launch(4321)
    assert started[0][1:] == [gpu_setup.__file__, "--wait-pid", "4321"]


def test_process_alive():
    import os
    assert gpu_setup.process_alive(os.getpid())
    finished = subprocess.Popen([sys.executable, "-c", "pass"])
    finished.wait()
    assert not gpu_setup.process_alive(finished.pid)


def test_the_file_runs_on_its_own_without_importing_cabana():
    # run as a program it must not load the cabana package (and with it PyTorch)
    code = ("import runpy, sys; sys.argv = ['x']; "
            f"runpy.run_path({gpu_setup.__file__!r}, run_name='loaded'); "
            "print('torch' in sys.modules, 'cabana' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert out.stdout.split() == ["False", "False"], out.stderr


class TestInstallerWindow:
    @pytest.fixture(scope="class")
    def app(self):
        pytest.importorskip("PyQt5")
        from PyQt5.QtWidgets import QApplication
        return QApplication.instance() or QApplication([])

    def _run(self, dialog):
        from PyQt5.QtCore import QTimer
        timer = QTimer()
        timer.timeout.connect(lambda: dialog.close_button.isEnabled() and dialog.accept())
        timer.start(50)
        QTimer.singleShot(60000, dialog.accept)
        dialog.exec_()
        timer.stop()
        return dialog.status_label.text(), dialog.log_box.toPlainText()

    def test_installs_then_reports_the_graphics_card_in_use(self, app, monkeypatch):
        monkeypatch.setattr(gpu_setup, "plan",
                            lambda: ([sys.executable, "-c", "print('Successfully installed')"], "2.6.0+cu124"))
        monkeypatch.setattr(gpu_setup, "check_installed", lambda: ("2.6.0+cu124", True))
        status, log = self._run(gpu_setup.installer_dialog())
        assert "GPU support is installed (PyTorch 2.6.0+cu124)" in status
        assert "Successfully installed" in log

    def test_pip_failure_keeps_the_previous_build(self, app, monkeypatch):
        monkeypatch.setattr(gpu_setup, "plan",
                            lambda: ([sys.executable, "-c", "import sys; print('ERROR: boom'); sys.exit(1)"], "x"))
        status, log = self._run(gpu_setup.installer_dialog())
        assert "could not be replaced" in status and "ERROR: boom" in log

    def test_installed_but_card_unusable(self, app, monkeypatch):
        monkeypatch.setattr(gpu_setup, "plan", lambda: ([sys.executable, "-c", "pass"], "x"))
        monkeypatch.setattr(gpu_setup, "check_installed", lambda: ("2.6.0+cu124", False))
        assert "Update the NVIDIA driver" in self._run(gpu_setup.installer_dialog())[0]

    def test_nothing_to_install_is_explained(self, app, monkeypatch):
        def refuse():
            raise gpu_setup.SetupError("No NVIDIA driver was found.")
        monkeypatch.setattr(gpu_setup, "plan", refuse)
        assert self._run(gpu_setup.installer_dialog())[0] == "No NVIDIA driver was found."

    def test_window_stays_open_while_pip_runs(self, app, monkeypatch):
        from PyQt5.QtCore import QTimer
        monkeypatch.setattr(gpu_setup, "plan",
                            lambda: ([sys.executable, "-c", "import time; time.sleep(1.5)"], "x"))
        monkeypatch.setattr(gpu_setup, "check_installed", lambda: ("x", True))
        dialog = gpu_setup.installer_dialog()
        closed_early = []
        QTimer.singleShot(500, lambda: (dialog.reject(), closed_early.append(dialog.result())))
        self._run(dialog)
        assert closed_early == [0] and dialog.close_button.isEnabled()
