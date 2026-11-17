"""Install the CUDA build of PyTorch into Cabana's environment (Help > Install GPU Support).

``pip install cabana`` takes PyTorch from PyPI, which on Windows is the
CPU-only build; the CUDA builds are on PyTorch's own index, one folder per
CUDA version. This module picks the newest build the NVIDIA driver supports
and installs it over the CPU-only one.

Windows locks PyTorch's files while a program has them loaded, so Cabana
cannot replace them itself: ``launch`` starts this file as a separate program
that waits until Cabana has closed, installs, checks the result and reports
in a small window. Run that way it must not import ``cabana`` (importing the
package loads PyTorch), so it uses only the standard library and PyQt5.
"""

import os
import sys

if __name__ == "__main__":
    # Run as a file, the script's folder (the cabana package) heads sys.path
    # and its modules would hide others of the same name: drop it first.
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.curdir) != _here]

import importlib.util
import re
import shutil
import site
import subprocess
import tempfile
import urllib.request
from pathlib import Path

INDEX_ROOT = "https://download.pytorch.org/whl/"
PYPI_INDEX = "https://pypi.org/simple"
MAX_CANDIDATES = 4        # CUDA folders compared, newest first, among those the driver supports
CHECK_CODE = "import torch; print(torch.__version__, torch.cuda.is_available())"


class SetupError(Exception):
    """GPU support cannot be installed; the message says why and what to do."""


def _quiet():
    """subprocess options that keep a console window from flashing up on Windows."""
    return {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}


def driver_cuda(run=subprocess.run):
    """Highest CUDA version the NVIDIA driver supports, as ``(major, minor)``,
    read from the header of ``nvidia-smi``; None without a driver."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        out = run([exe], capture_output=True, text=True, timeout=30, **_quiet()).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", out or "")
    return (int(match.group(1)), int(match.group(2))) if match else None


def _fetch(url, timeout=30):
    request = urllib.request.Request(url, headers={"User-Agent": "cabana-gpu-setup"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", "replace")


def _platform_pattern():
    """Regular expression of the wheel platform tags this computer accepts,
    or None where PyTorch has no CUDA builds (macOS uses the Apple GPU)."""
    if sys.platform == "win32":
        return "win_amd64"
    if sys.platform.startswith("linux"):
        return r"(?:manylinux\w+|linux)_x86_64"
    return None


def _version_key(version):
    return tuple(int(p) for p in version.split("."))


def find_build(driver, fetch=_fetch):
    """The newest PyTorch with a CUDA build that ``driver`` (its CUDA version)
    supports and a wheel for this Python: ``(index_url, 'X.Y.Z+cuNNN')``.
    Raises ``SetupError`` when there is none."""
    platform = _platform_pattern()
    if platform is None:
        raise SetupError("PyTorch has no CUDA builds for this operating system.")
    python = f"cp{sys.version_info.major}{sys.version_info.minor}"
    # index folders are named cu<major><minor>, e.g. cu128 = CUDA 12.8
    names = set(re.findall(r'href="(cu\d{3})/"', fetch(INDEX_ROOT)))
    builds = sorted(((int(n[2:-1]), int(n[-1])), n) for n in names)
    found = []
    for cuda, name in reversed(builds):
        if cuda > driver:
            continue
        wheel = re.compile(rf"torch-(\d+(?:\.\d+)+)(?:%2B|\+){name}-{python}-{python}-{platform}\.whl")
        versions = set(wheel.findall(fetch(f"{INDEX_ROOT}{name}/torch/")))
        if versions:
            found.append((_version_key(max(versions, key=_version_key)), cuda, name,
                          max(versions, key=_version_key)))
        if len(found) == MAX_CANDIDATES:
            break
    if found:
        # the newest CUDA folder does not always hold the newest PyTorch
        _, _, name, version = max(found)
        return INDEX_ROOT + name, f"{version}+{name}"
    raise SetupError(
        f"No CUDA build of PyTorch for Python {sys.version_info.major}.{sys.version_info.minor} "
        f"supports this NVIDIA driver (CUDA {driver[0]}.{driver[1]}). Update the NVIDIA driver and try again.")


def torch_site_packages():
    """Folder PyTorch is installed in, found without importing it; None when absent."""
    spec = importlib.util.find_spec("torch")
    if spec is None or not spec.origin:
        return None
    return Path(spec.origin).resolve().parent.parent


def _in_user_site(folder):
    try:
        user = site.getusersitepackages()
    except Exception:
        return False
    return bool(user) and Path(user).resolve() == Path(folder).resolve()


def _can_modify(folder):
    """True when this account may replace the PyTorch installed in ``folder``."""
    try:
        with tempfile.TemporaryFile(dir=folder):
            pass
        with open(Path(folder) / "torch" / "__init__.py", "r+b"):
            pass
    except OSError:
        return False
    return True


CONDA_PYTORCH = ("pytorch", "pytorch-cpu", "pytorch-gpu")      # conda package names of PyTorch itself
INSTALL_GUIDE = "https://pytorch.org/get-started/locally/"


def conda_pytorch(prefix=None):
    """Name of the conda package PyTorch was installed from in this
    environment (e.g. ``'pytorch-2.5.1-cpu_py312'``), or None when it came
    from pip. conda lists what it installed in ``conda-meta`` as
    ``<name>-<version>-<build>.json``; pip installs leave no entry there."""
    meta = Path(prefix or sys.prefix) / "conda-meta"
    for record in sorted(meta.glob("pytorch*.json")):
        if record.stem.rsplit("-", 2)[0] in CONDA_PYTORCH:
            return record.stem
    return None


def location_problem(folder=None):
    """None when PyTorch can be replaced where it is, else the reason it cannot."""
    folder = folder or torch_site_packages()
    if folder is None:
        return "PyTorch is not installed in this Python environment."
    package = conda_pytorch()
    if package:
        return (f"PyTorch in this environment was installed with conda ({package}). Replacing it "
                "with pip would leave conda's records wrong.\n"
                "Switch to a CUDA build with conda, or remove the conda package and install one "
                f"with pip as described at {INSTALL_GUIDE}")
    if not _can_modify(folder):
        return ("PyTorch is installed in a folder that this account is not allowed to change:\n"
                f"  {folder}\n"
                "Ask an administrator to choose Help > Install GPU Support in Cabana.")
    return None


def install_command(index_url, version, user=False):
    """pip command that replaces PyTorch by ``version`` (``'X.Y.Z+cuNNN'``) from
    ``index_url``, where PyTorch already is: the per-user folder when ``user``,
    else the environment itself. The version is pinned because pip regards the
    CPU-only build as good enough for a bare ``torch``; PyPI stays available
    for any dependency the new build adds."""
    return [sys.executable, "-m", "pip", "install", "--user" if user else "--no-user",
            f"torch=={version}", "--index-url", index_url, "--extra-index-url", PYPI_INDEX]


def plan(fetch=_fetch, run=subprocess.run):
    """What to install: ``(pip command, 'X.Y.Z+cuNNN')``. Raises ``SetupError``
    with the reason when GPU support cannot be installed."""
    folder = torch_site_packages()
    problem = location_problem(folder)
    if problem:
        raise SetupError(problem)
    driver = driver_cuda(run)
    if driver is None:
        raise SetupError("No NVIDIA driver was found (nvidia-smi did not report a CUDA version). "
                         "Install or update the NVIDIA driver and try again.")
    try:
        index_url, version = find_build(driver, fetch)
    except OSError as exc:
        raise SetupError(f"Could not reach {INDEX_ROOT} to look up the PyTorch builds.\n{exc}")
    return install_command(index_url, version, _in_user_site(folder)), version


def check_installed(run=subprocess.run):
    """``(version, cuda_available)`` of the installed PyTorch, asked of a fresh
    Python so this program never loads it; None when it cannot be imported."""
    try:
        proc = run([sys.executable, "-c", CHECK_CODE], capture_output=True, text=True,
                   timeout=300, cwd=str(Path.home()), **_quiet())
    except (OSError, subprocess.SubprocessError):
        return None
    parts = proc.stdout.split()
    if proc.returncode != 0 or len(parts) != 2:
        return None
    return parts[0], parts[1] == "True"


def launch_command(wait_pid):
    """Command that runs this file as a program of its own (windowed on
    Windows), waiting for process ``wait_pid`` to end first."""
    exe = Path(sys.executable)
    if sys.platform == "win32" and exe.name.lower() == "python.exe" and exe.with_name("pythonw.exe").exists():
        exe = exe.with_name("pythonw.exe")
    return [str(exe), str(Path(__file__).resolve()), "--wait-pid", str(wait_pid)]


def launch(wait_pid=None):
    """Start the installer as a separate program that outlives this one."""
    kwargs = {"cwd": str(Path.home()), "close_fds": True}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    subprocess.Popen(launch_command(os.getpid() if wait_pid is None else wait_pid), **kwargs)


def process_alive(pid):
    """True while process ``pid`` exists."""
    if sys.platform == "win32":
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x00100000, False, int(pid))     # SYNCHRONIZE
        if not handle:
            return False
        try:
            return kernel32.WaitForSingleObject(handle, 0) == 0x102   # WAIT_TIMEOUT: still running
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True


def installer_dialog(wait_pid=None):
    """The installer window: wait for process ``wait_pid`` (Cabana) to end,
    install, check the result and report. Returns the dialog, already started."""
    from PyQt5.QtCore import QProcess, QTimer
    from PyQt5.QtGui import QIcon
    from PyQt5.QtWidgets import QDialog, QLabel, QPlainTextEdit, QProgressBar, QPushButton, QVBoxLayout

    dialog = QDialog()
    dialog.setWindowTitle("Cabana GPU Support")
    icon = Path(__file__).resolve().parent / "cabana-logo.ico"
    if icon.exists():
        dialog.setWindowIcon(QIcon(str(icon)))
    dialog.setMinimumWidth(560)
    layout = QVBoxLayout(dialog)
    status = QLabel()
    status.setWordWrap(True)
    bar = QProgressBar()
    bar.setRange(0, 0)                      # busy: pip reports no progress without a console
    log = QPlainTextEdit()
    log.setReadOnly(True)
    log.setMinimumHeight(220)
    close = QPushButton("Close")
    close.setEnabled(False)
    close.clicked.connect(dialog.accept)
    for widget in (status, bar, log, close):
        layout.addWidget(widget)
    dialog.status_label, dialog.log_box, dialog.close_button = status, log, close

    process = QProcess(dialog)
    process.setProcessChannelMode(QProcess.MergedChannels)
    process.setWorkingDirectory(str(Path.home()))

    def finish(message):
        status.setText(message)
        bar.setRange(0, 1)
        bar.setValue(1)
        close.setEnabled(True)

    def read_output():
        text = bytes(process.readAllStandardOutput()).decode("utf-8", "replace").rstrip()
        if text:
            log.appendPlainText(text)

    def installed(code, _status):
        read_output()
        if code != 0:
            finish("PyTorch could not be replaced; the previous build is still in place. "
                   "See the details below.")
            return
        result = check_installed()
        if result is None:
            finish("PyTorch was installed but could not be loaded. See the details below.")
        elif result[1]:
            finish(f"GPU support is installed (PyTorch {result[0]}). Start Cabana again: "
                   "segmentation now runs on the graphics card.")
        else:
            finish(f"PyTorch {result[0]} is installed, but it cannot use the graphics card. "
                   "Update the NVIDIA driver, then start Cabana again.")

    def start():
        status.setText("Looking up the PyTorch build for this graphics card…")
        dialog.repaint()
        try:
            command, version = plan()
        except SetupError as exc:
            finish(str(exc))
            return
        status.setText(f"Downloading and installing PyTorch {version} (about 2.5 GB). "
                       "This can take several minutes; keep this window open.")
        log.appendPlainText(" ".join(command))
        process.start(command[0], command[1:])

    def wait_for_cabana():
        if wait_pid is not None and process_alive(wait_pid):
            status.setText("Waiting for Cabana to close…")
            QTimer.singleShot(500, wait_for_cabana)
        else:
            start()

    def failed_to_start(_error):
        if process.state() == QProcess.NotRunning and not close.isEnabled():
            finish(f"pip could not be started: {process.errorString()}")

    process.readyReadStandardOutput.connect(read_output)
    process.finished.connect(installed)
    process.errorOccurred.connect(failed_to_start)
    # pip must not be cut off half-way: the window closes only when it is done
    dialog.closeEvent = lambda event: event.accept() if close.isEnabled() else event.ignore()
    dialog.reject = lambda: QDialog.reject(dialog) if close.isEnabled() else None
    QTimer.singleShot(0, wait_for_cabana)
    return dialog


def main(argv=None):
    from PyQt5.QtWidgets import QApplication
    argv = sys.argv[1:] if argv is None else argv
    wait_pid = int(argv[argv.index("--wait-pid") + 1]) if "--wait-pid" in argv else None
    app = QApplication(sys.argv[:1])
    app.setApplicationName("Cabana")
    dialog = installer_dialog(wait_pid)
    dialog.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
