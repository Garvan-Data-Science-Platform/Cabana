"""Check PyPI for a newer Cabana and install it with pip (Help > Check for Updates)."""

import json
import re
import subprocess
import sys
import urllib.request
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

PYPI_URL = "https://pypi.org/pypi/cabana/json"
CHANGELOG_URL = "https://cabana.readthedocs.io/en/latest/changelog.html"


def latest_version(timeout=10):
    """Latest Cabana release on PyPI (raises ``OSError`` or ``ValueError`` when unreachable)."""
    request = urllib.request.Request(PYPI_URL, headers={"User-Agent": "cabana-update-check"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)["info"]["version"]


def _release(version):
    """Numeric release part of a version (``'0.3.10'`` -> ``(0, 3, 10)``), None when absent."""
    match = re.match(r"\s*v?(\d+(?:\.\d+)*)", str(version))
    return tuple(int(p) for p in match.group(1).split(".")) if match else None


def is_newer(latest, current):
    """True when ``latest`` is a later release than ``current``; False when either is unparsable."""
    a, b = _release(latest), _release(current)
    if a is None or b is None:
        return False
    n = max(len(a), len(b))
    return a + (0,) * (n - len(a)) > b + (0,) * (n - len(b))


_UPDATE_YOURSELF = "\nUpdate that copy yourself, e.g. with git pull or pip."


def install_records(site_packages):
    """The ``cabana-*.dist-info`` folders in ``site_packages``; normally exactly one."""
    return sorted(p for p in Path(site_packages).glob("*.dist-info")
                  if p.name.lower().startswith("cabana-"))


def install_problem():
    """None when pip can update the running copy of Cabana, else the reason it
    cannot, with what to do about it.

    The running code must be the installed package: a ``cabana`` source folder
    in the working folder, a git checkout or an editable install would be
    left unchanged by pip. More than one install record (a leftover from an
    interrupted upgrade) makes the reported version, read from the first
    record found, unreliable, so updates would seem not to take effect.
    """
    try:
        dist = distribution("cabana")
    except PackageNotFoundError:
        return "Cabana is running from source and is not installed as a package." + _UPDATE_YOURSELF
    running = Path(__file__).resolve().parent
    if Path(dist.locate_file("cabana")).resolve() != running:
        return f"Cabana is running from {running}, not from the installed package." + _UPDATE_YOURSELF
    if (running.parent / ".git").exists():
        return f"Cabana is running from a development checkout ({running.parent})." + _UPDATE_YOURSELF
    direct_url = dist.read_text("direct_url.json")
    if direct_url and json.loads(direct_url).get("dir_info", {}).get("editable"):
        return "Cabana is installed in editable (development) mode." + _UPDATE_YOURSELF
    records = install_records(running.parent)
    if len(records) > 1:
        return ("This environment has more than one Cabana install record, so the version shown "
                "may be wrong and updates may not take effect:\n"
                + "\n".join(f"  {p}" for p in records)
                + "\nClose Cabana, delete these folders, then run in this environment "
                  "(not pip uninstall):\n  pip install --force-reinstall --no-deps cabana")
    return None


def upgrade_command(version):
    """pip command that installs ``version`` into the running Python environment."""
    return [sys.executable, "-m", "pip", "install", f"cabana=={version}"]


def install_version(version, timeout=1800):
    """Install ``version`` with pip; returns (success, pip output)."""
    kwargs = {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}
    try:
        proc = subprocess.run(upgrade_command(version), capture_output=True, text=True,
                              timeout=timeout, **kwargs)
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)
    return proc.returncode == 0, (proc.stdout + proc.stderr).strip()


def restart_command():
    """Command that starts the Cabana GUI in this Python environment (what a
    pinned Windows taskbar button runs). On Windows it prefers pythonw.exe,
    so no console window opens, like ``cabana-gui``."""
    exe = Path(sys.executable)
    if sys.platform == "win32" and exe.name.lower() == "python.exe":
        windowed = exe.with_name("pythonw.exe")
        if windowed.exists():
            exe = windowed
    return [str(exe), "-m", "cabana"]
