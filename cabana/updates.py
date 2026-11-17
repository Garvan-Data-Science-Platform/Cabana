"""Check PyPI for a newer Cabana and install it with pip (Help > Check for Updates)."""

import json
import re
import shutil
import site
import subprocess
import sys
import tempfile
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


def _record_version(record):
    """Version named by an install record (``cabana-0.3.11.dist-info`` -> ``'0.3.11'``)."""
    return record.name[len("cabana-"):-len(".dist-info")]


def _site_packages():
    """The folder the running ``cabana`` package is in."""
    return Path(__file__).resolve().parent.parent


def needs_repair(site_packages=None):
    """True when the environment has more than one Cabana install record (a
    leftover from an interrupted upgrade). The version Cabana reports is read
    from the first record found, so it may be wrong and an update may seem not
    to take effect; ``install_version`` repairs it."""
    return len(install_records(site_packages or _site_packages())) > 1


def remove_stale_records(site_packages):
    """Delete every Cabana install record except the newest, which lists the
    most files for pip to replace; returns the removed folders. Raises
    ``OSError`` when one cannot be deleted."""
    records = sorted(install_records(site_packages),
                     key=lambda p: _release(_record_version(p)) or ())
    for record in records[:-1]:
        shutil.rmtree(record)
    return records[:-1]


def in_user_site(site_packages):
    """True when ``site_packages`` is the per-user folder (on Windows
    ``AppData\\Roaming\\Python``), where pip installs when it may not write
    to the environment. A copy there hides the environment's own copy for
    that user only."""
    try:
        user = site.getusersitepackages()
    except Exception:
        return False
    return bool(user) and Path(user).resolve() == Path(site_packages).resolve()


def can_modify(site_packages):
    """True when this account may change the Cabana installed in
    ``site_packages``: create a file there and write to an installed one
    (a shared folder such as ``C:\\ProgramData`` may allow the first only)."""
    try:
        with tempfile.TemporaryFile(dir=site_packages):
            pass
        with open(Path(site_packages) / "cabana" / "__init__.py", "r+b"):
            pass
    except OSError:
        return False
    return True


def install_problem():
    """None when pip can update the running copy of Cabana, else the reason it
    cannot, with what to do about it.

    The running code must be the installed package: a ``cabana`` source folder
    in the working folder, a git checkout or an editable install would be
    left unchanged by pip. The account must also be allowed to change the
    installation: without that pip would quietly install a second, per-user
    copy instead of updating the shared one.
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
    if not can_modify(running.parent):
        return ("Cabana is installed in a folder that this account is not allowed to change:\n"
                f"  {running.parent}\n"
                "Ask an administrator to update it by running, in that environment:\n"
                "  python -m pip install --upgrade cabana")
    return None


def install_note(site_packages=None):
    """A caution to show before updating, or None: the running copy is a
    per-user one, so the update will not reach the other users."""
    site_packages = site_packages or _site_packages()
    if in_user_site(site_packages):
        return ("This copy of Cabana is installed for your account only, in\n"
                f"  {site_packages}\n"
                f"and is used instead of the one in {sys.prefix}. "
                "The update applies to your account only.")
    return None


def upgrade_command(version, user=False, force=False):
    """pip command that installs ``version`` where Cabana runs from: the
    per-user folder when ``user``, else the Python environment itself
    (``--no-user`` stops pip switching to the per-user folder when it may
    not write to the environment). ``force`` reinstalls Cabana alone, even
    when pip believes that version is present."""
    extra = ["--force-reinstall", "--no-deps"] if force else []
    return [sys.executable, "-m", "pip", "install", "--user" if user else "--no-user",
            *extra, f"cabana=={version}"]


def install_version(version, timeout=1800, site_packages=None):
    """Install ``version`` with pip over the running copy; returns (success, output).

    Leftover install records are removed first and Cabana is then reinstalled
    in full. Afterwards the folder must hold exactly one record, of
    ``version``: pip reporting success is not enough, it may have installed
    somewhere else."""
    site_packages = Path(site_packages) if site_packages else _site_packages()
    user = in_user_site(site_packages)
    output = []
    commands = [upgrade_command(version, user)]
    if needs_repair(site_packages):
        try:
            removed = remove_stale_records(site_packages)
        except OSError as exc:
            return False, f"Could not remove a leftover install record: {exc}"
        output.append("Removed leftover install records: " + ", ".join(p.name for p in removed))
        commands.insert(0, upgrade_command(version, user, force=True))
    kwargs = {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}
    for command in commands:
        try:
            proc = subprocess.run(command, capture_output=True, text=True, timeout=timeout, **kwargs)
        except (OSError, subprocess.SubprocessError) as exc:
            return False, "\n".join(output + [str(exc)])
        output.append((proc.stdout + proc.stderr).strip())
        if proc.returncode != 0:
            return False, "\n".join(output)
    found = [_record_version(p) for p in install_records(site_packages)]
    if len(found) != 1 or _release(found[0]) != _release(version):
        output.append(f"pip reported success, but {site_packages} holds "
                      f"{'Cabana ' + ', '.join(found) if found else 'no Cabana'}, not Cabana {version} alone.")
        return False, "\n".join(output)
    return True, "\n".join(output)


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
