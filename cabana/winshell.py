"""Windows taskbar identity: Cabana's own app ID and taskbar icon.

Setting the process app ID alone groups the window under Cabana but leaves a
generic icon on the taskbar; the window's AppUserModel relaunch properties
name the icon (and the command and name used when the button is pinned).
Every function is a no-op that returns False off Windows or on failure.
"""

import ctypes
import subprocess
import uuid
from ctypes import POINTER, byref, c_void_p
from pathlib import Path

from .log import Log

APP_ID = "Garvan.Cabana"
ICON_PATH = Path(__file__).parent / "cabana-logo.ico"

_IID_PROPERTY_STORE = "{886D8EEB-8CF2-4446-8D02-CDBA1DBDCF99}"
_APP_USER_MODEL = "{9F4C2855-9F79-4B39-A8D0-E1D42DE1D5F3}"     # PKEY_AppUserModel_* format id
_RELAUNCH_COMMAND, _RELAUNCH_ICON, _RELAUNCH_NAME, _ID = 2, 3, 4, 5
_VT_LPWSTR = 31


class _GUID(ctypes.Structure):
    _fields_ = [("Data1", ctypes.c_ulong), ("Data2", ctypes.c_ushort),
                ("Data3", ctypes.c_ushort), ("Data4", ctypes.c_ubyte * 8)]


def _guid(text):
    g = _GUID()
    ctypes.memmove(byref(g), uuid.UUID(text).bytes_le, 16)
    return g


class _PROPERTYKEY(ctypes.Structure):
    _fields_ = [("fmtid", _GUID), ("pid", ctypes.c_ulong)]


class _PROPVARIANT(ctypes.Structure):
    _fields_ = [("vt", ctypes.c_ushort), ("r1", ctypes.c_ushort), ("r2", ctypes.c_ushort),
                ("r3", ctypes.c_ushort), ("pwszVal", ctypes.c_wchar_p), ("pad", c_void_p)]


def set_app_id(app_id=APP_ID):
    """Give the process its own taskbar identity (call before any window exists)."""
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
        return True
    except Exception as exc:
        Log.logger.debug("Could not set the app ID: %s", exc)
        return False


def set_taskbar_icon(hwnd, relaunch_command, icon_path=ICON_PATH, app_id=APP_ID, name="Cabana"):
    """Name ``icon_path`` as the taskbar icon of window ``hwnd`` (call before it is shown)."""
    try:
        ctypes.windll.ole32.CoInitialize(None)
        store = c_void_p()
        hr = ctypes.windll.shell32.SHGetPropertyStoreForWindow(
            c_void_p(hwnd), byref(_guid(_IID_PROPERTY_STORE)), byref(store))
        if hr != 0 or not store:
            raise OSError(f"SHGetPropertyStoreForWindow failed ({hr & 0xFFFFFFFF:#x})")
        vtbl = ctypes.cast(store, POINTER(POINTER(c_void_p))).contents   # IPropertyStore methods
        set_value = ctypes.WINFUNCTYPE(ctypes.c_long, c_void_p, POINTER(_PROPERTYKEY),
                                       POINTER(_PROPVARIANT))(vtbl[6])
        commit = ctypes.WINFUNCTYPE(ctypes.c_long, c_void_p)(vtbl[7])
        release = ctypes.WINFUNCTYPE(ctypes.c_ulong, c_void_p)(vtbl[2])
        try:
            for pid, text in ((_ID, app_id),
                              (_RELAUNCH_COMMAND, subprocess.list2cmdline(relaunch_command)),
                              (_RELAUNCH_NAME, name),
                              (_RELAUNCH_ICON, f"{icon_path},0")):
                key = _PROPERTYKEY(_guid(_APP_USER_MODEL), pid)
                hr = set_value(store, byref(key), byref(_PROPVARIANT(vt=_VT_LPWSTR, pwszVal=text)))
                if hr != 0:
                    raise OSError(f"IPropertyStore::SetValue({pid}) failed ({hr & 0xFFFFFFFF:#x})")
            commit(store)
        finally:
            release(store)
        return True
    except Exception as exc:
        Log.logger.debug("Could not set the taskbar icon: %s", exc)
        return False
