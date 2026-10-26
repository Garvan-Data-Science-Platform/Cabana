from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
import os
import sys
from .cabana_gui import MainWindow
from .winshell import ICON_PATH, set_app_id


def _set_macos_dock_name(name):
    """Set the macOS application name (menu bar and Dock) by writing
    ``CFBundleName`` into the main bundle's info dictionaries before the
    application is created. Without a bundled ``Info.plist`` macOS would
    otherwise show the executable name, e.g. ``python3``."""
    try:
        from ctypes import cdll, util, c_void_p, c_char_p
        objc = cdll.LoadLibrary(util.find_library('objc'))
        objc.objc_getClass.restype = c_void_p
        objc.objc_getClass.argtypes = [c_char_p]
        objc.sel_registerName.restype = c_void_p
        objc.sel_registerName.argtypes = [c_char_p]
        msg = objc.objc_msgSend
        msg.restype = c_void_p

        def send(obj, sel, *args):
            # Every argument is an object pointer or a C string.
            msg.argtypes = [c_void_p, c_void_p] + [
                c_char_p if isinstance(a, bytes) else c_void_p for a in args]
            return msg(obj, objc.sel_registerName(sel), *args)

        ns_string = objc.objc_getClass(b'NSString')
        key = send(ns_string, b'stringWithUTF8String:', b'CFBundleName')
        val = send(ns_string, b'stringWithUTF8String:', name.encode())
        bundle = send(objc.objc_getClass(b'NSBundle'), b'mainBundle')
        for selector in (b'infoDictionary', b'localizedInfoDictionary'):
            info = send(bundle, selector)
            if info:
                send(info, b'setObject:forKey:', val, key)
    except Exception:
        pass


def _ensure_std_streams():
    """Without a console (pythonw) sys.stdout/stderr are None, and libraries
    that write to them (tqdm progress bars) would raise; discard the output."""
    for name in ("stdout", "stderr"):
        if getattr(sys, name) is None:
            setattr(sys, name, open(os.devnull, "w"))


def main():
    _ensure_std_streams()

    from .log import setup_logging
    setup_logging()

    # ``python -m cabana tma ...`` runs the TMA preprocessing CLI headless.
    if len(sys.argv) > 1 and sys.argv[1] == 'tma':
        from .tma import main as tma_main
        tma_main(sys.argv[2:])
        return

    if sys.platform == 'darwin':
        _set_macos_dock_name('Cabana')
    elif sys.platform == 'win32':
        set_app_id()

    # Qt on macOS logs "Back buffer dpr of 2 doesn't match ... contents scale of 1"
    # whenever a window or popup is first shown on a Retina display and then fixes
    # it itself; the message is noise, so silence that category unless the user
    # has set their own logging rules.
    os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.backingstore=false")

    # Enable High DPI display before creating QApplication
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("Cabana")
    app.setWindowIcon(QIcon(str(ICON_PATH)))

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()