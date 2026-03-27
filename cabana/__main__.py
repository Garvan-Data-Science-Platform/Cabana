from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
import sys
from pathlib import Path
from .cabana_gui import MainWindow


def _set_macos_dock_name(name):
    """Set the macOS dock label via CFBundleName using the Objective-C runtime."""
    try:
        from ctypes import cdll, util, c_void_p, c_char_p
        objc = cdll.LoadLibrary(util.find_library('objc'))
        objc.objc_getClass.restype = c_void_p
        objc.sel_registerName.restype = c_void_p

        def send(obj, sel, *args):
            objc.objc_msgSend.restype = c_void_p
            objc.objc_msgSend.argtypes = [c_void_p, c_void_p] + [type(a) for a in args]
            return objc.objc_msgSend(obj, objc.sel_registerName(sel), *args)

        bundle = send(objc.objc_getClass(b'NSBundle'), b'mainBundle')
        info = send(bundle, b'infoDictionary')
        ns = objc.objc_getClass(b'NSString')
        key = send(ns, b'stringWithUTF8String:', c_char_p(b'CFBundleName'))
        val = send(ns, b'stringWithUTF8String:', c_char_p(name.encode()))
        send(info, b'setObject:forKey:', val, key)
    except Exception:
        pass


def main():
    if sys.platform == 'darwin':
        _set_macos_dock_name('Cabana')

    # Enable High DPI display before creating QApplication
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("Cabana")

    window = MainWindow()
    icon_path = Path(__file__).parent / "cabana-logo.ico"
    app.setWindowIcon(QIcon(str(icon_path)))
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()