"""Capture the side panel of every analysis page in the Light and Dark themes.

Run before and after a UI change to compare the two:

    python prototypes/capture_panels.py before   # -> prototypes/ui_compare/before/
    python prototypes/capture_panels.py after    # -> prototypes/ui_compare/after/
    python prototypes/capture_panels.py compare  # -> prototypes/ui_compare/<page>.png, before | after

Uses Qt's offscreen platform and a stand-in for QSettings, so the user's
saved settings (theme, folders) are never read or changed.
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT = os.path.join(HERE, "ui_compare")
sys.path.insert(0, REPO)

THEMES = ["Light", "Dark"]
SIZE = (1500, 980)


class _FakeSettings:
    store = {}

    def __init__(self, *a):
        pass

    def value(self, key, default=None, **kw):
        return self.store.get(key, default)

    def setValue(self, key, value):
        self.store[key] = value


def settle(app, ms=300):
    from PyQt5.QtCore import QEventLoop, QTimer
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec_()
    app.processEvents()


def capture(tag):
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)
    import cabana.cabana_gui as gui
    gui.QSettings = _FakeSettings
    gui.MainWindow.check_updates_on_start = False
    win = gui.MainWindow()
    win.confirm_on_close = False
    win.resize(*SIZE)
    win.show()
    settle(app)

    out = os.path.join(OUT, tag)
    os.makedirs(out, exist_ok=True)
    pages = [(i, win._page_titles[i]) for i in range(1, win.pages.count())]
    for theme in THEMES:
        win.theme_combo.setCurrentText(theme)
        settle(app)
        for index, title in pages:
            win.show_page(index)
            settle(app, 200)
            path = os.path.join(out, f"{title.lower().replace(' ', '_')}_{theme.lower()}.png")
            win.dock_contents.grab().save(path)
            print(path)
    win.close()


def compare():
    """Side-by-side sheets: one per page, rows = themes, columns = before | after."""
    from PyQt5.QtGui import QColor, QFont, QGuiApplication, QImage, QPainter
    from PyQt5.QtCore import QRect, Qt
    app = QGuiApplication.instance() or QGuiApplication(sys.argv)
    before, after = os.path.join(OUT, "before"), os.path.join(OUT, "after")
    names = sorted({f.rsplit("_", 1)[0] for f in os.listdir(before) if f.endswith(".png")})
    gap, head = 24, 44
    for name in names:
        cells = [[QImage(os.path.join(d, f"{name}_{t.lower()}.png")) for d in (before, after)] for t in THEMES]
        w = max(img.width() for row in cells for img in row)
        h = max(img.height() for row in cells for img in row)
        sheet = QImage(2 * w + 3 * gap, len(THEMES) * (h + head) + gap, QImage.Format_RGB32)
        sheet.fill(QColor("#8E969C"))
        p = QPainter(sheet)
        f = QFont()
        f.setPixelSize(20)
        f.setWeight(QFont.DemiBold)
        p.setFont(f)
        p.setPen(QColor("#FFFFFF"))
        for r, theme in enumerate(THEMES):
            y = r * (h + head)
            for c, label in enumerate(("Before", "After")):
                x = gap + c * (w + gap)
                p.drawText(QRect(x, y, w, head), Qt.AlignVCenter | Qt.AlignLeft, f"{label}, {theme}")
                if not cells[r][c].isNull():
                    p.drawImage(x, y + head, cells[r][c])
        p.end()
        path = os.path.join(OUT, f"{name}.png")
        sheet.save(path)
        print(path)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "before"
    compare() if what == "compare" else capture(what)
