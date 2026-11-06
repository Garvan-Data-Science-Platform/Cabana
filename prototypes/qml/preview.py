"""Preview or render the side-panel prototype (Panel.qml).

    python prototypes/qml/preview.py            # interactive window
        Left/Right: page   Up/Down: theme   [ and ]: layout variant
    python prototypes/qml/preview.py --render   # renders/NN_<variant>.png: 4 themes x 6 pages each,
                                                # plus renders/overview_<theme>.png: every variant, one page
"""
import os
import sys

from styles import THEMES, VARIANTS, tokens

HERE = os.path.dirname(os.path.abspath(__file__))
PANEL_W, PANEL_H = 400, 1100
SCALE = 1.5
PAGES = ["TMA Dearrayer", "Birefringence", "Segmentation", "Fibre Detection", "Gap Analysis", "Batch Run"]
THEME_NAMES = list(THEMES)


def main():
    render = "--render" in sys.argv
    if render:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
    os.environ["QT_QUICK_BACKEND"] = "software"

    from PyQt5.QtCore import QEventLoop, QRect, QSize, QTimer, QUrl, Qt
    from PyQt5.QtGui import QColor, QFont, QFontDatabase, QGuiApplication, QImage, QPainter
    from PyQt5.QtQuick import QQuickView

    app = QGuiApplication(sys.argv)
    fonts = os.path.join(HERE, "fonts")
    for f in sorted(os.listdir(fonts)) if os.path.isdir(fonts) else []:
        if f.endswith(".ttf"):
            QFontDatabase.addApplicationFont(os.path.join(fonts, f))
    system = app.font().family()

    view = QQuickView()
    state = {"page": 0, "theme": 0, "variant": 0}

    def apply():
        view.rootContext().setContextProperty(
            "t", tokens(THEME_NAMES[state["theme"]], state["variant"], system))
        if view.rootObject():
            view.rootObject().setProperty("page", state["page"])
        view.setTitle(f"{PAGES[state['page']]}  |  {THEME_NAMES[state['theme']]}  |  "
                      f"{VARIANTS[state['variant']][0]}   (arrows: page/theme, [ ]: layout)")

    apply()
    view.setResizeMode(QQuickView.SizeRootObjectToView)
    view.setSource(QUrl.fromLocalFile(os.path.join(HERE, "Panel.qml")))
    if view.status() != QQuickView.Ready:
        for e in view.errors():
            print(e.toString())
        sys.exit(1)
    root = view.rootObject()
    view.resize(PANEL_W, PANEL_H if render else 900)
    root.setWidth(PANEL_W)          # the view resizes it late offscreen
    root.setHeight(PANEL_H if render else 900)

    if not render:
        def on_key(k):
            step = {Qt.Key_Left: ("page", -1, 6), Qt.Key_Right: ("page", 1, 6),
                    Qt.Key_Up: ("theme", -1, 4), Qt.Key_Down: ("theme", 1, 4),
                    Qt.Key_BracketLeft: ("variant", -1, len(VARIANTS)),
                    Qt.Key_BracketRight: ("variant", 1, len(VARIANTS))}.get(k)
            if step:
                name, d, n = step
                state[name] = (state[name] + d) % n
                apply()
        root.key.connect(on_key)
        apply()
        view.show()
        sys.exit(app.exec_())

    view.show()

    def wait(ms):
        loop = QEventLoop()
        QTimer.singleShot(ms, loop.quit)
        loop.exec_()

    def grab():
        result = root.grabToImage(QSize(int(PANEL_W * SCALE), int(root.height() * SCALE)))
        loop = QEventLoop()
        result.ready.connect(loop.quit)
        QTimer.singleShot(5000, loop.quit)
        loop.exec_()
        return result.image()

    out = os.path.join(HERE, "renders")
    os.makedirs(out, exist_ok=True)
    pw = int(PANEL_W * SCALE)
    gap, head, side = 36, 70, 150
    names = [a for a in sys.argv[1:] if not a.startswith("--")]
    f = QFont(system)
    f.setPixelSize(30)
    f.setWeight(QFont.DemiBold)

    # overview: every variant of one page, one sheet per theme
    if len(VARIANTS) > 1 and (not names or "overview" in names):
        cols, page = 4, PAGES.index("Fibre Detection")
        tallest = 0
        for v in range(len(VARIANTS)):
            state.update(theme=0, variant=v, page=page)
            apply()
            wait(150)
            tallest = max(tallest, root.property("contentHeight"))
        root.setHeight(tallest)
        ph = int(tallest * SCALE)
        rows = -(-len(VARIANTS) // cols)
        for r, theme in enumerate(THEME_NAMES):
            sheet = QImage(gap + cols * (pw + gap), rows * (ph + head) + gap, QImage.Format_RGB32)
            sheet.fill(QColor("#8E969C"))
            p = QPainter(sheet)
            p.setFont(f)
            p.setPen(QColor("#FFFFFF"))
            for v, (vname, _, _) in enumerate(VARIANTS):
                state.update(theme=r, variant=v, page=page)
                apply()
                wait(250)
                x, y = gap + (v % cols) * (pw + gap), (v // cols) * (ph + head)
                p.drawText(QRect(x, y, pw, head), Qt.AlignVCenter | Qt.AlignLeft, f"{v}. {vname}")
                p.drawImage(x, y + head, grab())
            p.end()
            path = os.path.join(out, f"overview_{theme.lower().replace(' ', '_')}.png")
            sheet.save(path)
            print(path)

    for v, (vname, _, _) in enumerate(VARIANTS):
        if names and vname.lower().split()[0] not in [n.lower() for n in names] and "all" not in names:
            continue
        # panel height: the tallest page in this layout
        tallest = 0
        for i in range(len(PAGES)):
            state.update(theme=0, variant=v, page=i)
            apply()
            wait(150)
            tallest = max(tallest, root.property("contentHeight"))
        root.setHeight(tallest)
        ph = int(tallest * SCALE)
        sheet = QImage(side + len(PAGES) * (pw + gap), head + len(THEME_NAMES) * (ph + gap),
                       QImage.Format_RGB32)
        sheet.fill(QColor("#8E969C"))
        p = QPainter(sheet)
        p.setFont(f)
        p.setPen(QColor("#FFFFFF"))
        for i, page in enumerate(PAGES):
            p.drawText(QRect(side + i * (pw + gap), 0, pw, head), Qt.AlignVCenter | Qt.AlignLeft, page)
        for r, theme in enumerate(THEME_NAMES):
            y = head + r * (ph + gap)
            p.drawText(QRect(0, y, side - 16, 90), Qt.AlignTop | Qt.AlignRight, theme.replace(" ", "\n"))
            state.update(theme=r, variant=v)
            for i in range(len(PAGES)):
                state["page"] = i
                apply()
                wait(250)            # let the Loader and Canvas items settle
                p.drawImage(side + i * (pw + gap), y, grab())
        p.end()
        path = os.path.join(out, f"{v:02d}_{vname.lower().replace(' ', '_')}.png")
        sheet.save(path)
        print(path)


if __name__ == "__main__":
    main()
