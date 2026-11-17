"""Guided tour, first-start prompts and the card-style dialogs."""
import pytest

pytest.importorskip("PyQt5")
from PyQt5.QtCore import QEvent, QEventLoop, Qt, QTimer  # noqa: E402
from PyQt5.QtGui import QKeyEvent  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def settle(app, ms=50):
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec_()
    app.processEvents()


class _FakeSettings:
    """Stand-in for QSettings so tests never touch the user's real settings."""
    store = {}

    def __init__(self, *a):
        pass

    def value(self, key, default=None):
        return self.store.get(key, default)

    def setValue(self, key, value):
        self.store[key] = value


@pytest.fixture
def win(app, monkeypatch):
    import cabana.cabana_gui as gui
    _FakeSettings.store = {}
    monkeypatch.setattr(gui, "QSettings", _FakeSettings)
    w = gui.MainWindow()
    w.showNormal()
    w.resize(1400, 900)
    w.show()
    settle(app)
    yield w
    w.close()


def test_tour_walks_the_pipeline_and_highlights_each_step(win, app):
    from cabana.ui import SectionCard
    win.seg_training_card.set_open(False)                            # a folded card is opened for its step
    win.start_tour()
    tour = win._tour
    assert tour is not None and tour.isVisible()
    pages, targets = [], []
    for i in range(len(tour.steps)):
        assert tour.index == i and tour.count_label.text() == f"{i + 1} of {len(tour.steps)}"
        settle(app)
        pages.append(win.pages.currentWidget())
        targets.append(tour.steps[i].target())
        target = tour.target_rect()
        assert target is not None and win.rect().contains(target.center())   # no centred card
        assert not tour.card.geometry().intersects(target.adjusted(4, 4, -4, -4))
        assert tour.card.height() >= tour.card.layout().totalHeightForWidth(tour.CARD_WIDTH)
        tour.next()
    assert pages[0] is win.start_tab and targets[1] is win.page_rail and pages[-1] is win.bat_tab
    cards = [c for i in range(win.pages.count()) for c in win.pages.widget(i).findChildren(SectionCard)]
    assert cards and all(c in targets for c in cards)                # every card has a step
    assert win.seg_training_card.is_open()
    settle(app)
    assert win._tour is None                                         # Done ends the tour


def test_tour_keys_and_back(win, app):
    win.start_tour()
    tour = win._tour
    assert not tour.back_btn.isVisible() and tour.next_btn.text() == "Next"
    QApplication.sendEvent(tour, QKeyEvent(QEvent.KeyPress, Qt.Key_Right, Qt.NoModifier))
    assert tour.index == 1 and tour.back_btn.isVisible()
    QApplication.sendEvent(tour, QKeyEvent(QEvent.KeyPress, Qt.Key_Left, Qt.NoModifier))
    assert tour.index == 0
    tour.go(len(tour.steps) - 1)
    assert tour.next_btn.text() == "Done"
    QApplication.sendEvent(tour, QKeyEvent(QEvent.KeyPress, Qt.Key_Escape, Qt.NoModifier))
    settle(app)
    assert win._tour is None


def test_tour_changes_no_settings(win, app):
    before = {k: dict(v) if isinstance(v, dict) else v for k, v in win.yml_data.items()}
    win.start_tour()
    while win._tour is not None:
        win._tour.next()
        settle(app)
    assert win.yml_data == before


def test_first_start_offers_the_tour(win, monkeypatch):
    asked, started = [], []
    monkeypatch.setattr(win, "_ask_tour", lambda: asked.append(1) or True)
    monkeypatch.setattr(win, "start_tour", lambda: started.append(1))
    monkeypatch.setattr(win, "show_whats_new", lambda *a: pytest.fail("no What's New on a fresh install"))
    win._startup_messages()
    assert asked == [1] and started == [1]
    from cabana import __version__
    assert _FakeSettings.store[win.LAST_VERSION_KEY] == __version__
    asked.clear()
    win._startup_messages()                                          # second start: nothing
    assert asked == []


def test_upgrade_shows_whats_new_once(win, monkeypatch):
    from cabana import __version__, whats_new
    monkeypatch.setattr(whats_new, "NOTES", {__version__: ["Something new."]})
    shown = []
    monkeypatch.setattr(win, "show_whats_new", lambda *a, **k: shown.append(1))
    monkeypatch.setattr(win, "_ask_tour", lambda: pytest.fail("no tour offer after an upgrade"))
    _FakeSettings.store = {"theme": "Dark"}                          # a user of a version without the key
    win._startup_messages()
    assert shown == [1]
    win._startup_messages()
    assert shown == [1]


def test_dialogs_use_the_card_style(win, monkeypatch):
    import cabana.cabana_gui as gui
    from cabana import __version__, whats_new
    opened = []

    def record(dlg):
        opened.append(dlg)
        return 0
    monkeypatch.setattr(gui.CardDialog, "exec_", record)
    win._show_about_dialog()
    win.show_whats_new()
    about, news = opened
    assert about.windowTitle() == "About Cabana" and about.objectName() == "cardDialog"
    labels = [w.text() for w in about.findChildren(gui.QLabel)]
    assert f"Version {__version__}, MIT Licence" in labels and "—" not in " ".join(labels)
    assert [b.text() for b in about.findChildren(gui.QPushButton)] == ["Licence", "Close"]
    version, notes = whats_new.latest() if whats_new.notes_for(__version__) is None else (
        __version__, whats_new.notes_for(__version__))
    assert news.windowTitle() == f"What's new in Cabana {version}"
    assert all(n in [w.text() for w in news.findChildren(gui.QLabel)] for n in notes)


def test_start_page_is_the_intro_and_three_buttons(win):
    from cabana.ui import SectionCard
    assert not win.start_tab.findChildren(SectionCard)
    assert [b.text() for b in (win.start_slide_btn, win.start_open_btn, win.start_params_btn)] == [
        "Open TMA Slide…", "Open Image…", "Import Parameters…"]


def test_page_rail_opens_pages_and_folds_the_panel(win, app):
    from PyQt5.QtTest import QTest
    rail = win.page_rail
    assert len(rail.buttons) == win.pages.count() and rail.buttons[0].isChecked()
    seg = win.pages.indexOf(win.seg_tab)
    QTest.mouseClick(rail.buttons[seg], Qt.LeftButton)              # another page: show it
    settle(app)
    assert win.pages.currentIndex() == seg and win.panel_visible and rail.buttons[seg].isChecked()
    QTest.mouseClick(rail.buttons[seg], Qt.LeftButton)              # the open page again: fold
    settle(app)
    assert not win.panel_visible and not win.dock_contents.isVisible()
    QTest.mouseClick(rail.buttons[seg], Qt.LeftButton)              # and unfold
    settle(app)
    assert win.panel_visible and win.pages.currentIndex() == seg
    win.toggle_panel()                                              # folded, then another page
    QTest.mouseClick(rail.buttons[win.pages.indexOf(win.bat_tab)], Qt.LeftButton)
    settle(app)
    assert win.panel_visible and win.pages.currentWidget() is win.bat_tab
    win.show_page(win.pages.indexOf(win.det_tab))                   # the menu keeps the rail in step
    assert rail.buttons[win.pages.indexOf(win.det_tab)].isChecked()
    rail.help_btn.click()
    assert win._tour is not None
    win._tour.end()


def test_title_bar_controls_do_not_fold_the_card(win, app):
    from PyQt5.QtTest import QTest
    win.show_page(win.pages.indexOf(win.tma_tab))
    settle(app)
    card = win.tma_export_card
    box = win.tma_channel_cbs["BF"]
    assert not box.isEnabled() and card.header.isAncestorOf(box)    # disabled until Fit Cores
    QTest.mouseClick(box, Qt.LeftButton)
    assert card.is_open()
    QTest.mouseClick(win.tma_edit_card.header, Qt.LeftButton, pos=win.tma_edit_card.title_label.geometry().center())
    assert not win.tma_edit_card.is_open()                           # the title still folds it


def test_card_dialog_shows_every_wrapped_line(win, app):
    import cabana.cabana_gui as gui
    dlg = gui.CardDialog(win, "What's new", width=440)
    dlg.add_bullets(["A long note that wraps over two or three lines in a dialog of this width, "
                     "so its last line must not be cut off."] * 4)
    dlg.show()
    settle(app)
    labels = [l for l in dlg.findChildren(gui.QLabel) if l.objectName() == "dialogText"]
    assert labels and all(l.height() >= l.heightForWidth(l.width()) for l in labels)
    dlg.close()


def test_unused_graphics_card_is_reported_once_per_pytorch_build(win, monkeypatch):
    import torch
    import cabana.cabana_gui as gui
    shown = []
    monkeypatch.setattr(win, "_message", lambda title, text, **k: shown.append((title, text, k)))
    monkeypatch.setattr(gui, "gpu_unused_reason", lambda: None)
    win._gpu_hint()
    assert shown == []                                               # nothing to report
    monkeypatch.setattr(gui, "gpu_unused_reason",
                        lambda: ("CPU-only <build>.", f"See {gui.PYTORCH_INSTALL_URL}"))
    win._gpu_hint()
    win._gpu_hint()
    assert len(shown) == 1
    title, text, options = shown[0]
    assert "CPU-only &lt;build&gt;." in text and f"href='{gui.PYTORCH_INSTALL_URL}'" in text
    assert options.get("rich") is True
    monkeypatch.setattr(torch, "__version__", "0.0.0+cpu")           # another build: told again
    win._gpu_hint()
    assert len(shown) == 2


class TestInstallGpuSupport:
    @pytest.fixture
    def cpu_build(self, win, monkeypatch):
        """A Windows PC with an NVIDIA card and the CPU-only PyTorch."""
        import torch
        import cabana.cabana_gui as gui
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.version, "cuda", None)
        monkeypatch.setattr(gui, "gpu_unused_reason", lambda: ("CPU-only build.", "Install a CUDA build."))
        monkeypatch.setattr(gui.gpu_setup, "location_problem", lambda: None)
        win.launched, win.closed, win.shown, win.answer = [], [], [], None
        monkeypatch.setattr(gui.gpu_setup, "launch", lambda: win.launched.append(True))
        monkeypatch.setattr(win, "close", lambda: win.closed.append(True))
        monkeypatch.setattr(win, "_message",
                            lambda title, text, info="", buttons=("OK",), **k:
                            win.shown.append((text, info, list(buttons))) or win.answer)
        return win

    def test_hint_offers_the_install(self, cpu_build):
        cpu_build.answer = "Install GPU Support"
        cpu_build._gpu_hint()
        assert cpu_build.shown[0][2] == ["Install GPU Support", "Not Now"]
        assert cpu_build.launched and cpu_build.closed       # already confirmed: no second question
        assert len(cpu_build.shown) == 1

    def test_hint_declined(self, cpu_build):
        cpu_build.answer = "Not Now"
        cpu_build._gpu_hint()
        assert not cpu_build.launched and not cpu_build.closed

    def test_menu_asks_before_closing_cabana(self, cpu_build):
        cpu_build.answer = "Cancel"
        cpu_build.install_gpu_support()
        assert cpu_build.shown[0][2] == ["Close Cabana and Install", "Cancel"] and not cpu_build.launched
        cpu_build.answer = "Close Cabana and Install"
        cpu_build.install_gpu_support()
        assert cpu_build.launched and cpu_build.closed

    def test_refused_while_work_would_be_lost(self, cpu_build, monkeypatch):
        monkeypatch.setattr(cpu_build, "_work_at_risk", lambda: ["a batch run is still running"])
        cpu_build.install_gpu_support()
        assert "a batch run is still running" in cpu_build.shown[0][1] and not cpu_build.launched

    def test_folder_this_account_cannot_change(self, cpu_build, monkeypatch):
        import cabana.cabana_gui as gui
        monkeypatch.setattr(gui.gpu_setup, "location_problem", lambda: "not allowed to change")
        cpu_build.install_gpu_support()
        assert cpu_build.shown[0][1] == "not allowed to change" and not cpu_build.launched

    def test_nothing_to_install(self, cpu_build, monkeypatch):
        import torch
        import cabana.cabana_gui as gui
        monkeypatch.setattr(gui, "gpu_unused_reason", lambda: None)
        cpu_build.install_gpu_support()
        assert "No NVIDIA graphics card" in cpu_build.shown[0][0]
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        cpu_build.install_gpu_support()
        assert "already used" in cpu_build.shown[1][0]
        assert not cpu_build.launched
