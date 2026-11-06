import html
import os
os.environ["NUMEXPR_MAX_THREADS"] = "20"
import sys
import colorsys
from datetime import date
import numpy as np
import yaml
import imageio.v3 as iio
import tifffile as tiff
from pathlib import Path
from .utils import join_path, sanitize_filename

from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QSpinBox,
                             QVBoxLayout, QHBoxLayout, QCheckBox,
                             QPushButton, QFileDialog, QSizePolicy, QColorDialog,
                             QGroupBox, QComboBox, QWidget,
                             QStatusBar, QLineEdit, QDoubleSpinBox, QStackedWidget, QGridLayout,
                             QAction, QActionGroup, QDialog, QScrollArea, QFrame, QLayout,
                             QPlainTextEdit)
from PyQt5.QtGui import QIcon, QPalette, QFont, QFontMetrics, QKeySequence
from PyQt5.QtCore import QPointF, QSettings, QTimer, QUrl
from PyQt5.QtGui import QDesktopServices

from .ui import *
from .log import Log, LOGS_DIR
from .batch import read_checkpoint, run_settings_changes, saved_run_parameters
from .themes import THEMES, DEFAULT_THEME
from .tma import AUTO_START_DIAMETER_UM, ORIENTATIONS, EDITS_FILE
from .io import (DEFAULT_MAX_SIZE, block_grid, max_image_size, migrate_max_size, pixel_size_um,
                 save_visualisations)
from .segmenter import generate_rois, mask_from_score
from .stages import GAP_FILL_ALPHA, GAP_INTRA_RGB, GAP_OTHER_RGB, gap_base_image, gap_intra
from . import brf
from .brf import DEFAULTS as BRF_DEFAULTS
from .tma_editor import TMACoreEditor
from .tma_maps import MAP_FILE_FILTER, available_arrays, load_map_file, match_map_to_slide, write_templates
from .tma_editor import MapPreviewDialog
from . import __version__
from .updates import CHANGELOG_URL, install_problem, is_newer, restart_command
from .winshell import ICON_PATH, set_taskbar_icon
from . import whats_new
from .tour import TourOverlay, TourStep


class PageStack(QStackedWidget):
    """Stacked widget sized by its current page, not its tallest one. Inside a
    scroll area a short page then shows no empty scroll range, and a tall one
    scrolls instead of being squeezed: the minimum height is the height the
    page needs at its current width (wrapped labels included). The minimum
    width is the page's own minimum, so a long wrapped label (such as the TMA
    status line) wraps to the panel instead of widening the page past it."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._last_width = -1
        # QStackedLayout would otherwise set this widget's minimum height to
        # the tallest page of all, so every page would scroll like the tallest
        self.layout().setSizeConstraint(QLayout.SetNoConstraint)
        self.currentChanged.connect(lambda _i: self.updateGeometry())

    def sizeHint(self):
        page = self.currentWidget()
        return page.sizeHint() if page is not None else super().sizeHint()

    # QStackedLayout answers height-for-width with the maximum over all pages;
    # answer for the current page only, or a short page scrolls like the tallest
    def hasHeightForWidth(self):
        page = self.currentWidget()
        return page is not None and page.hasHeightForWidth()

    def _frame(self):
        """Width and height this widget adds around its page (border and margins)."""
        inner = self.layout().contentsRect()
        if inner.isValid():
            return max(0, self.width() - inner.width()), max(0, self.height() - inner.height())
        m = self.contentsMargins()
        return m.left() + m.right(), m.top() + m.bottom()

    def heightForWidth(self, width):
        page = self.currentWidget()
        if page is None:
            return super().heightForWidth(width)
        fw, fh = self._frame()
        inner = page.heightForWidth(width - fw) if page.hasHeightForWidth() else page.sizeHint().height()
        return inner + fh

    def minimumSizeHint(self):
        page = self.currentWidget()
        if page is None:
            return super().minimumSizeHint()
        fw, fh = self._frame()
        height = page.sizeHint().height()
        if page.hasHeightForWidth() and self.width() > 0:
            height = page.heightForWidth(self.width() - fw)
        return QSize(page.minimumSizeHint().width() + fw, max(height, page.minimumSizeHint().height()) + fh)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if event.size().width() != self._last_width:       # wrapped labels change height with width
            self._last_width = event.size().width()
            self.updateGeometry()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set window properties
        self.setWindowTitle(f"Cabana v{__version__}")
        self.setMinimumSize(800, 600)

        # Windows takes the taskbar icon from the window's properties, which
        # must be set before the window is first shown
        if sys.platform == 'win32':
            set_taskbar_icon(int(self.winId()), restart_command())

        # Make window full screen when starting
        self.showMaximized()  # This will maximize the window to full screen

        # Load saved theme and apply
        settings = QSettings('Cabana', 'CabanaGUI')
        self._current_theme = settings.value('theme', DEFAULT_THEME)
        if self._current_theme not in THEMES:
            self._current_theme = DEFAULT_THEME
        apply_theme(self._current_theme)
        self.set_theme()

        # Create the central widget with a splitter
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Create layout for central widget
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)

        # Create a custom splitter
        self.splitter = CustomSplitter(Qt.Horizontal)

        # Create and add the left dock widget to the splitter with Napari dock color
        self.dock_contents = QWidget()
        self.dock_contents.setObjectName("dockContents")
        self.dock_contents.setAttribute(Qt.WA_StyledBackground, True)   # panel colour from the stylesheet
        self.dock_contents.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.dock_contents.setAutoFillBackground(True)

        # Set Napari dock color
        dock_palette = self.dock_contents.palette()
        dock_palette.setColor(QPalette.Window, panel_colors()['panel'])
        self.dock_contents.setPalette(dock_palette)

        self.dock_layout = QVBoxLayout(self.dock_contents)
        self.dock_layout.setContentsMargins(6, 8, 6, 10)
        self.dock_layout.setSpacing(4)
        self.panel_visible = True

        self._setup_styles()
        self.dock_contents.setStyleSheet(self.panel_style)
        self._play_buttons = []              # primary buttons with a play icon (recoloured per theme)

        # Menu bar carries the Image / Parameters / Analysis / About commands,
        # so the side panel holds only the selected analysis page.
        self._setup_menu_bar()

        self.page_title = QLabel("")
        self.page_title.setStyleSheet(self.page_title_style)
        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 4, 0)
        title_row.addWidget(self.page_title)
        title_row.addStretch(1)
        # pages that can be switched off in the pipeline show their switch here
        self.page_title_right = QHBoxLayout()
        self.page_title_right.setSpacing(8)
        title_row.addLayout(self.page_title_right)
        self._title_switches = {}            # page widget -> its switch container
        self.dock_layout.addLayout(title_row)

        self.dock_inner_layout = QVBoxLayout()
        self.dock_inner_layout.setContentsMargins(0, 4, 0, 0)
        self.dock_inner_layout.setSpacing(6)
        self.dock_layout.addLayout(self.dock_inner_layout, 1)

        # Analysis pages live in a stacked widget selected from the Analysis menu,
        # inside a scroll area so a short screen scrolls the page instead of
        # squeezing its widgets below their natural height.
        self.pages = PageStack()
        self.pages.setStyleSheet(self.page_stack_style)
        self.pages_scroll = QScrollArea()
        self.pages_scroll.setWidgetResizable(True)
        self.pages_scroll.setFrameShape(QFrame.NoFrame)
        self.pages_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.pages_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.pages_scroll.setStyleSheet(self.scroll_area_style)
        self.pages_scroll.viewport().setAutoFillBackground(False)

        # Create pages
        self.start_tab = QWidget()
        self.tma_tab = QWidget()
        self.seg_tab = QWidget()
        self.brf_tab = QWidget()
        self.det_tab = QWidget()
        self.gap_tab = QWidget()
        self.bat_tab = QWidget()

        # Set up each page (TMA last: it refers to batch widgets)
        self.setup_segmentation_tab()
        self.setup_birefringence_tab()
        self.setup_detection_tab()
        self.setup_gap_analysis_tab()
        self.setup_batch_processing_tab()
        self.setup_tma_tab()
        self.setup_start_tab()

        # Page 0 is the Start page (not in the Analysis menu); the rest follow menu order
        self._page_titles = ["Start"]
        self.pages.addWidget(self.start_tab)
        for (title, action), page in zip(self._page_actions,
                                         (self.tma_tab, self.brf_tab, self.seg_tab, self.det_tab,
                                          self.gap_tab, self.bat_tab)):
            index = self.pages.addWidget(page)
            self._page_titles.append(title)
            action.triggered.connect(lambda _checked=False, i=index: self.show_page(i))
        self.pages_scroll.setWidget(self.pages)
        self.dock_inner_layout.addWidget(self.pages_scroll, 1)
        self._share_hints_with_labels()
        # buttons and checkboxes take focus from the keyboard (Tab) only, so a
        # mouse click leaves no focus ring behind
        for widget in self.dock_contents.findChildren((QPushButton, QCheckBox)):
            widget.setFocusPolicy(Qt.TabFocus)
        # scrolling the side panel never changes a setting the user has not clicked
        self._wheel_guard = WheelGuard(self)
        self._wheel_guard.guard(self.pages)
        self.show_page(0)

        # Add progress bar
        self.progress_bar = PercentageProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet(self.progressbar_style)
        self.dock_inner_layout.addWidget(self.progress_bar)

        # Create and add the image panel to the splitter
        self.image_panel = ImagePanel()
        self.image_panel.imageDropped.connect(self.load_original_image)

        # Page rail at the window's left edge: an icon per page; the open
        # page's icon folds the side panel and any icon unfolds it
        icons = ['page-start', 'page-tma', 'page-brf', 'page-seg', 'page-fibres', 'page-gaps', 'page-batch']
        tips = ["Start"] + [f"{title} ({action.shortcut().toString(QKeySequence.NativeText)})"
                            for title, action in self._page_actions]
        self.page_rail = PageRail(list(zip(icons, tips)))
        self.page_rail.page_clicked.connect(self._rail_clicked)
        self.page_rail.help_clicked.connect(self.start_tour)
        self.page_rail.set_current(self.pages.currentIndex())

        # Add widgets to splitter
        self.splitter.addWidget(self.dock_contents)
        self.splitter.addWidget(self.image_panel)

        # Let the image panel stretch, dock panel keeps its natural size
        self.splitter.setStretchFactor(0, 0)  # dock: don't stretch
        self.splitter.setStretchFactor(1, 1)  # image: stretch to fill

        # Compute the minimum dock width so tab labels are never clipped.
        # Match main-branch behavior: let the default tab bar manage elision inside the available width.
        # The panel never gets narrower than its widest page (plus the scroll
        # bar), so a short window squeezes the image panel instead of hiding
        # the right edge of a page
        widest = max(self.pages.widget(i).minimumSizeHint().width() for i in range(self.pages.count()))
        margins = self.dock_layout.contentsMargins()
        self.dock_contents.setMinimumWidth(max(200, widest + margins.left() + margins.right() + 20))

        # Set initial sizes so the left panel opens wide enough for the full Analysis tab labels.
        # Account for all nesting: dock_layout margins, QGroupBox stylesheet padding+border,
        # inner layout margins, and the splitter handle.
        dock_margins = self.dock_layout.contentsMargins()
        inner_margins = self.dock_inner_layout.contentsMargins()
        # QGroupBox CSS: padding 8px L/R + border 1px L/R = 18px total
        groupbox_chrome = 18
        initial_dock_width = max(
            self.dock_contents.minimumWidth(),
            400
            + dock_margins.left() + dock_margins.right()
            + groupbox_chrome
            + inner_margins.left() + inner_margins.right()
            + self.splitter.handleWidth(),
        )
        self.splitter.setSizes([initial_dock_width, 800])

        # Rail and splitter side by side in the main layout
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        body.addWidget(self.page_rail)
        body.addWidget(self.splitter, 1)
        self.main_layout.addLayout(body)

        # Set handle width thinner
        self.splitter.setHandleWidth(5)

        # Make handle transparent by default
        self.splitter.setStyleSheet("QSplitter::handle { background-color: transparent; }")

        self.img_path = None
        self.ori_img = None
        self.seg_img = None
        self._seg_score = None      # (score map, min_size) of the last segmentation, for re-thresholding
        self.frb_img = None
        self.wdt_img = None
        self.gap_img = None         # fibre mask view shown under the gap circles
        self.gap_circles = None     # (row, col, radius) px of the last gap analysis
        self._gap_intra = None      # per circle: centre inside the segmentation mask
        self._brf_hsb = None        # (hue, saturation, brightness) of the open image, for re-thresholding
        self._brf_masks = None      # (band masks, gate) at the current Birefringence settings
        self._brf_pixel_size = None  # µm/pixel of the open image, when its metadata has it
        self.brf_worker = None
        self.segmentation_worker = None
        self.detection_worker = None
        self.gap_analysis_worker = None
        self.load_default_params()
        self.panel_visible = True

        # --- Status bar ---
        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet(self.status_bar_style)
        self.setStatusBar(self.status_bar)
        self.status_file_label = QLabel("No image loaded")
        self.status_dims_label = QLabel("")
        self.status_zoom_label = QLabel("")
        self.status_bar.addWidget(self.status_file_label, 1)
        self.status_bar.addPermanentWidget(self.status_dims_label)
        self.status_bar.addPermanentWidget(self.status_zoom_label)

        # Theme switcher
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(THEMES.keys())
        self.theme_combo.setCurrentText(self._current_theme)
        self.theme_combo.setFixedWidth(100)
        self.theme_combo.setStyleSheet(self.theme_combo_style)
        self.theme_combo.currentTextChanged.connect(self._on_theme_changed)
        self.status_bar.addPermanentWidget(self.theme_combo)

        # Connect zoom updates from image panel
        self.image_panel.zoomChanged.connect(self._update_zoom_status)

        # Look for a newer release once a day, after the window is up
        self.update_worker = None
        if self.check_updates_on_start:
            QTimer.singleShot(3000, self._startup_update_check)
        # What's New once after an upgrade, the tour offer on the very first start
        self._tour = None
        if self.startup_messages:
            QTimer.singleShot(600, self._startup_messages)


    def _setup_menu_bar(self):
        """Build File / Parameters / Analysis / About menus.

        The actions replace the former Image and Parameters button groups and
        the page navigation rail. ``self.load_btn``-style aliases are kept so
        existing enable/disable logic keeps working on the actions.
        """
        bar = self.menuBar()
        bar.setNativeMenuBar(sys.platform == 'darwin')
        bar.setStyleSheet(self.menubar_style)

        # --- File ---
        file_menu = bar.addMenu("&File")
        self.load_action = QAction("&Open Image…", self)
        self.load_action.setShortcut(QKeySequence.Open)
        self.load_action.setStatusTip("Load an image file")
        self.load_action.triggered.connect(self.load_image)
        file_menu.addAction(self.load_action)

        self.reload_action = QAction("&Reload Image", self)
        self.reload_action.setShortcut(QKeySequence("Ctrl+R"))
        self.reload_action.setStatusTip("Reload the original image")
        self.reload_action.setEnabled(False)
        self.reload_action.triggered.connect(self.reload_image)
        file_menu.addAction(self.reload_action)

        file_menu.addSeparator()
        self.open_slide_action = QAction("Open &TMA Slide…", self)
        self.open_slide_action.setShortcut(QKeySequence("Ctrl+Shift+O"))
        self.open_slide_action.setStatusTip("Choose a whole-slide TMA scan on the TMA Dearrayer page")
        self.open_slide_action.triggered.connect(self._open_tma_slide_from_menu)
        file_menu.addAction(self.open_slide_action)
        self.save_map_templates_action = QAction("Generate TMA Map &Template…", self)
        self.save_map_templates_action.setStatusTip(
            "Write a template for your own TMA map")
        self.save_map_templates_action.triggered.connect(self.save_tma_map_templates)
        file_menu.addAction(self.save_map_templates_action)

        file_menu.addSeparator()
        quit_action = QAction("&Quit", self)
        quit_action.setShortcut(QKeySequence.Quit)
        quit_action.setMenuRole(QAction.QuitRole)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # --- Parameters ---
        params_menu = bar.addMenu("&Parameters")
        self.import_params_action = QAction("&Import…", self)
        self.import_params_action.setShortcut(QKeySequence("Ctrl+I"))
        self.import_params_action.setStatusTip("Load parameters from a YAML file")
        self.import_params_action.triggered.connect(self.import_parameters)
        params_menu.addAction(self.import_params_action)

        self.export_params_action = QAction("&Export…", self)
        self.export_params_action.setShortcut(QKeySequence("Ctrl+E"))
        self.export_params_action.setStatusTip("Save the current parameters to a YAML file")
        self.export_params_action.triggered.connect(self.export_parameters)
        params_menu.addAction(self.export_params_action)

        params_menu.addSeparator()
        reset_action = QAction("Restore &Defaults", self)
        reset_action.setStatusTip("Reset every parameter to the bundled defaults")
        reset_action.triggered.connect(self.restore_default_parameters)
        params_menu.addAction(reset_action)

        # --- Analysis ---
        analysis_menu = bar.addMenu("&Analysis")
        self._page_group = QActionGroup(self)
        # ExclusiveOptional lets the Start page leave every Analysis entry unchecked
        self._page_group.setExclusionPolicy(QActionGroup.ExclusionPolicy.ExclusiveOptional)
        self._page_actions = []
        for i, title in enumerate(("TMA Dearrayer", "Birefringence", "Segmentation", "Fibre Detection",
                                   "Gap Analysis", "Batch Run")):
            action = QAction(title, self)
            action.setCheckable(True)
            action.setShortcut(QKeySequence(f"Ctrl+{i + 1}"))
            action.setStatusTip(f"Show the {title} parameters in the side panel")
            self._page_group.addAction(action)
            analysis_menu.addAction(action)
            self._page_actions.append((title, action))
        analysis_menu.addSeparator()
        self.toggle_panel_action = QAction("Show Side &Panel", self)
        self.toggle_panel_action.setCheckable(True)
        self.toggle_panel_action.setChecked(True)
        self.toggle_panel_action.setShortcut(QKeySequence("Ctrl+B"))
        self.toggle_panel_action.triggered.connect(self.toggle_panel)
        analysis_menu.addAction(self.toggle_panel_action)

        # --- About ---
        about_menu = bar.addMenu("&Help")
        about_action = QAction("&About Cabana", self)
        about_action.setMenuRole(QAction.AboutRole)
        about_action.triggered.connect(self._show_about_dialog)
        about_menu.addAction(about_action)
        docs_action = QAction("&Documentation", self)
        docs_action.triggered.connect(
            lambda: QDesktopServices.openUrl(QUrl("https://cabana.readthedocs.io")))
        about_menu.addAction(docs_action)
        self.tour_action = QAction("Take the &Tour", self)
        self.tour_action.triggered.connect(self.start_tour)
        about_menu.addAction(self.tour_action)
        self.whats_new_action = QAction("What's &New", self)
        self.whats_new_action.triggered.connect(lambda: self.show_whats_new())
        about_menu.addAction(self.whats_new_action)
        issue_action = QAction("Report an &Issue", self)
        issue_action.triggered.connect(
            lambda: QDesktopServices.openUrl(QUrl("https://github.com/Garvan-Data-Science-Platform/Cabana/issues")))
        about_menu.addAction(issue_action)
        about_menu.addSeparator()
        self.check_updates_action = QAction("Check for &Updates…", self)
        # macOS moves it into the application menu, under About Cabana
        self.check_updates_action.setMenuRole(QAction.ApplicationSpecificRole)
        self.check_updates_action.triggered.connect(lambda: self.check_for_updates())
        about_menu.addAction(self.check_updates_action)
        version_action = QAction("&Version", self)
        version_action.setMenuRole(QAction.NoRole)
        version_action.triggered.connect(self._show_about_dialog)
        about_menu.addAction(version_action)
        license_action = QAction("&Licence", self)
        license_action.setMenuRole(QAction.NoRole)
        license_action.triggered.connect(self._show_license_dialog)
        about_menu.addAction(license_action)

        # Aliases used by existing enable/disable logic
        self.load_btn = self.load_action
        self.reload_btn = self.reload_action
        self.load_params_btn = self.import_params_action
        self.export_btn = self.export_params_action

    def show_page(self, index):
        """Show side-panel page ``index`` (0 = Start) and sync the Analysis menu."""
        self.pages.setCurrentIndex(index)
        self.page_title.setText(self._page_titles[index])
        for page, switch in self._title_switches.items():
            switch.setVisible(self.pages.widget(index) is page)
        if hasattr(self, 'page_rail'):
            self.page_rail.set_current(index)
        if index == 0:
            checked = self._page_group.checkedAction()
            if checked is not None:
                checked.setChecked(False)
        else:
            self._page_actions[index - 1][1].setChecked(True)
        if not self.panel_visible:
            self.toggle_panel()

    def setup_start_tab(self):
        """Landing page shown at launch: the three ways to begin."""
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 10, 6, 6)
        layout.setSpacing(10)

        intro = QLabel("Open a TMA slide to cut it into cores, open an image to tune parameters "
                       "on it, or import a saved parameter file. The analysis pages are on the bar "
                       "at the left and under the Analysis menu (Ctrl/Cmd+1 to 6).")
        intro.setWordWrap(True)
        intro.setStyleSheet(self.value_label_style)
        layout.addWidget(intro)

        self.start_actions = QWidget()          # what the tour's first step points at
        actions = QVBoxLayout(self.start_actions)
        actions.setContentsMargins(0, 0, 0, 0)
        actions.setSpacing(10)
        self.start_slide_btn = self._primary_button("Open TMA Slide…", play=False)
        self.start_slide_btn.setToolTip("Open a whole-slide TMA scan on the TMA Dearrayer page.")
        self.start_slide_btn.clicked.connect(self._open_tma_slide_from_menu)
        actions.addWidget(self.start_slide_btn)
        self.start_open_btn = QPushButton("Open Image…")
        self.start_open_btn.setStyleSheet(self.btn_style)
        self.start_open_btn.setToolTip("Open an image to tune the parameters on.")
        self.start_open_btn.clicked.connect(self.load_image)
        actions.addWidget(self.start_open_btn)
        self.start_params_btn = QPushButton("Import Parameters…")
        self.start_params_btn.setStyleSheet(self.btn_style)
        self.start_params_btn.setToolTip("Load a saved Parameters.yml into all pages.")
        self.start_params_btn.clicked.connect(self.import_parameters)
        actions.addWidget(self.start_params_btn)
        layout.addWidget(self.start_actions)

        layout.addStretch()
        self.start_tab.setLayout(layout)

    confirm_on_close = True     # tests and scripts switch the quit confirmation off

    def closeEvent(self, event):
        """Ask before quitting when work would be lost (a running job, unsaved
        TMA edits); otherwise close at once."""
        if not self.confirm_on_close or self._confirm_quit():
            event.accept()
        else:
            event.ignore()

    def _work_at_risk(self):
        """Plain-language list of running jobs and unsaved edits."""
        items = []
        running = [(getattr(self, "batch_worker", None), "a batch run"),
                   (getattr(self, "tma_worker", None), "a TMA fit or export"),
                   (getattr(self, "segmentation_worker", None), "segmentation"),
                   (getattr(self, "detection_worker", None), "fibre detection"),
                   (getattr(self, "gap_analysis_worker", None), "gap analysis"),
                   (getattr(self, "brf_folder_worker", None), "a birefringence folder run"),
                   (getattr(self, "update_worker", None), "the Cabana update")]
        for worker, label in running:
            if worker is not None and worker.isRunning():
                items.append(f"{label} is still running")
        if getattr(self, "tma_pre", None) is not None and getattr(self, "_tma_unsaved", False):
            items.append("hand edits of the TMA cores have not been exported or saved")
        return items

    def _confirm_quit(self):
        """True when quitting may go ahead: nothing is at risk, or the user confirmed."""
        risks = self._work_at_risk()
        if not risks:
            return True
        return self._ask_tma("Quit Cabana", "Quit Cabana?", "If you quit now:\n- " + "\n- ".join(risks),
                             "Confirm", "Cancel")

    TMA_LABEL_WIDTH = 86      # TMA labels are short: the narrower column leaves room for paired fields
    MAP_LOAD_ITEM = "__load_map__"

    def _tma_selected_map(self):
        """The array combo's map: None, a built-in array number or a loaded TMAMap."""
        data = self.tma_array_combo.currentData()
        return None if data == self.MAP_LOAD_ITEM else data

    def _tma_array_selected(self, index):
        """Array combo changed: 'Load map file…' loads a map, anything else relabels."""
        combo = self.tma_array_combo
        if combo.itemData(index) == self.MAP_LOAD_ITEM:
            chosen = self.load_tma_map_file()
            target = self._tma_array_prev
            if chosen is not None:
                target = next(i for i in range(combo.count()) if combo.itemData(i) is chosen)
            combo.blockSignals(True)
            combo.setCurrentIndex(target)
            combo.blockSignals(False)
            if chosen is None:
                return
        self._tma_array_prev = combo.currentIndex()
        self._tma_map_changed()

    def load_tma_map_file(self, path=None):
        """Read a map file, show it for checking and add its maps to the array
        combo. Returns the map chosen in the preview, or None."""
        if path is None:
            start = os.path.dirname(self.tma_slide) if self.tma_slide else ""
            path, _ = QFileDialog.getOpenFileName(self, "Load TMA Map", start, MAP_FILE_FILTER)
            if not path:
                return None
        map_file = load_map_file(path)
        slide = os.path.basename(self.tma_slide) if self.tma_slide else ""
        chosen = self._preview_map_file(map_file, match_map_to_slide(map_file.maps, slide))
        if chosen is None:
            return None
        self._add_tma_maps(map_file.maps)
        return chosen

    def _preview_map_file(self, map_file, preselect):
        """Show the map file for checking; the chosen map, or None if cancelled."""
        dlg = MapPreviewDialog(self, map_file, preselect, style=self.msgbox_style.replace("QMessageBox", "QDialog"))
        return dlg.selected_map() if dlg.exec_() == QDialog.Accepted else None

    def _add_tma_maps(self, maps):
        """Add loaded maps to the array combo before 'Load map file…', replacing
        a map of the same name from the same file."""
        combo = self.tma_array_combo
        combo.blockSignals(True)
        current = combo.currentData()
        for m in maps:
            for i in reversed(range(combo.count())):
                d = combo.itemData(i)
                if hasattr(d, "name") and d.name == m.name and d.source == m.source:
                    combo.removeItem(i)
            combo.insertItem(combo.findData(self.MAP_LOAD_ITEM), f"{m.name} ({m.source})", m)
        idx = next((i for i in range(combo.count()) if combo.itemData(i) is current), 0)
        combo.setCurrentIndex(idx)
        combo.blockSignals(False)

    def save_tma_map_templates(self):
        folder = QFileDialog.getExistingDirectory(self, "Generate TMA Map Template: Choose a Folder", "", QFileDialog.ShowDirsOnly)
        if not folder:
            return
        try:
            paths = write_templates(folder)
        except OSError as e:
            self._notify("TMA Map Template", "Could not write the template.", str(e), warning=True)
            return
        self._notify("TMA Map Template", "Wrote:\n" + "\n".join(paths),
                     "Open it in a spreadsheet, type or paste your map into the grid (the notes at "
                     "the top explain how), save it as CSV, then choose 'Load map file…' under Array Map.")

    def _open_tma_slide_from_menu(self):
        self.show_page(self.pages.indexOf(self.tma_tab))
        self.select_tma_slide()

    def _message(self, title, text, info="", buttons=("OK",), default=None, details=None, rich=False):
        """A card-style message: ``text`` (rich text when ``rich``), then ``info``
        in a fainter note and optional ``details`` behind a Show Details button. ``default`` (the first
        button when None) is the primary button, placed last on the right.
        Returns the label clicked, or None when the dialog is closed."""
        default = default or buttons[0]
        as_html = lambda t: html.escape(t).replace("\n", "<br>")  # noqa: E731
        link = color_to_stylesheet(COLORS['highlight'])
        dlg = CardDialog(self, title, width=460)
        dlg.add_text(text.replace("<a ", f"<a style='color:{link}; text-decoration:none;' ") if rich
                     else as_html(text))
        if info:
            dlg.add_text(as_html(info), kind="note")
        clicked = []
        if details:
            box = QPlainTextEdit(details)
            box.setObjectName("dialogDetails")
            box.setReadOnly(True)
            box.setMinimumHeight(160)
            box.setVisible(False)
            dlg.body_layout.addWidget(box)
            toggle = QPushButton("Show Details")
            toggle.setStyleSheet(self.btn_style)
            toggle.clicked.connect(lambda: (box.setVisible(not box.isVisible()),
                                            toggle.setText("Hide Details" if box.isVisible() else "Show Details")))
            dlg.footer_layout.insertWidget(0, toggle)
        for label in [b for b in buttons if b != default] + [default]:
            dlg.add_button(label, primary=label == default,
                           slot=lambda _=False, label=label: clicked.append(label))
        dlg.exec_()
        return clicked[0] if clicked else None

    def _notify(self, title, text, info="", warning=False):
        """One-button message. Monkeypatched in tests."""
        self._message(title, text, info)

    def restore_default_parameters(self):
        """Reload the bundled default parameters into the widgets."""
        self.load_default_params()
        self.apply_params_to_widgets()

    def _share_hints_with_labels(self):
        """Give every row label the tooltip of the control it names.

        Qt shows a tooltip only for the widget under the cursor, so a hint set
        on a slider is invisible when the user hovers its label. For each
        layout on every page the tooltip of a control is copied to the
        neighbouring QLabel(s) that have none, and every tooltip is mirrored
        as a status tip so it also appears in the status bar.
        """
        from PyQt5.QtWidgets import QLayout, QGridLayout as _Grid

        def items(layout):
            return [layout.itemAt(i) for i in range(layout.count())]

        def walk(layout):
            if layout is None:
                return
            if isinstance(layout, _Grid):
                for r in range(layout.rowCount()):
                    row = []
                    for c in range(layout.columnCount()):
                        it = layout.itemAtPosition(r, c)
                        if it is not None and it.widget() is not None:
                            row.append(it.widget())
                    pair_up(row)
            else:
                widgets = [it.widget() for it in items(layout) if it.widget() is not None]
                pair_up(widgets)
            for it in items(layout):
                if it.layout() is not None:
                    walk(it.layout())
                elif it.widget() is not None and it.widget().layout() is not None \
                        and not isinstance(it.widget(), QStackedWidget):
                    walk(it.widget().layout())

        def pair_up(widgets):
            # a label takes the hint of the nearest following non-label widget
            for i, w in enumerate(widgets):
                if isinstance(w, QLabel) and not w.toolTip():
                    for other in widgets[i + 1:]:
                        if not isinstance(other, QLabel) and other.toolTip():
                            w.setToolTip(other.toolTip())
                            break
            for w in widgets:
                if w.toolTip() and not w.statusTip():
                    w.setStatusTip(w.toolTip().replace("\n", " "))

        for i in range(self.pages.count()):
            walk(self.pages.widget(i).layout())

    def _update_zoom_status(self, zoom_factor):
        """Update the zoom display in the status bar"""
        self.status_zoom_label.setText(f"Zoom: {zoom_factor:.0%}  ")

    def _update_image_status(self):
        """Update status bar with current image info"""
        if self.img_path and self.ori_img is not None:
            filename = os.path.basename(self.img_path)
            h, w = self.ori_img.shape[:2]
            self.status_file_label.setText(f"  {filename}")
            self.status_dims_label.setText(f"{w} x {h}  ")
        else:
            self.status_file_label.setText("  No image loaded")
            self.status_dims_label.setText("")
        self._refresh_size_note()

    def toggle_panel(self):
        if self.panel_visible:
            self.dock_contents.hide()
        else:
            self.dock_contents.show()
        self.panel_visible = not self.panel_visible
        if hasattr(self, 'toggle_panel_action'):
            self.toggle_panel_action.setChecked(self.panel_visible)


    def _rail_clicked(self, index):
        """A page icon on the rail: the open page's icon folds the panel, any
        other icon (or any icon while folded) shows that page."""
        if index == self.pages.currentIndex() and self.panel_visible:
            self.toggle_panel()
        else:
            self.show_page(index)

    def _setup_styles(self) -> None:
        """Set up all style sheets"""
        # Button style
        self.btn_style = generate_button_style()

        # Page stack / combo box / menu bar / page title styles
        self.page_stack_style = generate_page_stack_style()
        self.scroll_area_style = generate_scroll_area_style()
        self.combo_style = generate_combo_style()
        self.menubar_style = generate_menubar_style()
        self.page_title_style = (
            f"color: {color_to_stylesheet(COLORS['text'])}; font-weight: 600; "
            f"font-size: {FONT_SIZES['title']}px; padding: 2px 4px 4px 4px;")

        # Progress bar style
        self.progressbar_style = generate_progressbar_style()

        # Spinner style
        self.spinner_style = generate_spinner_style()

        # Messagebox style
        self.msgbox_style = generate_messagebox_style()

        # Primary action button style (filled highlight)
        self.primary_btn_style = generate_primary_button_style()

        # Checkbox style
        self.checkbox_style = generate_checkbox_style()

        # Group box style
        self.group_style = generate_group_box_style()

        # Side panel: section cards, row labels, notes, option lists
        self.panel_style = generate_panel_style()


        # Value label style (slider readouts)
        self.value_label_style = (
            f"color: {color_to_stylesheet(COLORS['text_dim'])}; font-size: {FONT_SIZES['small']}px;"
        )

        # Status bar style
        self.status_bar_style = (
            f"QStatusBar {{ background-color: {color_to_stylesheet(COLORS['background'])}; "
            f"color: {color_to_stylesheet(COLORS['text_dim'])}; "
            f"border-top: 1px solid {color_to_stylesheet(COLORS['border'])}; "
            f"font-size: {FONT_SIZES['small']}px; padding: 2px 8px; }}"
            f"QStatusBar::item {{ border: none; }}"
        )

        # Theme combo style
        self.theme_combo_style = (
            f"QComboBox {{ background-color: {color_to_stylesheet(COLORS['dock'])}; "
            f"color: {color_to_stylesheet(COLORS['text'])}; "
            f"border: 1px solid {color_to_stylesheet(COLORS['border'])}; "
            f"border-radius: 3px; padding: 2px 6px; "
            f"font-size: {FONT_SIZES['small']}px; }}"
            f"QComboBox::drop-down {{ border: none; width: 16px; }}"
            f"QComboBox::down-arrow {{ image: none; border-left: 4px solid transparent; "
            f"border-right: 4px solid transparent; "
            f"border-top: 5px solid {color_to_stylesheet(COLORS['text_dim'])}; }}"
            f"QComboBox QAbstractItemView {{ background-color: {color_to_stylesheet(COLORS['elevated'])}; "
            f"color: {color_to_stylesheet(COLORS['text'])}; "
            f"border: 1px solid {color_to_stylesheet(COLORS['border'])}; "
            f"selection-background-color: {color_to_stylesheet(COLORS['highlight'])}; "
            f"selection-color: {color_to_stylesheet(COLORS['background'])}; }}"
        )

    @staticmethod
    def _page_layout(spacing=14):
        """Top-level layout of a side-panel page: a column of section cards."""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 4, 0, 8)
        layout.setSpacing(spacing)
        return layout

    def _set_play_icon(self, btn):
        icon = QIcon()
        icon.addFile(icon_path('play', COLORS['background']), QSize(), QIcon.Normal)
        icon.addFile(icon_path('play', COLORS['text_muted']), QSize(), QIcon.Disabled)
        btn.setIcon(icon)
        btn.setIconSize(QSize(12, 12))

    def _primary_button(self, text, play=True):
        """The main action of a section card (filled; a play icon for runs)."""
        btn = QPushButton(text)
        btn.setStyleSheet(self.primary_btn_style)
        if play:
            self._set_play_icon(btn)
            self._play_buttons.append(btn)
        return btn

    def setup_batch_processing_tab(self):
        """Set up the Batch Run page: a Files card (the image, mask and results
        folders and the parameter file) and a Run card with this run's
        settings and the action."""
        layout = self._page_layout()

        flow = SectionCard("Files")
        self.input_folder_path = PathEdit("Not selected")
        self.input_folder_path.setToolTip("Images to analyse (TIFF/PNG/JPEG); the pixel size is read from their metadata.")
        self.input_btn = self.input_folder_path.pick_action
        self.input_btn.setToolTip("Choose the image folder")
        self.input_btn.triggered.connect(self.select_input_folder)
        flow.add_path_row("Images", self.input_folder_path)

        # Optional ROI mask folder (e.g. TMA core circles); empty = no masks
        self.mask_folder_path = PathEdit("")
        self.mask_folder_path.setPlaceholderText("Optional")
        self.mask_folder_path.setToolTip(
            "Optional masks named like the images (white = analyse), e.g. a TMA export's\n"
            "Masks folder. Empty: no masks.")
        self.mask_btn = self.mask_folder_path.pick_action
        self.mask_btn.setToolTip("Choose the ROI mask folder")
        self.mask_btn.triggered.connect(self.select_mask_folder)
        self.mask_clear_action = add_clear_action(self.mask_folder_path, self.clear_mask_folder,
                                                  "Clear the ROI mask folder (no masks)")
        flow.add_path_row("ROI masks", self.mask_folder_path)
        self.mask_folder = None

        self.param_file_path = PathEdit("Not selected", icon="file")
        self.param_file_path.setToolTip("Parameters.yml applied to every image (Parameters > Export).")
        self.param_btn = self.param_file_path.pick_action
        self.param_btn.setToolTip("Choose the parameter file")
        self.param_btn.triggered.connect(self.select_param_file)
        flow.add_path_row("Parameters", self.param_file_path)

        self.output_folder_path = PathEdit("Not selected")
        self.output_folder_path.setToolTip("Where the results go; also holds the checkpoint for resuming a run.")
        self.output_btn = self.output_folder_path.pick_action
        self.output_btn.setToolTip("Choose the results folder")
        self.output_btn.triggered.connect(self.select_output_folder)
        flow.add_path_row("Results", self.output_folder_path)
        self.bat_files_card = flow
        layout.addWidget(flow)

        self.bat_run_card = run = SectionCard("Run")
        self.batch_size_spinner = QSpinBox()
        self.batch_size_spinner.setRange(1, 100)
        self.batch_size_spinner.setValue(5)
        self.batch_size_spinner.setFixedWidth(110)
        self.batch_size_spinner.setStyleSheet(self.spinner_style)
        self.batch_size_spinner.setToolTip("Images per batch; lower it for very large images to save memory.")
        run.add_row("Batch size", self.batch_size_spinner, stretch=False)

        # Max Size of the run, shown read-only: it comes from the parameter file
        # (set it on the Segmentation page and export), or from the interrupted
        # run on a resume, so the file kept with the results describes the run
        self.batch_max_size_label = QLabel("–")
        self.batch_max_size_label.setToolTip(
            "Larger images are split into blocks of this size. From the parameter file;\n"
            "set it on the Segmentation page.")
        run.add_row("Max size", self.batch_max_size_label)

        # the heading sits close to its list, a little apart from Max size
        save_title = QLabel("Save, besides the measurements")
        save_title.setObjectName("optionTitle")
        save_block = QVBoxLayout()
        save_block.setContentsMargins(0, 6, 0, 0)
        save_block.setSpacing(8)
        save_block.addWidget(save_title)
        self.visuals_cb = QCheckBox("Visualisations")
        self.visuals_cb.setChecked(True)
        self.visuals_cb.setToolTip(
            "Save the colour maps and gap images; untick to keep only the measurements.\n"
            "Starts from the parameter file; a change applies to this run only.")
        self.stats_cb = QCheckBox("Per-patient statistics")
        self.stats_cb.setChecked(False)
        self.stats_cb.setEnabled(False)
        self.stats_cb.setToolTip(
            "Per-patient mean, STD and SEM (QuantificationResults_MEAN_STD_SEM.csv).")
        self.scores_cb = QCheckBox("Risk scores")
        self.scores_cb.setChecked(False)
        self.scores_cb.setEnabled(False)
        self.scores_cb.setToolTip(
            "Rigidity and bundling risk scores (QuantificationResults_SCORES.csv).")
        options = OptionList()
        for cb, note in ((self.visuals_cb, "colour maps, gap images"),
                         (self.stats_cb, "mean, STD, SEM"),
                         (self.scores_cb, "rigidity, bundling")):
            cb.setStyleSheet(self.checkbox_style)
            options.add_option(cb, note)
        save_block.addWidget(options)
        run.add_layout(save_block)

        # Process / cancel share the footer row
        batch_btn_layout = QHBoxLayout()
        batch_btn_layout.setSpacing(8)
        self.process_batch_btn = self._primary_button("Process Batch")
        self.process_batch_btn.clicked.connect(self.run_batch_processing)
        self.process_batch_btn.setEnabled(False)
        batch_btn_layout.addWidget(self.process_batch_btn)
        self.cancel_batch_btn = QPushButton("Cancel")
        self.cancel_batch_btn.clicked.connect(self._cancel_batch_processing)
        self.cancel_batch_btn.setVisible(False)
        self.cancel_batch_btn.setStyleSheet(self.btn_style)
        batch_btn_layout.addWidget(self.cancel_batch_btn)
        run.add_footer(batch_btn_layout)
        # the run's progress, under its button: status (batch / stage / image) and bar
        self.progress_label = QLabel("")
        self.progress_label.setWordWrap(True)
        self.progress_label.setStyleSheet(self.value_label_style)
        self.progress_label.setVisible(False)
        run.add_footer(self.progress_label)
        self.batch_progress_bar = PercentageProgressBar()
        self.batch_progress_bar.setRange(0, 100)
        self.batch_progress_bar.setStyleSheet(self.progressbar_style)
        self.batch_progress_bar.setVisible(False)
        run.add_footer(self.batch_progress_bar)
        layout.addWidget(run)

        layout.addStretch()
        self.bat_tab.setLayout(layout)

    # ------------------------------------------------------------------
    # TMA page
    # ------------------------------------------------------------------
    def setup_tma_tab(self):
        """Set up the TMA preprocessing page: Fit, Filter, Edit and Export cards
        (load slide -> fit cores -> correct -> export)."""
        layout = self._page_layout(spacing=10)
        # two columns: short fields in the field column, a fixed block at the right edge
        font = QFont()
        font.setPixelSize(FONT_SIZES['base'])
        second = max(QFontMetrics(font).horizontalAdvance(t) for t in ("Orientation", "Min stain", "Mask shrink"))
        split = dict(field_width=112, label2_width=second + 2, right_width=100)

        # --- Fit -------------------------------------------------------------
        fit = SectionCard("Fit", compact=True)
        fit.LABEL_WIDTH = self.TMA_LABEL_WIDTH
        fit.title_label.setToolTip("Settings for Fit Cores; fit again after changing them.")
        self.tma_slide_path = PathEdit("", icon="file")
        self.tma_slide_path.setPlaceholderText("Olympus .vsi, TIFF or PNG")
        self.tma_slide_path.setToolTip(
            "Whole-slide TMA scan: Olympus .vsi (with its _<name>_ folder), TIFF or PNG.")
        self.tma_slide_btn = self.tma_slide_path.pick_action
        self.tma_slide_btn.setToolTip("Choose the slide file.")
        self.tma_slide_btn.triggered.connect(self.select_tma_slide)
        fit.add_row("Slide", self.tma_slide_path)

        self.tma_array_combo = QComboBox()
        self.tma_array_combo.addItem("None (grid position only)", None)
        for n in available_arrays():
            self.tma_array_combo.addItem(f"ICGC Array {n}", n)
        self.tma_array_combo.addItem("Load map file…", self.MAP_LOAD_ITEM)
        self._tma_array_prev = 0
        self.tma_array_combo.setStyleSheet(self.combo_style)
        self.tma_array_combo.setToolTip(
            "Map that names the cores by patient: a built-in ICGC/APGI array or your own\n"
            "(Load map file…; File > Generate TMA Map Template… makes a template).")
        fit.add_row("Array map", self.tma_array_combo)

        self.tma_orientation_combo = QComboBox()
        for o in ORIENTATIONS:
            self.tma_orientation_combo.addItem("Auto" if o == "auto" else o, o)
        self.tma_orientation_combo.setStyleSheet(self.combo_style)
        self.tma_orientation_combo.setToolTip(
            "How the printed map sits on the scan. Auto matches the missing cores; a full array\n"
            "can mistake a rotation for its mirror, so check the control cores. Relabels at once.")
        self.tma_orientation_combo.currentIndexChanged.connect(self._tma_map_changed)
        self.tma_array_combo.currentIndexChanged.connect(self._tma_array_selected)
        self.tma_orientation_combo.setFixedWidth(104)

        self.tma_pixel_size_spin = QDoubleSpinBox()
        self.tma_pixel_size_spin.setRange(0.01, 50.0)
        self.tma_pixel_size_spin.setDecimals(4)
        self.tma_pixel_size_spin.setSingleStep(0.01)
        self.tma_pixel_size_spin.setValue(0.2738)
        self.tma_pixel_size_spin.setSuffix(" µm")
        self.tma_pixel_size_spin.setFixedWidth(140)
        self.tma_pixel_size_spin.setStyleSheet(self.spinner_style)
        self.tma_pixel_size_spin.setToolTip("Filled from the slide metadata when available.")
        fit.add_split_row("Pixel size", self.tma_pixel_size_spin, self.tma_orientation_combo, "Orientation", **split)

        self.tma_core_auto_cb = QCheckBox("Auto")
        self.tma_core_auto_cb.setChecked(True)
        self.tma_core_auto_cb.setStyleSheet(self.checkbox_style)
        self.tma_core_auto_cb.setToolTip(
            "Measure the core diameter at every fit; untick to type it in.")
        self.tma_core_diameter_spin = QSpinBox()
        self.tma_core_diameter_spin.setRange(0, 5000)
        self.tma_core_diameter_spin.setSpecialValueText("—")     # shown at 0: Auto, not measured yet
        self.tma_core_diameter_spin.setSingleStep(50)
        self.tma_core_diameter_spin.setValue(0)
        self.tma_core_diameter_spin.setSuffix(" µm")
        self.tma_core_diameter_spin.setFixedWidth(120)
        self.tma_core_diameter_spin.setStyleSheet(self.spinner_style)
        self.tma_core_diameter_spin.setEnabled(False)
        self._set_tma_core_tooltip()
        self.tma_core_auto_cb.toggled.connect(self._tma_core_auto_toggled)
        fit.add_split_row("Core Ø", self.tma_core_diameter_spin, self.tma_core_auto_cb, **split)

        self.tma_sat_spin = QSpinBox()
        self.tma_sat_spin.setRange(2, 60)
        self.tma_sat_spin.setValue(15)
        self.tma_sat_spin.setFixedWidth(96)
        self.tma_sat_spin.setStyleSheet(self.spinner_style)
        self.tma_sat_spin.setToolTip(
            "Saturation above which a pixel is tissue (default 15). Lower it for pale cores,\n"
            "raise it if debris is picked up.")
        self.tma_recover_cb = QCheckBox("Recover faint")
        self.tma_recover_cb.setChecked(True)
        self.tma_recover_cb.setStyleSheet(self.checkbox_style)
        self.tma_recover_cb.setToolTip(
            "Look for pale cores at empty grid positions and add them (purple).")
        fit.add_split_row("Sensitivity", self.tma_sat_spin, self.tma_recover_cb, **split)

        self.tma_fit_btn = self._primary_button("Fit Cores")
        self.tma_fit_btn.setEnabled(False)
        self.tma_fit_btn.setToolTip("Find the cores and draw the overlay; run again after changing settings.")
        self.tma_fit_btn.clicked.connect(self.run_tma_fit)
        self._tma_fit_row = QHBoxLayout()
        self._tma_fit_row.setSpacing(8)
        self._tma_fit_row.addWidget(self.tma_fit_btn, 1)
        fit.add_footer(self._tma_fit_row)
        self.tma_fit_card = fit
        layout.addWidget(fit)

        # --- Filter ----------------------------------------------------------
        # QC filters re-evaluate instantly on the fitted cores (no refit).
        self.tma_filter_card = flt = SectionCard("Filter", compact=True)
        flt.LABEL_WIDTH = self.TMA_LABEL_WIDTH
        flt.title_label.setToolTip(
            "Quality filters; changes update the overlay at once. Excluded cores are grey with\n"
            "a cross and not exported. Filters never change patient IDs.")

        def _spin(lo, hi, val, step, suffix, tip, decimals=None, width=140):
            w = QDoubleSpinBox() if decimals is not None else QSpinBox()
            if decimals is not None:
                w.setDecimals(decimals)
            w.setRange(lo, hi)
            w.setSingleStep(step)
            w.setValue(val)
            w.setSuffix(suffix)
            w.setFixedWidth(width)
            w.setStyleSheet(self.spinner_style)
            w.setToolTip(tip)
            w.valueChanged.connect(self._tma_filters_changed)
            return w

        self.tma_offset_spin = _spin(0.05, 1.0, 0.35, 0.05, " pitch",
                                     "Exclude cores this far from their grid position (fraction of the core spacing);\n"
                                     "catches debris.", decimals=2, width=132)
        self.tma_stain_spin = _spin(0, 100, 2, 1, " %",
                                    "Exclude patient cores with less stained area than this; controls are exempt.",
                                    width=106)
        self.tma_dmin_spin = _spin(10, 100, 80, 5, " %",
                                   "Exclude cores smaller than this share of Core Ø; lower it to keep partial cores.",
                                   width=106)
        self.tma_dmax_spin = _spin(100, 300, 120, 5, " %",
                                   "Exclude cores larger than this share of Core Ø.", width=106)
        flt.add_split_row("Grid offset", self.tma_offset_spin, self.tma_stain_spin, "Min stain", **split)
        flt.add_split_row("Core size", self.tma_dmin_spin, self.tma_dmax_spin, "to", **split)
        layout.addWidget(flt)

        # --- Edit ------------------------------------------------------------
        # Hand corrections of the fitted circles; every edit re-snaps to the grid.
        self.tma_edit_card = edit = SectionCard("Edit", compact=True)
        edit.LABEL_WIDTH = self.TMA_LABEL_WIDTH
        edit.title_label.setToolTip(
            "Correct the fit by hand; a moved or added core takes the patient ID of its grid cell.\n"
            "Edits are saved with the export and offered back after refitting.")
        edit_layout = QHBoxLayout()
        edit_layout.setSpacing(18)
        self.tma_edit_cb = QCheckBox("Edit cores")
        self.tma_edit_cb.setStyleSheet(self.checkbox_style)
        self.tma_edit_cb.setEnabled(False)
        self.tma_edit_cb.setToolTip(
            "Click a core to select it; drag to move, drag its rim to resize. Double-click to\n"
            "include, exclude or add a core; Delete removes, Ctrl+Z undoes, right-click for more.")
        self.tma_edit_cb.toggled.connect(self._tma_edit_toggled)
        edit_layout.addWidget(self.tma_edit_cb)
        self.tma_links_cb = QCheckBox("Grid lines")
        self.tma_links_cb.setChecked(True)
        self.tma_links_cb.setStyleSheet(self.checkbox_style)
        self.tma_links_cb.setToolTip("Draw lines between neighbouring cores.")
        self.tma_links_cb.toggled.connect(self._tma_links_toggled)
        edit_layout.addWidget(self.tma_links_cb)
        edit_layout.addStretch()
        self.tma_undo_btn = QPushButton("Undo")
        self.tma_undo_btn.setStyleSheet(self.btn_style)
        self.tma_undo_btn.setEnabled(False)
        self.tma_undo_btn.setToolTip("Undo the last edit (Ctrl+Z).")
        self.tma_undo_btn.clicked.connect(lambda: self.tma_editor.undo())
        edit_layout.addWidget(self.tma_undo_btn)
        edit.add_layout(edit_layout)

        # loading and saving edits sit in the title bar: one row for the card
        self.tma_load_edits_btn = QPushButton("Load Edits…")
        self.tma_load_edits_btn.setObjectName("headerButton")
        self.tma_load_edits_btn.setEnabled(False)
        self.tma_load_edits_btn.setToolTip(
            f"Load hand edits saved earlier ({EDITS_FILE}), with the orientation and map position.")
        self.tma_load_edits_btn.clicked.connect(self.open_tma_edits)
        edit.add_header_widget(self.tma_load_edits_btn)
        self.tma_save_edits_btn = QPushButton("Save Edits…")
        self.tma_save_edits_btn.setObjectName("headerButton")
        self.tma_save_edits_btn.setEnabled(False)
        self.tma_save_edits_btn.setToolTip(
            f"Save the cores, orientation and map position as {EDITS_FILE} without exporting;\n"
            "they are offered back after the next fit.")
        self.tma_save_edits_btn.clicked.connect(self.save_tma_edits)
        edit.add_header_widget(self.tma_save_edits_btn)
        layout.addWidget(edit)

        # --- Export ----------------------------------------------------------
        # Settings below only affect Export Cores; changing them needs no refit.
        self.tma_export_card = export = SectionCard("Export", compact=True)
        export.LABEL_WIDTH = self.TMA_LABEL_WIDTH
        export.title_label.setToolTip("Settings for Export Cores; no refit needed.")
        self.tma_margin_spin = QSpinBox()
        self.tma_margin_spin.setRange(0, 1000)
        self.tma_margin_spin.setSingleStep(10)
        self.tma_margin_spin.setValue(30)
        self.tma_margin_spin.setSuffix(" µm")
        self.tma_margin_spin.setFixedWidth(122)
        self.tma_margin_spin.setStyleSheet(self.spinner_style)
        self.tma_margin_spin.setToolTip("Extra border around the fitted circle in each crop.")

        self.tma_erode_spin = QSpinBox()
        self.tma_erode_spin.setRange(0, 200)
        self.tma_erode_spin.setValue(8)
        self.tma_erode_spin.setSuffix(" px")
        self.tma_erode_spin.setFixedWidth(112)
        self.tma_erode_spin.setStyleSheet(self.spinner_style)
        self.tma_erode_spin.setToolTip("Shrinks the core mask so the core edge is left out.")
        export.add_split_row("Margin", self.tma_margin_spin, self.tma_erode_spin, "Mask shrink", **split)

        # the channels sit in the title bar
        self.tma_channels_box = QWidget()
        ch_layout = QHBoxLayout(self.tma_channels_box)
        ch_layout.setContentsMargins(0, 0, 0, 0)
        ch_layout.setSpacing(14)
        self.tma_channel_cbs = {}
        ch_hints = {"BF": "Bright-field layer; analyse it with Dark line on.",
                    "POL": "Polarised-light layer; separate its bands on the Birefringence page,\n"
                           "or analyse it with Dark line off."}
        for name in ("BF", "POL"):
            cb = QCheckBox(name)
            cb.setChecked(True)
            cb.setEnabled(False)
            cb.setStyleSheet(self.checkbox_style)
            cb.setToolTip(ch_hints[name] + "\nSlide layers to export, one image per core each; "
                                           "available after Fit Cores.")
            self.tma_channel_cbs[name] = cb
            ch_layout.addWidget(cb)
        export.add_header_widget(self.tma_channels_box)

        self.tma_output_path = PathEdit("")
        self.tma_output_path.setPlaceholderText("Defaults to <slide>_cores")
        self.tma_output_path.setToolTip(
            "Gets BF/ and POL/ folders (Patients, Controls), each with Images/ and Masks/,\n"
            "plus cores.csv and overlay.png. Defaults to <slide>_cores.")
        self.tma_output_btn = self.tma_output_path.pick_action
        self.tma_output_btn.setToolTip("Choose the output folder.")
        self.tma_output_btn.triggered.connect(self.select_tma_output)
        export.add_row("Output folder", self.tma_output_path)

        # Status of the fit and export, then the export action
        self.tma_status_label = QLabel("No slide loaded")
        self.tma_status_label.setWordWrap(True)
        self.tma_status_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.tma_status_label.setStyleSheet(self.value_label_style)
        self.tma_status_label.setToolTip(
            "Fit summary. Overlay: green = exported, purple = recovered, grey with a cross =\n"
            "excluded, grey = outside the map, dashed = no core, orange dot = hand-edited.")
        export.add_footer(self.tma_status_label)

        self.tma_export_btn = self._primary_button("Export Cores", play=False)
        self.tma_export_btn.setEnabled(False)
        self.tma_export_btn.setToolTip("Write one image and one mask per core and channel.")
        self.tma_export_btn.clicked.connect(self.run_tma_export)
        self._tma_export_row = QHBoxLayout()
        self._tma_export_row.setSpacing(8)
        self._tma_export_row.addWidget(self.tma_export_btn, 1)
        export.add_footer(self._tma_export_row)
        layout.addWidget(export)

        # Cancel is shown next to the action that is running (fit or export)
        self.tma_cancel_btn = QPushButton("Cancel")
        self.tma_cancel_btn.setStyleSheet(self.btn_style)
        self.tma_cancel_btn.setToolTip("Stop after the current core; files written so far are kept.")
        self.tma_cancel_btn.setVisible(False)
        self.tma_cancel_btn.clicked.connect(self.cancel_tma)
        self._tma_fit_row.addWidget(self.tma_cancel_btn)

        layout.addStretch()
        self.tma_tab.setLayout(layout)

        self.tma_slide = None
        self.tma_output = None
        self.tma_pre = None
        self.tma_reader = None
        self.tma_worker = None
        self._tma_unsaved = False   # hand edits changed since the last export, save or load
        self.tma_overlay = None
        self._tma_shown_pre = None
        self._tma_editor = None     # created on first use: the image panel does not exist yet

    @property
    def tma_editor(self):
        if self._tma_editor is None:
            self._tma_editor = TMACoreEditor(self.image_panel, lambda: self.tma_pre, self._tma_edited)
        return self._tma_editor

    def _tma_map_changed(self, *_):
        """Relabel the fitted cores for a new array or orientation without refitting."""
        pre = self.tma_pre
        if pre is None or (self.tma_worker is not None and self.tma_worker.isRunning()):
            return
        pre.array_number = self._tma_selected_map()
        pre.orientation = self.tma_orientation_combo.currentData()
        pre.map_to_array()
        self._show_tma_result(preserve_view=True)

    def _tma_edit_toggled(self, on):
        self.tma_editor.enabled = bool(on)
        self.tma_editor.hovered = None      # refresh the hover hint on the next mouse move
        self.image_panel.update()
        if self.tma_pre is not None:
            self._show_tma_result(preserve_view=True)

    def _tma_links_toggled(self, on):
        self.tma_editor.show_links = bool(on)
        self.image_panel.update()

    def _sync_tma_orientation_combo(self):
        """Show the preprocessor's orientation in the combo without re-triggering a remap."""
        if self.tma_pre is None:
            return
        idx = self.tma_orientation_combo.findData(self.tma_pre.orientation)
        if idx >= 0 and idx != self.tma_orientation_combo.currentIndex():
            self.tma_orientation_combo.blockSignals(True)
            self.tma_orientation_combo.setCurrentIndex(idx)
            self.tma_orientation_combo.blockSignals(False)

    def _tma_edited(self, full):
        """Editor callback: repaint, and after a completed edit refresh the summary."""
        if full:
            self._tma_unsaved = True
            # setting a map position (or undoing one) can change the orientation
            self._sync_tma_orientation_combo()
            self._show_tma_result(preserve_view=True)
        else:
            self.image_panel.update()
        self.tma_undo_btn.setEnabled(self.tma_editor.can_undo())

    def load_tma_edits(self, path):
        """Reinstate the cores saved by an earlier export (see _offer_saved_edits)."""
        if self.tma_pre is None:
            return
        try:
            self.tma_editor.push_undo()
            n = self.tma_pre.load_edits(path)
        except Exception as e:
            self._notify("Load Edits", f"Could not load {path}:\n{e}", warning=True)
            return
        self.tma_editor.selected = None
        self._tma_unsaved = False
        # the saved orientation is reapplied by load_edits
        self._sync_tma_orientation_combo()
        self._show_tma_result(preserve_view=True)
        self.tma_status_label.setText(f"Reinstated {n} cores ({self.tma_pre.manual_count()} hand-edited) from {path}. "
                                      + self.tma_status_label.text())

    def _offer_saved_edits(self):
        """After a fit, offer to reinstate hand edits saved by an earlier export."""
        pre = self.tma_pre
        if pre is None or not self.tma_output:
            return
        path = join_path(self.tma_output, EDITS_FILE)
        if not os.path.isfile(path):
            return
        saved = self._read_saved_edits(path)
        if saved is None or saved["slide"] != pre.slide_name or not saved["description"]:
            return
        if self._ask_tma("TMA Edits Found", f"Earlier edits of this slide were saved: {saved['description']}.",
                         f"{path}\n\nReinstate them in place of the new fit?", "Reinstate", "Keep new fit"):
            self.load_tma_edits(path)

    @staticmethod
    def _read_saved_edits(path):
        """Summary of an edits file: dict(slide, n_manual, anchor, description),
        or None when it cannot be read. ``description`` is empty when the file
        holds no hand edits and no hand-set map position."""
        try:
            import json
            with open(path, encoding="utf-8") as f:
                saved = json.load(f)
            n_manual = sum(1 for c in saved.get("cores", []) if c.get("manual") or c.get("override"))
        except Exception:
            return None
        anchor = saved.get("map_anchor") is not None
        parts = []
        if n_manual:
            parts.append(f"{n_manual} hand-edited core{'s' if n_manual != 1 else ''}")
        if anchor:
            parts.append("a map position set by hand")
        return {"slide": saved.get("slide"), "n_manual": n_manual, "anchor": anchor,
                "description": " and ".join(parts)}

    def _ask_tma(self, title, text, info, yes_label, no_label):
        """Two-button question; True when ``yes_label`` is chosen. Monkeypatched in tests."""
        return self._message(title, text, info, (yes_label, no_label)) == yes_label

    def _tma_edits_default_path(self):
        folder = self.tma_output or (os.path.dirname(self.tma_slide) if self.tma_slide else "")
        return join_path(folder, EDITS_FILE) if folder else EDITS_FILE

    def open_tma_edits(self):
        """Load Edits…: reinstate an edits file chosen by the user."""
        if self.tma_pre is None:
            return
        path, _ = QFileDialog.getOpenFileName(self, "Load TMA Edits", self._tma_edits_default_path(),
                                              "TMA edits (*.json);;All files (*)")
        if not path:
            return
        saved = self._read_saved_edits(path)
        if saved is None:
            self._notify("Load Edits", f"Could not read {path}.", warning=True)
            return
        if saved["slide"] != self.tma_pre.slide_name and not self._ask_tma(
                "Load Edits", f"These edits were saved for slide '{saved['slide']}', "
                              f"not '{self.tma_pre.slide_name}'.",
                f"{path}\n\nLoad them anyway (for example for a rescan of the same array)?", "Load", "Cancel"):
            return
        self.load_tma_edits(path)

    def save_tma_edits(self):
        """Save Edits…: write the current cores, orientation and map position without exporting."""
        pre = self.tma_pre
        if pre is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save TMA Edits", self._tma_edits_default_path(),
                                              "TMA edits (*.json)")
        if not path:
            return
        try:
            folder = os.path.dirname(path)
            if folder:
                os.makedirs(folder, exist_ok=True)
            pre.save_edits(path)
        except Exception as e:
            self._notify("Save Edits", f"Could not save {path}:\n{e}", warning=True)
            return
        self._tma_unsaved = False
        self.tma_status_label.setText(f"Saved {len(pre.cores)} cores ({pre.manual_count()} hand-edited"
                                      + (", map position set by hand" if pre.map_anchor else "")
                                      + f") to {path}. " + self.tma_status_label.text())

    def select_tma_slide(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select TMA Slide", "",
            "Slides (*.vsi *.tif *.tiff *.png *.jpg *.jpeg);;All files (*)")
        if not file_path:
            return
        self.open_tma_slide(file_path)

    TMA_OUTPUT_PARENT_KEY = "tma_output_parent"

    def _tma_default_output(self, slide_path):
        """Default output folder of a slide: ``<slide name>_cores`` inside the
        parent of the output folder chosen last (remembered across sessions),
        or next to the slide until one has been chosen."""
        parent = QSettings('Cabana', 'CabanaGUI').value(self.TMA_OUTPUT_PARENT_KEY, "")
        if not parent or not os.path.isdir(parent):
            parent = os.path.dirname(slide_path)
        return join_path(parent, sanitize_filename(os.path.splitext(os.path.basename(slide_path))[0]) + "_cores")

    def _set_tma_output(self, folder):
        self.tma_output = folder
        self.tma_output_path.setText(folder)
        self.tma_output_path.setToolTip(folder)

    def open_tma_slide(self, file_path):
        """Open a slide on the TMA page. The output folder always follows the
        slide (see :meth:`_tma_default_output`): one folder holds one slide's export."""
        self.tma_slide = file_path
        self.tma_slide_path.setText(file_path)
        self.tma_slide_path.setToolTip(file_path)
        if self.tma_pre is not None:
            self.tma_pre.close()
        self.tma_pre = None
        self.tma_reader = None
        self._show_tma_core_diameter()      # each array may use another needle: measure afresh
        self.tma_export_btn.setEnabled(False)
        self.tma_fit_btn.setEnabled(False)
        self._set_tma_output(self._tma_default_output(file_path))
        self._start_tma_preview()

    def _start_tma_preview(self):
        """Open the slide in the background and show a low-resolution preview."""
        options = {}
        if not self.tma_slide.lower().endswith(".vsi"):
            options['pixel_size_um'] = float(self.tma_pixel_size_spin.value())
        self._tma_set_busy(True)
        self.tma_status_label.setText("Opening slide…")
        self.tma_worker = TMAWorker('preview', slide_path=self.tma_slide, options=options)
        self.tma_worker.progress_updated.connect(self.progress_bar.setValue)
        self.tma_worker.status_updated.connect(self.tma_status_label.setText)
        self.tma_worker.preview_complete.connect(self.handle_tma_preview_complete)
        self.tma_worker.failed.connect(self.handle_tma_failed)
        self.tma_worker.start()

    def handle_tma_preview_complete(self, result):
        reader = result['reader']
        self.tma_reader = reader
        self._tma_set_busy(False)
        self.tma_fit_btn.setEnabled(True)
        if reader.pixel_size_um:
            self.tma_pixel_size_spin.setValue(float(reader.pixel_size_um))
        for name, cb in self.tma_channel_cbs.items():
            available = name in reader.channels
            cb.setEnabled(available)
            cb.setChecked(available)
        self.tma_overlay = np.ascontiguousarray(result['image'])
        self.image_panel.setImage(self.tma_overlay)
        h0, w0 = reader.level_shape(0)
        px = f"{reader.pixel_size_um:.4f} µm/px" if reader.pixel_size_um else "pixel size unknown"
        self.tma_status_label.setText(
            f"{os.path.basename(self.tma_slide)}: {w0} x {h0} px, {px}, channels {', '.join(reader.channels)}. "
            f"Choose the array map, then Fit Cores.")
        self.status_file_label.setText(f"  {os.path.basename(self.tma_slide)}")
        self.status_dims_label.setText(f"{w0} x {h0}  ")

    def select_tma_output(self):
        start = self.tma_output or (os.path.dirname(self.tma_slide) if self.tma_slide else "")
        folder = QFileDialog.getExistingDirectory(self, "Select TMA Output Folder", start,
                                                  QFileDialog.ShowDirsOnly)
        if folder:
            self._set_tma_output(folder)
            # the next slide's default output goes next to this folder
            QSettings('Cabana', 'CabanaGUI').setValue(self.TMA_OUTPUT_PARENT_KEY, os.path.dirname(folder))

    def _measured_core_diameter(self):
        """Core diameter measured by the last fit, or None (no fit, or typed in)."""
        pre = getattr(self, "tma_pre", None)       # the tooltip is set before the TMA state exists
        return pre.core_diameter_um if pre is not None and pre.core_diameter_auto else None

    def _set_tma_core_tooltip(self):
        spin = self.tma_core_diameter_spin
        if not self.tma_core_auto_cb.isChecked():
            spin.setToolTip("Core diameter used by the fit and the Min Ø / Max Ø filters.")
        elif self._measured_core_diameter():
            spin.setToolTip(f"Measured: median of {self.tma_pre.core_diameter_count} well-formed cores.")
        else:
            spin.setToolTip("Measured from the slide when the cores are fitted.")

    def _show_tma_core_diameter(self):
        """With Auto, show the measured diameter (— before a fit)."""
        if self.tma_core_auto_cb.isChecked():
            measured = self._measured_core_diameter()
            self.tma_core_diameter_spin.setValue(int(round(measured)) if measured else 0)
        self._set_tma_core_tooltip()

    def _tma_core_auto_toggled(self, auto):
        """Auto measures the diameter at the next fit; unticking starts the
        typed value from the measured one."""
        spin = self.tma_core_diameter_spin
        if auto:
            spin.setMinimum(0)
            spin.setSpecialValueText("—")
            self._show_tma_core_diameter()
        else:
            start = spin.value() or int(AUTO_START_DIAMETER_UM)
            spin.setSpecialValueText("")
            spin.setMinimum(100)
            spin.setValue(max(100, start))
            self._set_tma_core_tooltip()
        busy = self.tma_worker is not None and self.tma_worker.isRunning()
        spin.setEnabled(not auto and not busy)

    def _tma_options(self):
        return dict(
            array_number=self._tma_selected_map(),
            pixel_size_um=float(self.tma_pixel_size_spin.value()) if self.tma_pixel_size_spin.value() > 0 else None,
            core_diameter_um=None if self.tma_core_auto_cb.isChecked() else float(self.tma_core_diameter_spin.value()),
            margin_um=float(self.tma_margin_spin.value()),
            erode_px=int(self.tma_erode_spin.value()),
            orientation=self.tma_orientation_combo.currentData(),
            sat_thresh=int(self.tma_sat_spin.value()),
            recover_faint=self.tma_recover_cb.isChecked(),
            **self._tma_filter_values(),
        )

    def _tma_filter_values(self):
        return dict(
            max_grid_offset=float(self.tma_offset_spin.value()),
            min_diameter_frac=self.tma_dmin_spin.value() / 100.0,
            max_diameter_frac=self.tma_dmax_spin.value() / 100.0,
            min_stain_frac=self.tma_stain_spin.value() / 100.0,
        )

    def _tma_filters_changed(self, *_):
        """Re-apply QC filters to the fitted cores and redraw, without refitting."""
        pre = self.tma_pre
        if pre is None or (self.tma_worker is not None and self.tma_worker.isRunning()):
            return
        for k, v in self._tma_filter_values().items():
            setattr(pre, k, v)
        pre.apply_filters()          # never touches the patient-ID assignment
        self._show_tma_result(preserve_view=True)

    def _tma_set_busy(self, busy):
        for w in (self.tma_slide_btn, self.tma_output_btn, self.tma_array_combo,
                  self.tma_orientation_combo, self.tma_pixel_size_spin, self.tma_core_auto_cb,
                  self.tma_margin_spin, self.tma_erode_spin, self.tma_sat_spin, self.tma_recover_cb,
                  self.tma_offset_spin, self.tma_dmin_spin, self.tma_dmax_spin, self.tma_stain_spin):
            w.setEnabled(not busy)
        self.tma_core_diameter_spin.setEnabled(not busy and not self.tma_core_auto_cb.isChecked())
        fitted = not busy and self.tma_pre is not None
        for w in (self.tma_edit_cb, self.tma_links_cb, self.tma_load_edits_btn, self.tma_save_edits_btn):
            w.setEnabled(fitted)
        self.tma_undo_btn.setEnabled(fitted and self.tma_editor.can_undo())
        if busy:
            self.tma_editor.enabled = False
        else:
            self.tma_editor.enabled = fitted and self.tma_edit_cb.isChecked()
        self.tma_fit_btn.setEnabled(not busy and self.tma_slide is not None
                                    and (self.tma_reader is not None or self.tma_pre is not None))
        self.tma_export_btn.setEnabled(not busy and self.tma_pre is not None
                                       and (self.tma_pre.array_number is None or self.tma_pre.orientation_resolved))
        self.tma_cancel_btn.setVisible(busy)
        self.tma_cancel_btn.setEnabled(busy)
        self.tma_cancel_btn.setText("Cancel")
        if busy:
            self.show_progress_bar()
        else:
            self.hide_progress_bar()

    def run_tma_fit(self):
        if not self.tma_slide:
            return
        if self.tma_pre is not None:
            # the refit below reuses this reader (self.tma_reader): closing it
            # would make later full-resolution reads fail ("seek of closed file")
            if self.tma_pre.reader is not self.tma_reader:
                self.tma_pre.close()
            self.tma_pre = None
        options = self._tma_options()
        if self.tma_slide.lower().endswith(".vsi"):
            # a VSI carries its own calibration: let the reader's value stand
            options["pixel_size_um"] = None
        # reuse the reader opened for the preview so the slide is read once
        options["reader"] = self.tma_reader
        self.tma_reader = None
        self._tma_fit_row.addWidget(self.tma_cancel_btn)      # Cancel beside the running action
        self._tma_set_busy(True)
        self.tma_fit_btn.setText("Fitting…")
        self.tma_status_label.setText("Fitting cores…")
        self.tma_worker = TMAWorker('fit', slide_path=self.tma_slide, options=options)
        self.tma_worker.progress_updated.connect(self.progress_bar.setValue)
        self.tma_worker.status_updated.connect(self.tma_status_label.setText)
        self.tma_worker.fit_complete.connect(self.handle_tma_fit_complete)
        self.tma_worker.failed.connect(self.handle_tma_failed)
        self.tma_worker.start()

    def handle_tma_fit_complete(self, pre):
        self.tma_pre = pre
        self._tma_unsaved = False
        self.tma_reader = pre.reader
        self.tma_editor.reset()
        self.tma_fit_btn.setText("Fit Cores")
        self._tma_set_busy(False)
        reader = pre.reader
        if reader.pixel_size_um:
            self.tma_pixel_size_spin.setValue(float(reader.pixel_size_um))
        for name, cb in self.tma_channel_cbs.items():
            available = name in reader.channels
            cb.setEnabled(available)
            cb.setChecked(available)
        self._show_tma_core_diameter()
        self._show_tma_result(preserve_view=False)
        self.status_file_label.setText(f"  {os.path.basename(self.tma_slide)}")
        self.status_dims_label.setText(f"{reader.level_shape(0)[1]} x {reader.level_shape(0)[0]}  ")
        self._offer_saved_edits()

    def _show_tma_result(self, preserve_view=False):
        """Show the fit image with the live overlay and write the fit/filter summary to the status line."""
        pre = self.tma_pre
        if self._tma_shown_pre is not pre or not preserve_view or self.image_panel.overlay_painter is None:
            # the base image changes only with a refit; the overlay is painted live on top
            self.tma_overlay = np.ascontiguousarray(pre._fit_image[:, :, ::-1])
            self.image_panel.setImage(self.tma_overlay, preserve_view=preserve_view and self._tma_shown_pre is pre)
            self._tma_shown_pre = pre
        self.image_panel.set_overlay_painter(self.tma_editor.paint)
        self.image_panel.set_edit_handler(self.tma_editor)
        self.tma_undo_btn.setEnabled(self.tma_editor.can_undo())
        n_rows, n_cols = pre.grid_shape
        n_out = len([c for c in pre.cores if c.outside_map])
        n_export = len(pre.exportable_cores())
        msg = f"{len(pre.cores)} cores on a {n_rows}x{n_cols} grid, {n_export} to export"
        if pre.matched_orientation:
            others = [o for o in pre.orientation_ties if o != pre.matched_orientation]
            how = {"brain": "chosen by the Brain control", "replicates": "chosen by replicate similarity",
                   "manual": "set manually", "occupancy": "from missing-core pattern",
                   "anchor": "map position set by hand"}.get(pre.orientation_method, "")
            if pre.orientation_margin is not None and pre.orientation_method in ("brain", "replicates"):
                how += f", margin {pre.orientation_margin:.0%}"
            msg += f"; orientation {pre.matched_orientation}" + (f" ({how})" if how else "")
            if not pre.orientation_resolved:
                msg += (f" — UNRESOLVED: {', '.join(pre.orientation_ties)} fit equally well and appearance "
                        f"cannot separate them; confirm a control core against the printed map and set "
                        f"Orientation explicitly. Export is disabled until then.")
            elif others and pre.orientation_method not in ("manual", "anchor"):
                msg += f" (also tied on occupancy: {', '.join(others)}; verify a control core)"
        self.tma_export_btn.setEnabled(pre.array_number is None or pre.orientation_resolved)
        summary = pre.exclusion_summary()
        if summary:
            names = {"off_grid": "off-grid", "diameter": "diameter", "stain": "low stain"}
            msg += "; excluded " + ", ".join(f"{v} {names.get(k, k)}" for k, v in sorted(summary.items()))
        if n_out:
            msg += f"; {n_out} outside the map"
        n_missing = len(pre.missing_positions())
        if n_missing:
            msg += f"; {n_missing} map position{'s' if n_missing > 1 else ''} without a core (dashed)"
        n_manual = pre.manual_count()
        if n_manual:
            msg += f"; {n_manual} hand-edited"
        diams = [c.diameter_um for c in pre.cores if not c.outside_map]
        if pre.core_diameter_auto:
            msg += f"; Core Ø {pre.core_diameter_um:.0f} µm (auto)"
        elif diams:
            med = float(np.median(diams))
            msg += f"; median Ø {med:.0f} µm"
            if abs(med - pre.core_diameter_um) > 0.1 * pre.core_diameter_um:
                msg += (f" — WARNING: Core Ø is {pre.core_diameter_um:.0f} µm, so the diameter filter "
                        f"is misjudging cores; set Core Ø to ~{round(med, -1):.0f} or tick Auto, and refit")
        lost = [p for p, (k, t) in pre.replicate_counts().items() if k == 0]
        if lost:
            msg += f"; no core left for patient{'s' if len(lost) > 1 else ''} {', '.join(lost)}"
        recovered = [c.index for c in pre.cores if c.recovered]
        if recovered:
            msg += f"; recovered: {', '.join(map(str, recovered))}"
        self.tma_status_label.setText(msg)

    def run_tma_export(self):
        if self.tma_pre is None:
            return
        if not self.tma_output:
            self.select_tma_output()
            if not self.tma_output:
                return
        channels = [n for n, cb in self.tma_channel_cbs.items() if cb.isEnabled() and cb.isChecked()]
        if not channels:
            self._notify("TMA Export", "Select at least one channel to export.", warning=True)
            return
        if not self._confirm_output_slide() or not self._confirm_edits_overwrite():
            return
        # settings that only affect the crop can be changed without refitting
        self.tma_pre.margin_um = float(self.tma_margin_spin.value())
        self.tma_pre.erode_px = int(self.tma_erode_spin.value())
        self._tma_export_row.addWidget(self.tma_cancel_btn)   # Cancel beside the running action
        self._tma_set_busy(True)
        self.tma_export_btn.setText("Exporting…")
        self.tma_worker = TMAWorker('export', preprocessor=self.tma_pre, out_dir=self.tma_output,
                                    channels=channels)
        self.tma_worker.progress_updated.connect(self.progress_bar.setValue)
        self.tma_worker.status_updated.connect(self.tma_status_label.setText)
        self.tma_worker.export_complete.connect(self.handle_tma_export_complete)
        self.tma_worker.export_cancelled.connect(self.handle_tma_export_cancelled)
        self.tma_worker.failed.connect(self.handle_tma_failed)
        self.tma_worker.start()

    def _confirm_output_slide(self):
        """Ask before exporting into a folder that holds another slide's export
        (identified by the slide name in its edits file)."""
        path = join_path(self.tma_output, EDITS_FILE)
        if not os.path.isfile(path):
            return True
        saved = self._read_saved_edits(path)
        if saved is None or not saved["slide"] or saved["slide"] == self.tma_pre.slide_name:
            return True
        return self._ask_tma(
            "Output Folder Holds Another Slide",
            f"The output folder already holds the export of slide '{saved['slide']}', "
            f"not '{self.tma_pre.slide_name}'.",
            f"{self.tma_output}\n\nExporting here would overwrite its cores.csv, overlay and saved edits "
            f"and mix the two slides' images. Choose Cancel and pick another folder with Select.",
            "Export Anyway", "Cancel")

    def _confirm_edits_overwrite(self):
        """Export replaces the output folder's edits file. Ask first when that
        file holds hand edits of this slide but the cores on screen hold none
        (e.g. after a refit whose reinstate offer was declined); in every other
        case the screen is the newer state and is saved without asking."""
        pre = self.tma_pre
        path = join_path(self.tma_output, EDITS_FILE)
        if not os.path.isfile(path) or pre.manual_count() or pre.map_anchor is not None:
            return True
        saved = self._read_saved_edits(path)
        if saved is None or saved["slide"] != pre.slide_name or not saved["description"]:
            return True
        return self._ask_tma(
            "Replace Saved Edits?",
            f"The output folder holds saved edits of this slide ({saved['description']}), "
            f"but the cores on screen have no hand edits.",
            f"{path}\n\nExporting will replace those edits with the unedited fit. "
            f"Choose Cancel and use Load Edits… to bring them back first.",
            "Replace", "Cancel")

    def cancel_tma(self):
        self.tma_cancel_btn.setEnabled(False)
        self.tma_cancel_btn.setText("Cancelling…")
        if self.tma_worker is not None:
            self.tma_worker.cancel()

    def handle_tma_export_complete(self, out_dir):
        self._tma_unsaved = False           # the export wrote cores_edits.json
        self.tma_export_btn.setText("Export Cores")
        self._tma_set_busy(False)
        n = len(self.tma_pre.exportable_cores()) if self.tma_pre else 0  # excludes filtered cores
        self.tma_status_label.setText(f"Exported {n} cores to {out_dir}")
        channels = [c for c in ("BF", "POL") if os.path.isdir(join_path(out_dir, c))]
        batch_btns = {f"Use {c} in Batch Run": c for c in channels}
        clicked = self._message(
            "TMA Export Complete", "Core images and masks were written.",
            f"{out_dir}\n\nEach channel ({', '.join(channels)}) has Patients/ and Controls/ folders, "
            f"each holding Images/ and Masks/ usable directly as the input and ROI-mask folders of "
            f"Batch Run. Analyse BF with Dark line on and POL with it off.",
            [*batch_btns, "Open Folder", "OK"], default="OK")
        if clicked == "Open Folder":
            QDesktopServices.openUrl(QUrl.fromLocalFile(out_dir))
        elif clicked in batch_btns:
            self.use_tma_export_in_batch(out_dir, batch_btns[clicked])

    def use_tma_export_in_batch(self, out_dir, channel="BF"):
        """Point the Batch Run page at the patient cores of one channel of a TMA
        export (or the unmapped cores when no array map was used) and switch to it."""
        group = next((g for g in ("Patients", "Unmapped")
                      if os.path.isdir(join_path(out_dir, channel, g, 'Images'))), "Patients")
        images = join_path(out_dir, channel, group, 'Images')
        masks = join_path(out_dir, channel, group, 'Masks')
        self.input_folder = images
        self.input_folder_path.setText(images)
        self.input_folder_path.setToolTip(images)
        self.set_mask_folder(masks if os.path.isdir(masks) else None)
        self._check_batch_processing_ready()
        self.show_page(self.pages.indexOf(self.bat_tab))

    def handle_tma_export_cancelled(self):
        self.tma_export_btn.setText("Export Cores")
        self._tma_set_busy(False)
        self.tma_status_label.setText("Export cancelled. Files written so far were kept.")

    def handle_tma_failed(self, message):
        self.tma_fit_btn.setText("Fit Cores")
        self.tma_export_btn.setText("Export Cores")
        self._tma_set_busy(False)
        self.tma_status_label.setText(f"Error: {message}")
        self._notify("TMA Preprocessing", "TMA preprocessing failed.", message, warning=True)

    def select_param_file(self):
        """Open a file dialog to select a parameter file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Parameter File",
            os.path.expanduser('~/Documents'),
            "YAML Files (*.yml *.yaml)")

        if file_path:
            self.param_file = file_path
            self.param_file_path.setText(file_path)
            self.param_file_path.setToolTip(file_path)
            self._show_param_file_max_size(file_path)
            self._check_batch_processing_ready()

    def _show_param_file_max_size(self, path):
        """Show the parameter file's Max Size and Save Visualisations on the Batch Run page."""
        try:
            with open(path) as f:
                params = yaml.safe_load(f)
        except (OSError, yaml.YAMLError):
            return
        if isinstance(params, dict):
            self._show_batch_max_size(max_image_size(params))
            self.visuals_cb.setChecked(save_visualisations(params))

    def _show_batch_max_size(self, max_size, source=""):
        text = f"{int(max_size)} px" if max_size else "No limit"
        self.batch_max_size_label.setText(f"{text}{source}")

    def select_input_folder(self):
        """Open a file dialog to select an input folder"""
        # start beside the parameter file when one is chosen (any order is allowed)
        if getattr(self, 'param_file', None):
            start = str(Path(self.param_file).parent.parent)
        else:
            start = getattr(self, 'input_folder', None) or os.path.expanduser('~')
        folder = QFileDialog.getExistingDirectory(
            self, "Select Input Folder", start, QFileDialog.ShowDirsOnly
        )

        if folder:
            self.input_folder = folder
            # A TMA export is Images/ next to Masks/: offer the masks automatically
            sibling = join_path(os.path.dirname(folder), 'Masks')
            if os.path.basename(folder) == 'Images' and os.path.isdir(sibling) and not self.mask_folder:
                self.set_mask_folder(sibling)
            self.input_folder_path.setText(folder)
            self.input_folder_path.setToolTip(folder)
            self._check_batch_processing_ready()

    def select_mask_folder(self):
        """Open a folder dialog to select the optional ROI mask folder"""
        start = getattr(self, 'input_folder', None) or os.path.expanduser('~')
        folder = QFileDialog.getExistingDirectory(
            self, "Select ROI Mask Folder (optional)", str(Path(start).parent), QFileDialog.ShowDirsOnly
        )
        if folder:
            self.set_mask_folder(folder)

    def set_mask_folder(self, folder):
        self.mask_folder = folder or None
        self.mask_folder_path.setText(folder or "")
        self.mask_folder_path.setToolTip(folder or "")

    def clear_mask_folder(self):
        self.set_mask_folder(None)

    def select_output_folder(self):
        """Open a file dialog to select an output folder"""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Output Folder", "", QFileDialog.ShowDirsOnly
        )

        if folder:
            self.output_folder = folder
            self.output_folder_path.setText(folder)
            self.output_folder_path.setToolTip(folder)
            self._check_batch_processing_ready()

    def _check_batch_processing_ready(self):
        """Check if all necessary paths are selected to enable batch processing"""
        is_ready = hasattr(self, 'param_file') and hasattr(self, 'input_folder') and hasattr(self, 'output_folder')
        self.process_batch_btn.setEnabled(is_ready)
        self.stats_cb.setEnabled(is_ready)
        self.scores_cb.setEnabled(is_ready)

    def _check_batch_running_status(self):
        checkpoint_path = join_path(self.output_folder, '.CheckPoint.txt')

        # Default values
        resume = False
        batch_size = 5
        batch_num = 0
        # Oversized images (larger than Max Size) are split into blocks; a
        # checkpoint keeps the setting of the run it was written by.
        ignore_large = False
        max_size = None         # Max Size of the interrupted run (None: not recorded)
        save_visuals = None     # Save Visualisations of the interrupted run

        # Check if checkpoint file exists
        if not os.path.exists(checkpoint_path):
            Log.logger.info("No checkpoint file found. Starting a new run.")
            return resume, batch_size, batch_num, ignore_large, {}

        # Read checkpoint file
        Log.logger.info("A checkpoint file exists in the output folder.")
        ckpt = read_checkpoint(checkpoint_path)
        input_folder, batch_size, batch_num, ignore_large, max_size, save_visuals = (
            ckpt[k] for k in ("input_folder", "batch_size", "batch_num", "ignore_large",
                              "max_size", "save_visuals"))

        # Check if input folder matches
        if input_folder and os.path.exists(input_folder):
            resume = os.path.samefile(input_folder, self.input_folder)

        # Verify all batch folders exist
        for batch_idx in range(batch_num + 1):
            batch_path = join_path(self.output_folder, 'Batches', f'batch_{batch_idx}')
            if not os.path.exists(batch_path):
                Log.logger.warning('However, some necessary sub-folders are missing. A new run will start.')
                resume = False
                break

        # If validation passes, ask user about resuming
        if resume:
            text = "A checkpoint file was found. Do you want to resume from the last checkpoint?"
            changes = run_settings_changes(self.output_folder, self.param_file, self.mask_folder,
                                           save_visuals=self.visuals_cb.isChecked())
            if changes:
                text += ("\n\nThe selected settings differ from those the run started with:\n"
                         + "\n".join("• " + c for c in changes)
                         + "\n\nResuming uses the original settings. Choose No to start a new run "
                           "with the selected ones.")

            if self._message("Checkpoint Detected", text, buttons=("Yes", "No")) == "Yes":
                Log.logger.info('Resuming from last check point.')
                return True, batch_size, batch_num, ignore_large, {
                    k: v for k, v in (("max_size", max_size), ("save_visuals", save_visuals)) if v is not None}
            else:
                Log.logger.info("Starting a new run.")

        return False, batch_size, batch_num, ignore_large, {}

    def run_batch_processing(self):
        """Run batch processing in a background thread"""
        if not hasattr(self, 'param_file') or not hasattr(self, 'input_folder') or not hasattr(self, 'output_folder'):
            return

        # Disable inputs during processing; show Cancel button
        self.process_batch_btn.setVisible(False)
        self.cancel_batch_btn.setVisible(True)
        self.cancel_batch_btn.setEnabled(True)
        self.cancel_batch_btn.setText("Cancel")
        self.param_btn.setEnabled(False)
        self.input_btn.setEnabled(False)
        self.output_btn.setEnabled(False)
        self.mask_btn.setEnabled(False)
        self.mask_clear_action.setEnabled(False)
        self.stats_cb.setEnabled(False)
        self.scores_cb.setEnabled(False)
        self.visuals_cb.setEnabled(False)
        self._show_batch_progress(True)

        resume, batch_size, batch_num, ignore_large, kept = self._check_batch_running_status()

        # a new run deletes what the results folder holds (all but its Logs): ask first
        if not resume and self._results_folder_in_use(self.output_folder) and not self._ask_tma(
                "Process Batch", "The results folder is not empty.",
                f"{self.output_folder}\n\nA new run deletes everything in it except the Logs folder.",
                "Delete and Run", "Cancel"):
            self._show_batch_progress(False)
            self._restore_batch_buttons()
            return

        if not resume:
            batch_size = self.batch_size_spinner.value()
            batch_num = 0
        # a new run takes Max Size from the parameter file; a resumed run keeps the
        # settings it started with (all its parameters, when it saved them)
        max_size = kept.get("max_size")
        save_visuals = kept.get("save_visuals", self.visuals_cb.isChecked())
        if resume:
            saved = saved_run_parameters(self.output_folder)
            if saved is not None:
                self._show_batch_max_size(max_image_size(saved), " (from the interrupted run)")
            elif max_size is not None:
                self._show_batch_max_size(max_size, " (from the interrupted run)")

        self.batch_worker = BatchProcessingWorker(
            self.param_file, self.input_folder, self.output_folder,
            batch_size, batch_num, resume, ignore_large,
            generate_stats=self.stats_cb.isChecked(),
            generate_scores=self.scores_cb.isChecked(),
            mask_dir=self.mask_folder,
            max_size=max_size,
            save_visuals=save_visuals,
        )

        # Connect signals
        self.batch_worker.progress_updated.connect(self.batch_progress_bar.setValue)
        self.batch_worker.status_updated.connect(self.progress_label.setText)
        self.progress_label.setText("Starting batch processing…")
        self.progress_label.setVisible(True)
        self.batch_worker.batch_complete.connect(self.handle_batch_complete)
        self.batch_worker.batch_cancelled.connect(self.handle_batch_cancelled)
        self.batch_worker.batch_failed.connect(self.handle_batch_failed)

        # Start the worker thread
        self.batch_worker.start()

    def _show_batch_progress(self, shown):
        """The Run card's status line and bar, shown while a batch run is going."""
        self.batch_progress_bar.setValue(0)
        self.batch_progress_bar.setVisible(shown)
        self.progress_label.setVisible(shown)
        if not shown:
            self.progress_label.setText("")

    @staticmethod
    def _results_folder_in_use(folder):
        """True when ``folder`` holds anything a new run would delete (all but a Logs folder)."""
        try:
            return any(entry.lower() != LOGS_DIR.lower() for entry in os.listdir(folder))
        except OSError:
            return False

    def _restore_batch_buttons(self):
        self.cancel_batch_btn.setVisible(False)
        self.process_batch_btn.setVisible(True)
        self.process_batch_btn.setEnabled(True)
        self.param_btn.setEnabled(True)
        self.input_btn.setEnabled(True)
        self.output_btn.setEnabled(True)
        self.mask_btn.setEnabled(True)
        self.mask_clear_action.setEnabled(True)
        self.stats_cb.setEnabled(True)
        self.scores_cb.setEnabled(True)
        self.visuals_cb.setEnabled(True)
        # back to the selected file's Max Size after a resumed run showed its own
        try:
            with open(self.param_file) as f:
                params = yaml.safe_load(f)
            if isinstance(params, dict):
                self._show_batch_max_size(max_image_size(params))
        except (AttributeError, OSError, yaml.YAMLError):
            pass

    def _cancel_batch_processing(self):
        self.cancel_batch_btn.setEnabled(False)
        self.cancel_batch_btn.setText("Cancelling…")
        if hasattr(self, 'batch_worker'):
            self.batch_worker.cancel()

    def handle_batch_complete(self):
        self._show_batch_progress(False)
        self._restore_batch_buttons()

        output_folder = getattr(self, 'output_folder', '')
        if self._message("Batch Processing Complete", "Batch processing finished successfully.",
                         f"Results saved to:\n{output_folder}",
                         ("Open Folder", "OK"), default="OK") == "Open Folder":
            QDesktopServices.openUrl(QUrl.fromLocalFile(output_folder))

    def handle_batch_failed(self, message):
        self._show_batch_progress(False)
        self._restore_batch_buttons()

        output_folder = getattr(self, 'output_folder', '')
        if self._message("Batch Processing Failed", "Batch processing stopped with an error.",
                         f"{message}\n\n"
                         f"Completed batches and the checkpoint are kept in:\n{output_folder}\n\n"
                         "Fix the cause (for example free disk space) and run again with the same "
                         "output folder to resume.",
                         ("Open Folder", "OK"), default="OK") == "Open Folder":
            QDesktopServices.openUrl(QUrl.fromLocalFile(output_folder))

    def handle_batch_cancelled(self):
        self._show_batch_progress(False)
        self._restore_batch_buttons()

        output_folder = getattr(self, 'output_folder', '')
        if self._message("Batch Processing Cancelled", "Batch processing was cancelled.",
                         f"Partial results and a checkpoint are saved in:\n{output_folder}\n\n"
                         "You can resume from where you left off next time.",
                         ("Open Folder", "OK"), default="OK") == "Open Folder":
            QDesktopServices.openUrl(QUrl.fromLocalFile(output_folder))

    def _title_switch(self, page, label, button):
        """Put a page's on/off switch beside the page title (shown on that page only)."""
        box = QWidget()
        row = QHBoxLayout(box)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        row.addWidget(label)
        row.addWidget(button)
        box.setVisible(False)
        self.page_title_right.addWidget(box)
        self._title_switches[page] = box

    @staticmethod
    def _set_switch_label(label, on):
        label.setText("On" if on else "Off")
        color = COLORS['highlight'] if on else panel_colors()['faint']
        label.setStyleSheet(f"color: {color_to_stylesheet(color)}; font-size: {FONT_SIZES['small']}px;")

    def _slider_value_label(self, text):
        label = QLabel(text)
        label.setStyleSheet(self.value_label_style)
        return label

    def setup_segmentation_tab(self):
        """Set up the Segmentation page: Colour, Training and Large images cards;
        the on/off switch sits beside the page title."""
        layout = self._page_layout()

        self.toggle_seg_label = QLabel()
        self._set_switch_label(self.toggle_seg_label, True)
        self.toggle_seg_btn = ToggleButton()
        self.toggle_seg_btn.setChecked(True)
        self.toggle_seg_btn.toggled.connect(self.toggle_segmentation)
        self.toggle_seg_btn.setToolTip("Turn segmentation on or off.")
        self._title_switch(self.seg_tab, self.toggle_seg_label, self.toggle_seg_btn)

        colour = SectionCard("Colour")
        swatch = QHBoxLayout()
        swatch.setSpacing(10)
        self.color_btn = QPushButton("")
        self.color_btn.setStyleSheet(
            f"background-color: #f53282; border: 1px solid {color_to_stylesheet(COLORS['border'])}; border-radius: 4px;")
        self.color_btn.setFixedSize(QSize(18, 18))
        self.color_btn.clicked.connect(self.select_color)
        self.color_btn.setToolTip("Colour of the structures to keep, e.g. Picrosirius Red collagen; click to pick.")
        swatch.addWidget(self.color_btn)
        self.hue_label = QLabel("Hue 0.96")
        self.hue_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        swatch.addWidget(self.hue_label)
        colour.add_row("Colour of interest", swatch)

        self.color_thresh_slider = CustomSlider(Qt.Horizontal)
        self.color_thresh_slider.setRange(0, 100)
        self.color_thresh_slider.setValue(20)  # Default 0.2
        self.color_thresh_slider.valueChanged.connect(self.update_color_threshold)
        self.color_thresh_slider.setToolTip(
            "Minimum colour score of a region (1 = exactly the chosen colour). Lower it for paler\n"
            "tissue; below about 0.12 glass starts to count. Updates the result at once.")
        self.color_thresh_value = self._slider_value_label("0.2")
        colour.add_row("Colour threshold", self.color_thresh_slider, self.color_thresh_value)
        self.seg_colour_card = colour
        layout.addWidget(colour)

        self.seg_training_card = training = SectionCard("Training")
        self.num_labels_slider = CustomSlider(Qt.Horizontal)
        self.num_labels_slider.setRange(8, 96)
        self.num_labels_slider.setValue(32)  # Default
        self.num_labels_slider.valueChanged.connect(self.update_num_labels)
        self.num_labels_slider.setToolTip("Higher = finer segmentation.")
        self.num_labels_value = self._slider_value_label("32")
        training.add_row("Labels", self.num_labels_slider, self.num_labels_value)

        self.max_iters_slider = CustomSlider(Qt.Horizontal)
        self.max_iters_slider.setRange(10, 100)
        self.max_iters_slider.setValue(30)  # Default
        self.max_iters_slider.valueChanged.connect(self.update_max_iters)
        self.max_iters_slider.setToolTip("Lower = finer segmentation.")
        self.max_iters_value = self._slider_value_label("30")
        training.add_row("Max iterations", self.max_iters_slider, self.max_iters_value)
        layout.addWidget(training)

        self.seg_large_card = large = SectionCard("Large images")
        # Patch size (0 = segment the whole image at once)
        self.patch_size_spinner = QSpinBox()
        self.patch_size_spinner.setRange(0, 8192)
        self.patch_size_spinner.setSingleStep(256)
        self.patch_size_spinner.setSpecialValueText("Off")
        self.patch_size_spinner.setSuffix(" px")
        self.patch_size_spinner.setValue(1024)
        self.patch_size_spinner.setFixedWidth(130)
        self.patch_size_spinner.setStyleSheet(self.spinner_style)
        self.patch_size_spinner.valueChanged.connect(self.update_patch_size)
        self.patch_size_spinner.setToolTip(
            "Large images are segmented in patches of this size to keep fine fibres;\n"
            "smaller is finer but slower. Off: whole image at 512 px.")
        large.add_row("Patch size", self.patch_size_spinner, stretch=False)

        # Max Size (Configs: Max Size, 0 = no limit): batch runs split larger
        # images into blocks, each segmented as a separate image. Saved by
        # Parameters > Export; the Batch Run page only shows it.
        self.seg_max_size_spinner = MaxSizeSpinBox()
        self.seg_max_size_spinner.setSingleStep(1000)
        self.seg_max_size_spinner.setSuffix(" px")
        self.seg_max_size_spinner.setValue(DEFAULT_MAX_SIZE)
        self.seg_max_size_spinner.setFixedWidth(130)
        self.seg_max_size_spinner.setStyleSheet(self.spinner_style)
        self.seg_max_size_spinner.valueChanged.connect(self.update_max_size)
        self.seg_max_size_spinner.setToolTip(
            "Batch runs split larger images into blocks of this size; lower it if memory\n"
            "runs out. No limit: never split.")
        large.add_row("Max size", self.seg_max_size_spinner, stretch=False)

        # How Max Size and Patch Size apply (the open image, patches of blocks)
        self.seg_size_note = QLabel("")
        self.seg_size_note.setWordWrap(True)
        self.seg_size_note.setStyleSheet(self.value_label_style)
        self.seg_size_note.setVisible(False)
        large.add_widget(self.seg_size_note)

        # Display options and the action share the footer
        view_row = QHBoxLayout()
        self.white_bg_cb = QCheckBox("White background")
        self.white_bg_cb.setChecked(True)
        self.white_bg_cb.setStyleSheet(self.checkbox_style)
        self.white_bg_cb.stateChanged.connect(self.update_white_bg)
        self.white_bg_cb.setToolTip("Fill the area outside the region with white; untick for bright fibres on dark.")
        view_row.addWidget(self.white_bg_cb)
        view_row.addStretch()
        # Toggle checkbox for comparing with the original image
        self.toggle_img_cb = QCheckBox("Overlay original")
        self.toggle_img_cb.setChecked(False)
        self.toggle_img_cb.setStyleSheet(self.checkbox_style)
        self.toggle_img_cb.clicked.connect(self.compare_image)
        self.toggle_img_cb.setToolTip("Show the original image.")
        view_row.addWidget(self.toggle_img_cb)
        large.add_footer(view_row)

        self.segment_btn = self._primary_button("Segment")
        self.segment_btn.clicked.connect(self.run_segmentation)
        self.segment_btn.setEnabled(False)
        large.add_footer(self.segment_btn)
        layout.addWidget(large)

        layout.addStretch()
        self.seg_tab.setLayout(layout)

    # Hue axis of the band sliders: ImageJ's 0-255 hue scale shifted so that the
    # red-orange band can extend below 0 into the deep red hues 192-255
    # (stored in the parameter file as Min > Max, a wrapping range)
    BRF_HUE_AXIS = (-64, 191)

    # Fill colour of each band's hue slider
    BRF_BAND_COLOURS = {"Red-Orange": "#E0533D", "Yellow": "#E3B931", "Green": "#3FAF5A"}

    def setup_birefringence_tab(self):
        """Set up the Birefringence page: colour bands of polarised-light images
        (PR-BRF Analyser classification, see cabana.brf). A Bands card (used by
        the open image and the folder run) and a Folder run card."""
        layout = self._page_layout()

        # The page's own settings: the macro defaults at every start, not part of
        # the parameter file (Batch Run analyses the separated images like any other)
        self._brf_settings = dict(BRF_DEFAULTS)

        # --- Bands: used by both the open image and the folder run ---
        bands = SectionCard("Bands")
        bands.title_label.setToolTip("How pixels are sorted into bands; check them on the open image with Separate.")

        self.brf_brightness_slider = CustomSlider(Qt.Horizontal)
        self.brf_brightness_slider.setRange(0, 255)
        self.brf_brightness_slider.setValue(10)  # Default (as in the macro)
        self.brf_brightness_slider.valueChanged.connect(self.update_brf_brightness)
        self.brf_brightness_slider.setToolTip(
            "Darker pixels are background (default 10, as in the macro).")
        self.brf_brightness_value = self._slider_value_label("10")
        bands.add_row("Min brightness", self.brf_brightness_slider, self.brf_brightness_value)

        self.brf_saturation_slider = CustomSlider(Qt.Horizontal)
        self.brf_saturation_slider.setRange(0, 255)
        self.brf_saturation_slider.setValue(0)  # Default (as in the macro)
        self.brf_saturation_slider.valueChanged.connect(self.update_brf_saturation)
        self.brf_saturation_slider.setToolTip(
            "Less saturated pixels belong to no band (default 0: grey pixels count as red-orange).")
        self.brf_saturation_value = self._slider_value_label("0")
        bands.add_row("Min saturation", self.brf_saturation_slider, self.brf_saturation_value)

        # Hue bands: range sliders on one hue axis, filled in the band's colour; they never overlap
        self._brf_syncing = False
        self.brf_band_sliders, self.brf_band_values = [], []
        tips = {
            "Red-Orange": "Hue range of red-orange (thick fibres), default 0-27. Drag below 0 to\n"
                          "include deep red (hues 192-255).",
            "Yellow": "Hue range of yellow, default 28-47.",
            "Green": "Hue range of green (thin fibres), default 48-140.",
        }
        for index, band in enumerate(brf.BANDS):
            slider = RangeSlider(Qt.Horizontal)
            slider.fill_color = QColor(self.BRF_BAND_COLOURS[band])
            slider.setRange(*self.BRF_HUE_AXIS)
            lo, hi = brf.band_ranges(BRF_DEFAULTS)[index]
            slider.setValues(lo, hi)
            slider.valueChanged.connect(lambda lo, hi, i=index: self.update_brf_band(i, lo, hi))
            slider.setToolTip(tips[band] + "\n"
                                           "An edge stops at the neighbouring band.")
            value = self._slider_value_label(f"{lo}, {hi}")
            bands.add_row(band.capitalize(), slider, value)
            self.brf_band_sliders.append(slider)
            self.brf_band_values.append(value)

        # View, with the overlay checkbox on the same row
        view_row = QHBoxLayout()
        view_row.setSpacing(12)
        self.brf_view_combo = QComboBox()
        self.brf_view_combo.addItems(["Grid", "Total", *brf.BANDS, "Class map"])
        self.brf_view_combo.setStyleSheet(self.combo_style)
        self.brf_view_combo.currentIndexChanged.connect(self._show_brf_view)
        self.brf_view_combo.setToolTip(
            "Grid: original and the three bands. Total or a band: those pixels, as saved.\n"
            "Class map: one colour per band, grey for bright pixels in no band.")
        view_row.addWidget(self.brf_view_combo, 1)
        self.brf_original_cb = QCheckBox("Overlay")
        self.brf_original_cb.setChecked(False)
        self.brf_original_cb.setStyleSheet(self.checkbox_style)
        self.brf_original_cb.clicked.connect(self._show_brf_view)
        self.brf_original_cb.setToolTip("Overlay the original image.")
        view_row.addWidget(self.brf_original_cb)
        bands.add_row("View", view_row)

        # Note shown for bright-field images
        self.brf_note = QLabel("This looks like a bright-field image; birefringence bands need a "
                               "polarised-light image.")
        self.brf_note.setWordWrap(True)
        self.brf_note.setStyleSheet(self.value_label_style)
        self.brf_note.setVisible(False)
        bands.add_widget(self.brf_note)

        self.brf_btn = self._primary_button("Separate")
        self.brf_btn.clicked.connect(self.run_birefringence)
        self.brf_btn.setEnabled(False)
        bands.add_footer(self.brf_btn)
        # Summary of the current separation
        self.brf_summary_label = QLabel("")
        self.brf_summary_label.setWordWrap(True)
        self.brf_summary_label.setStyleSheet(self.value_label_style)
        self.brf_summary_label.setToolTip("Signal per band, share of the image in a band or in none, and band area.")
        bands.add_footer(self.brf_summary_label)
        self.brf_bands_card = bands
        layout.addWidget(bands)

        # --- Folder: colour separation only, no segmentation or fibre analysis ---
        self.brf_folder_card = folder = SectionCard("Folder run")
        folder.title_label.setToolTip("Separate every image of a folder (no segmentation or fibre analysis);\n"
                                      "analyse the band images afterwards with Batch Run.")

        def folder_row(label, tip, select, placeholder="Not selected"):
            edit = PathEdit("")
            edit.setPlaceholderText(placeholder)
            edit.setToolTip(tip)
            edit.pick_action.triggered.connect(select)
            folder.add_path_row(label, edit)
            return edit, edit.pick_action

        self.brf_input_path, self.brf_input_btn = folder_row(
            "Input folder", "Polarised-light images (TIFF/PNG/JPEG); the pixel size is read from their metadata.",
            self.select_brf_input_folder)
        self.brf_input_btn.setToolTip("Choose the input folder")
        self.brf_mask_path, self.brf_mask_btn = folder_row(
            "ROI masks", "Optional masks named like the images (white = measure). Empty: whole images.",
            self.select_brf_mask_folder, placeholder="Optional")
        self.brf_mask_btn.setToolTip("Choose the ROI mask folder")
        self.brf_mask_clear_action = add_clear_action(self.brf_mask_path, lambda: self.set_brf_mask_folder(None),
                                                      "Clear the ROI mask folder (whole images)")
        self.brf_output_path, self.brf_output_btn = folder_row(
            "Output folder", "Where the band images and tables go; files of the same names are replaced.",
            self.select_brf_output_folder)
        self.brf_output_btn.setToolTip("Choose the output folder")
        self.brf_input_folder = self.brf_mask_folder = self.brf_output_folder = None

        out_cb_layout = QHBoxLayout()
        self.brf_sort_cb = QCheckBox("Sort into folders")
        self.brf_sort_cb.setChecked(True)
        self.brf_sort_cb.setStyleSheet(self.checkbox_style)
        self.brf_sort_cb.stateChanged.connect(self.update_brf_sort)
        self.brf_sort_cb.setToolTip("Band images in Total, Red, Yellow and Green; tables in Stats.")
        out_cb_layout.addWidget(self.brf_sort_cb)
        self.brf_images_cb = QCheckBox("Save images")
        self.brf_images_cb.setChecked(True)
        self.brf_images_cb.setStyleSheet(self.checkbox_style)
        self.brf_images_cb.setToolTip("Write the band images; untick to write only the tables.")
        out_cb_layout.addStretch()
        out_cb_layout.addWidget(self.brf_images_cb)
        folder.add_layout(out_cb_layout)

        self.brf_run_btn = self._primary_button("Run Folder")
        self.brf_run_btn.clicked.connect(self.run_brf_folder)
        self.brf_run_btn.setEnabled(False)
        folder.add_footer(self.brf_run_btn)
        self.brf_progress_bar = PercentageProgressBar()
        self.brf_progress_bar.setStyleSheet(self.progressbar_style)
        self.brf_progress_bar.setVisible(False)
        folder.add_footer(self.brf_progress_bar)
        self.brf_folder_label = QLabel("")
        self.brf_folder_label.setWordWrap(True)
        self.brf_folder_label.setStyleSheet(self.value_label_style)
        folder.add_footer(self.brf_folder_label)
        self.brf_folder_worker = None
        layout.addWidget(folder)

        # the card titles (kept for tests and theme switching)
        self.brf_section_titles = [bands.title_label, folder.title_label]

        layout.addStretch()
        self.brf_tab.setLayout(layout)

    def update_brf_brightness(self):
        value = self.brf_brightness_slider.value()
        self.brf_brightness_value.setText(str(value))
        self._brf_settings["Brightness Threshold"] = value
        self._schedule_brf_refresh()

    def update_brf_saturation(self):
        value = self.brf_saturation_slider.value()
        self.brf_saturation_value.setText(str(value))
        self._brf_settings["Saturation Min"] = value
        self._schedule_brf_refresh()

    def update_brf_sort(self):
        self._brf_settings["Sort Into Folders"] = self.brf_sort_cb.isChecked()

    def update_brf_band(self, index, lo, hi):
        """A band slider moved: it stops at its neighbours' edges (bands never
        overlap and never move each other), then store all three ranges."""
        if self._brf_syncing:
            return
        self._brf_syncing = True
        try:
            sliders = self.brf_band_sliders
            if index > 0:
                lo = max(lo, sliders[index - 1].upper_value + 1)
            if index < len(sliders) - 1:
                hi = min(hi, sliders[index + 1].lower_value - 1)
            lo = min(lo, hi)
            slider = sliders[index]
            if (slider.lower_value, slider.upper_value) != (lo, hi):
                slider.setValues(lo, hi)
        finally:
            self._brf_syncing = False
        params = self._brf_settings
        for band, slider, value in zip(brf.BANDS, self.brf_band_sliders, self.brf_band_values):
            a, b = slider.lower_value, slider.upper_value
            a = a + 256 if a < 0 else a                        # below 0: wraps past 255
            params[f"{band} Min"], params[f"{band} Max"] = a, b
            value.setText(f"{a}, {b}")
        self._schedule_brf_refresh()

    def select_brf_input_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Input Folder", self.brf_input_folder or "",
                                                  QFileDialog.ShowDirsOnly)
        if folder:
            self.set_brf_input_folder(folder)

    def set_brf_input_folder(self, folder):
        self.brf_input_folder = folder or None
        self.brf_input_path.setText(folder or "")
        self.brf_input_path.setToolTip(folder or "")
        # a QuPath export is <region>/ next to <region>_mask/; a TMA export Images/ next to Masks/
        if folder and not self.brf_mask_folder:
            for sibling in (folder.rstrip("/\\") + "_mask", join_path(os.path.dirname(folder), "Masks")):
                if os.path.isdir(sibling) and (sibling.endswith("_mask") or os.path.basename(folder) == "Images"):
                    self.set_brf_mask_folder(sibling)
                    break
        self._check_brf_folder_ready()

    def select_brf_mask_folder(self):
        start = os.path.dirname(self.brf_input_folder) if self.brf_input_folder else ""
        folder = QFileDialog.getExistingDirectory(self, "Select ROI Mask Folder (optional)", start,
                                                  QFileDialog.ShowDirsOnly)
        if folder:
            self.set_brf_mask_folder(folder)

    def set_brf_mask_folder(self, folder):
        """The ROI mask folder of Folder Run; None (empty): whole images."""
        self.brf_mask_folder = folder or None
        self.brf_mask_path.setText(folder or "")
        self.brf_mask_path.setToolTip(folder or "")

    def select_brf_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder", "", QFileDialog.ShowDirsOnly)
        if folder:
            self.set_brf_output_folder(folder)

    def set_brf_output_folder(self, folder):
        self.brf_output_folder = folder or None
        self.brf_output_path.setText(folder or "")
        self.brf_output_path.setToolTip(folder or "")
        self._check_brf_folder_ready()

    def _check_brf_folder_ready(self):
        running = self.brf_folder_worker is not None and self.brf_folder_worker.isRunning()
        self.brf_run_btn.setEnabled(running or bool(self.brf_input_folder and self.brf_output_folder))

    def run_brf_folder(self):
        """Run Folder: colour separation of every image of the input folder (a
        worker thread); pressed again while running, it cancels."""
        if self.brf_folder_worker is not None and self.brf_folder_worker.isRunning():
            self.brf_folder_worker.cancel()
            self.brf_run_btn.setEnabled(False)
            self.brf_folder_label.setText("Cancelling after the current image...")
            return
        out = self.brf_output_folder
        if os.path.isdir(out) and os.listdir(out) and not self._ask_tma(
                "Run Folder", "The output folder is not empty.",
                f"{out}\n\nResults with the same names are replaced; other files are kept.",
                "Continue", "Cancel"):
            return
        self.brf_folder_worker = BRFFolderWorker(dict(self._brf_settings), self.brf_input_folder, out,
                                                 self.brf_mask_folder, self.brf_images_cb.isChecked())
        self.brf_folder_worker.progress_updated.connect(self._brf_folder_progress)
        self.brf_folder_worker.run_complete.connect(self.handle_brf_folder_complete)
        self.brf_folder_worker.failed.connect(self.handle_brf_folder_failed)
        self.brf_run_btn.setText("Cancel")
        self.brf_run_btn.setIcon(QIcon())
        self.brf_progress_bar.setValue(0)
        self.brf_progress_bar.setVisible(True)
        self.brf_folder_label.setText("Starting...")
        self.brf_folder_worker.start()

    def _brf_folder_progress(self, done, total, name):
        self.brf_progress_bar.setValue(int(done / max(1, total) * 100))
        if name:
            self.brf_folder_label.setText(f"{done + 1}/{total}: {name}")

    def _brf_folder_finished(self):
        self.brf_run_btn.setText("Run Folder")
        self._set_play_icon(self.brf_run_btn)
        self.brf_progress_bar.setVisible(False)
        self.brf_folder_label.setText("")
        self._check_brf_folder_ready()

    def handle_brf_folder_complete(self, summary):
        self._brf_folder_finished()
        title = "Run Folder"
        text = (f"Separated {summary['done']} of {summary['total']} images"
                + (" before it was cancelled." if summary["cancelled"] else "."))
        info = [summary["results"] or self.brf_output_folder]
        if summary["skipped"]:
            info.append(f"Not readable, skipped: {', '.join(summary['skipped'][:5])}"
                        + (" ..." if len(summary["skipped"]) > 5 else ""))
        if summary["no_pixel_size"]:
            info.append(f"{summary['no_pixel_size']} images have no pixel size in their metadata; "
                        f"their areas assume 1.0 µm/pixel.")
        self.brf_folder_label.setText(text)
        self._notify(title, text, "\n\n".join(info))

    def handle_brf_folder_failed(self, message):
        self._brf_folder_finished()
        self._notify("Run Folder", "The folder could not be separated.", message, warning=True)

    def run_birefringence(self):
        """Compute the HSB stack of the open image once (worker thread); the
        sliders then re-threshold it."""
        if self.ori_img is None:
            return
        if self._brf_hsb is not None:
            self._refresh_brf()
            return
        self.brf_btn.setEnabled(False)
        self.brf_btn.setText("Separating...")
        self.brf_worker = BRFWorker(self.ori_img)
        self.brf_worker.hsb_complete.connect(self.handle_brf_complete)
        self.brf_worker.failed.connect(self.handle_brf_failed)
        self.brf_worker.start()

    def handle_brf_complete(self, hsb):
        self._brf_hsb = hsb
        self.brf_btn.setText("Separate")
        self.brf_btn.setEnabled(self.ori_img is not None)
        self._refresh_brf()

    def handle_brf_failed(self, message):
        self.brf_btn.setText("Separate")
        self.brf_btn.setEnabled(self.ori_img is not None)
        self._notify("Birefringence failed", f"Birefringence did not complete:\n{message}", warning=True)

    def _schedule_brf_refresh(self):
        """Re-threshold (debounced) after a setting changed, once separated."""
        if self._brf_hsb is None:
            return
        if not hasattr(self, "_brf_timer"):
            self._brf_timer = QTimer(self)
            self._brf_timer.setSingleShot(True)
            self._brf_timer.timeout.connect(self._refresh_brf)
        self._brf_timer.start(150)

    def _refresh_brf(self):
        """Band masks, summary and view at the current settings."""
        if self._brf_hsb is None or self.ori_img is None:
            return
        params = self._brf_settings
        hue, sat, bright = self._brf_hsb
        masks, gate = brf.band_masks(hue, sat, bright, brf.band_ranges(params),
                                     int(params["Brightness Threshold"]), int(params["Saturation Min"]))
        self._brf_masks = (masks, gate)
        stats = brf.band_stats(masks, gate, self.ori_img, pixel_size_um=self._brf_pixel_size or 1.0)
        self.brf_summary_label.setText(self._brf_summary(stats))
        self._show_brf_view()

    def _brf_summary(self, stats):
        signal = " · ".join(f"{b} {stats[f'BRF {b} (% Signal)']:.1f} %" for b in brf.BANDS)
        text = (f"{signal} of the signal\n{stats['BRF Total (% Region)']:.2f} % of the image in the bands"
                f" · {stats['BRF Unclassified (% Region)']:.2f} % unclassified")
        if self._brf_pixel_size:
            text += f" · {stats['BRF Total Area (µm²)']:,.0f} µm²"
        return text

    def _show_brf_view(self):
        """Show the chosen view of the separation (or the original image)."""
        if self.ori_img is None:
            return
        if self.brf_original_cb.isChecked():
            self.image_panel.setImage(self.ori_img, preserve_view=True)
            return
        if self._brf_masks is None:
            return
        masks, gate = self._brf_masks
        view = self.brf_view_combo.currentText()
        if view == "Grid":
            image = brf.grid_view(self.ori_img, masks)
        elif view == "Class map":
            image = brf.class_map(masks, gate)
        elif view == "Total":
            image = brf.channel_image(self.ori_img, np.logical_or.reduce(masks))
        else:
            image = brf.channel_image(self.ori_img, masks[brf.BANDS.index(view)])
        self.image_panel.setImage(image, preserve_view=True)

    def _clear_brf_results(self):
        """A new image was opened: drop the separation of the previous one."""
        self._brf_hsb = None
        self._brf_masks = None
        self.brf_summary_label.setText("")
        self.brf_note.setVisible(self.ori_img is not None and brf.looks_brightfield(self.ori_img))
        self._brf_pixel_size = None
        if self.img_path:
            try:
                size = pixel_size_um(self.img_path)
                self._brf_pixel_size = float(size[0]) if size else None
            except Exception:
                self._brf_pixel_size = None
        self.brf_btn.setEnabled(self.ori_img is not None)

    def setup_detection_tab(self):
        """Set up the Fibre Detection page: a Detection card (ridge detection,
        the Detect preview) and a Measurements card (used by Batch Run)."""
        layout = self._page_layout()

        detection = SectionCard("Detection")
        self.line_width_range = RangeSlider(Qt.Horizontal)
        self.line_width_range.setRange(1, 15)
        self.line_width_range.setValues(5, 7)  # Default values
        self.line_width_range.valueChanged.connect(self.update_line_width_range)
        self.line_width_range.setToolTip("Widths of the fibres to detect; raise them for thicker fibres.")
        self.line_width_value = self._slider_value_label("5, 7 px")
        detection.add_row("Line width", self.line_width_range, self.line_width_value)

        self.line_step_slider = CustomSlider(Qt.Horizontal)
        self.line_step_slider.setRange(1, 5)
        self.line_step_slider.setValue(2)  # Default
        self.line_step_slider.valueChanged.connect(self.update_line_step)
        self.line_step_slider.setToolTip("Step between line widths; lower finds more fibres.")
        self.line_step_value = self._slider_value_label("2 px")
        detection.add_row("Line step", self.line_step_slider, self.line_step_value)

        self.contrast_range = RangeSlider(Qt.Horizontal)
        self.contrast_range.setRange(0, 255)
        self.contrast_range.setValues(100, 200)  # Default values
        self.contrast_range.valueChanged.connect(self.update_contrast_range)
        self.contrast_range.setToolTip("Lower it for faint fibres.")
        self.contrast_value = self._slider_value_label("100, 200")
        detection.add_row("Contrast", self.contrast_range, self.contrast_value)

        self.min_length_slider = CustomSlider(Qt.Horizontal)
        self.min_length_slider.setRange(1, 50)
        self.min_length_slider.setValue(5)  # Default
        self.min_length_slider.valueChanged.connect(self.update_min_length)
        self.min_length_slider.setToolTip("Shorter fibres are ignored.")
        self.min_length_value = self._slider_value_label("5 px")
        detection.add_row("Min line length", self.min_length_slider, self.min_length_value)

        checkbox_layout = QHBoxLayout()
        self.dark_line_cb = QCheckBox("Dark line")
        self.dark_line_cb.setChecked(True)
        self.dark_line_cb.stateChanged.connect(self.update_dark_line)
        self.dark_line_cb.setToolTip("Dark fibres on a light background (Picrosirius Red); untick for bright fibres.")
        self.extend_line_cb = QCheckBox("Extend line")
        self.extend_line_cb.setChecked(False)
        self.extend_line_cb.stateChanged.connect(self.update_extend_line)
        self.extend_line_cb.setToolTip("Also detect fibres near junctions (may add artefacts).")
        self.overlay_fibres_cb = QCheckBox("Overlay fibres")
        self.overlay_fibres_cb.setChecked(True)
        self.overlay_fibres_cb.stateChanged.connect(self.update_overlay_fibres)
        self.overlay_fibres_cb.setToolTip("Show the detected fibres on the image.")
        for cb in (self.dark_line_cb, self.extend_line_cb, self.overlay_fibres_cb):
            cb.setStyleSheet(self.checkbox_style)
            checkbox_layout.addWidget(cb)
            checkbox_layout.addStretch()
        detection.add_layout(checkbox_layout)

        self.detect_btn = self._primary_button("Detect")
        self.detect_btn.clicked.connect(self.run_detection)
        self.detect_btn.setEnabled(False)
        detection.add_footer(self.detect_btn)
        self.det_detection_card = detection
        layout.addWidget(detection)

        # Fibre measurements: used by the skeleton analysis of a full run
        # (Batch Run), not by the Detect preview above
        self.det_measurements_card = measurements = SectionCard("Measurements")
        measurements.setToolTip("Used by Batch Run, not by Detect.")
        note = QLabel("Used by Batch Run, not by Detect.")
        note.setObjectName("note")
        measurements.add_widget(note)

        self.min_branch_slider = CustomSlider(Qt.Horizontal)
        self.min_branch_slider.setRange(1, 100)
        self.min_branch_slider.setValue(5)  # Default
        self.min_branch_slider.valueChanged.connect(self.update_min_branch)
        self.min_branch_slider.setToolTip(
            "Shorter side branches are pruned before length, branching and curvature are measured.")
        self.min_branch_value = self._slider_value_label("5 px")
        measurements.add_row("Min branch length", self.min_branch_slider, self.min_branch_value)

        self.curve_window_range = RangeSlider(Qt.Horizontal)
        self.curve_window_range.setRange(2, 100)
        self.curve_window_range.setValues(10, 30)  # Default values
        self.curve_window_range.valueChanged.connect(self.update_curve_window_range)
        self.curve_window_range.setToolTip(
            "Curvature window sizes, from smaller to larger in steps of Window step;\n"
            "one Curvature column per window.")
        self.curve_window_value = self._slider_value_label("10, 30 px")
        measurements.add_row("Curvature window", self.curve_window_range, self.curve_window_value)

        self.curve_step_slider = CustomSlider(Qt.Horizontal)
        self.curve_step_slider.setRange(1, 100)
        self.curve_step_slider.setValue(10)  # Default
        self.curve_step_slider.valueChanged.connect(self.update_curve_step)
        self.curve_step_slider.setToolTip("Spacing between the curvature window sizes.")
        self.curve_step_value = self._slider_value_label("10 px")
        measurements.add_row("Window step", self.curve_step_slider, self.curve_step_value)
        layout.addWidget(measurements)

        layout.addStretch()
        self.det_tab.setLayout(layout)

    def setup_gap_analysis_tab(self):
        """Set up the Gap Analysis page: a Gaps card and an HDM card (the HDM
        settings stay active when gap analysis is off); the on/off switch sits
        beside the page title."""
        layout = self._page_layout()

        self.toggle_gap_label = QLabel()
        self._set_switch_label(self.toggle_gap_label, True)
        self.toggle_gap_btn = ToggleButton()
        self.toggle_gap_btn.setChecked(True)
        self.toggle_gap_btn.toggled.connect(self.toggle_gap_analysis)
        self.toggle_gap_btn.setToolTip("Turn gap analysis on or off.")
        self._title_switch(self.gap_tab, self.toggle_gap_label, self.toggle_gap_btn)

        gaps = SectionCard("Gaps")
        self.min_gap_slider = CustomSlider(Qt.Horizontal)
        self.min_gap_slider.setRange(5, 100)
        self.min_gap_slider.setValue(20)  # Default
        self.min_gap_slider.valueChanged.connect(self.update_min_gap)
        self.min_gap_slider.setToolTip("Smallest gap to report; lower finds smaller gaps.")
        self.min_gap_value = self._slider_value_label("20 px")
        gaps.add_row("Min gap diameter", self.min_gap_slider, self.min_gap_value)

        self.overlay_gaps_cb = QCheckBox("Overlay gaps")
        self.overlay_gaps_cb.setChecked(False)
        self.overlay_gaps_cb.setStyleSheet(self.checkbox_style)
        self.overlay_gaps_cb.stateChanged.connect(self.update_overlay_gaps)
        self.overlay_gaps_cb.setToolTip("Show the gaps over the tissue image.")
        gaps.add_widget(self.overlay_gaps_cb)

        self.analyze_btn = self._primary_button("Analyse")
        self.analyze_btn.clicked.connect(self.run_gap_analysis)
        self.analyze_btn.setEnabled(False)
        gaps.add_footer(self.analyze_btn)
        # Summary of the last gap analysis
        self.gap_summary_label = QLabel("")
        self.gap_summary_label.setWordWrap(True)
        self.gap_summary_label.setStyleSheet(self.value_label_style)
        self.gap_summary_label.setToolTip("Last analysis: gap count, median diameter and share of the image covered.")
        gaps.add_footer(self.gap_summary_label)
        self.gap_gaps_card = gaps
        layout.addWidget(gaps)

        self.gap_hdm_card = hdm = SectionCard("Hematoxylin density (HDM)")
        self.max_hdm_slider = CustomSlider(Qt.Horizontal)
        self.max_hdm_slider.setRange(100, 255)
        self.max_hdm_slider.setValue(230)  # Default
        self.max_hdm_slider.valueChanged.connect(self.update_max_hdm)
        self.max_hdm_slider.setToolTip("Highest grey level counted in the HDM; lower it to narrow the HDM area.")
        self.max_hdm_value = self._slider_value_label("230")
        hdm.add_row("Max display HDM", self.max_hdm_slider, self.max_hdm_value)

        # HDM saturation (Quantification: Contrast Enhancement, a fraction)
        self.hdm_saturation_slider = CustomSlider(Qt.Horizontal)
        self.hdm_saturation_slider.setRange(0, 50)
        self.hdm_saturation_slider.setValue(10)  # Default 0.1
        self.hdm_saturation_slider.valueChanged.connect(self.update_hdm_saturation)
        self.hdm_saturation_slider.setToolTip(
            "Share of pixels saturated when stretching the HDM contrast; used by Batch Run.")
        self.hdm_saturation_value = self._slider_value_label("10 %")
        hdm.add_row("Saturation", self.hdm_saturation_slider, self.hdm_saturation_value)
        layout.addWidget(hdm)

        layout.addStretch()
        self.gap_tab.setLayout(layout)

    def update_line_width_range(self, min_val, max_val):
        """Update line width range values"""
        self.line_width_value.setText(f"{min_val}, {max_val} px")
        self.yml_data["Detection"]["Min Line Width"] = min_val
        self.yml_data["Detection"]["Max Line Width"] = max_val

    def update_contrast_range(self, min_val, max_val):
        """Update contrast range values"""
        self.contrast_value.setText(f"{min_val}, {max_val}")
        self.yml_data["Detection"]["Low Contrast"] = min_val
        self.yml_data["Detection"]["High Contrast"] = max_val

    def select_color(self):
        """Open the colour picker"""
        current_color = QColor(self.color_btn.styleSheet().split("background-color: ")[1].split(";")[0])
        color = QColorDialog.getColor(current_color)

        if color.isValid():
            hex_color = color.name()
            self.color_btn.setStyleSheet(
                f"background-color: {hex_color}; border: 1px solid {color_to_stylesheet(COLORS['border'])}; border-radius: 4px;")

            # Calculate hue
            hue = hex_to_hue(hex_color)
            self.hue_label.setText(f"Hue {hue:.2f}")
            # self.hue_label.setStyleSheet(f"color: {hex_color}")

            # Update YAML data
            self.yml_data["Segmentation"]["Normalized Hue Value"] = float(f"{hue:.2f}")
            self._seg_score = None      # the stored scores belong to the previous hue

    def update_color_threshold(self):
        """Update color threshold value; after a segmentation, re-threshold its
        colour scores (debounced) instead of segmenting again."""
        value = self.color_thresh_slider.value() / 100.0
        self.color_thresh_value.setText(f"{value:.2f}")
        self.yml_data["Segmentation"]["Color Threshold"] = value
        if self._seg_score is not None:
            if not hasattr(self, "_rethreshold_timer"):
                self._rethreshold_timer = QTimer(self)
                self._rethreshold_timer.setSingleShot(True)
                self._rethreshold_timer.timeout.connect(self._apply_color_threshold)
            self._rethreshold_timer.start(150)

    def _apply_color_threshold(self):
        """Redraw the segmentation result at the current colour threshold from
        the stored colour scores; no retraining."""
        if self._seg_score is None or self.ori_img is None:
            return
        score, min_size = self._seg_score
        mask = mask_from_score(score, min_size, self.yml_data["Segmentation"]["Color Threshold"], self.ori_img.shape)
        self.seg_img = generate_rois(self.ori_img, mask, self.white_bg_cb.isChecked())
        if not self.toggle_img_cb.isChecked():
            self.image_panel.setImage(self.seg_img, preserve_view=True)

    def update_num_labels(self):
        """Update number of labels value"""
        value = self.num_labels_slider.value()
        self.num_labels_value.setText(str(value))
        self.yml_data["Segmentation"]["Number of Labels"] = value
        self._seg_score = None

    def update_max_iters(self):
        """Update max iterations value"""
        value = self.max_iters_slider.value()
        self.max_iters_value.setText(str(value))
        self.yml_data["Segmentation"]["Max Iterations"] = value
        self._seg_score = None

    def update_patch_size(self):
        """Update segmentation patch size (0 disables patch-wise segmentation)"""
        self.yml_data["Segmentation"]["Patch Size"] = int(self.patch_size_spinner.value())
        self._seg_score = None
        self._refresh_size_note()

    def update_max_size(self):
        """Update Configs: Max Size (0 = no limit), saved by Parameters > Export"""
        self.yml_data.setdefault("Configs", {})["Max Size"] = int(self.seg_max_size_spinner.value())
        self._refresh_size_note()

    def _size_note_text(self):
        """What Max Size and Patch Size mean for batch runs: the open image is
        split (its preview segments it whole), and Patch Size cannot apply to
        blocks no larger than the patch. Empty when neither applies."""
        max_size = int(self.seg_max_size_spinner.value())
        patch = int(self.patch_size_spinner.value())
        notes = []
        if self.ori_img is not None:
            h, w = self.ori_img.shape[:2]
            rows, cols, bh, bw = block_grid(h, w, max_size)
            if rows * cols > 1:
                notes.append(f"Batch runs split this image ({w} × {h} px) into {rows * cols} blocks of "
                             f"{bw} × {bh} px; the preview here segments it whole.")
        if max_size and patch and patch >= max_size:
            notes.append(f"Patch Size has no effect on split images: their blocks are at most {max_size} px.")
        return "\n".join(notes)

    def _refresh_size_note(self):
        text = self._size_note_text()
        self.seg_size_note.setText(text)
        self.seg_size_note.setVisible(bool(text))

    def update_white_bg(self):
        """Update white background setting"""
        self.yml_data["Segmentation"]["Dark Line"] = self.white_bg_cb.isChecked()
        self.dark_line_cb.setChecked(self.white_bg_cb.isChecked())

    def update_line_step(self):
        """Update line step value"""
        value = self.line_step_slider.value()
        self.line_step_value.setText(f"{value} px")
        self.yml_data["Detection"]["Line Width Step"] = value

    def update_min_length(self):
        """Update minimum line length value"""
        value = self.min_length_slider.value()
        self.min_length_value.setText(f"{value} px")
        self.yml_data["Detection"]["Minimum Line Length"] = value

    def update_dark_line(self):
        """Update dark line setting"""
        self.yml_data["Detection"]["Dark Line"] = self.dark_line_cb.isChecked()

    def update_extend_line(self):
        """Update extend line setting"""
        self.yml_data["Detection"]["Extend Line"] = self.extend_line_cb.isChecked()

    def update_min_gap(self):
        """Update minimum gap diameter value"""
        value = self.min_gap_slider.value()
        self.min_gap_value.setText(f"{value} px")
        self.yml_data["Gap Analysis"]["Minimum Gap Diameter"] = value

    def update_max_hdm(self):
        """Update maximum HDM display value"""
        value = self.max_hdm_slider.value()
        self.max_hdm_value.setText(str(value))
        self.yml_data["Quantification"]["Maximum Display HDM"] = value

    def update_min_branch(self):
        """Update Quantification: Minimum Branch Length"""
        value = self.min_branch_slider.value()
        self.min_branch_value.setText(f"{value} px")
        self.yml_data.setdefault("Quantification", {})["Minimum Branch Length"] = value

    def update_curve_window_range(self, min_val, max_val):
        """Update Quantification: Minimum/Maximum Curvature Window"""
        self.curve_window_value.setText(f"{min_val}, {max_val} px")
        quant = self.yml_data.setdefault("Quantification", {})
        quant["Minimum Curvature Window"] = min_val
        quant["Maximum Curvature Window"] = max_val

    def update_curve_step(self):
        """Update Quantification: Curvature Window Step"""
        value = self.curve_step_slider.value()
        self.curve_step_value.setText(f"{value} px")
        self.yml_data.setdefault("Quantification", {})["Curvature Window Step"] = value

    def update_hdm_saturation(self):
        """Update Quantification: Contrast Enhancement (slider in percent)"""
        value = self.hdm_saturation_slider.value()
        self.hdm_saturation_value.setText(f"{value} %")
        quant = self.yml_data.setdefault("Quantification", {})
        # keep a finer value from a parameter file that rounds to this percent
        if round(float(quant.get("Contrast Enhancement", -1)) * 100) != value:
            quant["Contrast Enhancement"] = value / 100

    def load_default_params(self):
        """Load default parameters from YAML file"""
        default_params_path = Path(os.path.join(os.path.dirname(__file__), "default_params.yml"))
        if default_params_path.exists():
            self.yml_data = yaml.safe_load(default_params_path.read_text())
        else:
            # Define default parameters if file doesn't exist
            self.yml_data = {
                "Configs": {
                    "Segmentation": True,
                    "Quantification": True,
                    "Gap Analysis": True,
                    "Max Size": DEFAULT_MAX_SIZE,
                    "Save Visualisations": True,
                },
                "Segmentation": {
                    "Number of Labels": 32,
                    "Max Iterations": 30,
                    "Color Threshold": 0.2,
                    "Min Size": 64,
                    "Patch Size": 1024,
                    "Normalized Hue Value": 0.96
                },
                "Detection": {
                    "Min Line Width": 5,
                    "Max Line Width": 13,
                    "Line Width Step": 2,
                    "Low Contrast": 100,
                    "High Contrast": 200,
                    "Minimum Line Length": 5,
                    "Maximum Line Length": 0,
                    "Dark Line": True,
                    "Extend Line": False,
                },
                "Gap Analysis": {
                    "Minimum Gap Diameter": 20,
                },
                "Quantification": {
                    "Maximum Display HDM": 230,
                    "Contrast Enhancement": 0.1,
                    "Minimum Branch Length": 5,
                    "Minimum Curvature Window": 10,
                    "Maximum Curvature Window": 30,
                    "Curvature Window Step": 10,
                }
            }

    def export_parameters(self):
        """Export parameters to YAML file"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Params", "Parameters.yml", "YAML Files (*.yml)"
        )

        if file_path:
            with open(file_path, 'w') as file:
                yaml.dump(self.yml_data, file)

    def import_parameters(self):
        """Import parameters from a YAML file and apply to GUI widgets"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Params", "", "YAML Files (*.yml *.yaml)"
        )

        if not file_path:
            return

        try:
            with open(file_path, 'r') as file:
                data = yaml.safe_load(file)
            if not isinstance(data, dict):
                return
            self.yml_data = migrate_max_size(data)      # pre-0.3.9 files kept Max Size under Segmentation
            self.apply_params_to_widgets()
        except Exception as e:
            Log.logger.error(f"Failed to load parameters: {e}")

    def apply_params_to_widgets(self):
        """Update all GUI widgets to reflect current yml_data values"""
        seg = self.yml_data.get("Segmentation", {})
        det = self.yml_data.get("Detection", {})
        gap = self.yml_data.get("Gap Analysis", {})
        quant = self.yml_data.get("Quantification", {})
        configs = self.yml_data.get("Configs", {})

        # Segmentation widgets
        if "Color Threshold" in seg:
            self.color_thresh_slider.setValue(int(seg["Color Threshold"] * 100))
        if "Number of Labels" in seg:
            self.num_labels_slider.setValue(seg["Number of Labels"])
        if "Max Iterations" in seg:
            self.max_iters_slider.setValue(seg["Max Iterations"])
        if "Patch Size" in seg:
            self.patch_size_spinner.setValue(int(seg["Patch Size"] or 0))
        self.seg_max_size_spinner.setValue(max_image_size(self.yml_data))
        if "Dark Line" in seg:
            self.white_bg_cb.setChecked(seg["Dark Line"])
        if "Normalized Hue Value" in seg:
            hue = seg["Normalized Hue Value"]
            self.hue_label.setText(f"Hue {hue:.2f}")
            # Update color button to reflect the hue
            r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
            hex_color = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
            self.color_btn.setStyleSheet(
                f"background-color: {hex_color}; border: 1px solid {color_to_stylesheet(COLORS['border'])}; border-radius: 4px;")

        # Segmentation toggle
        if "Segmentation" in configs:
            self.toggle_seg_btn.setChecked(configs["Segmentation"])

        # Detection widgets
        if "Min Line Width" in det and "Max Line Width" in det:
            self.line_width_range.setValues(det["Min Line Width"], det["Max Line Width"])
        if "Low Contrast" in det and "High Contrast" in det:
            self.contrast_range.setValues(det["Low Contrast"], det["High Contrast"])
        if "Line Width Step" in det:
            self.line_step_slider.setValue(det["Line Width Step"])
        if "Minimum Line Length" in det:
            self.min_length_slider.setValue(det["Minimum Line Length"])
        if "Dark Line" in det:
            self.dark_line_cb.setChecked(det["Dark Line"])
        if "Extend Line" in det:
            self.extend_line_cb.setChecked(det["Extend Line"])

        # Gap analysis widgets
        if "Minimum Gap Diameter" in gap:
            self.min_gap_slider.setValue(gap["Minimum Gap Diameter"])
        if "Gap Analysis" in configs:
            self.toggle_gap_btn.setChecked(configs["Gap Analysis"])

        # Quantification widgets
        if "Maximum Display HDM" in quant:
            self.max_hdm_slider.setValue(quant["Maximum Display HDM"])
        if "Contrast Enhancement" in quant:
            self.hdm_saturation_slider.setValue(int(round(float(quant["Contrast Enhancement"]) * 100)))
        if "Minimum Branch Length" in quant:
            self.min_branch_slider.setValue(int(quant["Minimum Branch Length"]))
        if "Minimum Curvature Window" in quant and "Maximum Curvature Window" in quant:
            self.curve_window_range.setValues(int(quant["Minimum Curvature Window"]),
                                              int(quant["Maximum Curvature Window"]))
        if "Curvature Window Step" in quant:
            self.curve_step_slider.setValue(int(quant["Curvature Window Step"]))

    def load_original_image(self, path):
        """Load the original image from a file path"""
        try:
            img = tiff.imread(path) if path.lower().endswith(('.tif', '.tiff')) else iio.imread(path)
            if img.dtype != np.uint8:
                self.ori_img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
            else:
                self.ori_img = img
            self.img_path = path
            if len(self.ori_img.shape) < 3:
                self.ori_img = np.repeat(self.ori_img[:, :, np.newaxis], 3, axis=2)
            elif len(self.ori_img.shape) == 3 and self.ori_img.shape[2] > 3:
                # Remove alpha channel if present
                self.ori_img = self.ori_img[:, :, :3]

            if self.pages.currentIndex() == 0:
                self.show_page(self.pages.indexOf(self.seg_tab))

            # Enable processing buttons (Segment also requires the toggle on)
            self.segment_btn.setEnabled(self.toggle_seg_btn.isChecked())
            self.detect_btn.setEnabled(True)
            self.reload_btn.setEnabled(True)
            self.seg_img = None
            self._seg_score = None
            self.frb_img = None
            self.wdt_img = None
            self._clear_gap_results()
            self._clear_brf_results()
            self._update_image_status()
        except Exception as e:
            self.ori_img = None
            self.img_path = None
            self.segment_btn.setEnabled(False)
            self.detect_btn.setEnabled(False)
            self.reload_btn.setEnabled(False)
            self._update_image_status()
            Log.logger.error(f"Error loading image: {e}")


    def load_image(self):
        """Open a file dialog to select an image file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.tif)")

        if file_path:
            img = tiff.imread(file_path) if file_path.lower().endswith(('.tif', '.tiff')) else iio.imread(file_path)
            if img.dtype != np.uint8:
                self.ori_img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
            else:
                self.ori_img = img
            self.image_panel.setImage(self.ori_img)
            self.img_path = file_path
            if len(self.ori_img.shape) < 3:
                self.ori_img = np.repeat(self.ori_img[:, :, np.newaxis], 3, axis=2)
            elif len(self.ori_img.shape) == 3 and self.ori_img.shape[2] > 3:
                self.ori_img = self.ori_img[:, :, :3]

            self.seg_img = None
            self._seg_score = None
            self.frb_img = None
            self.wdt_img = None
            self._clear_gap_results()
            self._clear_brf_results()
            self.segment_btn.setEnabled(self.toggle_seg_btn.isChecked())
            self.detect_btn.setEnabled(True)
            self.reload_btn.setEnabled(True)
            self._update_image_status()

    def reload_image(self):
        """Reload the original image"""
        if self.img_path:
            self.image_panel.setImage(self.img_path)
            self.segment_btn.setEnabled(self.toggle_seg_btn.isChecked())
            self.detect_btn.setEnabled(True)
            self.reload_btn.setEnabled(True)

    def compare_image(self):
        """Toggle the original image for comparison."""
        show_original = self.toggle_img_cb.isChecked()
        image = self.ori_img if show_original else self.seg_img
        label = "original" if show_original else "segmented"

        if image is not None:
            self.image_panel.setImage(image, preserve_view=True)
        else:
            self._notify("No Image", f"No {label} image {'loaded' if label == 'original' else 'available'}.",
                         warning=True)

    def update_overlay_fibres(self):
        """Overlay fibres on the image"""
        show_overlay = self.overlay_fibres_cb.isChecked()
        show_img = self.ori_img if (self.seg_img is None or not self.toggle_seg_btn.isChecked()) else self.seg_img
        image = self.wdt_img if show_overlay else show_img
        label = "fibre" if show_overlay else "original"

        if image is not None:
            self.image_panel.setImage(image, preserve_view=True)
        else:
            self._notify("No Image", f"No {label} image available.", warning=True)

    def update_overlay_gaps(self):
        """Show the gap circles over the tissue image (Overlay Gaps) or the fibre mask."""
        if self.gap_img is not None:
            self._show_gaps(preserve_view=True)
        else:
            self._notify("No Image", "No gap image available.", warning=True)

    def toggle_segmentation(self):
        """Disable segmentation and update the UI accordingly"""
        if not self.toggle_seg_btn.isChecked():
            self.yml_data["Configs"]["Segmentation"] = False
            self._set_switch_label(self.toggle_seg_label, False)
            self.segment_btn.setEnabled(False)
            self.color_btn.setEnabled(False)
            self.color_thresh_slider.setEnabled(False)
            self.num_labels_slider.setEnabled(False)
            self.max_iters_slider.setEnabled(False)
            self.patch_size_spinner.setEnabled(False)
            self.white_bg_cb.setEnabled(False)
            self.toggle_img_cb.setEnabled(False)
        else:
            self.yml_data["Configs"]["Segmentation"] = True
            self._set_switch_label(self.toggle_seg_label, True)
            # Segment still requires a loaded image
            self.segment_btn.setEnabled(self.ori_img is not None)
            self.color_btn.setEnabled(True)
            self.color_thresh_slider.setEnabled(True)
            self.num_labels_slider.setEnabled(True)
            self.max_iters_slider.setEnabled(True)
            self.patch_size_spinner.setEnabled(True)
            self.white_bg_cb.setEnabled(True)
            self.toggle_img_cb.setEnabled(True)

    def toggle_gap_analysis(self):
        """Disable gap analysis and update the UI accordingly"""
        if not self.toggle_gap_btn.isChecked():
            self.yml_data["Configs"]["Gap Analysis"] = False
            self._set_switch_label(self.toggle_gap_label, False)
            self.analyze_btn.setEnabled(False)
            self.min_gap_slider.setEnabled(False)
            self.overlay_gaps_cb.setEnabled(False)
        else:
            self.yml_data["Configs"]["Gap Analysis"] = True
            self._set_switch_label(self.toggle_gap_label, True)
            self.analyze_btn.setEnabled(True)
            self.min_gap_slider.setEnabled(True)
            self.overlay_gaps_cb.setEnabled(True)

    def run_segmentation(self):
        if self.ori_img is None:
            return

        # Disable all buttons during processing
        self.segment_btn.setEnabled(False)
        self.segment_btn.setText("Segmenting...")
        self.detect_btn.setEnabled(False)
        self.analyze_btn.setEnabled(False)
        self.show_progress_bar()

        seg_args = parse_args()
        seg_args.num_channels = self.yml_data["Segmentation"]["Number of Labels"]
        seg_args.max_iter = self.yml_data["Segmentation"]["Max Iterations"]
        seg_args.hue_value = self.yml_data["Segmentation"]["Normalized Hue Value"]
        seg_args.rt = self.yml_data["Segmentation"]["Color Threshold"]
        seg_args.patch_size = int(self.yml_data["Segmentation"].get("Patch Size", 0) or 0)
        seg_args.white_background = self.white_bg_cb.isChecked()

        # Create and configure the worker
        self.segmentation_worker = SegmentationWorker(self.ori_img, seg_args)

        # Connect signals
        self.segmentation_worker.progress_updated.connect(lambda value: self.progress_bar.setValue(value))
        self.segmentation_worker.segmentation_complete.connect(self.handle_segmentation_complete)
        self.segmentation_worker.failed.connect(lambda m: self.handle_worker_failed("Segmentation", m))

        # Start the worker thread
        self.segmentation_worker.start()

    def handle_segmentation_complete(self, result):
        """Handle the completed segmentation result"""
        # Store the result and the colour scores the Color Threshold slider re-thresholds
        self.seg_img, self._seg_score = result

        # Update the UI
        self.image_panel.setImage(self.seg_img)

        # Hide progress bar
        self.hide_progress_bar()

        # Re-enable buttons
        self.segment_btn.setEnabled(True)
        self.segment_btn.setText("Segment")
        self.detect_btn.setEnabled(True)

    def handle_detection_complete(self, result):
        """Handle the completed detection result"""

        # Store the binary mask
        self.frb_img = result[1]

        self.wdt_img = result[0]
        self.image_panel.setImage(self.wdt_img)
        self.overlay_fibres_cb.setChecked(True)

        # Hide progress bar
        self.hide_progress_bar()

        # Re-enable buttons
        self.load_btn.setEnabled(True)
        self.segment_btn.setEnabled(True)
        self.detect_btn.setEnabled(True)
        self.detect_btn.setText("Detect")
        self.analyze_btn.setEnabled(True)

    def handle_worker_failed(self, stage, message):
        """A single-image worker raised: restore the buttons and tell the user."""
        self.hide_progress_bar()
        self.load_btn.setEnabled(True)
        self.segment_btn.setEnabled(self.ori_img is not None)
        self.segment_btn.setText("Segment")
        self.detect_btn.setEnabled(self.ori_img is not None or self.seg_img is not None)
        self.detect_btn.setText("Detect")
        self.analyze_btn.setEnabled(self.frb_img is not None and self.toggle_gap_btn.isChecked())
        self.analyze_btn.setText("Analyse")
        self._notify(f"{stage} failed", f"{stage} did not complete:\n{message}", warning=True)

    def show_progress_bar(self):
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)

    def hide_progress_bar(self):
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        self.progress_label.setText("")

    def run_detection(self):
        if self.ori_img is None and self.seg_img is None:
            return

        # Determine which image to use
        input_image = self.ori_img if (self.seg_img is None or not self.toggle_seg_btn.isChecked()) else self.seg_img

        # Disable all buttons during processing
        self.load_btn.setEnabled(False)
        self.segment_btn.setEnabled(False)
        self.detect_btn.setEnabled(False)
        self.detect_btn.setText("Detecting...")
        self.analyze_btn.setEnabled(False)
        self.show_progress_bar()

        class DetectionArgs:
            def __init__(self):
                self.min_line_width = None
                self.max_line_width = None
                self.line_step = None
                self.low_contrast = None
                self.high_contrast = None
                self.min_length = None
                self.dark_line = None
                self.extend_line = None

        # Create and populate the args object
        det_args = DetectionArgs()
        det_args.min_line_width = self.yml_data["Detection"]["Min Line Width"]
        det_args.max_line_width = self.yml_data["Detection"]["Max Line Width"]
        det_args.line_step = self.yml_data["Detection"]["Line Width Step"]
        det_args.low_contrast = self.yml_data["Detection"]["Low Contrast"]
        det_args.high_contrast = self.yml_data["Detection"]["High Contrast"]
        det_args.min_length = self.yml_data["Detection"]["Minimum Line Length"]
        det_args.dark_line = self.yml_data["Detection"]["Dark Line"]
        det_args.extend_line = self.yml_data["Detection"]["Extend Line"]

        # Ensure line_step is valid
        if det_args.line_step > det_args.max_line_width - det_args.min_line_width:
            det_args.line_step = det_args.max_line_width - det_args.min_line_width

        # Create and configure the worker
        self.detection_worker = DetectionWorker(input_image, det_args)

        # Connect signals
        self.detection_worker.progress_updated.connect(lambda value: self.progress_bar.setValue(value))
        self.detection_worker.detection_complete.connect(self.handle_detection_complete)
        self.detection_worker.failed.connect(lambda m: self.handle_worker_failed("Fibre detection", m))

        # Start the worker thread
        self.detection_worker.start()

    def run_gap_analysis(self):
        if self.frb_img is None:
            return
        min_gap_diameter = self.yml_data["Gap Analysis"]["Minimum Gap Diameter"]

        # Disable all buttons during processing
        self.detect_btn.setEnabled(False)
        self.segment_btn.setEnabled(False)
        self.load_btn.setEnabled(False)
        self.analyze_btn.setText("Analysing…")
        self.analyze_btn.setEnabled(False)
        self.show_progress_bar()

        self.gap_analysis_worker = GapAnalysisWorker(self.frb_img, min_gap_diameter)
        self.gap_analysis_worker.progress_updated.connect(lambda value: self.progress_bar.setValue(value))
        self.gap_analysis_worker.gap_analysis_complete.connect(self.handle_gap_analysis_complete)
        self.gap_analysis_worker.failed.connect(lambda m: self.handle_worker_failed("Gap analysis", m))

        self.gap_analysis_worker.start()

    def _clear_gap_results(self):
        self.gap_img = None
        self.gap_circles = None
        self._gap_intra = None
        self.gap_summary_label.setText("")

    def _show_gaps(self, preserve_view):
        """Base image with the circles painted live on top, so they stay smooth
        at any zoom: the fibres in dark grey on white, or with Overlay Gaps the
        faded tissue image with the fibres drawn over it."""
        if self.overlay_gaps_cb.isChecked():
            tissue = self.ori_img if (self.seg_img is None or not self.toggle_seg_btn.isChecked()) else self.seg_img
            base = gap_base_image(self.frb_img, tissue)
        else:
            base = self.gap_img
        self.image_panel.setImage(base, preserve_view=preserve_view)
        self.image_panel.set_overlay_painter(self._paint_gaps)    # setImage clears it

    def _segmentation_mask(self):
        """The current segmentation mask, or None when segmentation is off or not run."""
        if self._seg_score is None or not self.toggle_seg_btn.isChecked():
            return None
        score, min_size = self._seg_score
        return mask_from_score(score, min_size, self.yml_data["Segmentation"]["Color Threshold"],
                               self.ori_img.shape)

    def _paint_gaps(self, painter, zoom, to_widget):
        if self.gap_circles is None:
            return
        for (row, col, radius), intra in zip(self.gap_circles, self._gap_intra):
            colour = QColor(*(GAP_INTRA_RGB if intra else GAP_OTHER_RGB))
            pen = QPen(colour)
            pen.setWidthF(1.5)
            pen.setCosmetic(True)
            if not intra:
                pen.setStyle(Qt.DashLine)
            fill = QColor(colour)
            fill.setAlpha(GAP_FILL_ALPHA)
            painter.setPen(pen)
            painter.setBrush(fill)
            painter.drawEllipse(QPointF(col + 0.5, row + 0.5), radius, radius)

    def _gap_summary(self, circles):
        """'N gaps (M intra-fibre) · median Ø … · …% of the image' for the side panel."""
        if len(circles) == 0:
            return f"No gaps of at least {self.yml_data['Gap Analysis']['Minimum Gap Diameter']} px found."
        diameter_px = 2 * float(np.median(circles[:, 2]))
        size = pixel_size_um(self.img_path) if self.img_path else None
        diameter = f"{diameter_px * size[0]:.1f} µm ({diameter_px:.0f} px)" if size else f"{diameter_px:.0f} px"
        covered = np.pi * np.sum(circles[:, 2] ** 2) / self.frb_img.size * 100
        return (f"{len(circles)} gaps ({int(np.sum(self._gap_intra))} intra-fibre) · "
                f"median Ø {diameter} · {covered:.0f} % of the image")

    def handle_gap_analysis_complete(self, circles):
        """Show the fitted gap circles and their summary."""
        self.gap_circles = circles
        self.gap_img = gap_base_image(self.frb_img)
        # Intra-fibre: centre inside the segmentation mask; all gaps without segmentation
        self._gap_intra = gap_intra(circles, self._segmentation_mask())
        self.gap_summary_label.setText(self._gap_summary(circles))
        self.overlay_gaps_cb.blockSignals(True)
        self.overlay_gaps_cb.setChecked(False)
        self.overlay_gaps_cb.blockSignals(False)
        self._show_gaps(preserve_view=False)

        self.hide_progress_bar()
        self.segment_btn.setEnabled(True)
        self.analyze_btn.setEnabled(True)
        self.detect_btn.setEnabled(True)
        self.load_btn.setEnabled(True)
        self.analyze_btn.setText("Analyse")

    def set_theme(self):
        """Apply Napari-inspired theme to the application"""
        # Global font — platform-native for zero overhead
        app_font = QFont()
        if sys.platform == 'darwin':
            app_font.setFamily('.AppleSystemUIFont')
        elif sys.platform == 'win32':
            app_font.setFamily('Segoe UI')
        else:
            app_font.setFamily('Ubuntu')
        app_font.setPointSize(FONT_SIZES['base'])
        QApplication.setFont(app_font)

        # Set Napari palette
        palette = QPalette()

        # Set color group
        palette.setColor(QPalette.Window, COLORS['background'])
        palette.setColor(QPalette.WindowText, COLORS['text'])
        palette.setColor(QPalette.Base, COLORS['background'])
        palette.setColor(QPalette.AlternateBase, COLORS['dock'])
        palette.setColor(QPalette.ToolTipBase, COLORS['elevated'])
        palette.setColor(QPalette.ToolTipText, COLORS['text'])
        palette.setColor(QPalette.Text, COLORS['text'])
        palette.setColor(QPalette.Button, COLORS['dock'])
        palette.setColor(QPalette.ButtonText, COLORS['text'])
        palette.setColor(QPalette.BrightText, COLORS['highlight'])
        palette.setColor(QPalette.Link, COLORS['highlight'])
        palette.setColor(QPalette.Highlight, COLORS['highlight'])
        palette.setColor(QPalette.HighlightedText, COLORS['background'])

        # Disabled state colors
        palette.setColor(QPalette.Disabled, QPalette.WindowText, COLORS['text_dim'])
        palette.setColor(QPalette.Disabled, QPalette.Text, COLORS['text_dim'])
        palette.setColor(QPalette.Disabled, QPalette.ButtonText, COLORS['text_dim'])
        palette.setColor(QPalette.Disabled, QPalette.Button, COLORS['background'])

        # Apply the palette
        self.setPalette(palette)

        # Comprehensive stylesheet
        self.setStyleSheet(f"""
            /* Tooltips */
            QToolTip {{
                color: {color_to_stylesheet(COLORS['text'])};
                background-color: {color_to_stylesheet(COLORS['elevated'])};
                border: 1px solid {color_to_stylesheet(COLORS['border'])};
                border-radius: 3px;
                padding: 4px 6px;
                font-size: {FONT_SIZES['small']}px;
            }}

            /* Labels */
            QLabel {{
                color: {color_to_stylesheet(COLORS['text'])};
                font-size: {FONT_SIZES['base']}px;
            }}

            /* Checkbox indicators */
            QCheckBox {{
                color: {color_to_stylesheet(COLORS['text'])};
                spacing: 6px;
                font-size: {FONT_SIZES['base']}px;
            }}
            QCheckBox::indicator {{
                width: 16px;
                height: 16px;
                border: 1px solid {color_to_stylesheet(COLORS['border'])};
                border-radius: 3px;
                background-color: {color_to_stylesheet(COLORS['dock'])};
            }}
            QCheckBox::indicator:hover {{
                border-color: {color_to_stylesheet(COLORS['highlight'])};
                background-color: {color_to_stylesheet(COLORS['elevated'])};
            }}
            QCheckBox::indicator:checked {{
                background-color: {color_to_stylesheet(COLORS['highlight'])};
                border-color: {color_to_stylesheet(COLORS['highlight'])};
            }}
            QCheckBox::indicator:disabled {{
                background-color: {color_to_stylesheet(COLORS['background'])};
                border-color: {color_to_stylesheet(COLORS['border_subtle'])};
            }}
            QCheckBox:disabled {{
                color: {color_to_stylesheet(COLORS['text_dim'])};
            }}

            /* Scrollbars — vertical */
            QScrollBar:vertical {{
                background: {color_to_stylesheet(COLORS['background'])};
                width: 10px;
                margin: 0;
                border: none;
            }}
            QScrollBar::handle:vertical {{
                background: {color_to_stylesheet(COLORS['border'])};
                min-height: 30px;
                border-radius: 4px;
                margin: 2px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: {color_to_stylesheet(COLORS['elevated'])};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0;
            }}
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
                background: none;
            }}

            /* Scrollbars — horizontal */
            QScrollBar:horizontal {{
                background: {color_to_stylesheet(COLORS['background'])};
                height: 10px;
                margin: 0;
                border: none;
            }}
            QScrollBar::handle:horizontal {{
                background: {color_to_stylesheet(COLORS['border'])};
                min-width: 30px;
                border-radius: 4px;
                margin: 2px;
            }}
            QScrollBar::handle:horizontal:hover {{
                background: {color_to_stylesheet(COLORS['elevated'])};
            }}
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
                width: 0;
            }}
            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
                background: none;
            }}
        """)

    def _show_about_dialog(self) -> None:
        """Show the About dialog with version and project links."""
        dlg = CardDialog(self, "About Cabana", width=400)
        head = QHBoxLayout()
        head.setSpacing(14)
        icon = QLabel()
        pixmap = QIcon(str(ICON_PATH)).pixmap(56, 56)
        if not pixmap.isNull():
            icon.setPixmap(pixmap)
            head.addWidget(icon, 0, Qt.AlignTop)
        names = QVBoxLayout()
        names.setSpacing(2)
        name = QLabel("Cabana")
        name.setObjectName("dialogTitle")
        names.addWidget(name)
        for line in ("CollAgen FiBre ANAlyzer", f"Version {__version__}, MIT Licence"):
            label = QLabel(line)
            label.setObjectName("dialogNote")
            names.addWidget(label)
        head.addLayout(names, 1)
        dlg.body_layout.addLayout(head)
        dlg.body_layout.addSpacing(4)
        dlg.add_text("A toolkit for analysing collagen fibre architecture in IHC, polarised-light "
                     "and fluorescence microscopy images.")
        link = color_to_stylesheet(COLORS['highlight'])
        dlg.add_text("&nbsp;&nbsp;&nbsp;".join(
            f"<a href='{url}' style='color:{link}; text-decoration:none;'>{text}</a>"
            for text, url in (("Documentation", "https://cabana.readthedocs.io"),
                              ("Changelog", CHANGELOG_URL),
                              ("PyPI", "https://pypi.org/project/cabana/"))))
        dlg.add_button("Licence", slot=self._show_license_dialog)
        dlg.add_button("Close", primary=True)
        dlg.exec_()

    def _show_license_dialog(self) -> None:
        """Show the licence text in a read-only dialog."""
        dlg = CardDialog(self, "Licence", width=520)
        text = QLabel(self._license_text())
        text.setObjectName("dialogNote")
        text.setTextFormat(Qt.PlainText)
        text.setWordWrap(True)
        text.setStyleSheet("font-family: monospace;")
        dlg.body_layout.addWidget(text)
        dlg.add_button("Close", primary=True)
        dlg.exec_()

    # ------------------------------------------------------------------
    # What's New, the tour and the first start
    # ------------------------------------------------------------------
    startup_messages = True         # tests and scripts switch the startup prompts off
    LAST_VERSION_KEY = 'last_seen_version'

    def _startup_messages(self):
        """Once after an upgrade show What's New; on the very first start offer the tour."""
        settings = QSettings('Cabana', 'CabanaGUI')
        last = settings.value(self.LAST_VERSION_KEY, "") or ""
        # users of versions before this key existed still have other settings
        known = bool(last) or any(settings.value(key, None) is not None
                                  for key in ('theme', self.UPDATE_CHECK_KEY, 'tma_output_parent'))
        settings.setValue(self.LAST_VERSION_KEY, __version__)
        if not known:
            if self._ask_tour():
                self.start_tour()
        elif whats_new.should_show(last or "0.0.0", __version__):
            self.show_whats_new()

    def _ask_tour(self):
        """First start: offer the tour. True to start it (monkeypatched in tests)."""
        dlg = CardDialog(self, "Take a quick tour?", width=380)
        dlg.add_text("See where each step of the analysis lives. It takes about a minute and "
                     "changes no settings.")
        dlg.add_button("Not Now")
        dlg.add_button("Start Tour", primary=True, accept=True)
        return dlg.exec_() == QDialog.Accepted

    def show_whats_new(self, version=None):
        """The short notes of ``version`` (default: this version, else the newest notes)."""
        version = version or __version__
        notes = whats_new.notes_for(version)
        if notes is None:
            version, notes = whats_new.latest()
        dlg = CardDialog(self, f"What's new in Cabana {version}", width=440)
        dlg.add_bullets(notes)
        dlg.add_button("Full Changelog", slot=lambda: QDesktopServices.openUrl(QUrl(CHANGELOG_URL)))
        dlg.add_button("Close", primary=True)
        self._whats_new_dialog = dlg          # for tests
        dlg.exec_()

    def _tour_steps(self):
        """The pipeline, one step per card: what to do there, pointing at the card."""
        def page(tab, card=None):
            def enter():
                self.show_page(self.pages.indexOf(tab))
                if card is not None:
                    if not card.is_open():
                        card.set_open(True)
                    self.pages_scroll.ensureWidgetVisible(card, 0, 0)
            return enter

        def card(tab, widget, title, text):
            return TourStep(title, text, page(tab, widget), lambda: widget)
        return [
            TourStep("Welcome to Cabana",
                     "This tour shows where each step of the analysis lives. It changes no settings; "
                     "press Esc to leave at any time.",
                     page(self.start_tab), lambda: self.start_actions),
            TourStep("Pages",
                     "Each icon opens a page of the side panel. Click the open page's icon to fold "
                     "the panel and give the image more room.",
                     page(self.start_tab), lambda: self.page_rail),
            card(self.tma_tab, self.tma_fit_card, "Cut a TMA slide into cores",
                 "Optional. Open a whole-slide TMA scan and fit its cores to the array map."),
            card(self.tma_tab, self.tma_filter_card, "Leave out poor cores",
                 "Quality filters exclude cores; excluded cores are grey with a cross on the overlay "
                 "and are not exported."),
            card(self.tma_tab, self.tma_edit_card, "Correct the fit by hand",
                 "Turn on Edit cores to move, resize, add or remove cores on the image."),
            card(self.tma_tab, self.tma_export_card, "Export the cores",
                 "Writes one image and mask per core, named by patient, ready for Batch Run."),
            card(self.brf_tab, self.brf_bands_card, "Separate birefringence bands",
                 "For polarised-light images: set the hues of the red-orange, yellow and green bands "
                 "and check them on the image."),
            card(self.brf_tab, self.brf_folder_card, "Separate a whole folder",
                 "Run Folder writes the band images of every image in a folder, inside ROI masks "
                 "if you choose them."),
            card(self.seg_tab, self.seg_colour_card, "Keep the stained tissue",
                 "Open a typical image, pick the stain colour and click Segment."),
            card(self.seg_tab, self.seg_training_card, "Fine or coarse",
                 "More labels or fewer iterations give a finer segmentation."),
            card(self.seg_tab, self.seg_large_card, "Large images",
                 "Big images are segmented in patches; images over Max size are split into blocks "
                 "in a batch run."),
            card(self.det_tab, self.det_detection_card, "Detect the fibres",
                 "Set the fibre widths and contrast, then Detect to check them on the image."),
            card(self.det_tab, self.det_measurements_card, "Measure the fibres",
                 "How Batch Run measures fibre length, branching and curvature."),
            card(self.gap_tab, self.gap_gaps_card, "Measure the gaps",
                 "Analyse fills the spaces between the fibres with circles."),
            card(self.gap_tab, self.gap_hdm_card, "Hematoxylin density",
                 "Settings of the hematoxylin density map (HDM) that Batch Run measures."),
            card(self.bat_tab, self.bat_files_card, "Choose the files",
                 "Pick the images, optional ROI masks, a parameter file and a results folder. "
                 "Parameters > Export Parameters saves the settings of the earlier pages as that file."),
            card(self.bat_tab, self.bat_run_card, "Run a whole folder",
                 "Choose what to save besides the measurements, then Process Batch. The progress "
                 "shows here."),
        ]

    def start_tour(self):
        """Walk through the analysis pages with a spotlight on each step."""
        if self._tour is not None:
            self._tour.end()
        self._tour = TourOverlay(self, self._tour_steps())
        self._tour.finished.connect(lambda: setattr(self, "_tour", None))
        self._tour.start()

    # ------------------------------------------------------------------
    # Check for Updates
    # ------------------------------------------------------------------
    check_updates_on_start = True   # tests and scripts switch the startup check off
    UPDATE_CHECK_KEY = 'update_last_check'
    UPDATE_SKIP_KEY = 'update_skipped_version'

    def _startup_update_check(self):
        """Quiet check at most once a day; speaks only when an update can be installed."""
        settings = QSettings('Cabana', 'CabanaGUI')
        today = date.today().isoformat()
        if settings.value(self.UPDATE_CHECK_KEY, "") == today or install_problem() is not None:
            return
        settings.setValue(self.UPDATE_CHECK_KEY, today)
        self.check_for_updates(quiet=True)

    def check_for_updates(self, quiet=False):
        """Ask PyPI for the latest version in the background. ``quiet`` (the
        startup check) shows nothing unless a new, not skipped, version exists."""
        if self.update_worker is not None and self.update_worker.isRunning():
            return
        self.update_worker = UpdateWorker('check')
        self.update_worker.checked.connect(lambda latest: self._update_checked(latest, quiet))
        self.update_worker.failed.connect(lambda error: self._update_check_failed(error, quiet))
        self.update_worker.start()

    def _update_check_failed(self, error, quiet):
        Log.logger.debug("Update check failed: %s", error)
        if not quiet:
            self._ask_update("Check for Updates", "Could not reach PyPI to check for updates.",
                             f"Check the internet connection and try again.\n\n{error}", ["OK"])

    def _update_checked(self, latest, quiet):
        if not is_newer(latest, __version__):
            if not quiet:
                self._ask_update("Check for Updates", f"Cabana {__version__} is the latest version.",
                                 "", ["OK"])
            return
        settings = QSettings('Cabana', 'CabanaGUI')
        if quiet and settings.value(self.UPDATE_SKIP_KEY, "") == latest:
            return
        text = (f"Cabana {latest} is available (you have {__version__}).<br>"
                f"<a href='{CHANGELOG_URL}'>See what's new</a>")
        problem = install_problem()
        if problem:
            self._ask_update("Check for Updates", text, problem, ["OK"])
            return
        buttons = ["Update", "Later"] + (["Skip This Version"] if quiet else [])
        choice = self._ask_update(
            "Check for Updates", text,
            "Update installs it into this Python environment with pip; "
            "restart Cabana afterwards to use it.", buttons)
        if choice == "Update":
            self.install_update(latest)
        elif choice == "Skip This Version":
            settings.setValue(self.UPDATE_SKIP_KEY, latest)

    def install_update(self, version):
        """Install ``version`` with pip in the background, unless a job is running."""
        running = [r for r in self._work_at_risk() if r.endswith("still running")]
        if running:
            self._ask_update("Check for Updates", "Cabana cannot be updated while work is running.",
                             "Finish or cancel it first:\n- " + "\n- ".join(running), ["OK"])
            return
        self.check_updates_action.setEnabled(False)
        self.status_bar.showMessage(f"Installing Cabana {version}…")
        self.update_worker = UpdateWorker('install', version)
        self.update_worker.installed.connect(lambda ok, output: self._update_installed(version, ok, output))
        self.update_worker.failed.connect(lambda error: self._update_installed(version, False, error))
        self.update_worker.start()

    def _update_installed(self, version, ok, output):
        # The result arrives just before the thread ends; wait for it, or quitting
        # right after this dialog would still list the update as running
        if self.update_worker is not None:
            self.update_worker.wait()
        self.check_updates_action.setEnabled(True)
        self.status_bar.clearMessage()
        if ok:
            Log.logger.info("Installed Cabana %s", version)
            Log.logger.debug("pip output:\n%s", output)
            warned = "WARNING" in output.upper()
            info = "Close Cabana and start it again to use the new version."
            if warned:
                info += "\npip reported warnings; see the details below."
            self._ask_update("Check for Updates", f"Cabana {version} is installed.", info,
                             ["OK"], details=output if warned else None)
        else:
            Log.logger.warning("Update to Cabana %s failed:\n%s", version, output)
            self._ask_update("Check for Updates", f"Cabana {version} could not be installed.",
                             "The update did not complete. Cabana keeps working; see the details below.",
                             ["OK"], details=output)

    def _ask_update(self, title, text, info, buttons, details=None):
        """Message with ``buttons`` (the first is the default); returns the
        label clicked, or None when closed. Monkeypatched in tests."""
        return self._message(title, text, info, buttons, details=details, rich=True)

    @staticmethod
    def _license_text() -> str:
        """Return the MIT license text: the repository LICENSE when running from
        source, else the copy bundled in the installed package metadata."""
        repo_license = Path(__file__).resolve().parent.parent / "LICENSE"
        if repo_license.is_file():
            return repo_license.read_text(encoding="utf-8")
        try:
            from importlib.metadata import distribution
            dist = distribution("cabana")
            for name in ("LICENSE", "licenses/LICENSE"):
                text = dist.read_text(name)
                if text:
                    return text
        except Exception:
            pass
        return ("Cabana is released under the MIT License.\n\n"
                "https://github.com/Garvan-Data-Science-Platform/Cabana/blob/main/LICENSE")

    def _on_theme_changed(self, theme_name: str) -> None:
        """Handle theme selection from combo box."""
        if theme_name == self._current_theme:
            return
        apply_theme(theme_name)
        self._current_theme = theme_name
        QSettings('Cabana', 'CabanaGUI').setValue('theme', theme_name)
        self._setup_styles()
        self.set_theme()
        self._reapply_widget_styles()

    def _reapply_widget_styles(self) -> None:
        """Re-apply cached styles to all individually-styled widgets after theme change."""
        # Menu bar and page title
        self.menuBar().setStyleSheet(self.menubar_style)
        self.page_title.setStyleSheet(self.page_title_style)

        # Buttons
        for btn in (self.start_open_btn, self.start_params_btn, self.cancel_batch_btn,
                     self.tma_cancel_btn, self.tma_undo_btn):
            btn.setStyleSheet(self.btn_style)

        # Primary buttons
        for btn in (self.segment_btn, self.detect_btn, self.analyze_btn, self.process_batch_btn,
                    self.tma_fit_btn, self.tma_export_btn, self.start_slide_btn, self.brf_btn,
                    self.brf_run_btn):
            btn.setStyleSheet(self.primary_btn_style)

        # Page stack and its scroll area
        self.pages.setStyleSheet(self.page_stack_style)
        self.pages_scroll.setStyleSheet(self.scroll_area_style)

        # Progress bars
        self.progress_bar.setStyleSheet(self.progressbar_style)
        self.batch_progress_bar.setStyleSheet(self.progressbar_style)

        # Spinboxes
        for spin in (self.batch_size_spinner, self.patch_size_spinner, self.seg_max_size_spinner, self.tma_pixel_size_spin,
                     self.tma_core_diameter_spin, self.tma_margin_spin, self.tma_erode_spin,
                     self.tma_sat_spin, self.tma_offset_spin, self.tma_dmin_spin, self.tma_dmax_spin,
                     self.tma_stain_spin):
            spin.setStyleSheet(self.spinner_style)

        # Combo boxes
        for combo in (self.tma_array_combo, self.tma_orientation_combo, self.brf_view_combo):
            combo.setStyleSheet(self.combo_style)

        # Checkboxes
        for cb in (self.white_bg_cb, self.toggle_img_cb, self.dark_line_cb,
                   self.extend_line_cb, self.overlay_fibres_cb, self.overlay_gaps_cb,
                   self.stats_cb, self.scores_cb, self.visuals_cb, self.tma_recover_cb, self.tma_core_auto_cb,
                   self.tma_edit_cb, self.tma_links_cb, *self.tma_channel_cbs.values(),
                   self.brf_sort_cb, self.brf_original_cb, self.brf_images_cb):
            cb.setStyleSheet(self.checkbox_style)

        # Path edits
        self.tma_status_label.setStyleSheet(self.value_label_style)
        self.progress_label.setStyleSheet(self.value_label_style)
        self.gap_summary_label.setStyleSheet(self.value_label_style)
        self.seg_size_note.setStyleSheet(self.value_label_style)
        self.brf_note.setStyleSheet(self.value_label_style)
        self.brf_summary_label.setStyleSheet(self.value_label_style)
        self.brf_folder_label.setStyleSheet(self.value_label_style)
        self.brf_progress_bar.setStyleSheet(self.progressbar_style)

        # Value labels (slider readouts)
        for label in (self.color_thresh_value, self.num_labels_value, self.max_iters_value,
                      self.line_width_value, self.line_step_value, self.contrast_value,
                      self.min_length_value, self.min_gap_value, self.max_hdm_value,
                      self.min_branch_value, self.curve_window_value, self.curve_step_value,
                      self.hdm_saturation_value, self.brf_brightness_value, self.brf_saturation_value,
                      *self.brf_band_values):
            label.setStyleSheet(self.value_label_style)

        # Status bar
        self.status_bar.setStyleSheet(self.status_bar_style)

        # Theme combo
        self.theme_combo.setStyleSheet(self.theme_combo_style)

        # Dock panel palette and card stylesheet
        dock_palette = self.dock_contents.palette()
        dock_palette.setColor(QPalette.Window, panel_colors()['panel'])
        self.dock_contents.setPalette(dock_palette)
        self.dock_contents.setStyleSheet(self.panel_style)
        self.page_rail.restyle()
        for widget in self.dock_contents.findChildren((SectionCard, PathEdit)):
            widget.restyle()
        for btn in self._play_buttons:
            self._set_play_icon(btn)

        # Image panel palette
        img_palette = self.image_panel.palette()
        img_palette.setColor(self.image_panel.backgroundRole(), COLORS['canvas'])
        self.image_panel.setPalette(img_palette)

        # Update toggle labels with current theme colors
        self._set_switch_label(self.toggle_seg_label, self.toggle_seg_btn.isChecked())
        self._set_switch_label(self.toggle_gap_label, self.toggle_gap_btn.isChecked())

        # Force repaint on all widgets
        for widget in self.findChildren(QWidget):
            widget.update()
