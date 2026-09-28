import os
os.environ["NUMEXPR_MAX_THREADS"] = "20"
import sys
import colorsys
import numpy as np
import yaml
import imageio.v3 as iio
import tifffile as tiff
from pathlib import Path
from .utils import join_path, sanitize_filename

from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QSpinBox,
                             QVBoxLayout, QHBoxLayout, QCheckBox,
                             QPushButton, QFileDialog, QSizePolicy, QColorDialog,
                             QMessageBox, QGroupBox, QComboBox, QWidget,
                             QStatusBar, QLineEdit, QDoubleSpinBox, QStackedWidget, QGridLayout,
                             QAction, QActionGroup)
from PyQt5.QtGui import QIcon, QPalette, QFont, QKeySequence
from PyQt5.QtCore import QSettings, QUrl
from PyQt5.QtGui import QDesktopServices

from .ui import *
from .themes import THEMES, DEFAULT_THEME
from .tma import ORIENTATIONS
from .tma_maps import available_arrays
from . import __version__


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set window properties
        self.setWindowTitle(f"Cabana v{__version__}")
        self.setMinimumSize(800, 600)

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
        self.dock_contents.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.dock_contents.setAutoFillBackground(True)

        # Set Napari dock color
        dock_palette = self.dock_contents.palette()
        dock_palette.setColor(QPalette.Window, COLORS['dock'])
        self.dock_contents.setPalette(dock_palette)

        self.dock_layout = QVBoxLayout(self.dock_contents)
        self.dock_layout.setContentsMargins(6, 8, 6, 10)
        self.dock_layout.setSpacing(4)
        self.panel_visible = True

        self._setup_styles()

        # Menu bar carries the Image / Parameters / Analysis / About commands,
        # so the side panel holds only the selected analysis page.
        self._setup_menu_bar()

        self.page_title = QLabel("")
        self.page_title.setStyleSheet(self.page_title_style)
        self.dock_layout.addWidget(self.page_title)

        self.dock_inner_layout = QVBoxLayout()
        self.dock_inner_layout.setContentsMargins(0, 4, 0, 0)
        self.dock_inner_layout.setSpacing(6)
        self.dock_layout.addLayout(self.dock_inner_layout, 1)

        # Analysis pages live in a stacked widget selected from the Analysis menu.
        self.pages = QStackedWidget()
        self.pages.setStyleSheet(self.page_stack_style)

        # Create pages
        self.start_tab = QWidget()
        self.tma_tab = QWidget()
        self.seg_tab = QWidget()
        self.det_tab = QWidget()
        self.gap_tab = QWidget()
        self.bat_tab = QWidget()

        # Set up each page (TMA last: it refers to batch widgets)
        self.setup_segmentation_tab()
        self.setup_detection_tab()
        self.setup_gap_analysis_tab()
        self.setup_batch_processing_tab()
        self.setup_tma_tab()
        self.setup_start_tab()

        # Page 0 is the Start page (not in the Analysis menu); the rest follow menu order
        self._page_titles = ["Start"]
        self.pages.addWidget(self.start_tab)
        for (title, action), page in zip(self._page_actions,
                                         (self.tma_tab, self.seg_tab, self.det_tab,
                                          self.gap_tab, self.bat_tab)):
            index = self.pages.addWidget(page)
            self._page_titles.append(title)
            action.triggered.connect(lambda _checked=False, i=index: self.show_page(i))
        self.dock_inner_layout.addWidget(self.pages)
        self._share_hints_with_labels()
        self.show_page(0)

        # Add a spacer to push content to the top
        self.dock_inner_layout.addStretch()

        # Progress status (which batch / stage / image), shown during Batch Run
        self.progress_label = QLabel("")
        self.progress_label.setWordWrap(True)
        self.progress_label.setStyleSheet(self.value_label_style)
        self.progress_label.setVisible(False)
        self.dock_inner_layout.addWidget(self.progress_label)

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

        content_layout = QVBoxLayout(self.image_panel)

        # Create toggle button
        self.toggle_button = PanelToggleButton()
        self.toggle_button.clicked.connect(self.toggle_panel)

        # Add toggle button and some content to the main area
        content_layout.addWidget(self.toggle_button, 0, Qt.AlignLeft)
        content_layout.addStretch(1)

        # Add widgets to splitter
        self.splitter.addWidget(self.dock_contents)
        self.splitter.addWidget(self.image_panel)

        # Let the image panel stretch, dock panel keeps its natural size
        self.splitter.setStretchFactor(0, 0)  # dock: don't stretch
        self.splitter.setStretchFactor(1, 1)  # image: stretch to fill

        # Compute the minimum dock width so tab labels are never clipped.
        # Match main-branch behavior: let the default tab bar manage elision inside the available width.
        self.dock_contents.setMinimumWidth(200)

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

        # Add splitter to the main layout
        self.main_layout.addWidget(self.splitter)

        # Set handle width thinner
        self.splitter.setHandleWidth(5)

        # Make handle transparent by default
        self.splitter.setStyleSheet("QSplitter::handle { background-color: transparent; }")

        self.img_path = None
        self.ori_img = None
        self.seg_img = None
        self.frb_img = None
        self.wdt_img = None
        self.gap_img = None
        self.gap_ovl = None
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
        self.open_slide_action.setStatusTip("Choose a whole-slide TMA scan on the TMA page")
        self.open_slide_action.triggered.connect(self._open_tma_slide_from_menu)
        file_menu.addAction(self.open_slide_action)

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
        for i, title in enumerate(("TMA", "Segmentation", "Fibre Detection",
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
        issue_action = QAction("Report an &Issue", self)
        issue_action.triggered.connect(
            lambda: QDesktopServices.openUrl(QUrl("https://github.com/lxfhfut/Cabana/issues")))
        about_menu.addAction(issue_action)
        about_menu.addSeparator()
        version_action = QAction("&Version", self)
        version_action.setMenuRole(QAction.NoRole)
        version_action.triggered.connect(self._show_about_dialog)
        about_menu.addAction(version_action)
        license_action = QAction("&License", self)
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
                       "on it, or import a saved parameter file. Analysis pages are also under the "
                       "Analysis menu (Ctrl/Cmd+1 to 5).")
        intro.setWordWrap(True)
        intro.setStyleSheet(self.value_label_style)
        layout.addWidget(intro)

        self.start_slide_btn = QPushButton("Open TMA Slide…")
        self.start_slide_btn.setStyleSheet(self.primary_btn_style)
        self.start_slide_btn.setToolTip("Choose a whole-slide TMA scan and switch to the TMA page.")
        self.start_slide_btn.clicked.connect(self._open_tma_slide_from_menu)
        layout.addWidget(self.start_slide_btn)

        self.start_open_btn = QPushButton("Open Image…")
        self.start_open_btn.setStyleSheet(self.btn_style)
        self.start_open_btn.setToolTip("Load a single image; the Segmentation page opens once it is loaded.")
        self.start_open_btn.clicked.connect(self.load_image)
        layout.addWidget(self.start_open_btn)

        self.start_params_btn = QPushButton("Import Parameters…")
        self.start_params_btn.setStyleSheet(self.btn_style)
        self.start_params_btn.setToolTip("Load a Parameters.yml exported earlier; all pages update to its values.")
        self.start_params_btn.clicked.connect(self.import_parameters)
        layout.addWidget(self.start_params_btn)

        layout.addStretch()
        self.start_tab.setLayout(layout)

    def _open_tma_slide_from_menu(self):
        self.show_page(self.pages.indexOf(self.tma_tab))
        self.select_tma_slide()

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

    def toggle_panel(self):
        if self.panel_visible:
            self.dock_contents.hide()
        else:
            self.dock_contents.show()
        self.panel_visible = not self.panel_visible
        self.toggle_button.setChecked(not self.panel_visible)
        if hasattr(self, 'toggle_panel_action'):
            self.toggle_panel_action.setChecked(self.panel_visible)


    def _setup_styles(self) -> None:
        """Set up all style sheets"""
        # Button style
        self.btn_style = generate_button_style()

        # Page stack / combo box / menu bar / page title styles
        self.page_stack_style = generate_page_stack_style()
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

        # Read-only path line edit style
        self.path_edit_style = (
            f"QLineEdit {{ background-color: {color_to_stylesheet(COLORS['background'])}; "
            f"color: {color_to_stylesheet(COLORS['text_dim'])}; "
            f"border: 1px solid {color_to_stylesheet(COLORS['border'])}; "
            f"border-radius: 3px; padding: 8px 6px; "
            f"font-size: {FONT_SIZES['small']}px; }}"
        )

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

    def setup_batch_processing_tab(self):
        """Set up the batch processing tab UI"""
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 10, 6, 6)
        layout.setSpacing(12)

        # Parameter file selection
        param_layout = QHBoxLayout()
        param_label = QLabel("Parameter File:")
        param_label.setFixedWidth(95)
        param_layout.addWidget(param_label)

        self.param_file_path = QLineEdit("Not selected")
        self.param_file_path.setReadOnly(True)
        self.param_file_path.setStyleSheet(self.path_edit_style)
        self.param_file_path.setToolTip("Parameters.yml to apply to every image; export one from Parameters > Export\n"
                                        "after tuning on a representative image.")
        param_layout.addWidget(self.param_file_path, 1)

        self.param_btn = QPushButton("Select")
        self.param_btn.clicked.connect(self.select_param_file)
        self.param_btn.setStyleSheet(self.btn_style)
        param_layout.addWidget(self.param_btn)

        layout.addLayout(param_layout)

        # Input folder selection
        input_layout = QHBoxLayout()
        input_label = QLabel("Input Folder:")
        input_label.setFixedWidth(95)
        input_layout.addWidget(input_label)

        self.input_folder_path = QLineEdit("Not selected")
        self.input_folder_path.setReadOnly(True)
        self.input_folder_path.setStyleSheet(self.path_edit_style)
        self.input_folder_path.setToolTip("Folder of images to analyse (TIFF/PNG/JPEG). Pixel size is read from the\n"
                                          "image metadata; a TMA export's <channel>/Patients/Images folder works directly.")
        input_layout.addWidget(self.input_folder_path, 1)

        self.input_btn = QPushButton("Select")
        self.input_btn.clicked.connect(self.select_input_folder)
        self.input_btn.setStyleSheet(self.btn_style)
        input_layout.addWidget(self.input_btn)

        layout.addLayout(input_layout)

        # Output folder selection
        output_layout = QHBoxLayout()
        output_label = QLabel("Output Folder:")
        output_label.setFixedWidth(95)
        output_layout.addWidget(output_label)

        self.output_folder_path = QLineEdit("Not selected")
        self.output_folder_path.setReadOnly(True)
        self.output_folder_path.setStyleSheet(self.path_edit_style)
        self.output_folder_path.setToolTip("Where results are written (QuantificationResults.csv, per-image exports,\n"
                                           "colour maps). A checkpoint here lets an interrupted run resume.")
        output_layout.addWidget(self.output_folder_path, 1)

        self.output_btn = QPushButton("Select")
        self.output_btn.clicked.connect(self.select_output_folder)
        self.output_btn.setStyleSheet(self.btn_style)
        output_layout.addWidget(self.output_btn)

        layout.addLayout(output_layout)

        # Optional ROI mask folder (e.g. TMA core circles)
        mask_layout = QHBoxLayout()
        mask_label = QLabel("ROI Masks:")
        mask_label.setFixedWidth(95)
        mask_label.setToolTip("Optional folder of binary masks, one <image name>.png per input image.")
        mask_layout.addWidget(mask_label)

        self.mask_folder_path = QLineEdit("")
        self.mask_folder_path.setReadOnly(True)
        self.mask_folder_path.setPlaceholderText("Optional: leave empty to use segmentation only")
        self.mask_folder_path.setStyleSheet(self.path_edit_style)
        self.mask_folder_path.setToolTip(
            "Optional. Binary masks (white = analyse, black = ignore) named like the input images,\n"
            "e.g. the Masks folder written by the TMA page. Leave empty to rely on segmentation alone.")
        mask_layout.addWidget(self.mask_folder_path, 1)

        self.mask_btn = QPushButton("Select")
        self.mask_btn.clicked.connect(self.select_mask_folder)
        self.mask_btn.setStyleSheet(self.btn_style)
        mask_layout.addWidget(self.mask_btn)

        self.mask_clear_btn = QPushButton("✕")
        self.mask_clear_btn.setToolTip("Clear the ROI mask folder")
        self.mask_clear_btn.clicked.connect(self.clear_mask_folder)
        self.mask_clear_btn.setStyleSheet(self.btn_style)
        self.mask_clear_btn.setFixedWidth(34)
        self.mask_clear_btn.setEnabled(False)
        mask_layout.addWidget(self.mask_clear_btn)

        layout.addLayout(mask_layout)
        self.mask_folder = None

        # Batch size spinbox
        batch_size_layout = QHBoxLayout()
        batch_size_label = QLabel("Batch Size:")
        batch_size_layout.addWidget(batch_size_label)

        self.batch_size_spinner = QSpinBox()
        self.batch_size_spinner.setRange(1, 100)
        self.batch_size_spinner.setValue(5)
        self.batch_size_spinner.setFixedWidth(50)
        self.batch_size_spinner.setStyleSheet(self.spinner_style)
        self.batch_size_spinner.setToolTip("Images processed per batch. Lower it for very large images (e.g. 2 for\n"
                                           "full-resolution TMA cores) to keep memory use flat.")
        batch_size_layout.addWidget(self.batch_size_spinner)

        layout.addLayout(batch_size_layout)

        # Post-processing options
        layout.addWidget(create_separator())
        options_layout = QHBoxLayout()
        options_layout.setSpacing(16)

        self.stats_cb = QCheckBox("Stats")
        self.stats_cb.setChecked(False)
        self.stats_cb.setEnabled(False)
        self.stats_cb.setStyleSheet(self.checkbox_style)
        self.stats_cb.setToolTip(
            "Generate per-patient MEAN, STD and SEM statistics\n"
            "(QuantificationResults_MEAN_STD_SEM.csv)")

        self.scores_cb = QCheckBox("Scores")
        self.scores_cb.setChecked(False)
        self.scores_cb.setEnabled(False)
        self.scores_cb.setStyleSheet(self.checkbox_style)
        self.scores_cb.setToolTip(
            "Generate collagen rigidity and bundling risk scores\n"
            "(QuantificationResults_SCORES.csv)")

        options_layout.addWidget(self.stats_cb)
        options_layout.addWidget(self.scores_cb)
        options_layout.addStretch()
        layout.addLayout(options_layout)

        # Batch processing / cancel buttons (share the same row)
        batch_btn_layout = QHBoxLayout()

        self.process_batch_btn = QPushButton("Process Batch")
        self.process_batch_btn.clicked.connect(self.run_batch_processing)
        self.process_batch_btn.setEnabled(False)
        self.process_batch_btn.setStyleSheet(self.primary_btn_style)
        batch_btn_layout.addWidget(self.process_batch_btn)

        self.cancel_batch_btn = QPushButton("Cancel")
        self.cancel_batch_btn.clicked.connect(self._cancel_batch_processing)
        self.cancel_batch_btn.setVisible(False)
        self.cancel_batch_btn.setStyleSheet(self.btn_style)
        batch_btn_layout.addWidget(self.cancel_batch_btn)

        layout.addLayout(batch_btn_layout)

        layout.addStretch()
        self.bat_tab.setLayout(layout)

    # ------------------------------------------------------------------
    # TMA page
    # ------------------------------------------------------------------
    def setup_tma_tab(self):
        """Set up the TMA preprocessing page: load slide -> fit cores -> export."""
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 10, 6, 6)
        layout.setSpacing(10)

        fit_title = QLabel("Fit")
        fit_title.setStyleSheet(self.value_label_style + " font-weight: 600;")
        fit_title.setToolTip("Settings used by Fit Cores; change them and fit again.")
        layout.addWidget(fit_title)

        # --- Slide -------------------------------------------------------
        slide_layout = QHBoxLayout()
        slide_label = QLabel("Slide:")
        slide_label.setFixedWidth(95)
        slide_layout.addWidget(slide_label)
        self.tma_slide_path = QLineEdit("")
        self.tma_slide_path.setReadOnly(True)
        self.tma_slide_path.setPlaceholderText("Olympus .vsi or whole-slide TIFF/PNG")
        self.tma_slide_path.setStyleSheet(self.path_edit_style)
        self.tma_slide_path.setToolTip(
            "Whole-slide scan of the tissue micro-array. Olympus .vsi files are read natively\n"
            "(keep the _<name>_ tile folder next to the .vsi); TIFF/PNG exports also work.")
        slide_layout.addWidget(self.tma_slide_path, 1)
        self.tma_slide_btn = QPushButton("Open")
        self.tma_slide_btn.setStyleSheet(self.btn_style)
        self.tma_slide_btn.setToolTip("Choose the slide file (File > Open TMA Slide).")
        self.tma_slide_btn.clicked.connect(self.select_tma_slide)
        slide_layout.addWidget(self.tma_slide_btn)
        layout.addLayout(slide_layout)

        # --- Array / orientation ------------------------------------------
        map_layout = QHBoxLayout()
        array_label = QLabel("Array Map:")
        array_label.setFixedWidth(95)
        map_layout.addWidget(array_label)
        self.tma_array_combo = QComboBox()
        self.tma_array_combo.addItem("None (grid position only)", None)
        for n in available_arrays():
            self.tma_array_combo.addItem(f"ICGC Array {n}", n)
        self.tma_array_combo.setStyleSheet(self.combo_style)
        self.tma_array_combo.setToolTip("Printed ICGC/APGI array map used to name cores by patient ID.")
        map_layout.addWidget(self.tma_array_combo, 1)
        layout.addLayout(map_layout)

        orient_layout = QHBoxLayout()
        orient_label = QLabel("Orientation:")
        orient_label.setFixedWidth(95)
        orient_layout.addWidget(orient_label)
        self.tma_orientation_combo = QComboBox()
        for o in ORIENTATIONS:
            self.tma_orientation_combo.addItem("Auto (from missing cores)" if o == "auto" else o, o)
        self.tma_orientation_combo.setStyleSheet(self.combo_style)
        self.tma_orientation_combo.setToolTip(
            "How the printed map sits on the scan. Auto compares the pattern of missing cores;\n"
            "a fully populated array cannot distinguish a rotation from its mirror image,\n"
            "so check the labels on the overlay and pick the orientation explicitly if needed.")
        orient_layout.addWidget(self.tma_orientation_combo, 1)
        layout.addLayout(orient_layout)

        # --- Geometry ------------------------------------------------------
        geo = QGridLayout()
        geo.setHorizontalSpacing(8)
        geo.setVerticalSpacing(8)
        geo.setColumnStretch(1, 1)
        geo.setColumnStretch(3, 1)

        geo.addWidget(QLabel("Pixel Size:"), 0, 0)
        self.tma_pixel_size_spin = QDoubleSpinBox()
        self.tma_pixel_size_spin.setRange(0.01, 50.0)
        self.tma_pixel_size_spin.setDecimals(4)
        self.tma_pixel_size_spin.setSingleStep(0.01)
        self.tma_pixel_size_spin.setValue(0.2738)
        self.tma_pixel_size_spin.setSuffix(" µm")
        self.tma_pixel_size_spin.setStyleSheet(self.spinner_style)
        self.tma_pixel_size_spin.setToolTip("Filled from the slide metadata when available.")
        geo.addWidget(self.tma_pixel_size_spin, 0, 1)

        geo.addWidget(QLabel("Core Ø:"), 0, 2)
        self.tma_core_diameter_spin = QSpinBox()
        self.tma_core_diameter_spin.setRange(100, 5000)
        self.tma_core_diameter_spin.setSingleStep(50)
        self.tma_core_diameter_spin.setValue(1250)
        self.tma_core_diameter_spin.setSuffix(" µm")
        self.tma_core_diameter_spin.setStyleSheet(self.spinner_style)
        self.tma_core_diameter_spin.setToolTip("Nominal core diameter.")
        geo.addWidget(self.tma_core_diameter_spin, 0, 3)

        geo.addWidget(QLabel("Sensitivity:"), 1, 0)
        self.tma_sat_spin = QSpinBox()
        self.tma_sat_spin.setRange(2, 60)
        self.tma_sat_spin.setValue(15)
        self.tma_sat_spin.setStyleSheet(self.spinner_style)
        self.tma_sat_spin.setToolTip(
            "HSV saturation above which a pixel counts as tissue (default 15).\n"
            "Lower it to catch paler cores; raise it if debris or shading is picked up.")
        geo.addWidget(self.tma_sat_spin, 1, 1)

        self.tma_recover_cb = QCheckBox("Recover faint")
        self.tma_recover_cb.setChecked(True)
        self.tma_recover_cb.setStyleSheet(self.checkbox_style)
        self.tma_recover_cb.setToolTip(
            "After the grid is known, test every empty grid position for pale tissue with a\n"
            "more permissive threshold and add such cores (purple circles, flag 'recovered').")
        geo.addWidget(self.tma_recover_cb, 1, 2, 1, 2)
        layout.addLayout(geo)

        # --- Filter ------------------------------------------------------
        # QC filters re-evaluate instantly on the fitted cores (no refit).
        layout.addWidget(create_separator())
        filter_title = QLabel("Filter")
        filter_title.setStyleSheet(self.value_label_style + " font-weight: 600;")
        filter_title.setToolTip(
            "Quality filters applied to the fitted cores; changes update the overlay instantly.\n"
            "Excluded cores are drawn grey with a cross, listed in cores.csv and not exported.\n"
            "Filters run after patient IDs are assigned and never change them.")
        layout.addWidget(filter_title)
        flt = QGridLayout()
        flt.setHorizontalSpacing(8)
        flt.setVerticalSpacing(8)
        flt.setColumnStretch(1, 1)
        flt.setColumnStretch(3, 1)

        def _spin(lo, hi, val, step, suffix, tip, decimals=None):
            w = QDoubleSpinBox() if decimals is not None else QSpinBox()
            if decimals is not None:
                w.setDecimals(decimals)
            w.setRange(lo, hi)
            w.setSingleStep(step)
            w.setValue(val)
            w.setSuffix(suffix)
            w.setStyleSheet(self.spinner_style)
            w.setToolTip(tip)
            w.valueChanged.connect(self._tma_filters_changed)
            return w

        self.tma_offset_spin = _spin(0.05, 1.0, 0.35, 0.05, " pitch",
                                     "Exclude cores whose centre is further than this from its grid position\n"
                                     "(fraction of the core spacing). Catches debris between cores.", decimals=2)
        self.tma_dmin_spin = _spin(10, 100, 80, 5, " %",
                                   "Exclude cores smaller than this percentage of Core Ø (lower it to keep partial cores).\n"
                                   "Set Core Ø to the real diameter first (see the median in the status line).")
        self.tma_dmax_spin = _spin(100, 300, 120, 5, " %",
                                   "Exclude cores larger than this percentage of Core Ø.\n"
                                   "Set Core Ø to the real diameter first (see the median in the status line).")
        self.tma_stain_spin = _spin(0, 100, 2, 1, " %",
                                    "Exclude patient cores whose stained area (saturation above 40) is below this\n"
                                    "fraction of the circle; catches empty and unstained cores.\n"
                                    "Control cores (e.g. Brain) are exempt.")

        flt.addWidget(QLabel("Grid Offset:"), 0, 0)
        flt.addWidget(self.tma_offset_spin, 0, 1)
        flt.addWidget(QLabel("Min Stain:"), 0, 2)
        flt.addWidget(self.tma_stain_spin, 0, 3)
        flt.addWidget(QLabel("Min Ø:"), 1, 0)
        flt.addWidget(self.tma_dmin_spin, 1, 1)
        flt.addWidget(QLabel("Max Ø:"), 1, 2)
        flt.addWidget(self.tma_dmax_spin, 1, 3)
        layout.addLayout(flt)

        # --- Export ------------------------------------------------------
        # Settings below only affect Export Cores; changing them needs no refit.
        layout.addWidget(create_separator())
        export_title = QLabel("Export")
        export_title.setStyleSheet(self.value_label_style + " font-weight: 600;")
        export_title.setToolTip("Settings used when writing the core images and masks; no refit needed.")
        layout.addWidget(export_title)
        exp = QGridLayout()
        exp.setHorizontalSpacing(8)
        exp.setVerticalSpacing(8)
        exp.setColumnStretch(1, 1)
        exp.setColumnStretch(3, 1)
        exp.addWidget(QLabel("Margin:"), 0, 0)
        self.tma_margin_spin = QSpinBox()
        self.tma_margin_spin.setRange(0, 1000)
        self.tma_margin_spin.setSingleStep(10)
        self.tma_margin_spin.setValue(30)
        self.tma_margin_spin.setSuffix(" µm")
        self.tma_margin_spin.setStyleSheet(self.spinner_style)
        self.tma_margin_spin.setToolTip("Extra border around the fitted circle in each crop.")
        exp.addWidget(self.tma_margin_spin, 0, 1)

        exp.addWidget(QLabel("Mask Shrink:"), 0, 2)
        self.tma_erode_spin = QSpinBox()
        self.tma_erode_spin.setRange(0, 200)
        self.tma_erode_spin.setValue(8)
        self.tma_erode_spin.setSuffix(" px")
        self.tma_erode_spin.setStyleSheet(self.spinner_style)
        self.tma_erode_spin.setToolTip("Shrink of the circular mask so the core edge stays out of the analysis.")
        exp.addWidget(self.tma_erode_spin, 0, 3)

        layout.addLayout(exp)

        ch_layout = QHBoxLayout()
        ch_layout.setSpacing(16)
        ch_label = QLabel("Channels:")
        ch_label.setFixedWidth(95)
        ch_layout.addWidget(ch_label)
        self.tma_channel_cbs = {}
        ch_hints = {"BF": "Bright-field layer (Picrosirius Red in transmitted light). Analyse with Dark Line on.",
                    "POL": "Polarised-light layer (collagen birefringence on black). Analyse with Dark Line off."}
        ch_label.setToolTip("Slide layers to export; each core gets one image per ticked channel.\n"
                            "Enabled after Fit Cores once the slide's layers are known.")
        for name in ("BF", "POL"):
            cb = QCheckBox(name)
            cb.setChecked(True)
            cb.setEnabled(False)
            cb.setStyleSheet(self.checkbox_style)
            cb.setToolTip(ch_hints[name])
            self.tma_channel_cbs[name] = cb
            ch_layout.addWidget(cb)
        ch_layout.addStretch()
        layout.addLayout(ch_layout)

        # --- Output --------------------------------------------------------
        out_layout = QHBoxLayout()
        out_label = QLabel("Output Folder:")
        out_label.setFixedWidth(95)
        out_layout.addWidget(out_label)
        self.tma_output_path = QLineEdit("")
        self.tma_output_path.setReadOnly(True)
        self.tma_output_path.setPlaceholderText("BF/, POL/ (Patients, Controls), cores.csv, overlay.png")
        self.tma_output_path.setStyleSheet(self.path_edit_style)
        self.tma_output_path.setToolTip(
            "Destination: one folder per channel (BF/, POL/), split into Patients/ and Controls/\n"
            "(Unmapped/ when no array map is used), each with Images/ (core crops) and Masks/\n"
            "(circle masks with the same names); plus cores.csv (manifest with patient IDs) and\n"
            "overlay.png at the top. Defaults to <slide>_cores.")
        out_layout.addWidget(self.tma_output_path, 1)
        self.tma_output_btn = QPushButton("Select")
        self.tma_output_btn.setStyleSheet(self.btn_style)
        self.tma_output_btn.setToolTip("Choose a different output folder.")
        self.tma_output_btn.clicked.connect(self.select_tma_output)
        out_layout.addWidget(self.tma_output_btn)
        layout.addLayout(out_layout)

        # --- Status + actions ----------------------------------------------
        self.tma_status_label = QLabel("No slide loaded")
        self.tma_status_label.setWordWrap(True)
        self.tma_status_label.setMinimumHeight(40)
        self.tma_status_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.tma_status_label.setStyleSheet(self.value_label_style)
        self.tma_status_label.setToolTip(
            "Fit summary: cores found and to export, grid size, matched orientation and any\n"
            "orientations that fit equally well, exclusions per filter and median diameter.\n"
            "Overlay: green = exported, purple = recovered at an empty grid position (exported),\n"
            "grey with a cross = excluded by a filter, grey = outside the printed map.")
        layout.addWidget(self.tma_status_label)

        btn_layout = QHBoxLayout()
        self.tma_fit_btn = QPushButton("Fit Cores")
        self.tma_fit_btn.setStyleSheet(self.primary_btn_style)
        self.tma_fit_btn.setEnabled(False)
        self.tma_fit_btn.setToolTip("Detect the cores and draw the numbered overlay. Re-run after changing settings.")
        self.tma_fit_btn.clicked.connect(self.run_tma_fit)
        btn_layout.addWidget(self.tma_fit_btn)
        self.tma_export_btn = QPushButton("Export Cores")
        self.tma_export_btn.setStyleSheet(self.primary_btn_style)
        self.tma_export_btn.setEnabled(False)
        self.tma_export_btn.setToolTip("Write one image and one mask per core and channel to the output folder.")
        self.tma_export_btn.clicked.connect(self.run_tma_export)
        btn_layout.addWidget(self.tma_export_btn)
        self.tma_cancel_btn = QPushButton("Cancel")
        self.tma_cancel_btn.setStyleSheet(self.btn_style)
        self.tma_cancel_btn.setToolTip("Stop the export after the current core; files already written are kept.")
        self.tma_cancel_btn.setVisible(False)
        self.tma_cancel_btn.clicked.connect(self.cancel_tma)
        btn_layout.addWidget(self.tma_cancel_btn)
        layout.addLayout(btn_layout)

        layout.addStretch()
        self.tma_tab.setLayout(layout)

        self.tma_slide = None
        self.tma_output = None
        self.tma_pre = None
        self.tma_reader = None
        self.tma_worker = None
        self.tma_overlay = None

    def select_tma_slide(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select TMA Slide", "",
            "Slides (*.vsi *.tif *.tiff *.png *.jpg *.jpeg);;All files (*)")
        if not file_path:
            return
        self.tma_slide = file_path
        self.tma_slide_path.setText(file_path)
        self.tma_slide_path.setToolTip(file_path)
        if self.tma_pre is not None:
            self.tma_pre.close()
        self.tma_pre = None
        self.tma_reader = None
        self.tma_export_btn.setEnabled(False)
        self.tma_fit_btn.setEnabled(False)
        if not self.tma_output:
            default_out = join_path(os.path.dirname(file_path),
                                    sanitize_filename(os.path.splitext(os.path.basename(file_path))[0]) + "_cores")
            self.tma_output = default_out
            self.tma_output_path.setText(default_out)
            self.tma_output_path.setToolTip(default_out)
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
            self.tma_output = folder
            self.tma_output_path.setText(folder)
            self.tma_output_path.setToolTip(folder)

    def _tma_options(self):
        return dict(
            array_number=self.tma_array_combo.currentData(),
            pixel_size_um=float(self.tma_pixel_size_spin.value()) if self.tma_pixel_size_spin.value() > 0 else None,
            core_diameter_um=float(self.tma_core_diameter_spin.value()),
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
                  self.tma_orientation_combo, self.tma_pixel_size_spin, self.tma_core_diameter_spin,
                  self.tma_margin_spin, self.tma_erode_spin, self.tma_sat_spin, self.tma_recover_cb,
                  self.tma_offset_spin, self.tma_dmin_spin, self.tma_dmax_spin, self.tma_stain_spin):
            w.setEnabled(not busy)
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
            self.tma_pre.close()
            self.tma_pre = None
        options = self._tma_options()
        if self.tma_slide.lower().endswith(".vsi"):
            # a VSI carries its own calibration: let the reader's value stand
            options["pixel_size_um"] = None
        # reuse the reader opened for the preview so the slide is read once
        options["reader"] = self.tma_reader
        self.tma_reader = None
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
        self.tma_reader = pre.reader
        self.tma_fit_btn.setText("Fit Cores")
        self._tma_set_busy(False)
        reader = pre.reader
        if reader.pixel_size_um:
            self.tma_pixel_size_spin.setValue(float(reader.pixel_size_um))
        for name, cb in self.tma_channel_cbs.items():
            available = name in reader.channels
            cb.setEnabled(available)
            cb.setChecked(available)
        self._show_tma_result(preserve_view=False)
        self.status_file_label.setText(f"  {os.path.basename(self.tma_slide)}")
        self.status_dims_label.setText(f"{reader.level_shape(0)[1]} x {reader.level_shape(0)[0]}  ")

    def _show_tma_result(self, preserve_view=False):
        """Draw the overlay and write the fit/filter summary to the status line."""
        pre = self.tma_pre
        self.tma_overlay = np.ascontiguousarray(pre.draw_overlay()[:, :, ::-1])
        self.image_panel.setImage(self.tma_overlay, preserve_view=preserve_view)
        n_rows, n_cols = pre.grid_shape
        n_out = len([c for c in pre.cores if c.outside_map])
        n_export = len(pre.exportable_cores())
        msg = f"{len(pre.cores)} cores on a {n_rows}x{n_cols} grid, {n_export} to export"
        if pre.matched_orientation:
            others = [o for o in pre.orientation_ties if o != pre.matched_orientation]
            how = {"brain": "chosen by the Brain control", "replicates": "chosen by replicate similarity",
                   "manual": "set manually", "occupancy": "from missing-core pattern"}.get(pre.orientation_method, "")
            if pre.orientation_margin is not None and pre.orientation_method in ("brain", "replicates"):
                how += f", margin {pre.orientation_margin:.0%}"
            msg += f"; orientation {pre.matched_orientation}" + (f" ({how})" if how else "")
            if not pre.orientation_resolved:
                msg += (f" — UNRESOLVED: {', '.join(pre.orientation_ties)} fit equally well and appearance "
                        f"cannot separate them; confirm a control core against the printed map and set "
                        f"Orientation explicitly. Export is disabled until then.")
            elif others and pre.orientation_method != "manual":
                msg += f" (also tied on occupancy: {', '.join(others)}; verify a control core)"
        self.tma_export_btn.setEnabled(pre.array_number is None or pre.orientation_resolved)
        summary = pre.exclusion_summary()
        if summary:
            names = {"off_grid": "off-grid", "diameter": "diameter", "stain": "low stain"}
            msg += "; excluded " + ", ".join(f"{v} {names.get(k, k)}" for k, v in sorted(summary.items()))
        if n_out:
            msg += f"; {n_out} outside the map"
        diams = [c.diameter_um for c in pre.cores if not c.outside_map]
        if diams:
            med = float(np.median(diams))
            msg += f"; median Ø {med:.0f} µm"
            if abs(med - pre.core_diameter_um) > 0.1 * pre.core_diameter_um:
                msg += (f" — WARNING: Core Ø is {pre.core_diameter_um:.0f} µm, so the diameter filter "
                        f"is misjudging cores; set Core Ø to ~{round(med, -1):.0f} and refit")
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
            QMessageBox.warning(self, "TMA Export", "Select at least one channel to export.")
            return
        # settings that only affect the crop can be changed without refitting
        self.tma_pre.margin_um = float(self.tma_margin_spin.value())
        self.tma_pre.erode_px = int(self.tma_erode_spin.value())
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

    def cancel_tma(self):
        self.tma_cancel_btn.setEnabled(False)
        self.tma_cancel_btn.setText("Cancelling…")
        if self.tma_worker is not None:
            self.tma_worker.cancel()

    def handle_tma_export_complete(self, out_dir):
        self.tma_export_btn.setText("Export Cores")
        self._tma_set_busy(False)
        n = len(self.tma_pre.exportable_cores()) if self.tma_pre else 0  # excludes filtered cores
        self.tma_status_label.setText(f"Exported {n} cores to {out_dir}")
        msg = QMessageBox(self)
        msg.setWindowTitle("TMA Export Complete")
        msg.setText("Core images and masks were written.")
        channels = [c for c in ("BF", "POL") if os.path.isdir(join_path(out_dir, c))]
        msg.setInformativeText(f"{out_dir}\n\nEach channel ({', '.join(channels)}) has Patients/ and Controls/ folders,\n"
                               f"each holding Images/ and Masks/ usable directly as the input and\n"
                               f"ROI-mask folders of Batch Run. Analyse BF with Dark Line on and\n"
                               f"POL with it off.")
        msg.setStyleSheet(self.msgbox_style)
        batch_btns = {msg.addButton(f"Use {c} in Batch Run", QMessageBox.ActionRole): c for c in channels}
        open_btn = msg.addButton("Open Folder", QMessageBox.ActionRole)
        msg.addButton(QMessageBox.Ok)
        self._fit_dialog_buttons(msg)
        msg.exec_()
        clicked = msg.clickedButton()
        if clicked == open_btn:
            QDesktopServices.openUrl(QUrl.fromLocalFile(out_dir))
        elif clicked in batch_btns:
            self.use_tma_export_in_batch(out_dir, batch_btns[clicked])

    @staticmethod
    def _fit_dialog_buttons(box):
        """Widen message-box buttons so their full text fits.

        The dialog stylesheet adds horizontal padding that Qt's size hint for
        QMessageBox buttons does not account for, which clips longer labels.
        """
        for btn in box.buttons():
            need = btn.fontMetrics().horizontalAdvance(btn.text().replace("&", "")) + 40
            btn.setMinimumWidth(max(btn.minimumWidth(), need))

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
        box = QMessageBox(self)
        box.setWindowTitle("TMA Preprocessing")
        box.setIcon(QMessageBox.Warning)
        box.setText("TMA preprocessing failed.")
        box.setInformativeText(message)
        box.setStyleSheet(self.msgbox_style)
        box.exec_()

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
            self._check_batch_processing_ready()

    def select_input_folder(self):
        """Open a file dialog to select an input folder"""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Input Folder", str(Path(self.param_file).parent.parent), QFileDialog.ShowDirsOnly
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
        self.mask_clear_btn.setEnabled(bool(folder))

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
        ignore_large = True

        # Check if checkpoint file exists
        if not os.path.exists(checkpoint_path):
            print("No checkpoint file found. Starting a new run.")
            return resume, batch_size, batch_num, ignore_large

        # Read checkpoint file
        print("A checkpoint file exists in the output folder.")
        with open(checkpoint_path, "r") as f:
            for line in f:
                key, value = line.rstrip().split(",")
                if key == "Input Folder":
                    input_folder = value
                elif key == "Batch Size":
                    batch_size = int(value)
                elif key == "Batch Number":
                    batch_num = int(value)
                elif key == "Ignore Large":
                    ignore_large = value.lower() == 'true'

        # Check if input folder matches
        if os.path.exists(input_folder):
            resume = os.path.samefile(input_folder, self.input_folder)

        # Verify all batch folders exist
        for batch_idx in range(batch_num + 1):
            batch_path = join_path(self.output_folder, 'Batches', f'batch_{batch_idx}')
            if not os.path.exists(batch_path):
                print('However, some necessary sub-folders are missing. A new run will start.')
                resume = False
                break

        # If validation passes, ask user about resuming
        if resume:
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Checkpoint Detected")
            msg_box.setText("A checkpoint file was found. Do you want to resume from the last checkpoint?")
            msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            msg_box.setDefaultButton(QMessageBox.Yes)
            msg_box.setStyleSheet(self.msgbox_style)

            if msg_box.exec_() == QMessageBox.Yes:
                print('Resuming from last check point.')
                return True, batch_size, batch_num, ignore_large
            else:
                print("Starting a new run.")

        return False, batch_size, batch_num, ignore_large

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
        self.mask_clear_btn.setEnabled(False)
        self.stats_cb.setEnabled(False)
        self.scores_cb.setEnabled(False)
        self.show_progress_bar()

        resume, batch_size, batch_num, ignore_large = self._check_batch_running_status()

        if not resume:
            batch_size = self.batch_size_spinner.value()
            batch_num = 0

        self.batch_worker = BatchProcessingWorker(
            self.param_file, self.input_folder, self.output_folder,
            batch_size, batch_num, resume, ignore_large,
            generate_stats=self.stats_cb.isChecked(),
            generate_scores=self.scores_cb.isChecked(),
            mask_dir=self.mask_folder
        )

        # Connect signals
        self.batch_worker.progress_updated.connect(lambda value: self.progress_bar.setValue(value))
        self.batch_worker.status_updated.connect(self.progress_label.setText)
        self.progress_label.setText("Starting batch processing…")
        self.progress_label.setVisible(True)
        self.batch_worker.batch_complete.connect(self.handle_batch_complete)
        self.batch_worker.batch_cancelled.connect(self.handle_batch_cancelled)

        # Start the worker thread
        self.batch_worker.start()

    def _restore_batch_buttons(self):
        self.cancel_batch_btn.setVisible(False)
        self.process_batch_btn.setVisible(True)
        self.process_batch_btn.setEnabled(True)
        self.param_btn.setEnabled(True)
        self.input_btn.setEnabled(True)
        self.output_btn.setEnabled(True)
        self.mask_btn.setEnabled(True)
        self.mask_clear_btn.setEnabled(bool(self.mask_folder))
        self.stats_cb.setEnabled(True)
        self.scores_cb.setEnabled(True)

    def _cancel_batch_processing(self):
        self.cancel_batch_btn.setEnabled(False)
        self.cancel_batch_btn.setText("Cancelling…")
        if hasattr(self, 'batch_worker'):
            self.batch_worker.cancel()

    def handle_batch_complete(self):
        self.hide_progress_bar()
        self._restore_batch_buttons()

        output_folder = getattr(self, 'output_folder', '')
        msg = QMessageBox(self)
        msg.setWindowTitle("Batch Processing Complete")
        msg.setText("Batch processing finished successfully.")
        msg.setInformativeText(f"Results saved to:\n{output_folder}")
        msg.setStyleSheet(self.msgbox_style)
        open_btn = msg.addButton("Open Folder", QMessageBox.ActionRole)
        msg.addButton(QMessageBox.Ok)
        self._fit_dialog_buttons(msg)

        msg.exec_()
        if msg.clickedButton() == open_btn:
            QDesktopServices.openUrl(QUrl.fromLocalFile(output_folder))

    def handle_batch_cancelled(self):
        self.hide_progress_bar()
        self._restore_batch_buttons()

        output_folder = getattr(self, 'output_folder', '')
        msg = QMessageBox(self)
        msg.setWindowTitle("Batch Processing Cancelled")
        msg.setText("Batch processing was cancelled.")
        msg.setInformativeText(
            f"Partial results and a checkpoint are saved in:\n{output_folder}\n\n"
            "You can resume from where you left off next time.")
        msg.setStyleSheet(self.msgbox_style)
        open_btn = msg.addButton("Open Folder", QMessageBox.ActionRole)
        msg.addButton(QMessageBox.Ok)
        self._fit_dialog_buttons(msg)

        msg.exec_()
        if msg.clickedButton() == open_btn:
            QDesktopServices.openUrl(QUrl.fromLocalFile(output_folder))

    def setup_segmentation_tab(self):
        """Set up the segmentation tab UI"""
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 10, 6, 6)
        layout.setSpacing(12)

        color_group_layout = QVBoxLayout()

        color_toggle_layout = QHBoxLayout()
        color_label = QLabel("Color of Interest:")
        color_label.setToolTip("Colour of the structures to keep (e.g. Picrosirius Red collagen). Click the swatch\n"
                               "to pick it; the normalised hue is written to the parameters.")
        self.toggle_seg_label = QLabel()
        self.toggle_seg_label.setText(
            f"Segmentation <b><span style='color: {COLORS['highlight'].name()};'>Enabled</span></b>")
        color_toggle_layout.addWidget(color_label)
        color_toggle_layout.addStretch()
        color_toggle_layout.addWidget(self.toggle_seg_label)
        color_group_layout.addLayout(color_toggle_layout)

        color_layout = QHBoxLayout()
        self.color_btn = QPushButton("")
        self.color_btn.setStyleSheet(
            f"background-color: #f53282; border: 1px solid {color_to_stylesheet(COLORS['border'])}; border-radius: 4px;")
        self.color_btn.setFixedSize(QSize(30, 30))
        self.color_btn.clicked.connect(self.select_color)
        self.color_btn.setToolTip("Select the color you want to segment.")
        color_layout.addWidget(self.color_btn)

        self.hue_label = QLabel("Normalized hue: 0.96")
        self.hue_label.setStyleSheet(f"font-weight: bold")
        self.hue_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        color_layout.addWidget(self.hue_label)

        # Toggle segmentation button
        self.toggle_seg_btn = ToggleButton()
        self.toggle_seg_btn.setChecked(True)
        self.toggle_seg_btn.toggled.connect(self.toggle_segmentation)
        self.toggle_seg_btn.setToolTip("Enable/Disable segmentation and update accordingly in the parameter file.")
        color_layout.addStretch()
        color_layout.addWidget(self.toggle_seg_btn)
        color_group_layout.addLayout(color_layout)

        layout.addLayout(color_group_layout)

        # Color threshold slider
        threshold_layout = QHBoxLayout()
        threshold_label = QLabel("Color Threshold:")
        threshold_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        threshold_label.setMinimumWidth(50)
        threshold_layout.addWidget(threshold_label)
        self.color_thresh_slider = CustomSlider(Qt.Horizontal)
        self.color_thresh_slider.setRange(0, 100)
        self.color_thresh_slider.setValue(20)  # Default 0.2
        self.color_thresh_slider.valueChanged.connect(self.update_color_threshold)
        self.color_thresh_slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.color_thresh_slider.setToolTip("Lower this threshold to preserve more areas of interest.")
        threshold_layout.addWidget(self.color_thresh_slider, 3)
        self.color_thresh_value = QLabel("0.2")
        self.color_thresh_value.setFixedWidth(35)
        self.color_thresh_value.setAlignment(Qt.AlignVCenter | Qt.AlignRight)
        self.color_thresh_value.setStyleSheet(
            self.value_label_style)
        threshold_layout.addWidget(self.color_thresh_value)
        layout.addLayout(threshold_layout)

        # Number of labels slider
        num_labels_layout = QHBoxLayout()
        num_labels_label = QLabel("No. of Labels:")
        num_labels_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        num_labels_label.setMinimumWidth(50)
        num_labels_layout.addWidget(num_labels_label)
        self.num_labels_slider = CustomSlider(Qt.Horizontal)
        self.num_labels_slider.setRange(8, 96)
        self.num_labels_slider.setValue(32)  # Default
        self.num_labels_slider.valueChanged.connect(self.update_num_labels)
        self.num_labels_slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.num_labels_slider.setToolTip("Increase this value for fine-granularity segmentation.")
        num_labels_layout.addWidget(self.num_labels_slider, 3)
        self.num_labels_value = QLabel("32")
        self.num_labels_value.setFixedWidth(30)
        self.num_labels_value.setAlignment(Qt.AlignVCenter | Qt.AlignRight)
        self.num_labels_value.setStyleSheet(
            self.value_label_style)
        num_labels_layout.addWidget(self.num_labels_value)
        layout.addLayout(num_labels_layout)

        # Max iterations slider
        max_iters_layout = QHBoxLayout()
        max_iters_label = QLabel("Max Iterations:")
        max_iters_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        max_iters_label.setMinimumWidth(50)
        max_iters_layout.addWidget(max_iters_label)
        self.max_iters_slider = CustomSlider(Qt.Horizontal)
        self.max_iters_slider.setRange(10, 100)
        self.max_iters_slider.setValue(30)  # Default
        self.max_iters_slider.valueChanged.connect(self.update_max_iters)
        self.max_iters_slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.max_iters_slider.setToolTip("Reduce this value for fine-granularity segmentation.")
        max_iters_layout.addWidget(self.max_iters_slider, 3)
        self.max_iters_value = QLabel("30")
        self.max_iters_value.setFixedWidth(30)
        self.max_iters_value.setAlignment(Qt.AlignVCenter | Qt.AlignRight)
        self.max_iters_value.setStyleSheet(
            self.value_label_style)
        max_iters_layout.addWidget(self.max_iters_value)
        layout.addLayout(max_iters_layout)

        # Patch size (0 = segment the whole image at once)
        patch_layout = QHBoxLayout()
        patch_label = QLabel("Patch Size:")
        patch_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        patch_label.setMinimumWidth(50)
        patch_layout.addWidget(patch_label)
        patch_layout.addStretch()
        self.patch_size_spinner = QSpinBox()
        self.patch_size_spinner.setRange(0, 8192)
        self.patch_size_spinner.setSingleStep(256)
        self.patch_size_spinner.setSpecialValueText("Off")
        self.patch_size_spinner.setSuffix(" px")
        self.patch_size_spinner.setValue(0)
        self.patch_size_spinner.setFixedWidth(90)
        self.patch_size_spinner.setStyleSheet(self.spinner_style)
        self.patch_size_spinner.valueChanged.connect(self.update_patch_size)
        self.patch_size_spinner.setToolTip(
            "Segment large images in overlapping square patches of this size\n"
            "instead of shrinking the whole image to 512 px. Off = whole image.")
        patch_layout.addWidget(self.patch_size_spinner)
        layout.addLayout(patch_layout)

        # White background checkbox
        layout.addWidget(create_separator())
        h_layout = QHBoxLayout()
        self.white_bg_cb = QCheckBox("White Background")
        self.white_bg_cb.setChecked(True)
        self.white_bg_cb.setStyleSheet(self.checkbox_style)
        self.white_bg_cb.stateChanged.connect(self.update_white_bg)
        self.white_bg_cb.setToolTip("Enable this option when detecting dark fibres in bright backgrounds.")
        h_layout.addWidget(self.white_bg_cb)

        # Add toggle checkbox for comparing with the original image
        self.toggle_img_cb = QCheckBox("Overlay Original")
        self.toggle_img_cb.setChecked(False)
        self.toggle_img_cb.setStyleSheet(self.checkbox_style)
        self.toggle_img_cb.clicked.connect(self.compare_image)
        self.toggle_img_cb.setToolTip("Toggle to overlay the original image.")
        h_layout.addStretch()
        h_layout.addWidget(self.toggle_img_cb)

        layout.addLayout(h_layout)

        # Segmentation button
        self.segment_btn = QPushButton("Segment")
        self.segment_btn.clicked.connect(self.run_segmentation)
        self.segment_btn.setEnabled(False)
        self.segment_btn.setStyleSheet(self.primary_btn_style)
        layout.addWidget(self.segment_btn)

        layout.addStretch()
        self.seg_tab.setLayout(layout)

    def setup_detection_tab(self):
        """Set up the detection tab UI with range sliders"""
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 10, 6, 6)
        layout.setSpacing(12)

        # Line width range slider
        line_width_layout = QHBoxLayout()
        line_width_label = QLabel("Line Width (px):")
        line_width_layout.addWidget(line_width_label)

        # Create range slider for line width
        self.line_width_range = RangeSlider(Qt.Horizontal)
        self.line_width_range.setRange(1, 15)
        self.line_width_range.setValues(5, 7)  # Default values
        self.line_width_range.valueChanged.connect(self.update_line_width_range)
        self.line_width_range.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.line_width_range.setToolTip("Increase line widths to detect thicker fibers.")
        line_width_layout.addWidget(self.line_width_range, 3)

        self.line_width_value = QLabel("(5, 7)")
        self.line_width_value.setFixedWidth(55)
        self.line_width_value.setAlignment(Qt.AlignVCenter | Qt.AlignRight)
        self.line_width_value.setStyleSheet(
            self.value_label_style)
        line_width_layout.addWidget(self.line_width_value)
        layout.addLayout(line_width_layout)

        # Line step slider
        line_step_layout = QHBoxLayout()
        line_step_label = QLabel("Line Step (px):")
        line_step_layout.addWidget(line_step_label)
        self.line_step_slider = CustomSlider(Qt.Horizontal)
        self.line_step_slider.setRange(1, 5)
        self.line_step_slider.setValue(2)  # Default
        self.line_step_slider.valueChanged.connect(self.update_line_step)
        self.line_step_slider.setToolTip("Reduce this value to detect more fibers.")
        line_step_layout.addWidget(self.line_step_slider)
        self.line_step_value = QLabel("2")
        self.line_step_value.setFixedWidth(30)
        self.line_step_value.setAlignment(Qt.AlignVCenter | Qt.AlignRight)
        self.line_step_value.setStyleSheet(
            self.value_label_style)
        line_step_layout.addWidget(self.line_step_value)
        layout.addLayout(line_step_layout)

        # Contrast range slider
        contrast_layout = QHBoxLayout()
        contrast_label = QLabel("Contrast:")
        contrast_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        contrast_layout.addWidget(contrast_label)

        # Create range slider for contrast
        self.contrast_range = RangeSlider(Qt.Horizontal)
        self.contrast_range.setRange(0, 255)
        self.contrast_range.setValues(100, 200)  # Default values
        self.contrast_range.valueChanged.connect(self.update_contrast_range)
        self.contrast_range.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.contrast_range.setToolTip("Reduce the values if fibre contrast is low.")
        contrast_layout.addWidget(self.contrast_range, 3)

        self.contrast_value = QLabel("(100, 200)")
        self.contrast_value.setFixedWidth(70)
        self.contrast_value.setAlignment(Qt.AlignVCenter | Qt.AlignRight)
        self.contrast_value.setStyleSheet(
            self.value_label_style)
        contrast_layout.addWidget(self.contrast_value)
        layout.addLayout(contrast_layout)

        # Minimum line length slider
        min_length_layout = QHBoxLayout()
        min_length_label = QLabel("Minimum Line Length:")
        min_length_layout.addWidget(min_length_label)
        self.min_length_slider = CustomSlider(Qt.Horizontal)
        self.min_length_slider.setRange(1, 50)
        self.min_length_slider.setValue(5)  # Default
        self.min_length_slider.valueChanged.connect(self.update_min_length)
        self.min_length_slider.setToolTip("Fibers shorter than this length will be ignored.")
        min_length_layout.addWidget(self.min_length_slider)
        self.min_length_value = QLabel("5")
        self.min_length_value.setFixedWidth(30)
        self.min_length_value.setAlignment(Qt.AlignVCenter | Qt.AlignRight)
        self.min_length_value.setStyleSheet(
            self.value_label_style)
        min_length_layout.addWidget(self.min_length_value)
        layout.addLayout(min_length_layout)

        # Checkboxes
        layout.addWidget(create_separator())
        checkbox_layout = QHBoxLayout()
        self.dark_line_cb = QCheckBox("Dark Line")
        self.dark_line_cb.setChecked(True)
        self.dark_line_cb.setStyleSheet(self.checkbox_style)
        self.dark_line_cb.stateChanged.connect(self.update_dark_line)
        self.dark_line_cb.setToolTip("Enable this option to detect dark fibers on bright backgrounds.")
        checkbox_layout.addWidget(self.dark_line_cb)

        self.extend_line_cb = QCheckBox("Extend Line")
        self.extend_line_cb.setChecked(False)
        self.extend_line_cb.setStyleSheet(self.checkbox_style)
        self.extend_line_cb.stateChanged.connect(self.update_extend_line)
        self.extend_line_cb.setToolTip("Enable to detect fibers near junctions.")
        checkbox_layout.addWidget(self.extend_line_cb)

        self.overlay_fibres_cb = QCheckBox("Overlay Fibres")
        self.overlay_fibres_cb.setChecked(True)
        self.overlay_fibres_cb.setStyleSheet(self.checkbox_style)
        self.overlay_fibres_cb.stateChanged.connect(self.update_overlay_fibres)
        self.overlay_fibres_cb.setToolTip("Toggle to overlay detected fibres on the image.")
        checkbox_layout.addWidget(self.overlay_fibres_cb)

        layout.addLayout(checkbox_layout)

        # Detection button
        self.detect_btn = QPushButton("Detect")
        self.detect_btn.clicked.connect(self.run_detection)
        self.detect_btn.setEnabled(False)
        self.detect_btn.setStyleSheet(self.primary_btn_style)
        layout.addWidget(self.detect_btn)

        layout.addStretch()
        self.det_tab.setLayout(layout)

    def setup_gap_analysis_tab(self):
        """Set up the gap analysis tab UI"""
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 10, 6, 6)
        layout.setSpacing(12)

        self.toggle_gap_label = QLabel()
        self.toggle_gap_label.setText(
            f"Gap Analysis <b><span style='color: {COLORS['highlight'].name()};'>Enabled</span></b>")
        # layout.addWidget(self.toggle_gap_label, 0, Qt.AlignRight)
        toggle_layout = QHBoxLayout()

        self.toggle_gap_btn = ToggleButton()
        self.toggle_gap_btn.setChecked(True)
        self.toggle_gap_btn.toggled.connect(self.toggle_gap_analysis)
        self.toggle_gap_btn.setToolTip("Enable/Disable gap analysis and update accordingly in the parameter file.")
        # layout.addWidget(self.toggle_gap_btn, 0, Qt.AlignRight)
        toggle_layout.addStretch()
        toggle_layout.addWidget(self.toggle_gap_label)
        toggle_layout.addWidget(self.toggle_gap_btn)
        layout.addLayout(toggle_layout)

        # Minimum gap diameter slider
        min_gap_layout = QHBoxLayout()
        min_gap_label = QLabel("Min Gap Diameter (px):")
        min_gap_layout.addWidget(min_gap_label)
        self.min_gap_slider = CustomSlider(Qt.Horizontal)
        self.min_gap_slider.setRange(5, 100)
        self.min_gap_slider.setValue(20)  # Default
        self.min_gap_slider.valueChanged.connect(self.update_min_gap)
        self.min_gap_slider.setToolTip("Lower this value for more detailed analysis.")
        min_gap_layout.addWidget(self.min_gap_slider)
        self.min_gap_value = QLabel("20")
        self.min_gap_value.setFixedWidth(30)
        self.min_gap_value.setAlignment(Qt.AlignVCenter | Qt.AlignRight)
        self.min_gap_value.setStyleSheet(
            self.value_label_style)
        min_gap_layout.addWidget(self.min_gap_value)
        layout.addLayout(min_gap_layout)

        # Max display HDM slider
        max_hdm_layout = QHBoxLayout()
        max_hdm_label = QLabel("Max Display HDM:")
        max_hdm_layout.addWidget(max_hdm_label)
        self.max_hdm_slider = CustomSlider(Qt.Horizontal)
        self.max_hdm_slider.setRange(100, 255)
        self.max_hdm_slider.setValue(230)  # Default
        self.max_hdm_slider.valueChanged.connect(self.update_max_hdm)
        self.max_hdm_slider.setToolTip("Reduce this value to narrow down the HDM area of interest.")
        max_hdm_layout.addWidget(self.max_hdm_slider)
        self.max_hdm_value = QLabel("230")
        self.max_hdm_value.setFixedWidth(30)
        self.max_hdm_value.setAlignment(Qt.AlignVCenter | Qt.AlignRight)
        self.max_hdm_value.setStyleSheet(
            self.value_label_style)
        max_hdm_layout.addWidget(self.max_hdm_value)
        layout.addLayout(max_hdm_layout)

        # Overlay checkbox
        layout.addWidget(create_separator())
        self.overlay_gaps_cb = QCheckBox("Overlay Gaps")
        self.overlay_gaps_cb.setChecked(False)
        self.overlay_gaps_cb.setStyleSheet(self.checkbox_style)
        self.overlay_gaps_cb.stateChanged.connect(self.update_overlay_gaps)
        self.overlay_gaps_cb.setToolTip("Toggle to overlay gap analysis results on image.")
        overlay_layout = QHBoxLayout()
        overlay_layout.addStretch()
        overlay_layout.addWidget(self.overlay_gaps_cb)
        layout.addLayout(overlay_layout)

        # Analysis button
        self.analyze_btn = QPushButton("Analyze")
        self.analyze_btn.clicked.connect(self.run_gap_analysis)
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.setStyleSheet(self.primary_btn_style)
        layout.addWidget(self.analyze_btn)

        layout.addStretch()
        self.gap_tab.setLayout(layout)

    def update_line_width_range(self, min_val, max_val):
        """Update line width range values"""
        self.line_width_value.setText(f"({min_val}, {max_val})")
        self.yml_data["Detection"]["Min Line Width"] = min_val
        self.yml_data["Detection"]["Max Line Width"] = max_val

    def update_contrast_range(self, min_val, max_val):
        """Update contrast range values"""
        self.contrast_value.setText(f"({min_val}, {max_val})")
        self.yml_data["Detection"]["Low Contrast"] = min_val
        self.yml_data["Detection"]["High Contrast"] = max_val

    def select_color(self):
        """Open color picker dialog"""
        current_color = QColor(self.color_btn.styleSheet().split("background-color: ")[1].split(";")[0])
        color = QColorDialog.getColor(current_color)

        if color.isValid():
            hex_color = color.name()
            self.color_btn.setStyleSheet(
                f"background-color: {hex_color}; border: 1px solid {color_to_stylesheet(COLORS['border'])}; border-radius: 4px;")

            # Calculate hue
            hue = hex_to_hue(hex_color)
            self.hue_label.setText(f"Normalized hue: {hue:.2f}")
            # self.hue_label.setStyleSheet(f"color: {hex_color}")

            # Update YAML data
            self.yml_data["Segmentation"]["Normalized Hue Value"] = float(f"{hue:.2f}")

    def update_color_threshold(self):
        """Update color threshold value"""
        value = self.color_thresh_slider.value() / 100.0
        self.color_thresh_value.setText(f"{value:.2f}")
        self.yml_data["Segmentation"]["Color Threshold"] = value

    def update_num_labels(self):
        """Update number of labels value"""
        value = self.num_labels_slider.value()
        self.num_labels_value.setText(str(value))
        self.yml_data["Segmentation"]["Number of Labels"] = value

    def update_max_iters(self):
        """Update max iterations value"""
        value = self.max_iters_slider.value()
        self.max_iters_value.setText(str(value))
        self.yml_data["Segmentation"]["Max Iterations"] = value

    def update_patch_size(self):
        """Update segmentation patch size (0 disables patch-wise segmentation)"""
        self.yml_data["Segmentation"]["Patch Size"] = int(self.patch_size_spinner.value())

    def update_white_bg(self):
        """Update white background setting"""
        self.yml_data["Segmentation"]["Dark Line"] = self.white_bg_cb.isChecked()
        self.dark_line_cb.setChecked(self.white_bg_cb.isChecked())

    def update_line_step(self):
        """Update line step value"""
        value = self.line_step_slider.value()
        self.line_step_value.setText(str(value))
        self.yml_data["Detection"]["Line Width Step"] = value

    def update_min_length(self):
        """Update minimum line length value"""
        value = self.min_length_slider.value()
        self.min_length_value.setText(str(value))
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
        self.min_gap_value.setText(str(value))
        self.yml_data["Gap Analysis"]["Minimum Gap Diameter"] = value

    def update_max_hdm(self):
        """Update maximum HDM display value"""
        value = self.max_hdm_slider.value()
        self.max_hdm_value.setText(str(value))
        self.yml_data["Quantification"]["Maximum Display HDM"] = value

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
                },
                "Segmentation": {
                    "Number of Labels": 32,
                    "Max Iterations": 30,
                    "Color Threshold": 0.2,
                    "Min Size": 64,
                    "Max Size": 2048,
                    "Patch Size": 0,
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
            self.yml_data = data
            self.apply_params_to_widgets()
        except Exception as e:
            print(f"Failed to load parameters: {e}")

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
        if "Dark Line" in seg:
            self.white_bg_cb.setChecked(seg["Dark Line"])
        if "Normalized Hue Value" in seg:
            hue = seg["Normalized Hue Value"]
            self.hue_label.setText(f"Normalized hue: {hue:.2f}")
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
            self.frb_img = None
            self.wdt_img = None
            self.gap_img = None
            self.gap_ovl = None
            self._update_image_status()
        except Exception as e:
            self.ori_img = None
            self.img_path = None
            self.segment_btn.setEnabled(False)
            self.detect_btn.setEnabled(False)
            self.reload_btn.setEnabled(False)
            self._update_image_status()
            print(f"Error loading image: {e}")


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
            self.frb_img = None
            self.wdt_img = None
            self.gap_img = None
            self.gap_ovl = None
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
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Warning")
            msg.setText(f"No {label} image {'loaded' if label == 'original' else 'available'}.")
            msg.setStyleSheet(f"""
                QMessageBox {{
                    background-color: {COLORS['background'].name()};
                    color: {COLORS['text'].name()};
                }}
            """)
            self._fit_dialog_buttons(msg)

            msg.exec_()

    def update_overlay_fibres(self):
        """Overlay fibres on the image"""
        show_overlay = self.overlay_fibres_cb.isChecked()
        show_img = self.ori_img if (self.seg_img is None or not self.toggle_seg_btn.isChecked()) else self.seg_img
        image = self.wdt_img if show_overlay else show_img
        label = "fibre" if show_overlay else "original"

        if image is not None:
            self.image_panel.setImage(image, preserve_view=True)
        else:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Warning")
            msg.setText(f"No {label} image available.")
            msg.setStyleSheet(f"""
                            QMessageBox {{
                                background-color: {COLORS['background'].name()};
                                color: {COLORS['text'].name()};
                            }}
                        """)
            self._fit_dialog_buttons(msg)

            msg.exec_()

    def update_overlay_gaps(self):
        """Overlay gaps on the image"""
        show_overlay = self.overlay_gaps_cb.isChecked()
        image = self.gap_ovl if show_overlay else self.gap_img

        if image is not None:
            self.image_panel.setImage(image, preserve_view=True)
        else:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Warning")
            msg.setText(f"No gap image available.")
            msg.setStyleSheet(f"""
                            QMessageBox {{ 
                                background-color: {COLORS['background'].name()}; 
                                color: {COLORS['text'].name()};
                            }}
                        """)
            self._fit_dialog_buttons(msg)

            msg.exec_()

    def toggle_segmentation(self):
        """Disable segmentation and update the UI accordingly"""
        if not self.toggle_seg_btn.isChecked():
            self.yml_data["Configs"]["Segmentation"] = False
            self.toggle_seg_label.setText(
                        f"Segmentation <b><span style='color: {COLORS['warning'].name()};'>Disabled</span></b>")
            self.segment_btn.setEnabled(False)
            self.color_btn.setEnabled(False)
            self.color_thresh_slider.setEnabled(False)
            self.num_labels_slider.setEnabled(False)
            self.max_iters_slider.setEnabled(False)
            self.white_bg_cb.setEnabled(False)
            self.toggle_img_cb.setEnabled(False)
        else:
            self.yml_data["Configs"]["Segmentation"] = True
            self.toggle_seg_label.setText(
                f"Segmentation <b><span style='color: {COLORS['highlight'].name()};'>Enabled</span></b>")
            # Segment still requires a loaded image
            self.segment_btn.setEnabled(self.ori_img is not None)
            self.color_btn.setEnabled(True)
            self.color_thresh_slider.setEnabled(True)
            self.num_labels_slider.setEnabled(True)
            self.max_iters_slider.setEnabled(True)
            self.white_bg_cb.setEnabled(True)
            self.toggle_img_cb.setEnabled(True)

    def toggle_gap_analysis(self):
        """Disable gap analysis and update the UI accordingly"""
        if not self.toggle_gap_btn.isChecked():
            self.yml_data["Configs"]["Gap Analysis"] = False
            self.toggle_gap_label.setText(
                f"Gap Analysis <b><span style='color: {COLORS['warning'].name()};'>Disabled</span></b>")
            self.analyze_btn.setEnabled(False)
            self.min_gap_slider.setEnabled(False)
            self.overlay_gaps_cb.setEnabled(False)
        else:
            self.yml_data["Configs"]["Gap Analysis"] = True
            self.toggle_gap_label.setText(
                f"Gap Analysis <b><span style='color: {COLORS['highlight'].name()};'>Enabled</span></b>")
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

        # Start the worker thread
        self.segmentation_worker.start()

    def handle_segmentation_complete(self, result):
        """Handle the completed segmentation result"""
        # Store the result
        self.seg_img = result

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
        self.analyze_btn.setText("Analyzing...")
        self.analyze_btn.setEnabled(False)
        self.show_progress_bar()

        self.gap_analysis_worker = GapAnalysisWorker(self.frb_img, min_gap_diameter)
        self.gap_analysis_worker.progress_updated.connect(lambda value: self.progress_bar.setValue(value))
        self.gap_analysis_worker.gap_analysis_complete.connect(self.handle_gap_analysis_complete)

        self.gap_analysis_worker.start()

    def handle_gap_analysis_complete(self, result):
        """Handle the completed gap analysis result"""
        self.gap_img = result
        self.image_panel.setImage(self.gap_img)
        overlay_img = self.ori_img if (self.seg_img is None or not self.toggle_seg_btn.isChecked()) else self.seg_img
        # Create mask: True where overlay is NOT white
        mask = ~((self.gap_img[:, :, 0] == 255) &
                 (self.gap_img[:, :, 1] == 255) &
                 (self.gap_img[:, :, 2] == 255))
        self.gap_ovl = overlay_img.copy()
        self.gap_ovl[mask] = self.gap_img[mask]

        # self.gap_ovl = cv2.addWeighted(overlay_img, 0.7, self.gap_img, 0.4, 10)
        self.overlay_gaps_cb.setChecked(False)

        self.hide_progress_bar()
        self.segment_btn.setEnabled(True)
        self.analyze_btn.setEnabled(True)
        self.detect_btn.setEnabled(True)
        self.load_btn.setEnabled(True)
        self.analyze_btn.setText("Analyze")

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
        """Show the About dialog with version and project info."""
        text_color = color_to_stylesheet(COLORS['text'])
        link_color = color_to_stylesheet(COLORS['highlight'])
        bg_color = color_to_stylesheet(COLORS['surface'])
        dlg = QMessageBox(self)
        dlg.setWindowTitle("About Cabana")
        dlg.setIcon(QMessageBox.Information)
        dlg.setTextFormat(Qt.RichText)
        dlg.setText(
            f"<div style='color:{text_color}; font-size:14px;'>"
            f"<b style='font-size:16px;'>Cabana</b> — CollAgen FiBre ANAlyzer<br>"
            f"Version {__version__} · MIT License<br><br>"
            "A Python toolkit for analyzing collagen fibre architecture "
            "in IHC and fluorescence microscopy images.<br><br>"
            f"<a href='https://cabana.readthedocs.io' "
            f"style='color:{link_color};'>Documentation</a> · "
            f"<a href='https://pypi.org/project/cabana/' "
            f"style='color:{link_color};'>PyPI</a>"
            "</div>"
        )
        dlg.setStyleSheet(
            f"QMessageBox {{ background-color: {bg_color}; }}"
            f"QLabel {{ color: {text_color}; }}"
            f"QPushButton {{ min-width: 60px; }}"
        )
        dlg.exec_()

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
                "https://github.com/lxfhfut/Cabana/blob/main/LICENSE")

    def _show_license_dialog(self) -> None:
        """Show the license text in a read-only dialog."""
        text_color = color_to_stylesheet(COLORS['text'])
        bg_color = color_to_stylesheet(COLORS['surface'])
        dlg = QMessageBox(self)
        dlg.setWindowTitle("License")
        dlg.setIcon(QMessageBox.NoIcon)
        dlg.setTextFormat(Qt.PlainText)
        dlg.setText(self._license_text())
        dlg.setStyleSheet(
            f"QMessageBox {{ background-color: {bg_color}; }}"
            f"QLabel {{ color: {text_color}; font-family: monospace; }}"
            f"QPushButton {{ min-width: 60px; }}"
        )
        dlg.exec_()

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
        for btn in (self.start_open_btn, self.start_params_btn,
                    self.param_btn, self.input_btn, self.output_btn, self.cancel_batch_btn,
                     self.mask_btn, self.mask_clear_btn, self.tma_slide_btn, self.tma_output_btn,
                     self.tma_cancel_btn):
            btn.setStyleSheet(self.btn_style)

        # Primary buttons
        for btn in (self.segment_btn, self.detect_btn, self.analyze_btn, self.process_batch_btn,
                    self.tma_fit_btn, self.tma_export_btn, self.start_slide_btn):
            btn.setStyleSheet(self.primary_btn_style)

        # Page stack
        self.pages.setStyleSheet(self.page_stack_style)

        # Progress bar
        self.progress_bar.setStyleSheet(self.progressbar_style)

        # Spinboxes
        for spin in (self.batch_size_spinner, self.patch_size_spinner, self.tma_pixel_size_spin,
                     self.tma_core_diameter_spin, self.tma_margin_spin, self.tma_erode_spin,
                     self.tma_sat_spin, self.tma_offset_spin, self.tma_dmin_spin, self.tma_dmax_spin,
                     self.tma_stain_spin):
            spin.setStyleSheet(self.spinner_style)

        # Combo boxes
        for combo in (self.tma_array_combo, self.tma_orientation_combo):
            combo.setStyleSheet(self.combo_style)

        # Checkboxes
        for cb in (self.white_bg_cb, self.toggle_img_cb, self.dark_line_cb,
                   self.extend_line_cb, self.overlay_fibres_cb, self.overlay_gaps_cb,
                   self.stats_cb, self.scores_cb, self.tma_recover_cb, *self.tma_channel_cbs.values()):
            cb.setStyleSheet(self.checkbox_style)

        # Path edits
        for edit in (self.param_file_path, self.input_folder_path, self.output_folder_path,
                     self.mask_folder_path, self.tma_slide_path, self.tma_output_path):
            edit.setStyleSheet(self.path_edit_style)
        self.tma_status_label.setStyleSheet(self.value_label_style)
        self.progress_label.setStyleSheet(self.value_label_style)

        # Value labels (slider readouts)
        for label in (self.color_thresh_value, self.num_labels_value, self.max_iters_value,
                      self.line_width_value, self.line_step_value, self.contrast_value,
                      self.min_length_value, self.min_gap_value, self.max_hdm_value):
            label.setStyleSheet(self.value_label_style)

        # Status bar
        self.status_bar.setStyleSheet(self.status_bar_style)

        # Theme combo
        self.theme_combo.setStyleSheet(self.theme_combo_style)

        # Dock panel palette
        dock_palette = self.dock_contents.palette()
        dock_palette.setColor(QPalette.Window, COLORS['dock'])
        self.dock_contents.setPalette(dock_palette)

        # Image panel palette
        img_palette = self.image_panel.palette()
        img_palette.setColor(self.image_panel.backgroundRole(), COLORS['canvas'])
        self.image_panel.setPalette(img_palette)

        # Update toggle labels with current theme colors
        if hasattr(self, 'toggle_seg_btn') and self.toggle_seg_btn.isChecked():
            self.toggle_seg_label.setText(
                f"Segmentation <b><span style='color: {COLORS['highlight'].name()};'>Enabled</span></b>")
        if hasattr(self, 'toggle_gap_btn') and self.toggle_gap_btn.isChecked():
            self.toggle_gap_label.setText(
                f"Gap Analysis <b><span style='color: {COLORS['highlight'].name()};'>Enabled</span></b>")

        # Force repaint on all widgets
        for widget in self.findChildren(QWidget):
            widget.update()
