import os
os.environ["NUMEXPR_MAX_THREADS"] = "20"
import sys
import colorsys
import yaml
import imageio.v3 as iio
import tifffile as tiff
from pathlib import Path
from .utils import join_path

from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QSpinBox,
                             QVBoxLayout, QHBoxLayout, QTabWidget, QCheckBox,
                             QPushButton, QFileDialog, QSizePolicy, QColorDialog,
                             QMessageBox, QGroupBox, QComboBox, QWidget,
                             QStatusBar, QLineEdit)
from PyQt5.QtGui import QIcon, QPalette, QFont
from PyQt5.QtCore import QSettings

from .ui import *
from .themes import THEMES, DEFAULT_THEME


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set window properties
        self.setWindowTitle("Cabana-GUI")
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

        self._setup_styles()

        # --- Image group ---
        self.image_group = QGroupBox("Image")
        self.image_group.setStyleSheet(self.group_style)
        image_btn_layout = QHBoxLayout(self.image_group)
        image_btn_layout.setContentsMargins(8, 4, 8, 4)

        self.load_btn = QPushButton("Open")
        self.load_btn.setStyleSheet(self.btn_style)
        self.load_btn.setToolTip("Load an image file")
        self.load_btn.clicked.connect(self.load_image)
        image_btn_layout.addWidget(self.load_btn)

        self.reload_btn = QPushButton("Reload")
        self.reload_btn.setStyleSheet(self.btn_style)
        self.reload_btn.setToolTip("Reload the original image")
        self.reload_btn.clicked.connect(self.reload_image)
        self.reload_btn.setEnabled(False)
        image_btn_layout.addWidget(self.reload_btn)

        self.dock_layout.addWidget(self.image_group)

        # --- Parameters group ---
        self.params_group = QGroupBox("Parameters")
        self.params_group.setStyleSheet(self.group_style)
        params_btn_layout = QHBoxLayout(self.params_group)
        params_btn_layout.setContentsMargins(8, 4, 8, 4)

        self.load_params_btn = QPushButton("Import")
        self.load_params_btn.setStyleSheet(self.btn_style)
        self.load_params_btn.setToolTip("Load parameters from YAML file")
        self.load_params_btn.clicked.connect(self.import_parameters)
        params_btn_layout.addWidget(self.load_params_btn)

        self.export_btn = QPushButton("Export")
        self.export_btn.setStyleSheet(self.btn_style)
        self.export_btn.setToolTip("Export parameters to YAML file")
        self.export_btn.clicked.connect(self.export_parameters)
        params_btn_layout.addWidget(self.export_btn)

        self.dock_layout.addWidget(self.params_group)
        self.dock_layout.addSpacing(6)

        # --- Analysis group ---
        self.analysis_group = QGroupBox("Analysis")
        self.analysis_group.setStyleSheet(self.group_style)
        self.dock_inner_layout = QVBoxLayout(self.analysis_group)
        self.dock_inner_layout.setContentsMargins(6, 10, 6, 6)
        self.dock_inner_layout.setSpacing(6)
        self.dock_layout.addWidget(self.analysis_group, 1)

        # Create tab widget
        self.tabs = QTabWidget()
        self.tabs.setTabBar(AutoWidthTabBar())
        self.tabs.setUsesScrollButtons(False)
        self.tabs.tabBar().setElideMode(Qt.ElideRight)
        self.tabs.setStyleSheet(self.tab_style)

        # Create tabs
        self.seg_tab = QWidget()
        self.det_tab = QWidget()
        self.gap_tab = QWidget()
        self.bat_tab = QWidget()

        # Set up each tab
        self.setup_segmentation_tab()
        self.setup_detection_tab()
        self.setup_gap_analysis_tab()
        self.setup_batch_processing_tab()

        # Add tabs to widget
        self.tabs.addTab(self.seg_tab, "Segment")
        self.tabs.addTab(self.det_tab, "Detect Fibres")
        self.tabs.addTab(self.gap_tab, "Analyse Gaps")
        self.tabs.addTab(self.bat_tab, "Batch Run")

        self.dock_inner_layout.addWidget(self.tabs)

        # Add a spacer to push content to the top
        self.dock_inner_layout.addStretch()

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
        self.tabs.tabBar().adjustSize()
        dock_margins = self.dock_layout.contentsMargins()
        inner_margins = self.dock_inner_layout.contentsMargins()
        # QGroupBox CSS: padding 8px L/R + border 1px L/R = 18px total
        groupbox_chrome = 18
        initial_dock_width = max(
            self.dock_contents.minimumWidth(),
            self.tabs.tabBar().sizeHint().width()
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


    def _setup_styles(self) -> None:
        """Set up all style sheets"""
        # Button style
        self.btn_style = generate_button_style()

        # Tab style
        self.tab_style = generate_tab_style()

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
            f"border-radius: 3px; padding: 5px 6px; "
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
        output_layout.addWidget(self.output_folder_path, 1)

        self.output_btn = QPushButton("Select")
        self.output_btn.clicked.connect(self.select_output_folder)
        self.output_btn.setStyleSheet(self.btn_style)
        output_layout.addWidget(self.output_btn)

        layout.addLayout(output_layout)

        # Batch size spinbox
        batch_size_layout = QHBoxLayout()
        batch_size_label = QLabel("Batch Size:")
        batch_size_layout.addWidget(batch_size_label)

        self.batch_size_spinner = QSpinBox()
        self.batch_size_spinner.setRange(1, 100)
        self.batch_size_spinner.setValue(5)
        self.batch_size_spinner.setFixedWidth(50)
        self.batch_size_spinner.setStyleSheet(self.spinner_style)
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

        # Batch processing button
        self.process_batch_btn = QPushButton("Process Batch")
        self.process_batch_btn.clicked.connect(self.run_batch_processing)
        self.process_batch_btn.setEnabled(False)
        self.process_batch_btn.setStyleSheet(self.primary_btn_style)
        layout.addWidget(self.process_batch_btn)

        layout.addStretch()
        self.bat_tab.setLayout(layout)

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
            self.input_folder_path.setText(folder)
            self.input_folder_path.setToolTip(folder)
            self._check_batch_processing_ready()

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

        # Disable all buttons during processing
        self.process_batch_btn.setEnabled(False)
        self.process_batch_btn.setText("Processing...")
        self.param_btn.setEnabled(False)
        self.input_btn.setEnabled(False)
        self.output_btn.setEnabled(False)
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
            generate_scores=self.scores_cb.isChecked()
        )

        # Connect signals
        self.batch_worker.progress_updated.connect(lambda value: self.progress_bar.setValue(value))
        self.batch_worker.batch_complete.connect(self.handle_batch_complete)

        # Start the worker thread
        self.batch_worker.start()

    def handle_batch_complete(self):
        """Handle the completed batch processing"""
        # Hide progress bar
        self.hide_progress_bar()

        # Re-enable buttons
        self.process_batch_btn.setEnabled(True)
        self.process_batch_btn.setText("Process Batch")
        self.param_btn.setEnabled(True)
        self.input_btn.setEnabled(True)
        self.output_btn.setEnabled(True)
        self.stats_cb.setEnabled(True)
        self.scores_cb.setEnabled(True)

        # # Show a message box to notify the user
        # QMessageBox.information(
        #     self, "Batch Processing Complete",
        #     "Batch processing has been completed successfully."
        # )

    def setup_segmentation_tab(self):
        """Set up the segmentation tab UI"""
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 10, 6, 6)
        layout.setSpacing(12)

        color_group_layout = QVBoxLayout()

        color_toggle_layout = QHBoxLayout()
        color_label = QLabel("Color of Interest:")
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

            # Enable processing buttons
            self.segment_btn.setEnabled(True)
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
            self.segment_btn.setEnabled(True)
            self.detect_btn.setEnabled(True)
            self.reload_btn.setEnabled(True)
            self._update_image_status()

    def reload_image(self):
        """Reload the original image"""
        if self.img_path:
            self.image_panel.setImage(self.img_path)
            self.segment_btn.setEnabled(True)
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
            self.segment_btn.setEnabled(True)
            self.segment_btn.setEnabled(True)
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
        # Group boxes
        for group in (self.image_group, self.params_group, self.analysis_group):
            group.setStyleSheet(self.group_style)

        # Buttons
        for btn in (self.load_btn, self.reload_btn, self.load_params_btn, self.export_btn,
                     self.param_btn, self.input_btn, self.output_btn):
            btn.setStyleSheet(self.btn_style)

        # Primary buttons
        for btn in (self.segment_btn, self.detect_btn, self.analyze_btn, self.process_batch_btn):
            btn.setStyleSheet(self.primary_btn_style)

        # Tabs
        self.tabs.setStyleSheet(self.tab_style)

        # Progress bar
        self.progress_bar.setStyleSheet(self.progressbar_style)

        # Spinboxes
        self.batch_size_spinner.setStyleSheet(self.spinner_style)

        # Checkboxes
        for cb in (self.white_bg_cb, self.toggle_img_cb, self.dark_line_cb,
                   self.extend_line_cb, self.overlay_fibres_cb, self.overlay_gaps_cb,
                   self.stats_cb, self.scores_cb):
            cb.setStyleSheet(self.checkbox_style)

        # Path edits
        for edit in (self.param_file_path, self.input_folder_path, self.output_folder_path):
            edit.setStyleSheet(self.path_edit_style)

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
