import os
os.environ["NUMEXPR_MAX_THREADS"] = "20"
import sys
import yaml
import imageio.v3 as iio
from pathlib import Path
from .utils import join_path

from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QSpinBox,
                             QVBoxLayout, QHBoxLayout, QTabWidget, QCheckBox,
                             QPushButton, QFileDialog, QSizePolicy, QColorDialog, QMessageBox)
from PyQt5.QtGui import QIcon, QPalette

from .ui import *


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set window properties
        self.setWindowTitle("Cabana-GUI")
        self.setMinimumSize(800, 600)

        # Make window full screen when starting
        self.showMaximized()  # This will maximize the window to full screen

        # Apply Napari-inspired theme to the entire application
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
        self.dock_contents.setAutoFillBackground(True)

        # Set Napari dock color
        dock_palette = self.dock_contents.palette()
        dock_palette.setColor(QPalette.Window, COLORS['dock'])
        self.dock_contents.setPalette(dock_palette)

        self.dock_layout = QVBoxLayout(self.dock_contents)

        # Create a label with Napari-style
        dock_label = QLabel("Configuration Panel")
        dock_label.setStyleSheet(
            f"color: {color_to_stylesheet(COLORS['text'])}; font-weight: bold; font-size: 14px;")
        self.dock_layout.addWidget(dock_label)

        # File layout
        file_layout = QHBoxLayout()
        self._setup_styles()

        # Add a button to load an image
        self.load_btn = QPushButton("Load New Image")
        self.load_btn.setStyleSheet(self.btn_style)
        self.load_btn.clicked.connect(self.load_image)
        file_layout.addWidget(self.load_btn)

        self.reload_btn = QPushButton("Reload Image")
        self.reload_btn.clicked.connect(self.reload_image)
        self.reload_btn.setEnabled(False)
        self.reload_btn.setStyleSheet(self.btn_style)
        self.reload_btn.setToolTip("Reload the original image")
        file_layout.addWidget(self.reload_btn)

        # Export button
        self.export_btn = QPushButton("Export Params")
        self.export_btn.setStyleSheet(self.btn_style)
        self.export_btn.clicked.connect(self.export_parameters)
        file_layout.addWidget(self.export_btn)

        # Add buttons to main layout of left dock
        self.dock_layout.addLayout(file_layout)

        # Create tab widget
        self.tabs = QTabWidget()
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
        self.tabs.addTab(self.seg_tab, "Segmentation")
        self.tabs.addTab(self.det_tab, "Fibre Detection")
        self.tabs.addTab(self.gap_tab, "Gap Analysis")
        self.tabs.addTab(self.bat_tab, "Batch Processing")

        self.dock_layout.addWidget(self.tabs)

        # Add a spacer to push content to the top
        self.dock_layout.addStretch()

        # Add progress bar
        self.progress_bar = PercentageProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet(self.progressbar_style)
        self.dock_layout.addWidget(self.progress_bar)

        # Create and add the image panel to the splitter
        self.image_panel = ImagePanel()
        self.image_panel.imageDropped.connect(self.load_original_image)

        content_layout = QVBoxLayout(self.image_panel)

        # Create toggle button
        self.toggle_button = QPushButton("|☰")
        self.toggle_button.setCheckable(True)
        self.toggle_button.setFixedWidth(40)
        self.toggle_button.setStyleSheet(self.btn_style)
        self.toggle_button.clicked.connect(self.toggle_panel)

        # Add toggle button and some content to the main area
        content_layout.addWidget(self.toggle_button, 0, Qt.AlignLeft)
        content_layout.addStretch(1)

        # Add widgets to splitter
        self.splitter.addWidget(self.dock_contents)
        self.splitter.addWidget(self.image_panel)

        # Set initial sizes (adjust as needed)
        self.splitter.setSizes([200, 800])

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

    def toggle_panel(self):
        if self.panel_visible:
            # Hide panel
            self.dock_contents.hide()
            self.toggle_button.setText("☰|")
        else:
            # Show panel
            self.dock_contents.show()
            self.toggle_button.setText("|☰")

        self.panel_visible = not self.panel_visible


    def _setup_styles(self) -> None:
        """Set up all style sheets"""
        # Button style
        self.btn_style = generate_button_style()

        # Tab style
        self.tab_style = generate_tab_style()

        # Progress bar style
        self.progressbar_style = generate_progressbar_style()

        # Spinnere style
        self.spinner_style = generate_spinner_style()

        # Messagebox style
        self.msgbox_style = generate_messagebox_style()

    def setup_batch_processing_tab(self):
        """Set up the batch processing tab UI"""
        layout = QVBoxLayout()

        # Parameter file selection
        param_layout = QHBoxLayout()
        param_label = QLabel("Parameter File:")
        param_layout.addWidget(param_label)

        self.param_file_path = QLabel("Not selected")
        self.param_file_path.setStyleSheet("color: gray")
        param_layout.addWidget(self.param_file_path, 1)

        self.param_btn = QPushButton("Select")
        self.param_btn.clicked.connect(self.select_param_file)
        self.param_btn.setStyleSheet(self.btn_style)
        param_layout.addWidget(self.param_btn)

        layout.addLayout(param_layout)

        # Input folder selection
        input_layout = QHBoxLayout()
        input_label = QLabel("Input Folder:")
        input_layout.addWidget(input_label)

        self.input_folder_path = QLabel("Not selected")
        self.input_folder_path.setStyleSheet("color: gray;")
        input_layout.addWidget(self.input_folder_path, 1)

        self.input_btn = QPushButton("Select")
        self.input_btn.clicked.connect(self.select_input_folder)
        self.input_btn.setStyleSheet(self.btn_style)
        input_layout.addWidget(self.input_btn)

        layout.addLayout(input_layout)

        # Output folder selection
        output_layout = QHBoxLayout()
        output_label = QLabel("Output Folder:")
        output_layout.addWidget(output_label)

        self.output_folder_path = QLabel("Not selected")
        self.output_folder_path.setStyleSheet("color: gray;")
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

        # Batch processing button
        self.process_batch_btn = QPushButton("Process Batch")
        self.process_batch_btn.clicked.connect(self.run_batch_processing)
        self.process_batch_btn.setEnabled(False)
        self.process_batch_btn.setStyleSheet(self.btn_style)
        layout.addWidget(self.process_batch_btn)

        layout.addStretch()
        self.bat_tab.setLayout(layout)

    def select_param_file(self):
        """Open a file dialog to select a parameter file"""
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("YAML Files (*.yml *.yaml)")
        file_dialog.setWindowTitle("Select Parameter File")
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setDirectory(os.path.expanduser('~/Documents'))

        # Style the dialog to match theme
        file_dialog.setStyleSheet(f"""
            QFileDialog {{
                background-color: {color_to_stylesheet(COLORS['background'])};
                color: {color_to_stylesheet(COLORS['text'])};
            }}
        """)

        if file_dialog.exec_():
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                self.param_file = selected_files[0]
                self.param_file_path.setText(
                    selected_files[0] if len(selected_files[0]) <= 40 else selected_files[0][:16] + "..." + selected_files[0][-16:]
                )
                self.param_file_path.setToolTip(selected_files[0])
                self.param_file_path.setStyleSheet("color: white;")
                self._check_batch_processing_ready()

    def select_input_folder(self):
        """Open a file dialog to select an input folder"""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Input Folder", str(Path(self.param_file).parent.parent), QFileDialog.ShowDirsOnly
        )

        if folder:
            self.input_folder = folder
            self.input_folder_path.setText(
                folder if len(folder) <= 40 else folder[:16] + "..." + folder[-16:]
            )
            self.input_folder_path.setToolTip(folder)
            self.input_folder_path.setStyleSheet("color: white;")
            self._check_batch_processing_ready()

    def select_output_folder(self):
        """Open a file dialog to select an output folder"""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Output Folder", "", QFileDialog.ShowDirsOnly
        )

        if folder:
            self.output_folder = folder
            self.output_folder_path.setText(
                folder if len(folder) <= 40 else folder[:16] + "..." + folder[-16:]
            )
            self.output_folder_path.setToolTip(folder)
            self.output_folder_path.setStyleSheet("color: white;")
            self._check_batch_processing_ready()

    def _check_batch_processing_ready(self):
        """Check if all necessary paths are selected to enable batch processing"""
        is_ready = hasattr(self, 'param_file') and hasattr(self, 'input_folder') and hasattr(self, 'output_folder')
        self.process_batch_btn.setEnabled(is_ready)

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
        self.show_progress_bar()

        resume, batch_size, batch_num, ignore_large = self._check_batch_running_status()

        if not resume:
            batch_size = self.batch_size_spinner.value()
            batch_num = 0

        self.batch_worker = BatchProcessingWorker(
            self.param_file, self.input_folder, self.output_folder,
            batch_size, batch_num, resume, ignore_large
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

        # # Show a message box to notify the user
        # QMessageBox.information(
        #     self, "Batch Processing Complete",
        #     "Batch processing has been completed successfully."
        # )

    def setup_segmentation_tab(self):
        """Set up the segmentation tab UI"""
        layout = QVBoxLayout()

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
        self.color_btn.setStyleSheet("background-color: #f53282; font-weight: bold; ")
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
        self.color_thresh_value.setFixedWidth(30)
        self.color_thresh_value.setAlignment(Qt.AlignVCenter | Qt.AlignRight)
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
        self.num_labels_value.setFixedWidth(25)
        self.num_labels_value.setAlignment(Qt.AlignVCenter | Qt.AlignRight)
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
        self.max_iters_value.setFixedWidth(25)
        self.max_iters_value.setAlignment(Qt.AlignVCenter | Qt.AlignRight)
        max_iters_layout.addWidget(self.max_iters_value)
        layout.addLayout(max_iters_layout)

        # White background checkbox
        h_layout = QHBoxLayout()
        self.white_bg_cb = QCheckBox("White Background")
        self.white_bg_cb.setChecked(True)
        self.white_bg_cb.setStyleSheet(f"color: {color_to_stylesheet(COLORS['text'])}; ")
        self.white_bg_cb.stateChanged.connect(self.update_white_bg)
        self.white_bg_cb.setToolTip("Enable this option when detecting dark fibres in bright backgrounds.")
        h_layout.addWidget(self.white_bg_cb)

        # Add toggle checkbox for comparing with the original image
        self.toggle_img_cb = QCheckBox("Compare With Original")
        self.toggle_img_cb.setChecked(False)
        self.toggle_img_cb.setStyleSheet(f"color: {color_to_stylesheet(COLORS['text'])}; ")
        self.toggle_img_cb.clicked.connect(self.compare_image)
        self.toggle_img_cb.setToolTip("Toggle to compare with the original image.")
        h_layout.addStretch()
        h_layout.addWidget(self.toggle_img_cb)

        layout.addLayout(h_layout)

        # Segmentation button
        self.segment_btn = QPushButton("Segment")
        self.segment_btn.clicked.connect(self.run_segmentation)
        self.segment_btn.setEnabled(False)
        self.segment_btn.setStyleSheet(self.btn_style)
        layout.addWidget(self.segment_btn)

        layout.addStretch()
        self.seg_tab.setLayout(layout)

    def setup_detection_tab(self):
        """Set up the detection tab UI with range sliders"""
        layout = QVBoxLayout()

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
        self.line_width_value.setFixedWidth(45)
        self.line_width_value.setAlignment(Qt.AlignVCenter | Qt.AlignRight)
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
        self.line_step_value.setFixedWidth(20)
        self.line_step_value.setAlignment(Qt.AlignVCenter | Qt.AlignRight)
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
        self.contrast_value.setFixedWidth(65)
        self.contrast_value.setAlignment(Qt.AlignVCenter | Qt.AlignRight)
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
        self.min_length_value.setFixedWidth(20)
        self.min_length_value.setAlignment(Qt.AlignVCenter | Qt.AlignRight)
        min_length_layout.addWidget(self.min_length_value)
        layout.addLayout(min_length_layout)

        # Checkboxes
        checkbox_layout = QHBoxLayout()
        self.dark_line_cb = QCheckBox("Dark Line")
        self.dark_line_cb.setChecked(True)
        self.dark_line_cb.setStyleSheet(f"color: {color_to_stylesheet(COLORS['text'])};")
        self.dark_line_cb.stateChanged.connect(self.update_dark_line)
        self.dark_line_cb.setToolTip("Enable this option to detect dark fibers on bright backgrounds.")
        checkbox_layout.addWidget(self.dark_line_cb)

        self.extend_line_cb = QCheckBox("Extend Line")
        self.extend_line_cb.setChecked(False)
        self.extend_line_cb.setStyleSheet(f"color: {color_to_stylesheet(COLORS['text'])}; ")
        self.extend_line_cb.stateChanged.connect(self.update_extend_line)
        self.extend_line_cb.setToolTip("Enable to detect fibers near junctions.")
        checkbox_layout.addWidget(self.extend_line_cb)

        self.overlay_fibres_cb = QCheckBox("Overlay Fibres")
        self.overlay_fibres_cb.setChecked(True)
        self.overlay_fibres_cb.setStyleSheet(f"color: {color_to_stylesheet(COLORS['text'])}; ")
        self.overlay_fibres_cb.stateChanged.connect(self.update_overlay_fibres)
        self.overlay_fibres_cb.setToolTip("Toggle to overlay detected fibres on the image.")
        checkbox_layout.addWidget(self.overlay_fibres_cb)

        layout.addLayout(checkbox_layout)

        # Detection button
        self.detect_btn = QPushButton("Detect")
        self.detect_btn.clicked.connect(self.run_detection)
        self.detect_btn.setEnabled(False)
        self.detect_btn.setStyleSheet(self.btn_style)
        layout.addWidget(self.detect_btn)

        layout.addStretch()
        self.det_tab.setLayout(layout)

    def setup_gap_analysis_tab(self):
        """Set up the gap analysis tab UI"""
        layout = QVBoxLayout()

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
        self.min_gap_value.setFixedWidth(25)
        self.min_gap_value.setAlignment(Qt.AlignVCenter | Qt.AlignRight)
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
        self.max_hdm_value.setFixedWidth(25)
        self.max_hdm_value.setAlignment(Qt.AlignVCenter | Qt.AlignRight)
        max_hdm_layout.addWidget(self.max_hdm_value)
        layout.addLayout(max_hdm_layout)

        # Overlay checkbox
        self.overlay_gaps_cb = QCheckBox("Overlay Gaps")
        self.overlay_gaps_cb.setChecked(False)
        self.overlay_gaps_cb.setStyleSheet(f"color: {color_to_stylesheet(COLORS['text'])}; ")
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
        self.analyze_btn.setStyleSheet(self.btn_style)
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
            self.color_btn.setStyleSheet(f"background-color: {hex_color}; ")

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

    def load_original_image(self, path):
        """Load the original image from a file path"""
        try:
            self.ori_img = iio.imread(path)
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
        except Exception as e:
            self.ori_img = None
            self.img_path = None
            self.segment_btn.setEnabled(False)
            self.detect_btn.setEnabled(False)
            self.reload_btn.setEnabled(False)
            print(f"Error loading image: {e}")


    def load_image(self):
        """Open a file dialog to select an image file"""
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg *.bmp *.tiff *.tif)")
        file_dialog.setWindowTitle("Select Image")
        file_dialog.setFileMode(QFileDialog.ExistingFile)

        # Style the file dialog to match the theme
        file_dialog.setStyleSheet(f"""
            QFileDialog {{
                background-color: {color_to_stylesheet(COLORS['background'])};
                color: {color_to_stylesheet(COLORS['text'])};
            }}
            QFileDialog QLabel {{ 
                color: {color_to_stylesheet(COLORS['text'])}; 
            }}
            QFileDialog QPushButton {{ 
                background-color: {color_to_stylesheet(COLORS['dock'])};
                color: {color_to_stylesheet(COLORS['text'])};
                border: 1px solid {color_to_stylesheet(COLORS['border'])};
                border-radius: 4px;
                padding: 4px 8px;
            }}
            QFileDialog QPushButton:hover {{ 
                background-color: {color_to_stylesheet(COLORS['highlight'])};
                color: {color_to_stylesheet(COLORS['background'])};
            }}
        """)

        if file_dialog.exec_():
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                self.image_panel.setImage(selected_files[0])
                self.ori_img = iio.imread(selected_files[0])
                self.img_path = selected_files[0]
                if len(self.ori_img.shape) < 3:
                    self.ori_img = np.repeat(self.ori_img[:, :, np.newaxis], 3, axis=2)
                elif len(self.ori_img.shape) == 3 and self.ori_img.shape[2] > 3:
                    # Remove alpha channel if present
                    self.ori_img = self.ori_img[:, :, :3]

                self.seg_img = None
                self.frb_img = None
                self.wdt_img = None
                self.gap_img = None
                self.gap_ovl = None
                self.segment_btn.setEnabled(True)
                self.detect_btn.setEnabled(True)
                self.reload_btn.setEnabled(True)

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
            self.image_panel.setImage(image)
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
            self.image_panel.setImage(image)
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
            self.image_panel.setImage(image)
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
        # Set Napari palette
        palette = QPalette()

        # Set color group
        palette.setColor(QPalette.Window, COLORS['background'])
        palette.setColor(QPalette.WindowText, COLORS['text'])
        palette.setColor(QPalette.Base, COLORS['background'])
        palette.setColor(QPalette.AlternateBase, COLORS['dock'])
        palette.setColor(QPalette.ToolTipBase, COLORS['dock'])
        palette.setColor(QPalette.ToolTipText, COLORS['text'])
        palette.setColor(QPalette.Text, COLORS['text'])
        palette.setColor(QPalette.Button, COLORS['dock'])
        palette.setColor(QPalette.ButtonText, COLORS['text'])
        palette.setColor(QPalette.BrightText, COLORS['highlight'])
        palette.setColor(QPalette.Link, COLORS['highlight'])
        palette.setColor(QPalette.Highlight, COLORS['highlight'])
        palette.setColor(QPalette.HighlightedText, COLORS['background'])

        # Apply the palette
        self.setPalette(palette)

        # Additional stylesheet for fine-tuning
        self.setStyleSheet(f"""
            QToolTip {{ 
                color: {color_to_stylesheet(COLORS['text'])}; 
                background-color: {color_to_stylesheet(COLORS['dock'])}; 
                border: 1px solid {color_to_stylesheet(COLORS['border'])}; 
            }}
            QLabel {{ color: {color_to_stylesheet(COLORS['text'])}; }}
        """)
