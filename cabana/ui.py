import cv2
import torch
import argparse
import numpy as np
import tifffile as tiff
import imageio.v3 as iio
from .log import Log
from .detector import FibreDetector
from .segmenter import generate_rois, segment_image
from skimage.feature import peak_local_max
from .batch import BatchProcessor
from sklearn.metrics.pairwise import euclidean_distances

from PyQt5.QtWidgets import (QSlider, QWidget, QSplitter, QSplitterHandle,
                             QMenu, QAction, QFileDialog, QMessageBox,
                             QProgressBar, QSizePolicy, QPushButton)
from PyQt5.QtCore import Qt, QSize, QEvent, QPoint, QRect, QPropertyAnimation, QEasingCurve, pyqtProperty
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QDragEnterEvent, QDropEvent, QImage, QBrush, QFont
from PyQt5.QtCore import QThread, pyqtSignal


SEED = 0
torch.use_deterministic_algorithms(True)

# Color scheme — mutable dict, repopulated by apply_theme()
from .themes import THEMES, DEFAULT_THEME

COLORS = {}

def apply_theme(theme_name: str) -> None:
    """Replace COLORS contents with the given theme's QColor values."""
    theme_data = THEMES[theme_name]
    COLORS.clear()
    for key, value in theme_data.items():
        COLORS[key] = QColor(*value)

apply_theme(DEFAULT_THEME)

FONT_SIZES = {
    'base': 12,
    'small': 11,
    'title': 14,
}


def color_to_stylesheet(color: QColor) -> str:
    """Convert QColor to stylesheet color string, with alpha support."""
    if color.alpha() < 255:
        return f"rgba({color.red()}, {color.green()}, {color.blue()}, {color.alpha() / 255:.2f})"
    return f"rgb({color.red()}, {color.green()}, {color.blue()})"


def generate_spinner_style():
    """Generate a styled QSpinBox with theme colors."""
    bg_color = COLORS['dock']
    text_color = COLORS['text']
    border_color = COLORS['border']
    highlight_color = COLORS['highlight']
    return f"""
        QSpinBox, QDoubleSpinBox {{
            background-color: rgb({bg_color.red()}, {bg_color.green()}, {bg_color.blue()});
            color: rgb({text_color.red()}, {text_color.green()}, {text_color.blue()});
            border: 1px solid rgb({border_color.red()}, {border_color.green()}, {border_color.blue()});
            border-radius: 4px;
            padding: 3px;
            min-width: 50px;
            font-size: {FONT_SIZES['base']}px;
        }}

        QSpinBox::up-button, QSpinBox::down-button,
        QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
            background-color: rgb({bg_color.red()}, {bg_color.green()}, {bg_color.blue()});
            border: 1px solid rgb({border_color.red()}, {border_color.green()}, {border_color.blue()});
            border-radius: 2px;
        }}

        QSpinBox::up-button:hover, QSpinBox::down-button:hover,
        QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {{
            background-color: rgb({highlight_color.red()}, {highlight_color.green()}, {highlight_color.blue()});
        }}

        QSpinBox::up-arrow {{
                    image: none;
                    width: 0;
                    height: 0;
                    border-style: solid;
                    border-width: 0 5px 5px 5px;
                    border-bottom: 6px solid rgb({highlight_color.red()}, {highlight_color.green()}, {highlight_color.blue()});
                }}

        QSpinBox::down-arrow {{
            image: none;
            width: 0;
            height: 0;
            border-style: solid;
            border-width: 5px 5px 0 5px;
            border-top: 6px solid rgb({highlight_color.red()}, {highlight_color.green()}, {highlight_color.blue()});
        }}

        QSpinBox:focus {{
            border: 1px solid {color_to_stylesheet(highlight_color)};
        }}

        QSpinBox:disabled {{
            background-color: {color_to_stylesheet(COLORS['background'])};
            color: {color_to_stylesheet(COLORS['text_dim'])};
            border-color: {color_to_stylesheet(COLORS['border_subtle'])};
        }}
    """

def generate_button_style():
    """Generate button stylesheet with specified colors."""
    bg_color = COLORS['dock']
    text_color = COLORS['text']
    border_color = COLORS['border']
    highlight_color = COLORS['highlight']
    return f"""
        QPushButton {{
            background-color: {color_to_stylesheet(bg_color)};
            color: {color_to_stylesheet(text_color)};
            border: 1px solid {color_to_stylesheet(border_color)};
            border-radius: 4px;
            padding: 8px 12px;
            font-size: {FONT_SIZES['base']}px;
            font-weight: 600;
        }}
        QPushButton:hover {{
            background-color: {color_to_stylesheet(COLORS['hover'])};
            color: {color_to_stylesheet(highlight_color)};
            border-color: {color_to_stylesheet(highlight_color)};
        }}
        QPushButton:pressed {{
            background-color: {color_to_stylesheet(COLORS['active'])};
            color: {color_to_stylesheet(highlight_color)};
            border-color: {color_to_stylesheet(highlight_color)};
        }}
        QPushButton:disabled {{
            background-color: {color_to_stylesheet(COLORS['surface'])};
            color: {color_to_stylesheet(COLORS['text_muted'])};
            border-color: {color_to_stylesheet(COLORS['border_subtle'])};
        }}
    """


def generate_primary_button_style():
    """Generate a filled primary action button stylesheet."""
    bg_color = COLORS['highlight']
    text_color = COLORS['background']
    border_color = COLORS['highlight']
    hover_color = COLORS['highlight_hover']
    pressed_color = COLORS['highlight_dim']
    return f"""
        QPushButton {{
            background-color: {color_to_stylesheet(bg_color)};
            color: {color_to_stylesheet(text_color)};
            border: 1px solid {color_to_stylesheet(border_color)};
            border-radius: 4px;
            padding: 9px 12px;
            font-size: {FONT_SIZES['base']}px;
            font-weight: 700;
        }}
        QPushButton:hover {{
            background-color: {color_to_stylesheet(hover_color)};
            border-color: {color_to_stylesheet(hover_color)};
        }}
        QPushButton:pressed {{
            background-color: {color_to_stylesheet(pressed_color)};
            border-color: {color_to_stylesheet(pressed_color)};
        }}
        QPushButton:disabled {{
            background-color: {color_to_stylesheet(COLORS['surface'])};
            color: {color_to_stylesheet(COLORS['text_muted'])};
            border-color: {color_to_stylesheet(COLORS['border_subtle'])};
        }}
    """


def generate_progressbar_style():
    """Generate progress bar stylesheet with specified colors."""
    bg_color = COLORS['background']
    text_color = COLORS['text']
    border_color = COLORS['border']
    progress_color = COLORS['highlight']
    return f"""
        QProgressBar {{
            background-color: {color_to_stylesheet(bg_color)};
            color: {color_to_stylesheet(text_color)};
            border: 1px solid {color_to_stylesheet(border_color)};
            border-radius: 5px;
            text-align: center;
            font-size: {FONT_SIZES['small']}px;
            font-weight: 600;
            min-height: 18px;
        }}

        QProgressBar::chunk {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 {color_to_stylesheet(progress_color.darker(110))},
                stop:1 {color_to_stylesheet(progress_color)});
            border-radius: 5px;
        }}
    """


def generate_messagebox_style():
    """Generate QMessageBox stylesheet with specified colors."""
    bg_color = COLORS['background']
    text_color = COLORS['text']
    border_color = COLORS['border']
    highlight_color = COLORS['highlight']
    highlight_text_color = COLORS['background']
    button_bg_color = COLORS['dock']
    return f"""
        QMessageBox {{
            background-color: {color_to_stylesheet(bg_color)};
            color: {color_to_stylesheet(text_color)};
            border: 1px solid {color_to_stylesheet(border_color)};
            border-radius: 6px;
        }}

        QMessageBox QLabel {{
            color: {color_to_stylesheet(text_color)};
            font-size: {FONT_SIZES['base']}px;
        }}

        QMessageBox QPushButton {{
            background-color: {color_to_stylesheet(button_bg_color)};
            color: {color_to_stylesheet(text_color)};
            border: 1px solid {color_to_stylesheet(border_color)};
            border-radius: 4px;
            padding: 6px 12px;
            min-width: 80px;
            font-size: {FONT_SIZES['base']}px;
            font-weight: bold;
        }}

        QMessageBox QPushButton:hover {{
            background-color: {color_to_stylesheet(highlight_color)};
            color: {color_to_stylesheet(highlight_text_color)};
        }}

        QMessageBox QPushButton:pressed {{
            background-color: {color_to_stylesheet(highlight_color.darker(120))};
        }}

        QMessageBox QCheckBox {{
            color: {color_to_stylesheet(text_color)};
            padding: 2px;
        }}

        QMessageBox QCheckBox::indicator {{
            width: 16px;
            height: 16px;
            border: 1px solid {color_to_stylesheet(border_color)};
            border-radius: 3px;
            background-color: {color_to_stylesheet(button_bg_color)};
        }}

        QMessageBox QCheckBox::indicator:checked {{
            background-color: {color_to_stylesheet(highlight_color)};
            image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='10' viewBox='0 0 10 10'%3E%3Cpath fill='%23{highlight_text_color.red():02x}{highlight_text_color.green():02x}{highlight_text_color.blue():02x}' d='M1,5 L3.5,7.5 L9,2'/%3E%3C/svg%3E");
        }}

        QMessageBox QTextEdit {{
            background-color: {color_to_stylesheet(QColor(bg_color.red() - 5, bg_color.green() - 5, bg_color.blue() - 5))};
            color: {color_to_stylesheet(text_color)};
            border: 1px solid {color_to_stylesheet(border_color)};
            border-radius: 3px;
        }}
    """

def generate_toolbar_style():
    """Generate a compact toolbar stylesheet."""
    bg_color = COLORS['dock']
    text_color = COLORS['text']
    border_color = COLORS['border']
    highlight_color = COLORS['highlight']
    highlight_text_color = COLORS['background']
    return f"""
        QToolBar {{
            background-color: {color_to_stylesheet(bg_color)};
            border: none;
            border-bottom: 1px solid {color_to_stylesheet(border_color)};
            spacing: 2px;
            padding: 2px 4px;
        }}
        QToolButton {{
            background-color: transparent;
            color: {color_to_stylesheet(text_color)};
            border: 1px solid transparent;
            border-radius: 4px;
            padding: 4px 8px;
            font-size: {FONT_SIZES['base']}px;
            font-weight: 500;
        }}
        QToolButton:hover {{
            background-color: {color_to_stylesheet(highlight_color)};
            color: {color_to_stylesheet(highlight_text_color)};
            border-color: {color_to_stylesheet(highlight_color)};
        }}
        QToolButton:pressed {{
            background-color: {color_to_stylesheet(highlight_color.darker(120))};
            color: {color_to_stylesheet(highlight_text_color)};
        }}
        QToolButton:disabled {{
            color: {color_to_stylesheet(COLORS['text_dim'])};
        }}
        QToolBar::separator {{
            background-color: {color_to_stylesheet(border_color)};
            width: 1px;
            margin: 4px 4px;
        }}
    """


def generate_section_header_style():
    """Generate a section header label stylesheet."""
    text_color = COLORS['text']
    border_color = COLORS['border']
    return (f"color: {color_to_stylesheet(text_color)}; font-weight: 600; "
            f"font-size: {FONT_SIZES['title']}px; "
            f"padding: 4px 0px 2px 0px; "
            f"border-bottom: 1px solid {color_to_stylesheet(border_color)}; "
            f"margin-bottom: 2px;")


def generate_checkbox_style():
    """Generate a styled QCheckBox with hover and checked states."""
    text_color = COLORS['text']
    border_color = COLORS['border']
    bg_color = COLORS['dock']
    highlight_color = COLORS['highlight']
    highlight_text_color = COLORS['background']
    return f"""
        QCheckBox {{
            color: {color_to_stylesheet(text_color)};
            spacing: 6px;
            padding: 4px 6px;
            border-radius: 4px;
            font-size: {FONT_SIZES['base']}px;
        }}
        QCheckBox:hover {{
            color: {color_to_stylesheet(highlight_color)};
            background-color: {color_to_stylesheet(COLORS['hover'])};
            border-radius: 4px;
        }}
        QCheckBox::indicator {{
            width: 16px;
            height: 16px;
            border: 1px solid {color_to_stylesheet(border_color)};
            border-radius: 3px;
            background-color: {color_to_stylesheet(bg_color)};
        }}
        QCheckBox::indicator:hover {{
            border-color: {color_to_stylesheet(highlight_color)};
        }}
        QCheckBox::indicator:checked {{
            background-color: {color_to_stylesheet(highlight_color)};
            border-color: {color_to_stylesheet(highlight_color)};
            image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='10' viewBox='0 0 10 10'%3E%3Cpath fill='%23{highlight_text_color.red():02x}{highlight_text_color.green():02x}{highlight_text_color.blue():02x}' d='M1,5 L3.5,7.5 L9,2'/%3E%3C/svg%3E");
        }}
        QCheckBox:disabled {{
            color: {color_to_stylesheet(COLORS['text_muted'])};
        }}
        QCheckBox::indicator:disabled {{
            background-color: {color_to_stylesheet(COLORS['background'])};
            border-color: {color_to_stylesheet(COLORS['border_subtle'])};
        }}
        QCheckBox::indicator:checked:disabled {{
            background-color: {color_to_stylesheet(COLORS['text_muted'])};
            border-color: {color_to_stylesheet(COLORS['text_muted'])};
            image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='10' viewBox='0 0 10 10'%3E%3Cpath fill='%23{COLORS['text'].red():02x}{COLORS['text'].green():02x}{COLORS['text'].blue():02x}' d='M1,5 L3.5,7.5 L9,2'/%3E%3C/svg%3E");
        }}
    """


def generate_group_box_style():
    """Generate a QGroupBox stylesheet with thin border and inset title."""
    text_color = COLORS['text_dim']
    border_color = COLORS['border']
    return f"""
        QGroupBox {{
            border: 1px solid {color_to_stylesheet(border_color)};
            border-radius: 6px;
            margin-top: 6px;
            padding: 14px 8px 8px 8px;
            font-size: {FONT_SIZES['base']}px;
            font-weight: 600;
            color: {color_to_stylesheet(text_color)};
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 6px;
            left: 10px;
        }}
    """


def generate_page_stack_style():
    """Stylesheet for the QStackedWidget holding the analysis pages; matches
    the former tab pane (thin border, surface background)."""
    return f"""
        QStackedWidget {{
            border: 1px solid {color_to_stylesheet(COLORS['border'])};
            border-radius: 6px;
            background-color: {color_to_stylesheet(COLORS['surface'])};
        }}
    """


def generate_combo_style():
    """Stylesheet for QComboBox controls inside the dock (base font size)."""
    return (
        f"QComboBox {{ background-color: {color_to_stylesheet(COLORS['dock'])}; "
        f"color: {color_to_stylesheet(COLORS['text'])}; "
        f"border: 1px solid {color_to_stylesheet(COLORS['border'])}; "
        f"border-radius: 4px; padding: 4px 8px; "
        f"font-size: {FONT_SIZES['base']}px; }}"
        f"QComboBox:hover {{ border-color: {color_to_stylesheet(COLORS['highlight'])}; }}"
        f"QComboBox:disabled {{ color: {color_to_stylesheet(COLORS['text_muted'])}; }}"
        f"QComboBox::drop-down {{ border: none; width: 18px; }}"
        f"QComboBox::down-arrow {{ image: none; border-left: 4px solid transparent; "
        f"border-right: 4px solid transparent; "
        f"border-top: 5px solid {color_to_stylesheet(COLORS['text_dim'])}; }}"
        f"QComboBox QAbstractItemView {{ background-color: {color_to_stylesheet(COLORS['elevated'])}; "
        f"color: {color_to_stylesheet(COLORS['text'])}; "
        f"border: 1px solid {color_to_stylesheet(COLORS['border'])}; "
        f"selection-background-color: {color_to_stylesheet(COLORS['highlight'])}; "
        f"selection-color: {color_to_stylesheet(COLORS['background'])}; }}"
    )


def generate_context_menu_style():
    """Stylesheet for pop-up menus (image panel context menus) in the current theme."""
    return f"""
        QMenu {{
            background-color: {color_to_stylesheet(COLORS['dock'])};
            color: {color_to_stylesheet(COLORS['text'])};
            border: 1px solid {color_to_stylesheet(COLORS['border'])};
            border-radius: 4px;
            padding: 4px;
        }}
        QMenu::item {{
            background-color: transparent;
            padding: 6px 12px;
            border-radius: 2px;
        }}
        QMenu::item:selected {{
            background-color: {color_to_stylesheet(COLORS['highlight'])};
            color: {color_to_stylesheet(COLORS['background'])};
        }}
        QMenu::item:disabled {{
            color: {color_to_stylesheet(COLORS['border'])};
        }}
        QMenu::separator {{
            height: 1px;
            background-color: {color_to_stylesheet(COLORS['border'])};
            margin: 4px 0px;
        }}
    """


def generate_menubar_style():
    """Stylesheet for the in-window menu bar (ignored by the native macOS bar)."""
    return f"""
        QMenuBar {{
            background-color: {color_to_stylesheet(COLORS['background'])};
            color: {color_to_stylesheet(COLORS['text'])};
            border-bottom: 1px solid {color_to_stylesheet(COLORS['border'])};
            font-size: {FONT_SIZES['base']}px;
            padding: 2px 4px;
        }}
        QMenuBar::item {{
            background: transparent;
            padding: 4px 10px;
            border-radius: 4px;
        }}
        QMenuBar::item:selected {{
            background-color: {color_to_stylesheet(COLORS['hover'])};
            color: {color_to_stylesheet(COLORS['highlight'])};
        }}
        QMenu {{
            background-color: {color_to_stylesheet(COLORS['elevated'])};
            color: {color_to_stylesheet(COLORS['text'])};
            border: 1px solid {color_to_stylesheet(COLORS['border'])};
            padding: 4px;
        }}
        QMenu::item {{
            padding: 6px 24px 6px 12px;
            border-radius: 4px;
        }}
        QMenu::item:selected {{
            background-color: {color_to_stylesheet(COLORS['highlight'])};
            color: {color_to_stylesheet(COLORS['background'])};
        }}
        QMenu::item:disabled {{
            color: {color_to_stylesheet(COLORS['text_muted'])};
        }}
        QMenu::separator {{
            height: 1px;
            background: {color_to_stylesheet(COLORS['border'])};
            margin: 4px 8px;
        }}
    """


def create_separator():
    """Create a thin horizontal separator line."""
    from PyQt5.QtWidgets import QFrame
    sep = QFrame()
    sep.setFrameShape(QFrame.HLine)
    sep.setFrameShadow(QFrame.Plain)
    sep.setFixedHeight(1)
    sep.setStyleSheet(f"background-color: {color_to_stylesheet(COLORS['border'])}; border: none;")
    return sep


class PercentageProgressBar(QProgressBar):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)  # Center the text
        self.setTextVisible(True)  # Make sure text is visible

    def format(self):
        return "%p%"  # This shows the percentage with % sign

    def text(self):
        return f"{self.value()}%"  # Custom text format


class BatchProcessingWorker(QThread):
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    batch_complete = pyqtSignal()
    batch_failed = pyqtSignal(str)
    batch_cancelled = pyqtSignal()

    def __init__(self, param_file, input_folder, output_folder, batch_size=5,
                 batch_num=0, resume=False, ignore_large=False,
                 generate_stats=False, generate_scores=False, mask_dir=None):
        super().__init__()
        self.mask_dir = mask_dir
        self.param_file = param_file
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.batch_size = batch_size
        self.batch_num = batch_num
        self.resume = resume
        self.ignore_large = ignore_large
        self.generate_stats = generate_stats
        self.generate_scores = generate_scores
        self._cancel_requested = False

    def cancel(self):
        self._cancel_requested = True

    def run(self):
        self.progress_updated.emit(1)
        batch_processor = BatchProcessor(self.param_file, self.input_folder,
                                         self.output_folder, self.batch_size,
                                         self.batch_num, self.resume, self.ignore_large,
                                         self.generate_stats, self.generate_scores,
                                         mask_dir=self.mask_dir)

        batch_processor.progress_callback = self.update_progress
        batch_processor.status_callback = self.status_updated.emit
        batch_processor.cancel_check = lambda: self._cancel_requested

        try:
            was_cancelled = batch_processor.run()
        except Exception as exc:   # an exception escaping QThread.run aborts the process
            import traceback
            Log.logger.error("Batch processing failed:\n" + traceback.format_exc())
            self.batch_failed.emit(f"{type(exc).__name__}: {exc}")
            return
        if was_cancelled:
            self.batch_cancelled.emit()
        else:
            self.progress_updated.emit(100)
            self.batch_complete.emit()

    def update_progress(self, value):
        self.progress_updated.emit(value)


class GapAnalysisWorker(QThread):
    progress_updated = pyqtSignal(int)
    gap_analysis_complete = pyqtSignal(object)

    def __init__(self, image, min_gap_diameter):
        super().__init__()
        self.image = image
        self.min_gap_diameter = min_gap_diameter

    def run(self):
        min_gap_radius = self.min_gap_diameter / 2
        min_dist = int(np.max([1, min_gap_radius]))
        mask = self.image.copy()

        # set border pixels to zero to avoid partial circles
        mask[0, :] = mask[-1, :] = mask[:, :1] = mask[:, -1:] = 0

        final_circles = []
        downsample_factor = 2
        while True:
            dist_map = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

            # downsample distance map and upscale detected centers to original image size
            dist_map_downscaled = cv2.resize(dist_map, None, fx=1 / downsample_factor, fy=1 / downsample_factor)
            centers_downscaled = peak_local_max(dist_map_downscaled, min_distance=min_dist, exclude_border=False)
            centers = centers_downscaled * downsample_factor

            # centers = peak_local_max(dist_map, min_distance=min_dist, exclude_border=False)
            radius = dist_map[centers[:, 0], centers[:, 1]]

            eligible_centers = centers[radius > min_gap_radius, :]
            eligible_radius = radius[radius > min_gap_radius]
            eligible_circles = np.hstack([eligible_centers, eligible_radius[:, None]])

            if len(eligible_circles) == 0:
                break

            result = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            while len(eligible_circles) > 0:
                if eligible_circles[1:, :].size > 0:
                    pw_euclidean_dist = \
                        euclidean_distances(eligible_circles[[0], :2], eligible_circles[1:, :2])[0]
                    pw_radius_sum = eligible_circles[0, 2] + eligible_circles[1:, 2]
                    neighbor_idx = np.nonzero(pw_euclidean_dist < pw_radius_sum)[0] + 1
                    eligible_circles = np.delete(eligible_circles, neighbor_idx, axis=0)

                circle = eligible_circles[0, :]
                result = cv2.circle(result, (int(circle[1]), int(circle[0])), int(circle[2]), (0, 0, 0), -1)
                final_circles.append(eligible_circles[0, :])
                eligible_circles = np.delete(eligible_circles, 0, axis=0)

            mask = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            self.progress_updated.emit(int((np.count_nonzero(mask == 0) / float(np.prod(mask.shape[:2])))*100))

        final_result = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
        for circle in final_circles:
            final_result = cv2.circle(final_result, (int(circle[1]), int(circle[0])),
                                      int(circle[2]), (0, 95, 95), 1)
        final_result[self.image == 0] = (255, 255, 0)  # Set background to yellow
        # Emit the completed result
        self.gap_analysis_complete.emit(final_result)

class DetectionWorker(QThread):
    progress_updated = pyqtSignal(int)
    detection_complete = pyqtSignal(list)

    def __init__(self, image, args):
        super().__init__()
        self.image = image
        self.args = args

    def run(self):
        if self.args.min_line_width == self.args.max_line_width:
            line_widths = np.array([self.args.min_line_width])
        else:
            line_widths = np.arange(self.args.min_line_width, self.args.max_line_width, self.args.line_step)
        det = FibreDetector(
            line_widths= line_widths,
            low_contrast=self.args.low_contrast,
            high_contrast=self.args.high_contrast,
            dark_line=self.args.dark_line,
            extend_line=self.args.extend_line,
            correct_pos=False,
            min_len=self.args.min_length
        )

        # Normalize to uint8 if needed
        if self.image.dtype != np.uint8:
            self.image = ((self.image - self.image.min()) / (self.image.max() - self.image.min()) * 255).astype(np.uint8)

        # Convert to grayscale
        det.image = self.image.copy()
        det.gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY) if self.image.ndim == 3 else self.image
        self.progress_updated.emit(1)

        det.apply_filtering()
        self.progress_updated.emit(30)

        det.compute_line_points()
        self.progress_updated.emit(31)

        det.compute_contours()
        self.progress_updated.emit(50)

        det.compute_line_width()
        self.progress_updated.emit(70)

        det.prune_contours()
        self.progress_updated.emit(75)

        # Get results
        _, width_image, binary_contours, _, _ = det.get_results()
        self.progress_updated.emit(99)

        # Emit the completed result
        self.detection_complete.emit([width_image, binary_contours])

class SegmentationWorker(QThread):
    progress_updated = pyqtSignal(int)
    segmentation_complete = pyqtSignal(object)

    def __init__(self, image, args):
        super().__init__()
        self.ori_img = image          # RGB array as loaded by the GUI
        self.args = args

    def run(self):
        def _progress(it, max_iter):
            self.progress_updated.emit(int((it + 1.0) / max(1, max_iter) * 100))

        # segment_image expects BGR (OpenCV) input; the GUI holds RGB
        bgr = cv2.cvtColor(self.ori_img, cv2.COLOR_RGB2BGR)
        mask = segment_image(bgr, self.args, iter_callback=_progress)
        roi_img = generate_rois(self.ori_img, mask, self.args.white_background)
        self.segmentation_complete.emit(roi_img)


class TMAWorker(QThread):
    """Background worker for the TMA page.

    ``mode='preview'`` opens the slide and returns a low-resolution image of
    its first channel; ``mode='fit'`` fits the cores and matches the array
    map (reusing ``options['reader']`` when given); ``mode='export'`` writes
    the per-core images and masks. Progress is reported in percent, with a
    short status text for the status bar.
    """
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    preview_complete = pyqtSignal(object)  # dict(reader, image RGB, level)
    fit_complete = pyqtSignal(object)      # TMAPreprocessor
    export_complete = pyqtSignal(str)      # output folder
    export_cancelled = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(self, mode, preprocessor=None, slide_path=None, options=None,
                 out_dir=None, channels=None):
        super().__init__()
        self.mode = mode
        self.preprocessor = preprocessor
        self.slide_path = slide_path
        self.options = options or {}
        self.out_dir = out_dir
        self.channels = channels
        self._cancel_requested = False

    def cancel(self):
        self._cancel_requested = True

    def run(self):
        from .tma import TMAPreprocessor
        from .wsi import open_slide
        try:
            if self.mode == 'preview':
                self.progress_updated.emit(10)
                self.status_updated.emit("Opening slide…")
                reader = open_slide(self.slide_path, pixel_size_um=self.options.get('pixel_size_um'))
                # coarsest level that is still at least ~2000 px on its longer side
                level = 0
                for lv in range(reader.level_count):
                    if max(reader.level_shape(lv)) >= 2000:
                        level = lv
                self.progress_updated.emit(40)
                self.status_updated.emit("Reading preview…")
                bgr = reader.read_level(level, channel=0)
                self.progress_updated.emit(100)
                self.preview_complete.emit({'reader': reader, 'image': bgr[:, :, ::-1].copy(),
                                            'level': level})
            elif self.mode == 'fit':
                self.progress_updated.emit(5)
                self.status_updated.emit("Opening slide…")
                pre = TMAPreprocessor(self.slide_path, **self.options)
                self.progress_updated.emit(20)
                self.status_updated.emit("Fitting cores…")
                pre.fit()
                self.progress_updated.emit(80)
                self.status_updated.emit("Matching array map…")
                pre.map_to_array()
                self.progress_updated.emit(100)
                self.fit_complete.emit(pre)
            else:
                pre = self.preprocessor

                def _progress(done, total, stem):
                    self.progress_updated.emit(int(done / max(1, total) * 100))
                    self.status_updated.emit(f"Exporting {stem} ({done}/{total})")

                ok = pre.export(self.out_dir, channels=self.channels, progress=_progress,
                                cancel=lambda: self._cancel_requested)
                if ok:
                    self.export_complete.emit(self.out_dir)
                else:
                    self.export_cancelled.emit()
        except Exception as exc:   # surfaced to the user by the GUI
            self.failed.emit(str(exc))


def parse_args():
    parser = argparse.ArgumentParser(description='Self-Supervised Semantic Segmentation')
    parser.add_argument('--num_channels', default=48, type=int,
                        help='Number of channels')
    parser.add_argument('--max_iter', default=200, type=int,
                        help='Number of maximum iterations')
    parser.add_argument('--min_labels', default=2, type=int,
                        help='Minimum number of labels')
    parser.add_argument('--hue_value', default=1.0, type=float, help='Hue value of the color of interest')
    parser.add_argument('--lr', default=0.1, type=float,
                        help='Learning rate')
    parser.add_argument('--sz_filter', default=5, type=int,
                        help='CRF filter size')
    parser.add_argument('--rt', default=0.25, type=float,
                        help='Relative color threshold')
    parser.add_argument('--mode', type=str, default="both")
    parser.add_argument('--min_size', default=64, type=int,
                        help='The smallest allowable object size')
    parser.add_argument('--max_size', default=2048, type=int,
                        help='The maximal allowable image size')
    parser.add_argument('--white_background', default=True, type=bool, help='Used to set background color')
    parser.add_argument('--input', type=str, help='Input image path', required=False)
    args, _ = parser.parse_known_args()
    return args


class RangeSlider(QWidget):
    """
    Custom Range Slider widget that allows selecting a range between min and max values.
    Based on two handles that can be dragged to set the lower and upper values.
    """
    valueChanged = pyqtSignal(int, int)  # Signal to emit when values change (min, max)

    def __init__(self, orientation=Qt.Horizontal, parent=None):
        super().__init__(parent)
        self.orientation = orientation
        self.min_val = 0
        self.max_val = 100
        self.lower_value = 0
        self.upper_value = 100
        self.lower_pos = 0
        self.upper_pos = 0
        self.offset = 0
        self.moving_lower = False
        self.moving_upper = False
        self.handle_radius = 8
        self.hover_handle = None  # 0 for none, 1 for lower, 2 for upper

        # Set focus policy to accept focus and keyboard input
        self.setFocusPolicy(Qt.StrongFocus)
        # Set mouse tracking to capture hover events
        self.setMouseTracking(True)

        # Set minimum size
        if orientation == Qt.Horizontal:
            self.setMinimumSize(100, 30)
        else:
            self.setMinimumSize(30, 100)

    def setRange(self, min_val, max_val):
        """Set the range of the slider"""
        self.min_val = min_val
        self.max_val = max_val
        self.update()

    def setValues(self, lower, upper):
        """Set the current values of the slider"""
        if lower > upper:
            lower, upper = upper, lower

        self.lower_value = max(self.min_val, min(lower, self.max_val))
        self.upper_value = max(self.min_val, min(upper, self.max_val))
        self.update_positions()
        self.update()
        self.valueChanged.emit(self.lower_value, self.upper_value)

    def update_positions(self):
        """Update the pixel positions based on values"""
        if self.orientation == Qt.Horizontal:
            avail_width = self.width() - 2 * self.handle_radius
            if self.max_val == self.min_val:
                self.lower_pos = 0
                self.upper_pos = avail_width
            else:
                self.lower_pos = int(avail_width * (self.lower_value - self.min_val) / (self.max_val - self.min_val))
                self.upper_pos = int(avail_width * (self.upper_value - self.min_val) / (self.max_val - self.min_val))
        else:
            avail_height = self.height() - 2 * self.handle_radius
            if self.max_val == self.min_val:
                self.lower_pos = avail_height
                self.upper_pos = 0
            else:
                self.lower_pos = int(
                    avail_height * (1 - (self.lower_value - self.min_val) / (self.max_val - self.min_val)))
                self.upper_pos = int(
                    avail_height * (1 - (self.upper_value - self.min_val) / (self.max_val - self.min_val)))

    def paintEvent(self, event):
        """Draw the slider on screen"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Update handle positions based on current values
        self.update_positions()

        # Draw background track
        track_color = COLORS['border']
        painter.setBrush(track_color)
        painter.setPen(Qt.NoPen)

        if self.orientation == Qt.Horizontal:
            track_height = 4
            track_y = (self.height() - track_height) // 2
            painter.drawRoundedRect(self.handle_radius, track_y, self.width() - 2 * self.handle_radius, track_height, 2, 2)

            # Draw active range
            highlight_color = COLORS['highlight']
            painter.setBrush(highlight_color)
            painter.drawRoundedRect(self.handle_radius + self.lower_pos, track_y,
                             self.upper_pos - self.lower_pos, track_height, 2, 2)

            # Draw handles
            lower_handle_x = self.handle_radius + self.lower_pos
            upper_handle_x = self.handle_radius + self.upper_pos
            handle_y = self.height() // 2

            # Draw upper handle shadow
            painter.setBrush(QColor(0, 0, 0, 40))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QPoint(upper_handle_x, handle_y + 1), self.handle_radius, self.handle_radius)

            # Draw upper handle (so lower is on top)
            if self.hover_handle == 2 or self.moving_upper:
                painter.setBrush(QColor(COLORS['highlight'].lighter(130)))
                painter.setPen(QPen(COLORS['highlight'], 2))
            else:
                painter.setBrush(COLORS['highlight'])
                painter.setPen(QPen(COLORS['text'], 1))
            painter.drawEllipse(QPoint(upper_handle_x, handle_y), self.handle_radius, self.handle_radius)

            # Draw lower handle shadow
            painter.setBrush(QColor(0, 0, 0, 40))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QPoint(lower_handle_x, handle_y + 1), self.handle_radius, self.handle_radius)

            # Draw lower handle
            if self.hover_handle == 1 or self.moving_lower:
                painter.setBrush(QColor(COLORS['highlight'].lighter(130)))
                painter.setPen(QPen(COLORS['highlight'], 2))
            else:
                painter.setBrush(COLORS['highlight'])
                painter.setPen(QPen(COLORS['text'], 1))
            painter.drawEllipse(QPoint(lower_handle_x, handle_y), self.handle_radius, self.handle_radius)

        else:  # Vertical orientation
            track_width = 4
            track_x = (self.width() - track_width) // 2
            painter.drawRoundedRect(track_x, self.handle_radius, track_width, self.height() - 2 * self.handle_radius, 2, 2)

            # Draw active range
            highlight_color = COLORS['highlight']
            painter.setBrush(highlight_color)
            painter.drawRoundedRect(track_x, self.handle_radius + self.upper_pos,
                             track_width, self.lower_pos - self.upper_pos, 2, 2)

            # Draw handles
            handle_x = self.width() // 2
            lower_handle_y = self.handle_radius + self.lower_pos
            upper_handle_y = self.handle_radius + self.upper_pos

            # Draw upper handle shadow
            painter.setBrush(QColor(0, 0, 0, 40))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QPoint(handle_x, upper_handle_y + 1), self.handle_radius, self.handle_radius)

            # Draw upper handle (top)
            if self.hover_handle == 2 or self.moving_upper:
                painter.setBrush(QColor(COLORS['highlight'].lighter(130)))
                painter.setPen(QPen(COLORS['highlight'], 2))
            else:
                painter.setBrush(COLORS['highlight'])
                painter.setPen(QPen(COLORS['text'], 1))
            painter.drawEllipse(QPoint(handle_x, upper_handle_y), self.handle_radius, self.handle_radius)

            # Draw lower handle shadow
            painter.setBrush(QColor(0, 0, 0, 40))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QPoint(handle_x, lower_handle_y + 1), self.handle_radius, self.handle_radius)

            # Draw lower handle (bottom)
            if self.hover_handle == 1 or self.moving_lower:
                painter.setBrush(QColor(COLORS['highlight'].lighter(130)))
                painter.setPen(QPen(COLORS['highlight'], 2))
            else:
                painter.setBrush(COLORS['highlight'])
                painter.setPen(QPen(COLORS['text'], 1))
            painter.drawEllipse(QPoint(handle_x, lower_handle_y), self.handle_radius, self.handle_radius)

    def mousePressEvent(self, event):
        """Handle mouse press events to start dragging handles"""
        if event.button() == Qt.LeftButton:
            # Determine which handle was clicked (if any)
            handle = self.handle_at_position(event.pos())

            if handle == 1:  # Lower handle
                self.moving_lower = True
                self.offset = self.handle_radius + self.lower_pos - (
                    event.x() if self.orientation == Qt.Horizontal else event.y())
            elif handle == 2:  # Upper handle
                self.moving_upper = True
                self.offset = self.handle_radius + self.upper_pos - (
                    event.x() if self.orientation == Qt.Horizontal else event.y())
            else:
                # Click on track - move nearest handle to this position
                pos = event.x() if self.orientation == Qt.Horizontal else event.y()
                pos -= self.handle_radius  # Adjust for handle radius

                # Convert position to value
                if self.orientation == Qt.Horizontal:
                    avail_width = self.width() - 2 * self.handle_radius
                    if avail_width <= 0:
                        return
                    normalized_pos = max(0, min(1, pos / avail_width))
                    value = self.min_val + normalized_pos * (self.max_val - self.min_val)
                else:
                    avail_height = self.height() - 2 * self.handle_radius
                    if avail_height <= 0:
                        return
                    normalized_pos = max(0, min(1, 1 - (pos / avail_height)))
                    value = self.min_val + normalized_pos * (self.max_val - self.min_val)

                # Find which handle to move (closest one)
                if abs(value - self.lower_value) <= abs(value - self.upper_value):
                    self.lower_value = value
                    self.moving_lower = True
                else:
                    self.upper_value = value
                    self.moving_upper = True

                # Update both positions
                self.update_positions()
                self.update()
                self.valueChanged.emit(int(self.lower_value), int(self.upper_value))

    def mouseMoveEvent(self, event):
        """Handle mouse move events for dragging handles and hover effects"""
        if self.moving_lower or self.moving_upper:
            if self.orientation == Qt.Horizontal:
                pos = event.x() - self.offset  # Adjusted position
                avail_width = self.width() - 2 * self.handle_radius
                if avail_width <= 0:
                    return
                normalized_pos = max(0, min(1, pos / avail_width))
                value = self.min_val + normalized_pos * (self.max_val - self.min_val)
            else:
                pos = event.y() - self.offset  # Adjusted position
                avail_height = self.height() - 2 * self.handle_radius
                if avail_height <= 0:
                    return
                normalized_pos = max(0, min(1, 1 - (pos / avail_height)))
                value = self.min_val + normalized_pos * (self.max_val - self.min_val)

            if self.moving_lower:
                self.lower_value = max(self.min_val, min(value, self.upper_value))
            else:  # moving upper
                self.upper_value = max(self.lower_value, min(value, self.max_val))

            self.update_positions()
            self.update()
            self.valueChanged.emit(int(self.lower_value), int(self.upper_value))
        else:
            # Update hover state
            prev_hover = self.hover_handle
            self.hover_handle = self.handle_at_position(event.pos())
            if prev_hover != self.hover_handle:
                self.update()

    def mouseReleaseEvent(self, event):
        """Handle mouse release events to stop dragging"""
        if event.button() == Qt.LeftButton:
            self.moving_lower = False
            self.moving_upper = False

    def handle_at_position(self, pos):
        """Determine which handle (if any) is at the given position"""
        handle_size = self.handle_radius + 2  # Add a little extra for better click experience

        if self.orientation == Qt.Horizontal:
            lower_handle_x = self.handle_radius + self.lower_pos
            upper_handle_x = self.handle_radius + self.upper_pos
            handle_y = self.height() // 2

            lower_rect = QRect(lower_handle_x - handle_size, handle_y - handle_size,
                               handle_size * 2, handle_size * 2)
            upper_rect = QRect(upper_handle_x - handle_size, handle_y - handle_size,
                               handle_size * 2, handle_size * 2)

            if lower_rect.contains(pos):
                return 1
            elif upper_rect.contains(pos):
                return 2
        else:
            handle_x = self.width() // 2
            lower_handle_y = self.handle_radius + self.lower_pos
            upper_handle_y = self.handle_radius + self.upper_pos

            lower_rect = QRect(handle_x - handle_size, lower_handle_y - handle_size,
                               handle_size * 2, handle_size * 2)
            upper_rect = QRect(handle_x - handle_size, upper_handle_y - handle_size,
                               handle_size * 2, handle_size * 2)

            if lower_rect.contains(pos):
                return 1
            elif upper_rect.contains(pos):
                return 2

        return 0  # No handle at position

    def resizeEvent(self, event):
        """Handle resize events to update positions"""
        super().resizeEvent(event)
        self.update_positions()

    def minimumSizeHint(self):
        """Provide a minimum size hint for layout management"""
        if self.orientation == Qt.Horizontal:
            return QSize(30, 20)
        else:
            return QSize(20, 30)


class CustomSlider(QSlider):
    """
    Custom QSlider implementation with the same appearance as RangeSlider
    but with simplified behavior for a single handle.
    """

    def __init__(self, orientation=Qt.Horizontal, parent=None):
        super().__init__(orientation, parent)
        self.is_dragging = False
        self.handle_radius = 8
        self.hover = False

        # Set focus policy to accept focus and keyboard input
        self.setFocusPolicy(Qt.StrongFocus)
        # Set mouse tracking to capture hover events
        self.setMouseTracking(True)

        # Set minimum size
        if orientation == Qt.Horizontal:
            self.setMinimumSize(100, 30)
        else:
            self.setMinimumSize(30, 100)

    def paintEvent(self, event):
        """Draw the slider with same style as RangeSlider"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Get the current position as a fraction of the available range
        value_range = self.maximum() - self.minimum()
        if value_range == 0:
            normalized_position = 0
        else:
            normalized_position = (self.value() - self.minimum()) / value_range

        # Draw background track
        track_color = COLORS['border']
        painter.setBrush(track_color)
        painter.setPen(Qt.NoPen)

        if self.orientation() == Qt.Horizontal:
            track_height = 4
            track_y = (self.height() - track_height) // 2
            track_width = self.width() - 2 * self.handle_radius
            painter.drawRoundedRect(self.handle_radius, track_y, track_width, track_height, 2, 2)

            # Draw active portion of the track
            highlight_color = COLORS['highlight']
            painter.setBrush(highlight_color)
            handle_pos = int(normalized_position * track_width)
            painter.drawRoundedRect(self.handle_radius, track_y, handle_pos, track_height, 2, 2)

            # Draw handle
            handle_x = self.handle_radius + handle_pos
            handle_y = self.height() // 2

            # Draw subtle handle shadow
            painter.setBrush(QColor(0, 0, 0, 40))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QPoint(handle_x, handle_y + 1), self.handle_radius, self.handle_radius)

            # Set handle appearance based on hover/drag state
            if self.hover or self.is_dragging:
                painter.setBrush(QColor(COLORS['highlight'].lighter(130)))
                painter.setPen(QPen(COLORS['highlight'], 2))
            else:
                painter.setBrush(COLORS['highlight'])
                painter.setPen(QPen(COLORS['text'], 1))

            painter.drawEllipse(QPoint(handle_x, handle_y), self.handle_radius, self.handle_radius)

        else:  # Vertical orientation
            track_width = 4
            track_x = (self.width() - track_width) // 2
            track_height = self.height() - 2 * self.handle_radius
            painter.drawRoundedRect(track_x, self.handle_radius, track_width, track_height, 2, 2)

            # Draw active portion of the track
            highlight_color = COLORS['highlight']
            painter.setBrush(highlight_color)
            handle_pos = int((1 - normalized_position) * track_height)
            painter.drawRoundedRect(track_x, self.handle_radius + handle_pos, track_width, track_height - handle_pos, 2, 2)

            # Draw handle
            handle_x = self.width() // 2
            handle_y = self.handle_radius + handle_pos

            # Draw subtle handle shadow
            painter.setBrush(QColor(0, 0, 0, 40))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QPoint(handle_x, handle_y + 1), self.handle_radius, self.handle_radius)

            # Set handle appearance based on hover/drag state
            if self.hover or self.is_dragging:
                painter.setBrush(QColor(COLORS['highlight'].lighter(130)))
                painter.setPen(QPen(COLORS['highlight'], 2))
            else:
                painter.setBrush(COLORS['highlight'])
                painter.setPen(QPen(COLORS['text'], 1))

            painter.drawEllipse(QPoint(handle_x, handle_y), self.handle_radius, self.handle_radius)

    def mousePressEvent(self, event):
        """Handle mouse press events without propagation"""
        if event.button() == Qt.LeftButton:
            self.is_dragging = True
            # Calculate position and set value directly
            self.setValue(self.valueFromPosition(self.positionFromEvent(event)))
            self.update()
            event.accept()  # Accept this event to prevent propagation
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Handle mouse move events and hover effects"""
        prev_hover = self.hover

        # Check if mouse is over the handle
        handle_rect = self.handleRect()
        self.hover = handle_rect.contains(event.pos())

        if self.is_dragging:
            self.setValue(self.valueFromPosition(self.positionFromEvent(event)))
            self.update()
            event.accept()
        elif prev_hover != self.hover:
            # Update only if hover state changed
            self.update()
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """Handle mouse release events"""
        if event.button() == Qt.LeftButton and self.is_dragging:
            self.is_dragging = False
            self.update()
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def handleRect(self):
        """Get the rectangle representing the handle area"""
        handle_size = self.handle_radius + 2  # Add a little extra for better click experience

        # Get the current position as a fraction of the available range
        value_range = self.maximum() - self.minimum()
        if value_range == 0:
            normalized_position = 0
        else:
            normalized_position = (self.value() - self.minimum()) / value_range

        if self.orientation() == Qt.Horizontal:
            track_width = self.width() - 2 * self.handle_radius
            handle_pos = int(normalized_position * track_width)
            handle_x = self.handle_radius + handle_pos
            handle_y = self.height() // 2

            return QRect(handle_x - handle_size, handle_y - handle_size,
                         handle_size * 2, handle_size * 2)
        else:
            track_height = self.height() - 2 * self.handle_radius
            handle_pos = int((1 - normalized_position) * track_height)
            handle_x = self.width() // 2
            handle_y = self.handle_radius + handle_pos

            return QRect(handle_x - handle_size, handle_y - handle_size,
                         handle_size * 2, handle_size * 2)

    def positionFromEvent(self, event):
        """Convert mouse event position to normalized slider position (0-1)"""
        if self.orientation() == Qt.Horizontal:
            pos = max(self.handle_radius, min(event.x(), self.width() - self.handle_radius))
            track_width = self.width() - 2 * self.handle_radius
            if track_width <= 0:
                return 0
            return (pos - self.handle_radius) / track_width
        else:
            pos = max(self.handle_radius, min(event.y(), self.height() - self.handle_radius))
            track_height = self.height() - 2 * self.handle_radius
            if track_height <= 0:
                return 0
            return 1 - ((pos - self.handle_radius) / track_height)

    def valueFromPosition(self, normalized_position):
        """Convert normalized position (0-1) to slider value"""
        return self.minimum() + round(normalized_position * (self.maximum() - self.minimum()))

    def minimumSizeHint(self):
        """Provide a minimum size hint for layout management"""
        if self.orientation() == Qt.Horizontal:
            return QSize(30, 20)
        else:
            return QSize(20, 30)


def hex_to_hue(hex_value):
    # Convert hex to RGB
    r = int(hex_value[1:3], 16) / 255.0
    g = int(hex_value[3:5], 16) / 255.0
    b = int(hex_value[5:7], 16) / 255.0

    # Find maximum and minimum values
    max_val = max(r, g, b)
    min_val = min(r, g, b)

    # Calculate delta
    delta = max_val - min_val

    # Calculate hue
    if delta == 0:
        return 0
    elif max_val == r:
        hue = ((g - b) / delta) % 6
    elif max_val == g:
        hue = ((b - r) / delta) + 2
    else:
        hue = ((r - g) / delta) + 4

    hue /= 6.0

    return hue


class CustomSplitterHandle(QSplitterHandle):
    def __init__(self, orientation, parent=None):
        super().__init__(orientation, parent)
        self.is_hover = False
        self.setMouseTracking(True)
        # Install event filter to track mouse enter/leave events
        self.installEventFilter(self)

    def eventFilter(self, obj, event):
        """Filter events to detect mouse hover"""
        if event.type() == QEvent.Enter:
            self.is_hover = True
            self.update()
            return True
        elif event.type() == QEvent.Leave:
            self.is_hover = False
            self.update()
            return True
        return super().eventFilter(obj, event)

    def paintEvent(self, event):
        """Override paint event to draw the handle based on hover state"""
        painter = QPainter(self)

        # Set background to border color by default (blends with both dark/light themes)
        if not self.is_hover:
            # When not hovering, draw subtle border-colored background with dots
            painter.fillRect(self.rect(), COLORS['border'])

            # Set up the painter for the dots - only when hovering
            painter.setBrush(COLORS['secondary'])

            # Draw three dots in the center
            width = self.width()
            height = self.height()
            center_y = height // 2

            # Calculate positions for three dots
            if self.orientation() == Qt.Horizontal:
                # For horizontal splitter (vertical separators)
                center_x = width // 2
                dot_spacing = 6
                dot_size = 3  # Dot size

                # Draw circles for dots
                painter.drawEllipse(center_x - dot_size // 2, center_y - dot_spacing - dot_size // 2, dot_size,
                                    dot_size)
                painter.drawEllipse(center_x - dot_size // 2, center_y - dot_size // 2, dot_size, dot_size)
                painter.drawEllipse(center_x - dot_size // 2, center_y + dot_spacing - dot_size // 2, dot_size,
                                    dot_size)
            else:
                # For vertical splitter (horizontal separators)
                center_x = width // 2
                dot_spacing = 4
                dot_size = 2  # Dot size

                # Draw circles for dots
                painter.drawEllipse(center_x - dot_spacing - dot_size // 2, center_y - dot_size // 2, dot_size,
                                    dot_size)
                painter.drawEllipse(center_x - dot_size // 2, center_y - dot_size // 2, dot_size, dot_size)
                painter.drawEllipse(center_x + dot_spacing - dot_size // 2, center_y - dot_size // 2, dot_size,
                                    dot_size)
        else:
            # When hovering, draw a visible splitter bar
            # Fill with a napari border color
            painter.fillRect(self.rect(), COLORS['border'])

            # Draw a highlight line
            painter.setPen(QPen(COLORS['highlight'], 1))
            if self.orientation() == Qt.Horizontal:
                # Vertical line
                painter.drawLine(self.width() // 2, 0, self.width() // 2, self.height())
            else:
                # Horizontal line
                painter.drawLine(0, self.height() // 2, self.width(), self.height() // 2)

            # Draw the dots on top with highlight color
            painter.setPen(Qt.NoPen)
            painter.setBrush(COLORS['highlight'])

            width = self.width()
            height = self.height()
            center_y = height // 2

            # Calculate positions for three dots
            if self.orientation() == Qt.Horizontal:
                # For horizontal splitter (vertical separators)
                center_x = width // 2
                dot_spacing = 6
                dot_size = 3  # Dot size

                # Draw circles for dots
                painter.drawEllipse(center_x - dot_size // 2, center_y - dot_spacing - dot_size // 2, dot_size,
                                    dot_size)
                painter.drawEllipse(center_x - dot_size // 2, center_y - dot_size // 2, dot_size, dot_size)
                painter.drawEllipse(center_x - dot_size // 2, center_y + dot_spacing - dot_size // 2, dot_size,
                                    dot_size)
            else:
                # For vertical splitter (horizontal separators)
                center_x = width // 2
                dot_spacing = 4
                dot_size = 2  # Smaller size for thinner handle

                # Draw circles for dots
                painter.drawEllipse(center_x - dot_spacing - dot_size // 2, center_y - dot_size // 2, dot_size,
                                    dot_size)
                painter.drawEllipse(center_x - dot_size // 2, center_y - dot_size // 2, dot_size, dot_size)
                painter.drawEllipse(center_x + dot_spacing - dot_size // 2, center_y - dot_size // 2, dot_size,
                                    dot_size)


class CustomSplitter(QSplitter):
    def __init__(self, orientation, parent=None):
        super().__init__(orientation, parent)

    def createHandle(self):
        """Override to create our custom handle with the three dots"""
        return CustomSplitterHandle(self.orientation(), self)


class PanelToggleButton(QPushButton):
    """A toggle button that draws a guillemet arrow pointing left (panel visible)
    or right (panel hidden)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setFixedSize(20, 36)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("Toggle side panel")

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Background
        bg = COLORS['dock'] if not self.underMouse() else COLORS['elevated']
        painter.setBrush(bg)
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(self.rect(), 4, 4)

        # Draw guillemet
        color = COLORS['highlight'] if self.underMouse() else COLORS['secondary']
        painter.setPen(QPen(color, 1))
        font = painter.font()
        font.setPixelSize(14)
        painter.setFont(font)
        symbol = '»' if self.isChecked() else '«'
        painter.drawText(self.rect(), Qt.AlignCenter, symbol)

    def enterEvent(self, event):
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.update()
        super().leaveEvent(event)


class ImagePanel(QWidget):
    imageDropped = pyqtSignal(str)
    zoomChanged = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 300)

        # Set canvas background
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(self.backgroundRole(), COLORS['canvas'])
        self.setPalette(palette)

        # Initially, we'll just show a placeholder
        self.image = None
        self.scale_mode = Qt.KeepAspectRatio  # Scale while maintaining aspect ratio

        # Enable drag and drop
        self.setAcceptDrops(True)

        # Drag and drop state
        self.drag_active = False

        # Zoom parameters
        self.zoom_factor = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 10.0
        self.zoom_step = 0.1

        # For panning when zoomed in
        self.panning = False
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.offset_x = 0
        self.offset_y = 0

        # Enable mouse tracking for zoom and pan operations
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

        # Optional overlay drawn on top of the image in image coordinates, and an
        # editing handler that may consume mouse events (see set_edit_handler).
        self.overlay_painter = None
        self.edit_handler = None
        self._edit_drag = False

    # -- overlay and editing hooks ------------------------------------------
    def set_overlay_painter(self, painter_fn):
        """``painter_fn(painter, zoom, to_widget)`` is called after the image is
        drawn, with the painter transformed to image coordinates (use cosmetic
        pens so line widths stay constant on screen). ``to_widget(x, y)`` maps
        image to widget coordinates for text drawn with the transform reset.
        Cleared by :meth:`setImage`."""
        self.overlay_painter = painter_fn
        self.update()

    def set_edit_handler(self, handler):
        """``handler`` receives ``press/move/release/double_click(x, y, event)``
        in image coordinates and ``key(event)`` / ``context_menu(x, y, global_pos)``;
        the mouse methods return True to consume the event (no panning).
        Cleared by :meth:`setImage`."""
        self.edit_handler = handler
        self._edit_drag = False
        self.update()

    def image_origin(self):
        """Widget coordinates of the image's top-left corner and the zoom."""
        scaled_width = self.image.width() * self.zoom_factor
        scaled_height = self.image.height() * self.zoom_factor
        x = self.width() / 2 - scaled_width / 2 + self.offset_x
        y = self.height() / 2 - scaled_height / 2 + self.offset_y
        return x, y, self.zoom_factor

    def widget_to_image(self, wx, wy):
        x0, y0, z = self.image_origin()
        return (wx - x0) / z, (wy - y0) / z

    def image_to_widget(self, ix, iy):
        x0, y0, z = self.image_origin()
        return x0 + ix * z, y0 + iy * z

    def setImage(self, image, preserve_view=False):
        """
        Set an image to display - accepts either a file path string or numpy array

        Parameters:
        -----------
        image : str or numpy.ndarray
            Either a path to an image file, or a numpy array containing image data
        preserve_view : bool
            If True, keep the current zoom factor and pan offsets
        """
        if image is None:
            return

        # overlays belong to the image they were set for
        self.overlay_painter = None
        self.edit_handler = None
        self._edit_drag = False
        self.setToolTip("")

        if not preserve_view:
            # Reset offset to 0 when setting a new image
            self.offset_x = 0
            self.offset_y = 0

        if isinstance(image, str):
            # If image is a file path
            self.image = QPixmap(image)
            if not self.image.isNull():
                if not preserve_view:
                    # Calculate zoom factor to fit the image in the panel
                    self.calculateFitZoomFactor()
                self.update()
        else:
            # If image is a numpy array
            try:
                # Convert numpy array to QImage
                height, width = image.shape[:2]

                # Handle different image formats
                if len(image.shape) == 2:  # Grayscale
                    bytes_per_line = width
                    q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
                elif len(image.shape) == 3 and image.shape[2] == 3:  # RGB
                    bytes_per_line = 3 * width
                    q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
                elif len(image.shape) == 3 and image.shape[2] == 4:  # RGBA
                    bytes_per_line = 4 * width
                    q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGBA8888)
                else:
                    raise ValueError("Unsupported image format")

                # Convert QImage to QPixmap
                self.image = QPixmap.fromImage(q_image)
                if not preserve_view:
                    # Calculate zoom factor to fit the image in the panel
                    self.calculateFitZoomFactor()
                self.update()  # Trigger a repaint
            except Exception as e:
                print(f"Error converting numpy array to QPixmap: {str(e)}")
                return

    def calculateFitZoomFactor(self):
        """Calculate the zoom factor to fit the image in the panel"""
        if not self.image or self.image.isNull():
            self.zoom_factor = 1.0
            return

        # Get the dimensions of the panel and image
        panel_width = self.width()
        panel_height = self.height()
        image_width = self.image.width()
        image_height = self.image.height()

        # Calculate the zoom factor to fit the image in the panel
        # Leave a small margin (5%) around the image
        width_ratio = (panel_width * 0.95) / image_width
        height_ratio = (panel_height * 0.95) / image_height

        # Use the smaller ratio to ensure the image fits entirely
        self.zoom_factor = min(width_ratio, height_ratio)

        # Ensure the zoom factor is within bounds
        self.zoom_factor = max(self.min_zoom, min(self.zoom_factor, self.max_zoom))
        self.zoomChanged.emit(self.zoom_factor)

    def wheelEvent(self, event):
        """Handle mouse wheel events for zooming"""
        if self.image and not self.image.isNull():
            # Get mouse position
            mouse_x = event.x()
            mouse_y = event.y()

            # Calculate the point in the image that the mouse is over before zooming
            center_x = self.width() / 2
            center_y = self.height() / 2

            # Current image position
            current_image_x = center_x - (self.image.width() * self.zoom_factor / 2) + self.offset_x
            current_image_y = center_y - (self.image.height() * self.zoom_factor / 2) + self.offset_y

            # Point in image coordinates (relative to image, not widget)
            image_point_x = (mouse_x - current_image_x) / self.zoom_factor
            image_point_y = (mouse_y - current_image_y) / self.zoom_factor

            # Calculate zoom delta based on wheel movement
            delta = event.angleDelta().y() / 120  # 120 units per step

            # Calculate new zoom factor
            old_zoom = self.zoom_factor
            new_zoom = self.zoom_factor + (self.zoom_step * delta)

            # Clamp zoom factor within limits
            new_zoom = max(self.min_zoom, min(new_zoom, self.max_zoom))

            # Apply the zoom
            self.zoom_factor = new_zoom

            # Calculate new image position after zoom
            new_image_x = center_x - (self.image.width() * self.zoom_factor / 2)
            new_image_y = center_y - (self.image.height() * self.zoom_factor / 2)

            # Calculate where the image point would be after zoom
            new_point_x = new_image_x + (image_point_x * self.zoom_factor)
            new_point_y = new_image_y + (image_point_y * self.zoom_factor)

            # Adjust offset to keep the mouse point at the same screen position
            self.offset_x = mouse_x - new_point_x
            self.offset_y = mouse_y - new_point_y

            # Constrain the offset to keep the image partially visible
            self.constrain_offset()

            # Update the display
            self.update()
            self.zoomChanged.emit(self.zoom_factor)

    def mousePressEvent(self, event):
        """Handle mouse press events for editing (if a handler consumes it) or panning"""
        if self.image and not self.image.isNull():
            self.setFocus()
            if self.edit_handler is not None and event.button() == Qt.LeftButton:
                ix, iy = self.widget_to_image(event.x(), event.y())
                if self.edit_handler.press(ix, iy, event):
                    self._edit_drag = True
                    self.update()
                    return
            # Support both left and middle mouse button for panning
            if event.button() == Qt.LeftButton or event.button() == Qt.MiddleButton:
                self.panning = True
                self.pan_start_x = event.x()
                self.pan_start_y = event.y()
                self.setCursor(Qt.ClosedHandCursor)

    def mouseReleaseEvent(self, event):
        """Handle mouse release events for editing or panning"""
        if self._edit_drag and event.button() == Qt.LeftButton:
            self._edit_drag = False
            if self.edit_handler is not None:
                ix, iy = self.widget_to_image(event.x(), event.y())
                self.edit_handler.release(ix, iy, event)
            self.update()
            return
        # Support both left and middle mouse button for panning
        if event.button() == Qt.LeftButton or event.button() == Qt.MiddleButton:
            self.panning = False
            self.setCursor(Qt.ArrowCursor)

    def mouseDoubleClickEvent(self, event):
        if (self.edit_handler is not None and self.image and not self.image.isNull()
                and event.button() == Qt.LeftButton):
            ix, iy = self.widget_to_image(event.x(), event.y())
            if self.edit_handler.double_click(ix, iy, event):
                self._edit_drag = False
                self.panning = False
                self.update()
                return
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event):
        if self.edit_handler is not None and self.edit_handler.key(event):
            self.update()
            return
        super().keyPressEvent(event)

    def mouseMoveEvent(self, event):
        """Handle mouse move events for editing or panning"""
        if self.edit_handler is not None and self.image and not self.image.isNull():
            ix, iy = self.widget_to_image(event.x(), event.y())
            if self._edit_drag:
                self.edit_handler.move(ix, iy, event)
                self.update()
                return
            cursor = self.edit_handler.cursor_at(ix, iy)
            if cursor is not None and not self.panning:
                self.setCursor(cursor)
        if self.panning and self.image and not self.image.isNull():
            # Calculate the movement delta
            delta_x = event.x() - self.pan_start_x
            delta_y = event.y() - self.pan_start_y

            # Update the offset with the delta
            self.offset_x += delta_x
            self.offset_y += delta_y

            # Constrain the offset to keep the image partially visible
            self.constrain_offset()

            # Update the starting position for the next movement
            self.pan_start_x = event.x()
            self.pan_start_y = event.y()

            # Update the display
            self.update()

    def constrain_offset(self):
        """Constrain the offset to keep the image partially visible"""
        if self.image and not self.image.isNull():
            # Calculate the maximum allowed offsets to keep at least 25% of the image visible
            image_width = self.image.width() * self.zoom_factor
            image_height = self.image.height() * self.zoom_factor

            max_offset_x = (image_width / 2) + (self.width() / 4)
            max_offset_y = (image_height / 2) + (self.height() / 4)

            # Constrain the offset
            self.offset_x = max(-max_offset_x, min(self.offset_x, max_offset_x))
            self.offset_y = max(-max_offset_y, min(self.offset_y, max_offset_y))

    def resizeEvent(self, event):
        """Handle resize events to ensure the image is properly scaled"""
        super().resizeEvent(event)

        # Recalculate the fit zoom factor when the panel is resized
        if self.image and not self.image.isNull():
            self.calculateFitZoomFactor()

        self.update()

    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter events for image files"""
        # Check if the dragged data contains URLs (files)
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                # Convert URL to local path
                file_path = url.toLocalFile()
                # Check if it's an image file
                if self.isImageFile(file_path):
                    self.drag_active = True
                    event.acceptProposedAction()
                    self.update()
                    return
        event.ignore()

    def dragLeaveEvent(self, event):
        """Handle drag leave events"""
        self.drag_active = False
        self.update()

    def dragMoveEvent(self, event):
        """Handle drag move events"""
        # Accept the event to allow the drop
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        """Handle drop events for image files"""
        self.drag_active = False

        # Get the dropped file's URL
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if self.isImageFile(file_path):
                img = tiff.imread(file_path) if file_path.lower().endswith(('.tif', '.tiff')) else iio.imread(file_path)
                if img.dtype != np.uint8:
                    img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)

                self.setImage(img)
                self.imageDropped.emit(file_path)
                event.acceptProposedAction()
                return

        # If we get here, no valid image was found
        self.update()

    def isImageFile(self, file_path):
        """Check if the file is a supported image format"""
        valid_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
        return any(file_path.lower().endswith(ext) for ext in valid_extensions)

    def paintEvent(self, event):
        """Override paint event to draw the image and drag & drop visual feedback"""
        painter = QPainter(self)

        if self.image and not self.image.isNull():
            # Get the dimensions of the scaled image
            scaled_width = self.image.width() * self.zoom_factor
            scaled_height = self.image.height() * self.zoom_factor

            # Calculate position including offset for panning
            center_x = self.width() / 2
            center_y = self.height() / 2
            x = center_x - (scaled_width / 2) + self.offset_x
            y = center_y - (scaled_height / 2) + self.offset_y

            # Create a scaled version of the image
            scaled_pixmap = self.image.scaled(
                int(scaled_width),
                int(scaled_height),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )

            # Draw the image at the calculated position
            painter.drawPixmap(int(x), int(y), scaled_pixmap)

            if self.overlay_painter is not None:
                painter.save()
                painter.setRenderHint(QPainter.Antialiasing, True)
                painter.translate(x, y)
                painter.scale(self.zoom_factor, self.zoom_factor)
                try:
                    self.overlay_painter(painter, self.zoom_factor,
                                         lambda ix, iy: (x + ix * self.zoom_factor, y + iy * self.zoom_factor))
                finally:
                    painter.restore()
        else:
            # Draw placeholder text
            painter.setPen(COLORS['highlight'])
            font = QFont("Arial", 20)  # Family and size
            font.setBold(True)  # Optional: make it bold
            painter.setFont(font)
            text = "Drag & Drop Image Here or Use 'Load Image' Button"
            painter.drawText(self.rect(), Qt.AlignCenter, text)

        # Draw drag and drop highlight overlay when active
        if self.drag_active:
            # Dashed border
            pen = QPen(COLORS['highlight'])
            pen.setWidth(3)
            pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
            painter.drawRect(self.rect().adjusted(4, 4, -4, -4))

    def contextMenuEvent(self, event):
        """Handle right-click context menu events"""
        # Only show context menu if there's an image loaded
        if self.image and not self.image.isNull():
            if self.edit_handler is not None:
                ix, iy = self.widget_to_image(event.x(), event.y())
                if self.edit_handler.context_menu(ix, iy, event.globalPos()):
                    self.update()
                    return
            context_menu = QMenu(self)

            # Apply the same styling as other UI elements
            context_menu.setStyleSheet(generate_context_menu_style())

            # Create download action
            download_action = QAction("Save Image As...", self)
            download_action.triggered.connect(self.saveImage)
            context_menu.addAction(download_action)

            # Show context menu at cursor position
            context_menu.exec_(event.globalPos())

    def saveImage(self):
        """Save the current image to a file"""
        if not self.image or self.image.isNull():
            return

        # Open file dialog to choose save location
        file_dialog = QFileDialog(self)
        file_dialog.setAcceptMode(QFileDialog.AcceptSave)
        file_dialog.setDefaultSuffix("png")
        file_dialog.setNameFilter("PNG Files (*.png);;JPEG Files (*.jpg);;BMP Files (*.bmp);;TIFF Files (*.tiff)")
        file_dialog.setWindowTitle("Save Image As")

        # Apply dark theme styling to file dialog
        file_dialog.setStyleSheet(f"""
            QFileDialog {{
                background-color: {color_to_stylesheet(COLORS['background'])};
                color: {color_to_stylesheet(COLORS['text'])};
            }}
            QFileDialog QListView {{
                background-color: {color_to_stylesheet(COLORS['canvas'])};
                color: {color_to_stylesheet(COLORS['text'])};
                border: 1px solid {color_to_stylesheet(COLORS['border'])};
            }}
            QFileDialog QTreeView {{
                background-color: {color_to_stylesheet(COLORS['canvas'])};
                color: {color_to_stylesheet(COLORS['text'])};
                border: 1px solid {color_to_stylesheet(COLORS['border'])};
            }}
            QFileDialog QLineEdit {{
                background-color: {color_to_stylesheet(COLORS['dock'])};
                color: {color_to_stylesheet(COLORS['text'])};
                border: 1px solid {color_to_stylesheet(COLORS['border'])};
                border-radius: 4px;
                padding: 4px;
            }}
            QFileDialog QPushButton {{
                {generate_button_style()}
            }}
        """)

        if file_dialog.exec_() == QFileDialog.Accepted:
            file_path = file_dialog.selectedFiles()[0]

            try:
                # Save the image
                success = self.image.save(file_path)

                if success:
                    # Show success message
                    msg_box = QMessageBox(self)
                    msg_box.setIcon(QMessageBox.Information)
                    msg_box.setWindowTitle("Save Successful")
                    msg_box.setText(f"Image saved successfully to:\n{file_path}")
                    msg_box.setStyleSheet(generate_messagebox_style())
                    msg_box.exec_()
                else:
                    # Show error message
                    msg_box = QMessageBox(self)
                    msg_box.setIcon(QMessageBox.Critical)
                    msg_box.setWindowTitle("Save Failed")
                    msg_box.setText("Failed to save the image. Please check the file path and permissions.")
                    msg_box.setStyleSheet(generate_messagebox_style())
                    msg_box.exec_()

            except Exception as e:
                # Show error message for any exceptions
                msg_box = QMessageBox(self)
                msg_box.setIcon(QMessageBox.Critical)
                msg_box.setWindowTitle("Save Error")
                msg_box.setText(f"An error occurred while saving the image:\n{str(e)}")
                msg_box.setStyleSheet(generate_messagebox_style())
                msg_box.exec_()


class ToggleButton(QWidget):
    """Custom toggle button widget that responds properly to resize events"""

    toggled = pyqtSignal(bool)  # Signal emitted when state changes

    def __init__(self, parent=None):
        super().__init__(parent)

        # Fixed size so every toggle renders identically regardless of the
        # surrounding layout (rows previously compressed some instances)
        self.setFixedSize(60, 25)

        self.setCheckable(True)
        self._checked = False
        self._enabled = True

        # Initialize slider position BEFORE creating animation
        self._slider_position = 0.0

        # Animation for smooth toggle
        self._animation = QPropertyAnimation(self, b"slider_position")
        self._animation.setDuration(150)
        self._animation.setEasingCurve(QEasingCurve.OutCubic)

        # Colors — read from COLORS at paint time for theme support
        self.bg_color_off = QColor(100, 100, 100)

    def setCheckable(self, checkable):
        """Set whether the button is checkable"""
        self._checkable = checkable

    def isCheckable(self):
        """Return whether the button is checkable"""
        return getattr(self, '_checkable', True)

    def setChecked(self, checked):
        """Set the checked state"""
        if self._checked != checked:
            self._checked = checked
            self._animate_toggle()
            self.toggled.emit(checked)
            self.update()

    def isChecked(self):
        """Return the checked state"""
        return self._checked

    def setEnabled(self, enabled):
        """Override setEnabled to update appearance"""
        super().setEnabled(enabled)
        self._enabled = enabled
        self.update()

    def toggle(self):
        """Toggle the button state"""
        if self.isCheckable():
            self.setChecked(not self._checked)

    def _animate_toggle(self):
        """Animate the slider movement"""
        start_pos = self._slider_position
        end_pos = 1.0 if self._checked else 0.0

        # Only animate if there's actually a change in position
        if abs(start_pos - end_pos) > 0.01:
            self._animation.setStartValue(start_pos)
            self._animation.setEndValue(end_pos)
            self._animation.start()
        else:
            # If no animation needed, just set the final position
            self._slider_position = end_pos
            self.update()

    @pyqtProperty(float)
    def slider_position(self):
        """Get current slider position for animation"""
        return getattr(self, '_slider_position', 0.0)

    @slider_position.setter
    def slider_position(self, position):
        """Set slider position for animation"""
        self._slider_position = position
        self.update()

    def resizeEvent(self, event):
        """Handle resize events properly"""
        super().resizeEvent(event)
        # Force a repaint when resized to ensure proper scaling
        self.update()

    def sizeHint(self):
        """Provide a reasonable default size"""
        return QSize(60, 25)

    def minimumSizeHint(self):
        """Provide minimum size"""
        return QSize(50, 20)

    def heightForWidth(self, width):
        """Calculate height based on width to maintain proportions"""
        return int(width * 0.43)

    def mousePressEvent(self, event):
        """Handle mouse press events"""
        if event.button() == Qt.LeftButton and self._enabled:
            self.toggle()

    def paintEvent(self, event):
        """Custom paint event for the toggle button"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Get current dimensions
        w = self.width()
        h = self.height()

        # Calculate proportional dimensions
        radius = h * 0.45
        track_radius = h * 0.4
        slider_radius = h * 0.35

        # Track rectangle
        track_rect = QRect(int(radius), int((h - track_radius * 2) / 2),
                           int(w - radius * 2), int(track_radius * 2))

        # Background color based on state
        if self._checked:
            bg_color = QColor(COLORS['highlight'])
        else:
            bg_color = QColor(self.bg_color_off)

        # Adjust opacity if disabled
        if not self._enabled:
            bg_color.setAlpha(100)

        # Draw track background
        painter.setBrush(QBrush(bg_color))
        painter.setPen(QPen(COLORS['background'], 1))
        painter.drawRoundedRect(track_rect, track_radius, track_radius)

        # Calculate slider position
        slider_x = radius + (w - radius * 2 - slider_radius * 2) * self._slider_position
        slider_y = (h - slider_radius * 2) / 2

        # Draw slider with color based on state
        slider_color = QColor(COLORS['text'])

        if not self._enabled:
            slider_color = QColor(slider_color)  # Create copy to avoid modifying original
            slider_color.setAlpha(150)

        painter.setBrush(QBrush(slider_color))
        painter.setPen(QPen(COLORS['background'], 1))
        painter.drawEllipse(int(slider_x), int(slider_y),
                            int(slider_radius * 2), int(slider_radius * 2))
