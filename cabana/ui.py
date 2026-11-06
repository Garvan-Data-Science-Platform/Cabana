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
                             QProgressBar, QSizePolicy, QPushButton, QSpinBox,
                             QAbstractSpinBox, QAbstractSlider, QComboBox, QScrollBar,
                             QAbstractScrollArea, QApplication, QFrame, QLabel, QLineEdit,
                             QHBoxLayout, QVBoxLayout, QLayout, QToolButton, QDialog, QButtonGroup)
from PyQt5.QtCore import Qt, QSize, QEvent, QPoint, QRect, QRectF, QPropertyAnimation, QEasingCurve, pyqtProperty
from PyQt5.QtGui import (QPixmap, QPainter, QPen, QColor, QDragEnterEvent, QDropEvent, QImage, QBrush,
                         QFont, QFontMetrics, QIcon)
from PyQt5.QtCore import QThread, pyqtSignal, QObject


SEED = 0
torch.use_deterministic_algorithms(True)

# Color scheme — mutable dict, repopulated by apply_theme()
from .themes import THEMES, DEFAULT_THEME
from .io import MIN_MAX_SIZE

COLORS = {}

def apply_theme(theme_name: str) -> None:
    """Replace COLORS contents with the given theme's QColor values."""
    theme_data = THEMES[theme_name]
    COLORS.clear()
    for key, value in theme_data.items():
        COLORS[key] = QColor(*value)

apply_theme(DEFAULT_THEME)

FONT_SIZES = {
    'base': 13,
    'small': 12,
    'title': 15,
}


def color_to_stylesheet(color: QColor) -> str:
    """Convert QColor to stylesheet color string, with alpha support."""
    if color.alpha() < 255:
        return f"rgba({color.red()}, {color.green()}, {color.blue()}, {color.alpha() / 255:.2f})"
    return f"rgb({color.red()}, {color.green()}, {color.blue()})"


def is_light_theme() -> bool:
    """True when the current theme has dark text on a light background."""
    return COLORS['text'].lightness() < COLORS['background'].lightness()


def _mix(a: QColor, b: QColor, f: float) -> QColor:
    """``a`` blended toward ``b`` by ``f`` (0 = a, 1 = b)."""
    return QColor(round(a.red() + (b.red() - a.red()) * f),
                  round(a.green() + (b.green() - a.green()) * f),
                  round(a.blue() + (b.blue() - a.blue()) * f))


def contrast_ratio(a: QColor, b: QColor) -> float:
    """WCAG contrast ratio of two colours (1 to 21)."""
    def lum(c):
        def ch(v):
            v /= 255
            return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
        return 0.2126 * ch(c.red()) + 0.7152 * ch(c.green()) + 0.0722 * ch(c.blue())
    hi, lo = sorted((lum(a), lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def _with_contrast(color: QColor, backgrounds, target: float, toward: QColor) -> QColor:
    """``color`` moved toward ``toward`` just far enough to reach ``target``
    contrast against every one of ``backgrounds``."""
    for step in range(21):
        c = _mix(color, toward, step / 20)
        if all(contrast_ratio(c, bg) >= target for bg in backgrounds):
            return c
    return QColor(toward)


def panel_colors() -> dict:
    """Colours of the card-style side panel, derived from the current theme:
    cards sit one layer above the panel, inputs are inset (filled) on the
    card, and each card header is a tinted band."""
    c = COLORS
    light = is_light_theme()
    panel = c['dock'] if light else c['canvas']
    card = c['elevated'] if light else c['surface']
    field = c['background']
    # filled buttons: the accent, darkened where its label would fall below 4.5:1
    button = _with_contrast(c['highlight'], [c['background']], 4.5, QColor(0, 0, 0) if light else c['highlight'])
    shade = QColor(0, 0, 0) if light else QColor(255, 255, 255)
    return {
        'panel': panel,
        'card': card,
        'header': _mix(card, c['dock'], 0.7) if light else _mix(card, panel, 0.8),
        'card_border': c['border_subtle'] if light else c['border'],
        'field': field,
        # outlines of fields, checkboxes and buttons: 3:1 against the card (WCAG 1.4.11)
        'outline': _with_contrast(c['border'], [card, field], 3.0, c['text_dim']),
        'border_hover': _with_contrast(_mix(c['border'], c['text_dim'], 0.5), [card, field], 3.5, c['text']),
        # secondary text (paths, notes) and placeholders: 4.5:1 on cards and fields
        'faint': _with_contrast(_mix(c['text_dim'], c['text_muted'], 0.45), [card, field], 4.5, c['text']),
        'placeholder': _with_contrast(c['text_muted'], [field, card], 4.5, c['text']),
        'handle': c['elevated'] if light else c['text'],
        'button': button,
        'button_hover': _mix(button, shade, 0.12) if button != c['highlight'] else c['highlight_hover'],
        'button_pressed': _mix(button, QColor(0, 0, 0), 0.2),
    }


# Line icons on a 14 x 14 grid, written once per colour as SVG files that
# stylesheets (url()) and QIcon can load.
_ICON_SHAPES = {
    'check': '<polyline points="3.3,7.3 5.9,9.8 10.7,4.4" fill="none" stroke="{c}" stroke-width="1.8" '
             'stroke-linecap="round" stroke-linejoin="round"/>',
    'chevron-down': '<polyline points="4,5.5 7,8.5 10,5.5" fill="none" stroke="{c}" stroke-width="1.4" '
                    'stroke-linecap="round" stroke-linejoin="round"/>',
    'chevron-right': '<polyline points="5.5,4 8.5,7 5.5,10" fill="none" stroke="{c}" stroke-width="1.4" '
                     'stroke-linecap="round" stroke-linejoin="round"/>',
    'minus': '<line x1="3.5" y1="7" x2="10.5" y2="7" stroke="{c}" stroke-width="1.4" stroke-linecap="round"/>',
    'plus': '<path d="M3.5 7H10.5M7 3.5V10.5" stroke="{c}" stroke-width="1.4" stroke-linecap="round"/>',
    'folder': '<path d="M1.5 3H5.3L6.8 4.6H12.5V11.5H1.5Z" fill="none" stroke="{c}" stroke-width="1.3" '
              'stroke-linejoin="round"/>',
    'file': '<path d="M3.2 1.5H8.3L11 4.2V12.5H3.2Z M8.3 1.5V4.2H11" fill="none" stroke="{c}" '
            'stroke-width="1.3" stroke-linejoin="round"/>',
    'play': '<path d="M3.5 2L12.5 7L3.5 12Z" fill="{c}"/>',
    'clear': '<circle cx="7" cy="7" r="5.5" fill="{c}"/><path d="M5 5L9 9M9 5L5 9" stroke="{c2}" '
             'stroke-width="1.3" stroke-linecap="round"/>',
    # side-panel pages (the page rail)
    'page-start': '<path d="M2 6.6L7 2.6L12 6.6V12H8.6V9H5.4V12H2Z"/>',
    'page-tma': ''.join(f'<circle cx="{x}" cy="{y}" r="1.35"/>' for x in (3.2, 7, 10.8) for y in (3.2, 7, 10.8)),
    'page-brf': '<circle cx="7" cy="7" r="5.2"/><path d="M7 1.8A5.2 5.2 0 0 1 7 12.2Z" fill="{c}"/>',
    'page-seg': '<path d="M7 1.8C7 1.8 3 6.4 3 8.9A4 4 0 0 0 11 8.9C11 6.4 7 1.8 7 1.8Z"/>',
    'page-fibres': '<path d="M1.6 4.2C3.4 2.4 5 5.6 7 3.8S10.6 2.4 12.4 4.2M1.6 7.6C3.4 5.8 5 9 7 7.2'
                   'S10.6 5.8 12.4 7.6M1.6 11C3.4 9.2 5 12.4 7 10.6S10.6 9.2 12.4 11"/>',
    'page-gaps': '<rect x="1.8" y="1.8" width="10.4" height="10.4" rx="2.2"/><circle cx="7" cy="7" r="2.9"/>',
    'page-batch': '<rect x="4" y="1.8" width="8.2" height="7.2" rx="1.4"/><path d="M1.8 4.6V11.2A1 1 0 0 0 2.8 12.2H9.6"/>',
    'page-help': '<circle cx="7" cy="7" r="5.5"/><path d="M5.3 5.5A1.8 1.8 0 1 1 7.7 7.2C7.2 7.4 7 7.8 7 8.3V8.6"/>'
                 '<circle cx="7" cy="10.4" r="0.45" fill="{c}"/>',
}


def icon_path(name: str, color: QColor, color2: QColor = None) -> str:
    """Path of an SVG file drawing icon ``name`` in ``color`` (``color2`` for
    the cut-out of two-tone icons), with forward slashes for stylesheets."""
    import os
    import tempfile
    folder = os.path.join(tempfile.gettempdir(), "cabana_icons")
    os.makedirs(folder, exist_ok=True)
    c, c2 = color.name(), (color2 or color).name()
    path = os.path.join(folder, f"{name}_{c[1:]}_{c2[1:]}.svg")
    if not os.path.isfile(path):
        body = _ICON_SHAPES[name].format(c=c, c2=c2)
        if name.startswith('page-'):          # line icons: one stroke style for the whole set
            body = (f'<g fill="none" stroke="{c}" stroke-width="1.25" stroke-linecap="round" '
                    f'stroke-linejoin="round">{body}</g>')
        with open(path, "w") as f:
            f.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 14 14">{body}</svg>')
    return path.replace("\\", "/")


def themed_icon(name: str, color: QColor, color2: QColor = None):
    from PyQt5.QtGui import QIcon
    return QIcon(icon_path(name, color, color2))


def generate_spinner_style():
    """Spin boxes as inset fields with − and + steppers on the right."""
    pc = panel_colors()
    field, text = color_to_stylesheet(pc['field']), color_to_stylesheet(COLORS['text'])
    plus, minus = icon_path('plus', COLORS['text_dim']), icon_path('minus', COLORS['text_dim'])
    plus_off, minus_off = icon_path('plus', COLORS['text_muted']), icon_path('minus', COLORS['text_muted'])
    return f"""
        QSpinBox, QDoubleSpinBox {{
            background-color: {field};
            color: {text};
            border: 1px solid {color_to_stylesheet(pc['outline'])};
            border-radius: 5px;
            padding: 0px 28px 0px 8px;      /* Qt reserves one button width more */
            min-height: 26px;
            font-size: {FONT_SIZES['base']}px;
            selection-background-color: {color_to_stylesheet(COLORS['highlight'])};
        }}
        QSpinBox:hover, QDoubleSpinBox:hover {{ border-color: {color_to_stylesheet(pc['border_hover'])}; }}
        QSpinBox:focus, QDoubleSpinBox:focus {{ border-color: {color_to_stylesheet(COLORS['highlight'])}; }}
        QSpinBox::up-button, QDoubleSpinBox::up-button,
        QSpinBox::down-button, QDoubleSpinBox::down-button {{
            subcontrol-origin: border;
            subcontrol-position: center right;
            width: 24px;
            height: 24px;
            border: none;
            border-radius: 4px;
            background: transparent;
        }}
        QSpinBox::up-button, QDoubleSpinBox::up-button {{ right: 1px; }}
        QSpinBox::down-button, QDoubleSpinBox::down-button {{ right: 25px; }}
        QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
        QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
            background: {color_to_stylesheet(COLORS['border'])};
        }}
        QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{ image: url({plus}); width: 11px; height: 11px; }}
        QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{ image: url({minus}); width: 11px; height: 11px; }}
        QSpinBox::up-arrow:disabled, QSpinBox::up-arrow:off,
        QDoubleSpinBox::up-arrow:disabled, QDoubleSpinBox::up-arrow:off {{ image: url({plus_off}); }}
        QSpinBox::down-arrow:disabled, QSpinBox::down-arrow:off,
        QDoubleSpinBox::down-arrow:disabled, QDoubleSpinBox::down-arrow:off {{ image: url({minus_off}); }}
        QSpinBox:disabled, QDoubleSpinBox:disabled {{
            color: {color_to_stylesheet(COLORS['text_muted'])};
        }}
    """


def generate_button_style():
    """Secondary buttons: outlined, no fill until hovered."""
    pc = panel_colors()
    return f"""
        QPushButton {{
            background-color: transparent;
            color: {color_to_stylesheet(COLORS['text'])};
            border: 1px solid {color_to_stylesheet(pc['outline'])};
            border-radius: 6px;
            padding: 5px 12px;
            min-height: 18px;
            font-size: {FONT_SIZES['base']}px;
            font-weight: 500;
        }}
        QPushButton:hover {{
            background-color: {color_to_stylesheet(pc['field'])};
            border-color: {color_to_stylesheet(pc['border_hover'])};
        }}
        QPushButton:pressed {{
            background-color: {color_to_stylesheet(COLORS['border'])};
        }}
        QPushButton:focus {{ border: 2px solid {color_to_stylesheet(COLORS['highlight'])}; }}
        QPushButton:disabled {{
            color: {color_to_stylesheet(COLORS['text_muted'])};
            border-color: {color_to_stylesheet(COLORS['border_subtle'])};
        }}
    """


def generate_primary_button_style():
    """The main action of a section: filled with the theme's accent."""
    pc = panel_colors()
    return f"""
        QPushButton {{
            background-color: {color_to_stylesheet(pc['button'])};
            color: {color_to_stylesheet(COLORS['background'])};
            border: none;
            border-radius: 8px;
            padding: 8px 12px;
            min-height: 18px;
            font-size: {FONT_SIZES['base']}px;
            font-weight: 600;
        }}
        QPushButton:hover {{ background-color: {color_to_stylesheet(pc['button_hover'])}; }}
        QPushButton:pressed {{ background-color: {color_to_stylesheet(pc['button_pressed'])}; }}
        QPushButton:focus {{ border: 2px solid {color_to_stylesheet(COLORS['text'])}; }}
        QPushButton:disabled {{
            background-color: {color_to_stylesheet(pc['field'])};
            color: {color_to_stylesheet(COLORS['text_muted'])};
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
    """Checkboxes: an inset box that fills with the accent and a tick."""
    pc = panel_colors()
    tick = icon_path('check', COLORS['background'])
    tick_off = icon_path('check', pc['field'])
    return f"""
        QCheckBox {{
            color: {color_to_stylesheet(COLORS['text'])};
            spacing: 8px;
            padding: 2px 0px;
            background: transparent;
            font-size: {FONT_SIZES['base']}px;
        }}
        QCheckBox::indicator {{
            width: 15px;
            height: 15px;
            border: 1px solid {color_to_stylesheet(pc['outline'])};
            border-radius: 4px;
            background-color: {color_to_stylesheet(pc['field'])};
        }}
        QCheckBox::indicator:hover {{ border-color: {color_to_stylesheet(COLORS['highlight'])}; }}
        QCheckBox:focus {{ color: {color_to_stylesheet(COLORS['highlight'])}; }}
        QCheckBox::indicator:focus {{ border: 2px solid {color_to_stylesheet(COLORS['highlight'])}; }}
        QCheckBox::indicator:checked {{
            background-color: {color_to_stylesheet(COLORS['highlight'])};
            border-color: {color_to_stylesheet(COLORS['highlight'])};
            image: url({tick});
        }}
        QCheckBox:disabled {{ color: {color_to_stylesheet(COLORS['text_muted'])}; }}
        QCheckBox::indicator:disabled {{ border-color: {color_to_stylesheet(COLORS['border_subtle'])}; }}
        QCheckBox::indicator:checked:disabled {{
            background-color: {color_to_stylesheet(COLORS['text_muted'])};
            border-color: {color_to_stylesheet(COLORS['text_muted'])};
            image: url({tick_off});
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


def generate_scroll_area_style():
    """Stylesheet for the scroll area around the analysis pages. Styling the
    scroll bars makes Qt draw them as ordinary bars in their own column
    (never as macOS transient overlays on top of the content), slim, without
    arrows, and with a small gap between the page and the bar."""
    handle = color_to_stylesheet(COLORS['text_dim'])
    hover = color_to_stylesheet(COLORS['highlight'])
    return f"""
        QScrollArea {{ background: transparent; border: none; }}
        QScrollBar:vertical {{
            background: transparent; width: 12px; margin: 0 0 0 4px; border: none;
        }}
        QScrollBar::handle:vertical {{
            background: {handle}; min-height: 24px; border-radius: 4px;
        }}
        QScrollBar::handle:vertical:hover {{ background: {hover}; }}
        QScrollBar:horizontal {{
            background: transparent; height: 12px; margin: 4px 0 0 0; border: none;
        }}
        QScrollBar::handle:horizontal {{
            background: {handle}; min-width: 24px; border-radius: 4px;
        }}
        QScrollBar::handle:horizontal:hover {{ background: {hover}; }}
        QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
        QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}
    """


def generate_page_stack_style():
    """The analysis pages sit directly on the panel; their cards carry the surfaces."""
    return "QStackedWidget { border: none; background: transparent; }"


def generate_combo_style():
    """Drop-downs as inset fields with a chevron."""
    pc = panel_colors()
    chevron = icon_path('chevron-down', COLORS['text_dim'])
    return (
        f"QComboBox {{ background-color: {color_to_stylesheet(pc['field'])}; "
        f"color: {color_to_stylesheet(COLORS['text'])}; "
        f"border: 1px solid {color_to_stylesheet(pc['outline'])}; "
        f"border-radius: 5px; padding: 0px 8px; min-height: 26px; "
        f"font-size: {FONT_SIZES['base']}px; }}"
        f"QComboBox:hover {{ border-color: {color_to_stylesheet(pc['border_hover'])}; }}"
        f"QComboBox:focus {{ border-color: {color_to_stylesheet(COLORS['highlight'])}; }}"
        f"QComboBox:disabled {{ color: {color_to_stylesheet(COLORS['text_muted'])}; }}"
        f"QComboBox::drop-down {{ border: none; width: 26px; }}"
        f"QComboBox::down-arrow {{ image: url({chevron}); width: 12px; height: 12px; }}"
        f"QComboBox QAbstractItemView {{ background-color: {color_to_stylesheet(COLORS['elevated'])}; "
        f"color: {color_to_stylesheet(COLORS['text'])}; "
        f"border: 1px solid {color_to_stylesheet(COLORS['border'])}; "
        f"selection-background-color: {color_to_stylesheet(COLORS['highlight'])}; "
        f"selection-color: {color_to_stylesheet(COLORS['background'])}; }}"
    )


def generate_path_edit_style():
    """Read-only path fields (PathEdit paints the name and folder itself)."""
    pc = panel_colors()
    return (
        f"QLineEdit {{ background-color: {color_to_stylesheet(pc['field'])}; color: transparent; "
        f"border: 1px solid {color_to_stylesheet(pc['outline'])}; border-radius: 5px; "
        f"padding: 0px 4px; min-height: 26px; font-size: {FONT_SIZES['base']}px; }}"
        f"QLineEdit:hover {{ border-color: {color_to_stylesheet(pc['border_hover'])}; }}"
        f"QLineEdit:focus {{ border: 2px solid {color_to_stylesheet(COLORS['highlight'])}; }}"
    )


def generate_panel_style():
    """Side-panel stylesheet: section cards, their header bands, row labels,
    value readouts, notes and option lists (all selected by object name)."""
    pc = panel_colors()
    card, border = color_to_stylesheet(pc['card']), color_to_stylesheet(pc['card_border'])
    faint, small = color_to_stylesheet(pc['faint']), FONT_SIZES['small']
    return f"""
        QWidget#dockContents {{ background-color: {color_to_stylesheet(pc['panel'])}; }}
        QFrame#sectionCard {{ background-color: {card}; border: 1px solid {border}; border-radius: 10px; }}
        QFrame#sectionHeader {{
            background-color: {color_to_stylesheet(pc['header'])};
            border: none; border-bottom: 1px solid {border};
            border-top-left-radius: 9px; border-top-right-radius: 9px;
        }}
        QFrame#sectionHeader:focus {{ border: 2px solid {color_to_stylesheet(COLORS['highlight'])}; }}
        QFrame#sectionHeader[collapsed="true"] {{
            border-bottom: none; border-bottom-left-radius: 9px; border-bottom-right-radius: 9px;
        }}
        QLabel#sectionTitle {{ color: {color_to_stylesheet(COLORS['text'])}; background: transparent;
            font-size: {FONT_SIZES['base'] + 1}px; font-weight: 600; }}
        QLabel#rowValue, QLabel#note, QLabel#optionNote {{
            color: {faint}; background: transparent; font-size: {small}px; }}
        QLabel#rowLabel, QLabel#optionTitle {{ color: {color_to_stylesheet(COLORS['text_dim'])};
            background: transparent; font-size: {FONT_SIZES['base']}px; }}
        QLabel#optionTitle {{ font-size: {small}px; }}
        QFrame#cardDivider, QFrame#optionSep {{ background-color: {border}; border: none; }}
        QFrame#optionList {{ background-color: {color_to_stylesheet(pc['field'])}; border: none;
            border-radius: 5px; }}
    """


def generate_dialog_style():
    """Card-style dialogs (About, What's New, the tour prompt)."""
    pc = panel_colors()
    border = color_to_stylesheet(pc['card_border'])
    return f"""
        QDialog#cardDialog {{ background-color: {color_to_stylesheet(pc['card'])}; }}
        QFrame#dialogHeader {{ background-color: {color_to_stylesheet(pc['header'])};
            border: none; border-bottom: 1px solid {border}; }}
        QLabel#dialogTitle {{ color: {color_to_stylesheet(COLORS['text'])}; background: transparent;
            font-size: {FONT_SIZES['title'] + 1}px; font-weight: 600; }}
        QLabel#dialogText {{ color: {color_to_stylesheet(COLORS['text'])}; background: transparent;
            font-size: {FONT_SIZES['base'] + 1}px; }}
        QLabel#dialogNote {{ color: {color_to_stylesheet(pc['faint'])}; background: transparent;
            font-size: {FONT_SIZES['base']}px; }}
        QLabel#dialogBullet {{ color: {color_to_stylesheet(COLORS['highlight'])}; background: transparent;
            font-size: {FONT_SIZES['base'] + 1}px; }}
        QFrame#cardDivider {{ background-color: {border}; border: none; }}
    """


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


# Knob radius (px) of CustomSlider and RangeSlider
SLIDER_HANDLE_RADIUS = 7


def create_separator():
    """Create a thin horizontal separator line."""
    from PyQt5.QtWidgets import QFrame
    sep = QFrame()
    sep.setFrameShape(QFrame.HLine)
    sep.setFrameShadow(QFrame.Plain)
    sep.setFixedHeight(1)
    sep.setStyleSheet(f"background-color: {color_to_stylesheet(COLORS['border'])}; border: none;")
    return sep


def add_clear_action(edit, on_clear, tip="Clear"):
    """A small clear icon inside a (read-only) path field, shown only while the
    field holds a path; clicking it calls ``on_clear``. Returns the action."""
    from PyQt5.QtWidgets import QLineEdit
    pc = panel_colors()
    action = edit.addAction(themed_icon('clear', pc['faint'], pc['field']), QLineEdit.TrailingPosition)
    edit._clear_action = action          # PathEdit.restyle() recolours it
    action.setToolTip(tip)
    action.triggered.connect(on_clear)
    action.setVisible(bool(edit.text()))
    edit.textChanged.connect(lambda text: action.setVisible(bool(text)))
    return action


class PathEdit(QLineEdit):
    """Read-only path field: the file or folder name, then its parent folder
    in a fainter colour. ``pick_action`` is the trailing folder (or file) icon
    that chooses a new path; a click anywhere on the field triggers it too.
    ``text()`` is the full path, as for any QLineEdit."""

    def __init__(self, text="", icon="folder", parent=None):
        super().__init__(text, parent)
        self.setReadOnly(True)
        self._icon = icon
        self._placeholder = ""
        self.pick_action = self.addAction(QIcon(), QLineEdit.TrailingPosition)
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.TabFocus)       # Tab reaches it; Enter or Space opens the picker
        self.restyle()

    def setPlaceholderText(self, text):
        # drawn by paintEvent: the stylesheet hides the line edit's own text
        self._placeholder = text
        self.update()

    def placeholderText(self):
        return self._placeholder

    def restyle(self):
        """Apply the current theme (call after a theme change)."""
        self.setStyleSheet(generate_path_edit_style())
        self.pick_action.setIcon(themed_icon(self._icon, COLORS['text_dim']))
        clear = getattr(self, '_clear_action', None)
        if clear is not None:
            pc = panel_colors()
            clear.setIcon(themed_icon('clear', pc['faint'], pc['field']))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.isEnabled():
            self.pick_action.trigger()
            event.accept()
            return
        super().mousePressEvent(event)

    def focusInEvent(self, event):
        super().focusInEvent(event)
        self.deselect()                        # the hidden text must not show a selection

    def keyPressEvent(self, event):
        key = event.key()
        if key in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space) and self.isEnabled():
            self.pick_action.trigger()
            return
        clear = getattr(self, '_clear_action', None)
        if (key in (Qt.Key_Delete, Qt.Key_Backspace) and clear is not None
                and clear.isVisible() and clear.isEnabled()):
            clear.trigger()
            return
        super().keyPressEvent(event)

    def _text_rect(self):
        r = self.rect().adjusted(9, 0, -6, 0)
        for button in self.findChildren(QToolButton):        # the trailing icons
            if button.isVisible():
                r.setRight(min(r.right(), button.geometry().left() - 6))
        return r

    def paintEvent(self, event):
        super().paintEvent(event)
        import os
        pc = panel_colors()
        painter = QPainter(self)
        r = self._text_rect()
        text = self.text()
        if not text or text == "Not selected":
            painter.setPen(pc['placeholder'])
            label = self._placeholder or text
            painter.drawText(r, Qt.AlignVCenter | Qt.AlignLeft,
                             QFontMetrics(self.font()).elidedText(label, Qt.ElideRight, r.width()))
            return
        path = text.rstrip("/\\") or text
        name, folder = os.path.basename(path) or path, os.path.dirname(path)
        home = os.path.expanduser("~")
        if folder.startswith(home):
            folder = "~" + folder[len(home):]
        name_font = QFont(self.font())
        name_font.setWeight(QFont.Medium)
        fm = QFontMetrics(name_font)
        name = fm.elidedText(name, Qt.ElideMiddle, r.width())
        painter.setFont(name_font)
        painter.setPen(COLORS['text'] if self.isEnabled() else COLORS['text_muted'])
        painter.drawText(r, Qt.AlignVCenter | Qt.AlignLeft, name)
        used = fm.horizontalAdvance(name) + 8
        if folder and used < r.width() - 24:
            folder_font = QFont(self.font())
            folder_font.setPixelSize(FONT_SIZES['small'])
            rest = QRect(r.left() + used, r.top(), r.width() - used, r.height())
            painter.setFont(folder_font)
            painter.setPen(pc['faint'])
            painter.drawText(rest, Qt.AlignVCenter | Qt.AlignLeft,
                             QFontMetrics(folder_font).elidedText(folder, Qt.ElideLeft, rest.width()))


class PathDot(QWidget):
    """The dot before a path row: filled when its field holds a path, hollow
    when it is empty."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(22)
        self.filled = True

    def set_filled(self, filled):
        self.filled = bool(filled)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        accent = COLORS['highlight']
        cy = self.height() / 2
        painter.setBrush(accent if self.filled else panel_colors()['card'])
        painter.setPen(QPen(accent if self.filled else panel_colors()['faint'], 1.5))
        painter.drawEllipse(QRectF(2.5, cy - 4.5, 9, 9))


class SectionCard(QFrame):
    """A collapsible section card: a tinted header band with the title, the
    rows, and a footer (behind a divider) for the section's main action and
    its result. Styled by ``generate_panel_style`` through object names."""

    LABEL_WIDTH = 118
    VALUE_WIDTH = 58

    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setObjectName("sectionCard")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(1, 1, 1, 1)
        outer.setSpacing(0)

        self.header = QFrame()
        self.header.setObjectName("sectionHeader")
        self.header.setCursor(Qt.PointingHandCursor)
        self.header.setFocusPolicy(Qt.TabFocus)
        self.header.mousePressEvent = lambda event: self.set_open(not self._open)
        self.header.keyPressEvent = self._header_key
        head = QHBoxLayout(self.header)
        head.setContentsMargins(10, 10, 12, 10)
        head.setSpacing(6)
        self.chevron = QLabel()
        self.chevron.setFixedSize(14, 14)
        self.title_label = QLabel(title)
        self.title_label.setObjectName("sectionTitle")
        self.header_right = QHBoxLayout()
        self.header_right.setSpacing(8)
        head.addWidget(self.chevron, 0, Qt.AlignVCenter)
        head.addWidget(self.title_label, 1)
        head.addLayout(self.header_right)
        outer.addWidget(self.header)

        self.body = QWidget()
        self.body_layout = QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(12, 14, 12, 16)
        self.body_layout.setSpacing(14)
        outer.addWidget(self.body)

        self.footer = QWidget()
        footer_outer = QVBoxLayout(self.footer)
        footer_outer.setContentsMargins(0, 0, 0, 0)
        footer_outer.setSpacing(0)
        divider = QFrame()
        divider.setObjectName("cardDivider")
        divider.setFixedHeight(1)
        footer_outer.addWidget(divider)
        self.footer_layout = QVBoxLayout()
        self.footer_layout.setContentsMargins(12, 14, 12, 16)
        self.footer_layout.setSpacing(8)
        footer_outer.addLayout(self.footer_layout)
        self.footer.setVisible(False)
        outer.addWidget(self.footer)

        self._paths = None
        self._open = True
        self.restyle()

    # --- content -------------------------------------------------------
    def _row_label(self, text, width):
        label = QLabel(text)
        label.setObjectName("rowLabel")
        label.setFixedWidth(width)
        return label

    def add_row(self, label, control, value=None, stretch=True):
        """A labelled row; ``control`` is a widget or a layout. ``value`` is an
        optional readout label shown right-aligned (sliders). Returns the row
        label."""
        row = QHBoxLayout()
        row.setSpacing(8)
        lab = self._row_label(label, self.LABEL_WIDTH)
        row.addWidget(lab)
        self._add_control(row, control, stretch)
        if value is not None:
            value.setObjectName("rowValue")
            value.setFixedWidth(self.VALUE_WIDTH)
            value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            row.addWidget(value)
        self.body_layout.addLayout(row)
        return lab

    @staticmethod
    def _add_control(row, control, stretch):
        if isinstance(control, QLayout):
            row.addLayout(control, 1 if stretch else 0)
        else:
            row.addWidget(control, 1 if stretch else 0)
        if not stretch:
            row.addStretch(1)

    def add_path_row(self, label, edit):
        """A path row with a dot that is filled while ``edit`` holds a path."""
        if self._paths is None:
            self._paths = QVBoxLayout()
            self._paths.setContentsMargins(0, 0, 0, 0)
            self._paths.setSpacing(0)
            self.body_layout.addLayout(self._paths)
        node = PathDot()
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        row.addWidget(node)
        inner = QHBoxLayout()
        inner.setContentsMargins(0, 7, 0, 7)
        inner.setSpacing(8)
        lab = self._row_label(label, self.LABEL_WIDTH - node.width())
        inner.addWidget(lab)
        inner.addWidget(edit, 1)
        row.addLayout(inner, 1)
        self._paths.addLayout(row)
        filled = lambda text: bool(text) and text != "Not selected"
        node.set_filled(filled(edit.text()))
        edit.textChanged.connect(lambda text: node.set_filled(filled(text)))
        return lab

    def add_widget(self, widget, alignment=None):
        if alignment is None:
            self.body_layout.addWidget(widget)
        else:
            self.body_layout.addWidget(widget, 0, alignment)

    def add_layout(self, layout):
        self.body_layout.addLayout(layout)

    def add_footer(self, item):
        """Add the section's main action (or its result text) below the divider."""
        self.footer.setVisible(self._open)
        if isinstance(item, QLayout):
            self.footer_layout.addLayout(item)
        else:
            self.footer_layout.addWidget(item)

    # --- state ---------------------------------------------------------
    def _header_key(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space):
            self.set_open(not self._open)
        else:
            QFrame.keyPressEvent(self.header, event)

    def set_open(self, open_):
        self._open = bool(open_)
        self.body.setVisible(self._open)
        self.footer.setVisible(self._open and self.footer_layout.count() > 0)
        self.header.setProperty("collapsed", not self._open)
        self.header.style().unpolish(self.header)
        self.header.style().polish(self.header)
        self.restyle()

    def is_open(self):
        return self._open

    def restyle(self):
        """Apply the current theme to the parts not covered by the stylesheet."""
        icon = themed_icon('chevron-down' if self._open else 'chevron-right', COLORS['text_dim'])
        self.chevron.setPixmap(icon.pixmap(14, 14))


class OptionList(QFrame):
    """An inset list of on/off options, one per row: the checkbox on the left
    and a short note on what it produces on the right."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("optionList")
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)

    def add_option(self, checkbox, note=""):
        if self._layout.count():
            sep_row = QHBoxLayout()
            sep_row.setContentsMargins(33, 0, 0, 0)
            sep = QFrame()
            sep.setObjectName("optionSep")
            sep.setFixedHeight(1)
            sep_row.addWidget(sep)
            self._layout.addLayout(sep_row)
        row = QHBoxLayout()
        row.setContentsMargins(12, 7, 12, 7)
        row.addWidget(checkbox)
        row.addStretch(1)
        label = QLabel(note)
        label.setObjectName("optionNote")
        row.addWidget(label)
        self._layout.addLayout(row)
        return label


class CardDialog(QDialog):
    """A dialog in the side panel's card style: a tinted header band with the
    title, a body, and the buttons behind a divider."""

    def __init__(self, parent, title, width=420):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setObjectName("cardDialog")
        self.setMinimumWidth(width)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        header = QFrame()
        header.setObjectName("dialogHeader")
        head = QHBoxLayout(header)
        head.setContentsMargins(20, 14, 20, 14)
        self.title_label = QLabel(title)
        self.title_label.setObjectName("dialogTitle")
        head.addWidget(self.title_label)
        outer.addWidget(header)
        body = QWidget()
        self.body_layout = QVBoxLayout(body)
        self.body_layout.setContentsMargins(20, 18, 20, 18)
        self.body_layout.setSpacing(10)
        outer.addWidget(body, 1)
        divider = QFrame()
        divider.setObjectName("cardDivider")
        divider.setFixedHeight(1)
        outer.addWidget(divider)
        self.footer_layout = QHBoxLayout()
        self.footer_layout.setContentsMargins(20, 12, 20, 14)
        self.footer_layout.setSpacing(8)
        self.footer_layout.addStretch(1)
        outer.addLayout(self.footer_layout)
        self.setStyleSheet(generate_dialog_style())
        self._stretched = False

    def showEvent(self, event):
        # spare height goes below the content, not between its rows
        if not self._stretched:
            self.body_layout.addStretch(1)
            self._stretched = True
        super().showEvent(event)

    def add_text(self, text, kind="text"):
        """A wrapped paragraph; ``kind`` is "text" or "note" (fainter)."""
        label = QLabel(text)
        label.setObjectName("dialogNote" if kind == "note" else "dialogText")
        label.setWordWrap(True)
        label.setTextFormat(Qt.RichText)
        label.setOpenExternalLinks(True)
        self.body_layout.addWidget(label)
        return label

    def add_bullets(self, items):
        """A short list, one point per row, with accent bullets."""
        for item in items:
            row = QHBoxLayout()
            row.setSpacing(10)
            dot = QLabel("\u2022")
            dot.setObjectName("dialogBullet")
            row.addWidget(dot, 0, Qt.AlignTop)
            text = QLabel(item)
            text.setObjectName("dialogText")
            text.setWordWrap(True)
            row.addWidget(text, 1, Qt.AlignTop)
            self.body_layout.addLayout(row)

    def add_button(self, text, primary=False, slot=None, accept=False):
        """A footer button; ``accept`` closes the dialog with Accepted."""
        btn = QPushButton(text)
        btn.setStyleSheet(generate_primary_button_style() if primary else generate_button_style())
        btn.setMinimumWidth(88)
        if slot is not None:
            btn.clicked.connect(slot)
        btn.clicked.connect(self.accept if accept else self.reject)
        if primary:
            btn.setDefault(True)
        self.footer_layout.addWidget(btn)
        return btn


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
                 generate_stats=False, generate_scores=False, mask_dir=None, max_size=None,
                 save_visuals=None):
        super().__init__()
        self.mask_dir = mask_dir
        self.max_size = max_size
        self.save_visuals = save_visuals
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
        # An exception escaping QThread.run aborts the process, so both steps
        # are guarded. A failure inside run() is logged, with its traceback,
        # by the run's log (Log.run_log); setup happens before that log opens.
        try:
            batch_processor = BatchProcessor(self.param_file, self.input_folder,
                                             self.output_folder, self.batch_size,
                                             self.batch_num, self.resume, self.ignore_large,
                                             self.generate_stats, self.generate_scores,
                                             mask_dir=self.mask_dir, max_size=self.max_size,
                                             save_visuals=self.save_visuals)
        except Exception as exc:
            Log.logger.exception("Batch setup failed")
            self.batch_failed.emit(f"{type(exc).__name__}: {exc}")
            return

        batch_processor.progress_callback = self.update_progress
        batch_processor.status_callback = self.status_updated.emit
        batch_processor.cancel_check = lambda: self._cancel_requested

        try:
            was_cancelled = batch_processor.run()
        except Exception as exc:
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

    failed = pyqtSignal(str)

    def __init__(self, image, min_gap_diameter):
        super().__init__()
        self.image = image
        self.min_gap_diameter = min_gap_diameter

    def run(self):
        try:
            self._run()
        except Exception as e:      # an escaping exception would abort the whole GUI process
            Log.logger.exception("Gap analysis failed")
            self.failed.emit(str(e))

    def _run(self):
        from .stages import fit_gap_circles
        # The GUI previews the segmentation mask only (no external ROI mask)
        circles = fit_gap_circles(self.image, self.min_gap_diameter,
                                  progress=lambda f: self.progress_updated.emit(int(f * 100)))
        self.gap_analysis_complete.emit(circles)

class DetectionWorker(QThread):
    progress_updated = pyqtSignal(int)
    detection_complete = pyqtSignal(list)

    failed = pyqtSignal(str)

    def __init__(self, image, args):
        super().__init__()
        self.image = image
        self.args = args

    def run(self):
        try:
            self._run()
        except Exception as e:
            Log.logger.exception("Fibre detection failed")
            self.failed.emit(str(e))

    def _run(self):
        # the same scales as the batch pipeline (stages.build_fibre_detector): the upper bound is included
        if self.args.min_line_width == self.args.max_line_width or self.args.line_step <= 0:
            line_widths = np.array([self.args.min_line_width])
        else:
            line_widths = np.arange(self.args.min_line_width, self.args.max_line_width + self.args.line_step,
                                    self.args.line_step)
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
            lo, hi = float(np.nanmin(self.image)), float(np.nanmax(self.image))
            if hi <= lo:
                raise ValueError("The image has no dynamic range; nothing to detect.")
            self.image = ((np.nan_to_num(self.image, nan=lo) - lo) / (hi - lo) * 255).astype(np.uint8)

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

    failed = pyqtSignal(str)

    def __init__(self, image, args):
        super().__init__()
        self.ori_img = image          # RGB array as loaded by the GUI
        self.args = args

    def run(self):
        try:
            self._run()
        except Exception as e:
            Log.logger.exception("Segmentation failed")
            self.failed.emit(str(e))

    def _run(self):
        def _progress(it, max_iter):
            self.progress_updated.emit(int((it + 1.0) / max(1, max_iter) * 100))

        # segment_image expects BGR (OpenCV) input; the GUI holds RGB
        bgr = cv2.cvtColor(self.ori_img, cv2.COLOR_RGB2BGR)
        mask, score = segment_image(bgr, self.args, iter_callback=_progress, return_score=True)
        roi_img = generate_rois(self.ori_img, mask, self.args.white_background)
        self.segmentation_complete.emit((roi_img, score))


class BRFWorker(QThread):
    """Computes ImageJ's HSB stack of the open image for the Birefringence page;
    the band masks are then re-thresholded from it without this worker."""
    hsb_complete = pyqtSignal(object)       # (hue, saturation, brightness) uint8 arrays
    failed = pyqtSignal(str)

    def __init__(self, image):
        super().__init__()
        self.image = image                  # RGB array as loaded by the GUI

    def run(self):
        try:
            from .brf import hsb
            self.hsb_complete.emit(hsb(self.image))
        except Exception as e:
            Log.logger.exception("Birefringence failed")
            self.failed.emit(str(e))


class BRFFolderWorker(QThread):
    """Run Folder of the Birefringence page: colour separation of every image
    of a folder (cabana.stages.run_brf_folder); cancel() stops between images."""
    progress_updated = pyqtSignal(int, int, str)     # images done, total, image being separated
    run_complete = pyqtSignal(object)                 # the run_brf_folder summary
    failed = pyqtSignal(str)

    def __init__(self, settings, input_dir, output_dir, mask_dir=None, visuals=True):
        super().__init__()
        self.settings, self.input_dir, self.output_dir = settings, input_dir, output_dir
        self.mask_dir, self.visuals = mask_dir, visuals
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            from .stages import run_brf_folder
            summary = run_brf_folder(self.settings, self.input_dir, self.output_dir, mask_dir=self.mask_dir,
                                     visuals=self.visuals, cancel=lambda: self._cancel,
                                     progress=lambda done, total, name: self.progress_updated.emit(done, total, name))
            self.run_complete.emit(summary)
        except Exception as e:
            Log.logger.exception("Birefringence folder run failed")
            self.failed.emit(str(e))


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


class UpdateWorker(QThread):
    """Background worker for Check for Updates: ``mode='check'`` asks PyPI for
    the latest version, ``mode='install'`` installs ``version`` with pip."""
    checked = pyqtSignal(str)             # latest version on PyPI
    installed = pyqtSignal(bool, str)     # success, pip output
    failed = pyqtSignal(str)

    def __init__(self, mode, version=None):
        super().__init__()
        self.mode = mode
        self.version = version

    def run(self):
        from .updates import install_version, latest_version
        try:
            if self.mode == 'check':
                self.checked.emit(latest_version())
            else:
                self.installed.emit(*install_version(self.version))
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
    parser.add_argument('--white_background', default=True, type=bool, help='Used to set background color')
    parser.add_argument('--input', type=str, help='Input image path', required=False)
    args, _ = parser.parse_known_args()
    return args


def _focus_ring(painter, centre, radius):
    """Keyboard focus ring around a slider handle."""
    ring = QColor(COLORS['highlight'])
    ring.setAlpha(150)
    painter.setBrush(Qt.NoBrush)
    painter.setPen(QPen(ring, 2))
    painter.drawEllipse(centre, radius + 4, radius + 4)


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
        self.handle_radius = SLIDER_HANDLE_RADIUS
        self.hover_handle = None  # 0 for none, 1 for lower, 2 for upper
        self.fill_color = None    # colour of the selected span; None = theme accent
        self._kbd_focus = False   # focus came from the keyboard: draw the focus ring
        self._key_handle = 1      # handle the arrow keys move (1 lower, 2 upper); Space switches

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
        fill = QColor(self.fill_color) if self.fill_color is not None else COLORS['highlight']

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
            painter.setBrush(fill)
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
                painter.setBrush(panel_colors()['handle'])
                painter.setPen(QPen(fill, 3))
            else:
                painter.setBrush(panel_colors()['handle'])
                painter.setPen(QPen(fill, 2))
            painter.drawEllipse(QPoint(upper_handle_x, handle_y), self.handle_radius, self.handle_radius)

            # Draw lower handle shadow
            painter.setBrush(QColor(0, 0, 0, 40))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QPoint(lower_handle_x, handle_y + 1), self.handle_radius, self.handle_radius)

            # Draw lower handle
            if self.hover_handle == 1 or self.moving_lower:
                painter.setBrush(panel_colors()['handle'])
                painter.setPen(QPen(fill, 3))
            else:
                painter.setBrush(panel_colors()['handle'])
                painter.setPen(QPen(fill, 2))
            painter.drawEllipse(QPoint(lower_handle_x, handle_y), self.handle_radius, self.handle_radius)
            if self._kbd_focus and self.hasFocus():
                cx = lower_handle_x if self._key_handle == 1 else upper_handle_x
                _focus_ring(painter, QPoint(cx, handle_y), self.handle_radius)

        else:  # Vertical orientation
            track_width = 4
            track_x = (self.width() - track_width) // 2
            painter.drawRoundedRect(track_x, self.handle_radius, track_width, self.height() - 2 * self.handle_radius, 2, 2)

            # Draw active range
            painter.setBrush(fill)
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
                painter.setBrush(panel_colors()['handle'])
                painter.setPen(QPen(fill, 3))
            else:
                painter.setBrush(panel_colors()['handle'])
                painter.setPen(QPen(fill, 2))
            painter.drawEllipse(QPoint(handle_x, upper_handle_y), self.handle_radius, self.handle_radius)

            # Draw lower handle shadow
            painter.setBrush(QColor(0, 0, 0, 40))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QPoint(handle_x, lower_handle_y + 1), self.handle_radius, self.handle_radius)

            # Draw lower handle (bottom)
            if self.hover_handle == 1 or self.moving_lower:
                painter.setBrush(panel_colors()['handle'])
                painter.setPen(QPen(fill, 3))
            else:
                painter.setBrush(panel_colors()['handle'])
                painter.setPen(QPen(fill, 2))
            painter.drawEllipse(QPoint(handle_x, lower_handle_y), self.handle_radius, self.handle_radius)

    def focusInEvent(self, event):
        self._kbd_focus = event.reason() in (Qt.TabFocusReason, Qt.BacktabFocusReason, Qt.ShortcutFocusReason)
        self.update()
        super().focusInEvent(event)

    def focusOutEvent(self, event):
        self._kbd_focus = False
        self.update()
        super().focusOutEvent(event)

    def keyPressEvent(self, event):
        """Arrows (Page Up/Down for a tenth of the range, Home/End) move the
        active handle; Space switches between the lower and upper handle."""
        key, span = event.key(), max(1, (self.max_val - self.min_val) // 10)
        steps = {Qt.Key_Left: -1, Qt.Key_Down: -1, Qt.Key_Right: 1, Qt.Key_Up: 1,
                 Qt.Key_PageDown: -span, Qt.Key_PageUp: span}
        if key == Qt.Key_Space:
            self._key_handle = 2 if self._key_handle == 1 else 1
            self._kbd_focus = True
            self.update()
            return
        lower, upper = self.lower_value, self.upper_value
        if key in steps or key in (Qt.Key_Home, Qt.Key_End):
            if key == Qt.Key_Home:
                value = self.min_val
            elif key == Qt.Key_End:
                value = self.max_val
            else:
                value = (lower if self._key_handle == 1 else upper) + steps[key]
            if self._key_handle == 1:
                lower = max(self.min_val, min(value, upper))
            else:
                upper = min(self.max_val, max(value, lower))
            self._kbd_focus = True
            self.setValues(lower, upper)
            return
        super().keyPressEvent(event)

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
        self.handle_radius = SLIDER_HANDLE_RADIUS
        self.hover = False
        self.fill_color = None    # colour of the filled track; None = theme accent
        self._kbd_focus = False   # focus came from the keyboard: draw the focus ring

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
        fill = QColor(self.fill_color) if self.fill_color is not None else COLORS['highlight']

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
            painter.setBrush(fill)
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
                painter.setBrush(panel_colors()['handle'])
                painter.setPen(QPen(fill, 3))
            else:
                painter.setBrush(panel_colors()['handle'])
                painter.setPen(QPen(fill, 2))

            painter.drawEllipse(QPoint(handle_x, handle_y), self.handle_radius, self.handle_radius)
            if self._kbd_focus and self.hasFocus():
                _focus_ring(painter, QPoint(handle_x, handle_y), self.handle_radius)

        else:  # Vertical orientation
            track_width = 4
            track_x = (self.width() - track_width) // 2
            track_height = self.height() - 2 * self.handle_radius
            painter.drawRoundedRect(track_x, self.handle_radius, track_width, track_height, 2, 2)

            # Draw active portion of the track
            painter.setBrush(fill)
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
                painter.setBrush(panel_colors()['handle'])
                painter.setPen(QPen(fill, 3))
            else:
                painter.setBrush(panel_colors()['handle'])
                painter.setPen(QPen(fill, 2))

            painter.drawEllipse(QPoint(handle_x, handle_y), self.handle_radius, self.handle_radius)

    def focusInEvent(self, event):
        self._kbd_focus = event.reason() in (Qt.TabFocusReason, Qt.BacktabFocusReason, Qt.ShortcutFocusReason)
        self.update()
        super().focusInEvent(event)

    def focusOutEvent(self, event):
        self._kbd_focus = False
        self.update()
        super().focusOutEvent(event)

    def keyPressEvent(self, event):
        self._kbd_focus = True
        super().keyPressEvent(event)

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


class PageRail(QFrame):
    """Icon bar at the window's left edge: one icon per side-panel page, the
    tour at the bottom. A click on a page's icon opens that page; the open
    page's icon folds the side panel (the window decides, via
    ``page_clicked``)."""

    page_clicked = pyqtSignal(int)
    help_clicked = pyqtSignal()

    def __init__(self, pages, parent=None):
        """``pages``: (icon name, tooltip) per page, in page order."""
        super().__init__(parent)
        self.setObjectName("pageRail")
        self.setFixedWidth(48)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 8, 6, 10)
        layout.setSpacing(4)
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self.buttons = []
        for index, (icon, tip) in enumerate(pages):
            btn = self._button(icon, tip, checkable=True)
            self.group.addButton(btn, index)
            btn.clicked.connect(lambda _checked=False, i=index: self.page_clicked.emit(i))
            layout.addWidget(btn, 0, Qt.AlignHCenter)
            self.buttons.append(btn)
            if index == 0:
                layout.addSpacing(6)          # Start, then the analysis pages
        layout.addStretch(1)
        self.help_btn = self._button('page-help', "Take the Tour", checkable=False)
        self.help_btn.clicked.connect(self.help_clicked.emit)
        layout.addWidget(self.help_btn, 0, Qt.AlignHCenter)
        self.restyle()

    @staticmethod
    def _button(icon, tip, checkable):
        btn = QToolButton()
        btn.setCheckable(checkable)
        btn.setFixedSize(36, 36)
        btn.setIconSize(QSize(18, 18))
        btn.setToolTip(tip)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setFocusPolicy(Qt.NoFocus)        # Ctrl/Cmd+1 to 6 switch pages from the keyboard
        btn.setProperty('icon_name', icon)
        return btn

    def set_current(self, index):
        self.buttons[index].setChecked(True)

    def restyle(self):
        """Apply the current theme: icons accent when checked, dim otherwise."""
        pc = panel_colors()
        for btn in self.buttons + [self.help_btn]:
            name = btn.property('icon_name')
            icon = QIcon()
            icon.addFile(icon_path(name, COLORS['text_dim']), QSize(), QIcon.Normal, QIcon.Off)
            icon.addFile(icon_path(name, COLORS['highlight']), QSize(), QIcon.Normal, QIcon.On)
            icon.addFile(icon_path(name, COLORS['highlight']), QSize(), QIcon.Active, QIcon.Off)
            btn.setIcon(icon)
        self.setStyleSheet(
            f"QFrame#pageRail {{ background-color: {color_to_stylesheet(pc['panel'])};"
            f" border: none; border-right: 1px solid {color_to_stylesheet(pc['card_border'])}; }}"
            f"QToolButton {{ border: none; border-radius: 8px; background: transparent; }}"
            f"QToolButton:hover {{ background-color: {color_to_stylesheet(pc['field'])}; }}"
            f"QToolButton:checked {{ background-color:"
            f" {color_to_stylesheet(_mix(pc['card'], COLORS['highlight'], 0.18))}; }}")


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
                # QImage needs a C-contiguous 8-bit buffer (a sliced RGBA view is not)
                image = np.ascontiguousarray(image)
                if image.dtype != np.uint8:
                    image = np.clip(image, 0, 255).astype(np.uint8)
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
                Log.logger.error(f"Error converting numpy array to QPixmap: {str(e)}")
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

            # Draw only the visible part of the image, scaled on the fly: scaling the
            # whole pixmap at high zoom would allocate a gigantic intermediate image
            visible = QRectF(0, 0, self.width(), self.height()).intersected(
                QRectF(x, y, scaled_width, scaled_height))
            if not visible.isEmpty():
                z = self.zoom_factor
                source = QRectF((visible.left() - x) / z, (visible.top() - y) / z,
                                visible.width() / z, visible.height() / z)
                painter.setRenderHint(QPainter.SmoothPixmapTransform, z < 1.0)
                painter.drawPixmap(visible, self.image, source)

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
            font = QFont(self.font())
            font.setPixelSize(22)
            font.setBold(True)
            painter.setFont(font)
            text = "Drag & Drop an Image Here or Use File > Open Image…"
            # wrapped inside a margin, so a narrow viewer never cuts it off
            painter.drawText(self.rect().adjusted(32, 0, -32, 0), Qt.AlignCenter | Qt.TextWordWrap, text)

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
        self.setFixedSize(40, 24)

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
        return QSize(40, 24)

    def minimumSizeHint(self):
        """Provide minimum size"""
        return QSize(40, 24)

    def heightForWidth(self, width):
        """Calculate height based on width to maintain proportions"""
        return int(width * 0.43)

    def mousePressEvent(self, event):
        """Handle mouse press events"""
        if event.button() == Qt.LeftButton and self._enabled:
            self.toggle()

    def paintEvent(self, event):
        """A pill track (accent when on) with a round knob."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        track = QRectF(1, (h - 22) / 2, w - 2, 22)
        on = QColor(COLORS['highlight']) if self._checked else QColor(COLORS['border'])
        knob = QColor(COLORS['background']) if self._checked else QColor(panel_colors()['handle'])
        if not self._enabled:
            on.setAlpha(110)
            knob.setAlpha(160)
        painter.setPen(Qt.NoPen)
        painter.setBrush(on)
        painter.drawRoundedRect(track, 11, 11)
        d = 18
        x = track.left() + 2 + (track.width() - d - 4) * self._slider_position
        painter.setBrush(knob)
        painter.drawEllipse(QRectF(x, track.top() + 2, d, d))


class MaxSizeSpinBox(QSpinBox):
    """Max Size box: 0 (shown as *No limit*) or MIN_MAX_SIZE px and above.

    The arrows step to the next multiple of the single step (2048 -> 3000 ->
    4000 with a step of 1000) and between *No limit* and MIN_MAX_SIZE
    directly; a typed value below MIN_MAX_SIZE is raised to it when editing
    finishes.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRange(0, 20000)
        self.setSpecialValueText("No limit")
        self.editingFinished.connect(self._raise_to_minimum)

    def stepBy(self, steps):
        value, step = self.value(), max(1, self.singleStep())
        for _ in range(abs(steps)):
            if steps > 0:
                value = MIN_MAX_SIZE if value == 0 else (value // step + 1) * step
            elif value > MIN_MAX_SIZE:
                value = max(MIN_MAX_SIZE, (value - 1) // step * step)
            else:
                value = 0
        self.setValue(min(value, self.maximum()))

    def _raise_to_minimum(self):
        if 0 < self.value() < MIN_MAX_SIZE:
            self.setValue(MIN_MAX_SIZE)


class WheelGuard(QObject):
    """Event filter that keeps the mouse wheel from changing a spin box, slider
    or combo box the user has not clicked: over such a widget the wheel
    scrolls the enclosing scroll area instead, so scrolling the side panel
    never alters a setting by accident.

    ``guard(root)`` installs it on every such widget under ``root``.
    """

    TYPES = (QAbstractSpinBox, QAbstractSlider, QComboBox)

    def guard(self, root):
        for widget in root.findChildren(self.TYPES):
            if isinstance(widget, QScrollBar):
                continue
            widget.setFocusPolicy(Qt.StrongFocus)       # the wheel does not give focus
            widget.installEventFilter(self)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Wheel and isinstance(obj, self.TYPES) and not obj.hasFocus():
            area = obj.parentWidget()
            while area is not None and not isinstance(area, QAbstractScrollArea):
                area = area.parentWidget()
            if area is not None:
                QApplication.sendEvent(area.verticalScrollBar(), event)
            return True
        return super().eventFilter(obj, event)
