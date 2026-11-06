"""Guided tour: a spotlight over the main window that walks through the
analysis one step at a time.

Each ``TourStep`` names a side-panel page and a widget to highlight; the
overlay dims everything else, rings the widget in the theme's accent and
puts a callout card beside it with the step text and Back / Next. The tour
only switches pages: it changes no setting and needs no image.
"""
from dataclasses import dataclass
from typing import Callable, Optional

from PyQt5.QtCore import QEvent, QPoint, QRect, QRectF, QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QPainter, QPainterPath, QPen
from PyQt5.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QToolButton, QVBoxLayout, QWidget

from .ui import (COLORS, FONT_SIZES, color_to_stylesheet, generate_button_style,
                 generate_primary_button_style, is_light_theme, panel_colors)


@dataclass
class TourStep:
    title: str
    text: str
    enter: Optional[Callable[[], None]] = None            # e.g. show the step's page
    target: Optional[Callable[[], Optional[QWidget]]] = None   # widget to highlight; None = centred card


class TourOverlay(QWidget):
    """Spotlight overlay over ``window`` for ``steps``; Esc, × or Done ends it."""

    finished = pyqtSignal()
    PAD = 6           # space between the highlighted widget and the ring
    CARD_WIDTH = 300

    def __init__(self, window, steps):
        super().__init__(window)
        self.window_ = window
        self.steps = list(steps)
        self.index = 0
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        self._build_card()
        window.installEventFilter(self)
        self.setGeometry(window.rect())

    # --- card -------------------------------------------------------------
    def _build_card(self):
        pc = panel_colors()
        self.card = QFrame(self)
        self.card.setObjectName("tourCard")
        self.card.setFixedWidth(self.CARD_WIDTH)
        self.card.setStyleSheet(
            f"QFrame#tourCard {{ background-color: {color_to_stylesheet(pc['card'])};"
            f" border: 1px solid {color_to_stylesheet(pc['card_border'])}; border-radius: 12px; }}"
            f"QLabel {{ background: transparent; }}"
            f"QLabel#tourTitle {{ color: {color_to_stylesheet(COLORS['text'])};"
            f" font-size: {FONT_SIZES['title'] + 1}px; font-weight: 600; }}"
            f"QLabel#tourText {{ color: {color_to_stylesheet(COLORS['text_dim'])};"
            f" font-size: {FONT_SIZES['base'] + 1}px; }}"
            f"QLabel#tourCount {{ color: {color_to_stylesheet(pc['faint'])};"
            f" font-size: {FONT_SIZES['small']}px; }}"
            f"QToolButton {{ border: none; background: transparent;"
            f" color: {color_to_stylesheet(pc['faint'])}; font-size: 16px; }}"
            f"QToolButton:hover {{ color: {color_to_stylesheet(COLORS['text'])}; }}")
        layout = QVBoxLayout(self.card)
        layout.setContentsMargins(18, 14, 14, 16)
        layout.setSpacing(8)

        top = QHBoxLayout()
        self.title_label = QLabel()
        self.title_label.setObjectName("tourTitle")
        self.title_label.setWordWrap(True)
        top.addWidget(self.title_label, 1)
        self.close_btn = QToolButton()
        self.close_btn.setText("×")
        self.close_btn.setToolTip("End the tour (Esc)")
        self.close_btn.setCursor(Qt.PointingHandCursor)
        self.close_btn.clicked.connect(self.end)
        top.addWidget(self.close_btn, 0, Qt.AlignTop)
        layout.addLayout(top)

        self.text_label = QLabel()
        self.text_label.setObjectName("tourText")
        self.text_label.setWordWrap(True)
        layout.addWidget(self.text_label)
        layout.addSpacing(6)

        bottom = QHBoxLayout()
        bottom.setSpacing(8)
        self.count_label = QLabel()
        self.count_label.setObjectName("tourCount")
        bottom.addWidget(self.count_label, 1, Qt.AlignVCenter)
        self.back_btn = QPushButton("Back")
        self.back_btn.setStyleSheet(generate_button_style())
        self.back_btn.clicked.connect(self.back)
        bottom.addWidget(self.back_btn)
        self.next_btn = QPushButton("Next")
        self.next_btn.setStyleSheet(generate_primary_button_style())
        self.next_btn.setMinimumWidth(72)
        self.next_btn.clicked.connect(self.next)
        bottom.addWidget(self.next_btn)
        layout.addLayout(bottom)

    # --- navigation -------------------------------------------------------
    def start(self):
        self.show()
        self.raise_()
        self.setFocus()
        self.go(0)

    def go(self, index):
        self.index = max(0, min(index, len(self.steps) - 1))
        step = self.steps[self.index]
        if step.enter is not None:
            step.enter()
        self.title_label.setText(step.title)
        self.text_label.setText(step.text)
        self.count_label.setText(f"{self.index + 1} of {len(self.steps)}")
        self.back_btn.setVisible(self.index > 0)
        self.next_btn.setText("Done" if self.index == len(self.steps) - 1 else "Next")
        # the page switch lays out on the next pass: place the card after it
        QTimer.singleShot(0, self._place)
        self._place()

    def next(self):
        if self.index == len(self.steps) - 1:
            self.end()
        else:
            self.go(self.index + 1)

    def back(self):
        self.go(self.index - 1)

    def end(self):
        self.window_.removeEventFilter(self)
        self.hide()
        self.finished.emit()
        self.deleteLater()

    # --- geometry ---------------------------------------------------------
    def target_rect(self):
        """The highlighted area in overlay coordinates, or None (centred step)."""
        step = self.steps[self.index]
        widget = step.target() if step.target is not None else None
        if widget is None or not widget.isVisible():
            return None
        top_left = widget.mapTo(self.window_, QPoint(0, 0))
        rect = QRect(top_left, widget.size()).adjusted(-self.PAD, -self.PAD, self.PAD, self.PAD)
        return rect.intersected(self.rect().adjusted(2, 2, -2, -2))

    def _place(self):
        self.setGeometry(self.window_.rect())
        # wrapped labels: the card's height follows its width
        self.card.setFixedHeight(self.card.layout().totalHeightForWidth(self.CARD_WIDTH))
        target, card = self.target_rect(), self.card.size()
        area = self.rect().adjusted(16, 16, -16, -16)
        if target is None:
            pos = QPoint(area.center().x() - card.width() // 2, area.center().y() - card.height() // 2)
        elif target.right() + 20 + card.width() <= area.right():           # beside the target
            pos = QPoint(target.right() + 20, target.top())
        elif target.bottom() + 16 + card.height() <= area.bottom():        # or below it
            pos = QPoint(target.left(), target.bottom() + 16)
        else:                                                               # or above it
            pos = QPoint(target.left(), target.top() - 16 - card.height())
        pos.setX(max(area.left(), min(pos.x(), area.right() - card.width())))
        pos.setY(max(area.top(), min(pos.y(), area.bottom() - card.height())))
        self.card.move(pos)
        self.update()

    # --- painting and events ---------------------------------------------
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        shade = QPainterPath()
        shade.addRect(QRectF(self.rect()))
        target = self.target_rect()
        if target is not None:
            hole = QPainterPath()
            hole.addRoundedRect(QRectF(target), 12, 12)
            shade = shade.subtracted(hole)
        painter.fillPath(shade, QColor(0, 0, 0, 115 if is_light_theme() else 185))
        if target is not None:
            painter.setPen(QPen(COLORS['highlight'], 2))
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(QRectF(target), 12, 12)

    def eventFilter(self, obj, event):
        if obj is self.window_ and event.type() == QEvent.Resize:
            QTimer.singleShot(0, self._place)
        return False

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Escape:
            self.end()
        elif key in (Qt.Key_Right, Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space):
            self.next()
        elif key == Qt.Key_Left and self.index > 0:
            self.back()
        else:
            super().keyPressEvent(event)

    def mousePressEvent(self, event):
        event.accept()        # the window stays still during the tour

    def wheelEvent(self, event):
        event.accept()

