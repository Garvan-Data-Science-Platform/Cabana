"""Interactive overlay and hand editing of fitted TMA cores for the GUI.

:class:`TMACoreEditor` draws the cores of a :class:`~cabana.tma.TMAPreprocessor`
on an :class:`~cabana.ui.ImagePanel` (circles, grid lines between neighbouring
cells, dashed placeholders at map positions without a core, labels) and, when
enabled, lets the user move a core (drag inside it), resize it (drag its rim),
include or exclude it (double-click), add one (double-click empty space or a
placeholder), delete it (Delete key) and undo (Ctrl+Z). Every edit goes
through the preprocessor's editing API, so the core re-snaps to its grid cell,
takes the patient ID of that cell and is re-filtered; the user never types a
label.

The panel shows the fit-level image; core coordinates are level-0 pixels, so
the editor converts with the fit level's downsample.
"""

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QRectF
from PyQt5.QtGui import QColor, QFont, QPen
from PyQt5.QtWidgets import QAction, QMenu

from .tma import OVERLAY_COLOURS


def _rgb(bgr):
    return QColor(bgr[2], bgr[1], bgr[0])


class TMACoreEditor:
    """Overlay painter and edit handler for the TMA page.

    Parameters
    ----------
    panel : ImagePanel
        The panel showing the fit-level image.
    get_pre : callable
        Returns the current TMAPreprocessor (or None).
    on_change : callable(full: bool)
        Called after the cores changed; ``full`` is False during a drag (repaint
        only) and True once an edit is complete (refresh the status line too).
    """
    HIT_TOL_PX = 12         # screen pixels around the rim that count as "on the rim"
    HANDLE_PX = 7           # size of the resize handle on the selected circle
    RIM_PX = 3.0            # screen width of a core's ring
    MAX_UNDO = 50
    LABEL_MIN_RADIUS_PX = 10

    def __init__(self, panel, get_pre, on_change):
        self.panel = panel
        self.get_pre = get_pre
        self.on_change = on_change
        self.enabled = False
        self.show_links = True
        self.selected = None
        self.hovered = None
        self._drag = None           # ("move", dx, dy) or ("resize",)
        self._undo = []

    # -- geometry --------------------------------------------------------------
    def _ds(self):
        pre = self.get_pre()
        return pre.reader.level_downsample(pre._fit_level)

    def _zoom(self):
        try:
            return max(self.panel.image_origin()[2], 1e-6)
        except Exception:
            return 1.0

    def hit(self, ix, iy):
        """``(core, "rim" | "inside")`` under image point ``(ix, iy)``, or ``(None, None)``."""
        pre = self.get_pre()
        if pre is None or not pre.cores:
            return None, None
        ds = self._ds()
        x0, y0 = ix * ds, iy * ds
        tol = self.HIT_TOL_PX / self._zoom() * ds
        best, kind, best_d = None, None, None
        for c in pre.cores:
            d = float(np.hypot(c.cx - x0, c.cy - y0))
            if abs(d - c.radius) <= min(tol, 0.4 * c.radius):
                k, score = "rim", abs(d - c.radius)
            elif d < c.radius:
                k, score = "inside", d
            else:
                continue
            if best is None or (k == "rim" and kind != "rim") or (k == kind and score < best_d):
                best, kind, best_d = c, k, score
        return best, kind

    def placeholder_hit(self, ix, iy):
        """The missing map position whose placeholder circle contains the point."""
        pre = self.get_pre()
        if pre is None or pre._pitch_px is None:
            return None
        ds = self._ds()
        x0, y0 = ix * ds, iy * ds
        rad = pre.placeholder_radius()
        for item in pre.empty_positions():
            centre = item[3]
            if centre is not None and np.hypot(centre[0] - x0, centre[1] - y0) <= rad:
                return item
        return None

    # -- undo ------------------------------------------------------------------
    def push_undo(self):
        pre = self.get_pre()
        if pre is None:
            return
        self._undo.append(pre.snapshot())
        del self._undo[:-self.MAX_UNDO]

    def can_undo(self):
        return bool(self._undo)

    def undo(self):
        pre = self.get_pre()
        if pre is None or not self._undo:
            return False
        pre.restore(self._undo.pop())
        self.selected = None
        self._drag = None
        self.on_change(True)
        return True

    def reset(self):
        """Forget the selection and undo history (after a refit)."""
        self.selected = None
        self.hovered = None
        self._drag = None
        self._undo = []

    # -- edit handler protocol (called by ImagePanel) -------------------------
    def press(self, ix, iy, event):
        if not self.enabled:
            return False
        core, kind = self.hit(ix, iy)
        if core is None:
            if self.selected is not None:
                self.selected = None
                self.on_change(False)
            return False                        # let the panel pan
        self.push_undo()
        self.selected = core
        ds = self._ds()
        if kind == "rim" or event.modifiers() & Qt.ShiftModifier:
            self._drag = ("resize",)
        else:
            self._drag = ("move", core.cx - ix * ds, core.cy - iy * ds)
        self.on_change(False)
        return True

    def move(self, ix, iy, event):
        if self._drag is None or self.selected is None:
            return
        core, ds = self.selected, self._ds()
        if self._drag[0] == "move":
            core.cx, core.cy = ix * ds + self._drag[1], iy * ds + self._drag[2]
        else:
            core.radius = max(1.0, float(np.hypot(core.cx - ix * ds, core.cy - iy * ds)))
        self.on_change(False)

    def release(self, ix, iy, event):
        if self._drag is None or self.selected is None:
            return
        self.move(ix, iy, event)            # final position, even if no move event arrived
        pre, core = self.get_pre(), self.selected
        if self._drag[0] == "move":
            pre.move_core(core, core.cx, core.cy)
        else:
            pre.resize_core(core, core.radius)
        self._drag = None
        self.on_change(True)

    def double_click(self, ix, iy, event):
        if not self.enabled:
            return False
        pre = self.get_pre()
        if pre is None:
            return False
        core, _ = self.hit(ix, iy)
        self.push_undo()
        if core is not None:
            pre.set_override(core, "include" if core.excluded else "exclude")
            self.selected = core
        else:
            ds = self._ds()
            item = self.placeholder_hit(ix, iy)
            if item is not None and item[3] is not None:
                self.selected = pre.add_core(item[3][0], item[3][1])
            else:
                self.selected = pre.add_core(ix * ds, iy * ds)
        self.on_change(True)
        return True

    def key(self, event):
        if not self.enabled:
            return False
        pre = self.get_pre()
        if pre is None:
            return False
        k = event.key()
        if k in (Qt.Key_Delete, Qt.Key_Backspace) and self.selected is not None:
            self.push_undo()
            pre.remove_core(self.selected)
            self.selected = None
            self.on_change(True)
            return True
        if k == Qt.Key_Z and event.modifiers() & Qt.ControlModifier:
            return self.undo()
        if k == Qt.Key_Escape and self.selected is not None:
            self.selected = None
            self.on_change(False)
            return True
        return False

    def cursor_at(self, ix, iy):
        core, kind = self.hit(ix, iy)
        pre = self.get_pre()
        if core is not self.hovered:
            self.hovered = core
            self.panel.setToolTip(pre.core_tooltip(core) if core is not None and pre is not None else "")
            self.panel.update()
        if not self.enabled:
            return None
        if kind == "rim":
            return Qt.SizeFDiagCursor
        if kind == "inside":
            return Qt.SizeAllCursor
        return Qt.ArrowCursor

    def context_menu(self, ix, iy, global_pos):
        if not self.enabled:
            return False
        pre = self.get_pre()
        if pre is None:
            return False
        core, _ = self.hit(ix, iy)
        from .ui import generate_context_menu_style
        menu = QMenu(self.panel)
        menu.setStyleSheet(generate_context_menu_style())
        if core is not None:
            self.selected = core
            info = pre.core_info(core)
            title = QAction(f"Core {core.index}" + (f" ({info.position})" if info and not info.empty else ""), menu)
            title.setEnabled(False)
            menu.addAction(title)
            menu.addSeparator()
            include = QAction("Include", menu)
            include.setCheckable(True)
            include.setChecked(core.override == "include")
            include.triggered.connect(lambda: self._override(core, "include"))
            exclude = QAction("Exclude", menu)
            exclude.setCheckable(True)
            exclude.setChecked(core.override == "exclude")
            exclude.triggered.connect(lambda: self._override(core, "exclude"))
            auto = QAction("Let filters decide", menu)
            auto.setCheckable(True)
            auto.setChecked(core.override == "")
            auto.triggered.connect(lambda: self._override(core, ""))
            for a in (include, exclude, auto):
                menu.addAction(a)
            menu.addSeparator()
            delete = QAction("Delete core", menu)
            delete.triggered.connect(lambda: self._delete(core))
            menu.addAction(delete)
        else:
            item = self.placeholder_hit(ix, iy)
            ds = self._ds()
            if item is not None and item[3] is not None:
                add = QAction(f"Add core at {item[0].position}", menu)
                add.triggered.connect(lambda: self._add(item[3][0], item[3][1]))
            else:
                add = QAction("Add core here", menu)
                add.triggered.connect(lambda: self._add(ix * ds, iy * ds))
            menu.addAction(add)
        menu.addSeparator()
        undo = QAction("Undo", menu)
        undo.setEnabled(self.can_undo())
        undo.triggered.connect(self.undo)
        menu.addAction(undo)
        menu.exec_(global_pos)
        return True

    def _override(self, core, value):
        self.push_undo()
        self.get_pre().set_override(core, value)
        self.on_change(True)

    def _delete(self, core):
        self.push_undo()
        self.get_pre().remove_core(core)
        if self.selected is core:
            self.selected = None
        self.on_change(True)

    def _add(self, cx, cy):
        self.push_undo()
        self.selected = self.get_pre().add_core(cx, cy)
        self.on_change(True)

    # -- painting ---------------------------------------------------------------
    def paint(self, painter, zoom, to_widget):
        pre = self.get_pre()
        if pre is None or pre._fit_image is None:
            return
        ds = self._ds()
        inv = 1.0 / ds

        def pen(colour, width=2.0, style=Qt.SolidLine):
            p = QPen(colour)
            p.setWidthF(width)
            p.setCosmetic(True)
            p.setStyle(style)
            return p

        painter.setBrush(Qt.NoBrush)
        if self.show_links:
            painter.setPen(pen(QColor(70, 95, 130, 210), 1.5))
            for seg in pre.grid_segments():
                painter.drawLine(int(seg[0][0] * inv), int(seg[0][1] * inv),
                                 int(seg[1][0] * inv), int(seg[1][1] * inv))

        placeholders = []
        if pre._pitch_px:
            rad = pre.placeholder_radius() * inv
            painter.setPen(pen(QColor(110, 110, 110), 2.0, Qt.DashLine))
            for info, row, col, centre in pre.empty_positions():
                if centre is None:
                    continue
                cx, cy = centre[0] * inv, centre[1] * inv
                painter.drawEllipse(int(cx - rad), int(cy - rad), int(2 * rad), int(2 * rad))
                placeholders.append((cx, cy, pre.position_label(info, row, col)))

        labels = []
        radii = [c.radius * inv * zoom for c in pre.cores]
        typical = float(np.median(radii)) if radii else 0.0
        short_labels = typical < 2.5 * self.LABEL_MIN_RADIUS_PX   # one rule for the whole frame
        for c in pre.cores:
            colour = _rgb(OVERLAY_COLOURS[c.flag])
            cx, cy, r = c.cx * inv, c.cy * inv, c.radius * inv
            painter.setPen(pen(colour, self.RIM_PX + 1 if c is self.hovered else self.RIM_PX))
            painter.drawEllipse(QRectF(cx - r, cy - r, 2 * r, 2 * r))
            if c.manual or c.override:
                # hand-edited mark: a filled dot on the rim at 12 o'clock, clear of the label
                painter.setPen(pen(QColor(255, 255, 255), 1.0))
                painter.setBrush(_rgb(OVERLAY_COLOURS["manual"]))
                dot = 6.0 / zoom
                painter.drawEllipse(QRectF(cx - dot, cy - r - dot, 2 * dot, 2 * dot))
                painter.setBrush(Qt.NoBrush)
            if c.excluded:
                painter.setPen(pen(colour, self.RIM_PX))
                k = 0.5 * r
                painter.drawLine(int(cx - k), int(cy - k), int(cx + k), int(cy + k))
                painter.drawLine(int(cx - k), int(cy + k), int(cx + k), int(cy - k))
            if c is self.selected:
                # selection halo outside the circle, so the status colour stays readable,
                # and a handle on the rim to grab for resizing
                painter.setPen(pen(QColor(255, 220, 0), 1.5, Qt.DashLine))
                halo = r + 5.0 / zoom
                painter.drawEllipse(int(cx - halo), int(cy - halo), int(2 * halo), int(2 * halo))
                painter.setPen(pen(QColor(255, 220, 0), 1.5))
                painter.setBrush(QColor(255, 220, 0))
                h = self.HANDLE_PX / zoom
                painter.drawRect(int(cx + r - h / 2), int(cy - h / 2), int(h), int(h))
                painter.setBrush(Qt.NoBrush)
            if r * zoom >= self.LABEL_MIN_RADIUS_PX:
                lines = pre.core_labels(c)
                if short_labels:
                    lines = [lines[0].split(" #")[0]]     # too small for the number and tissue
                labels.append((cx, cy, lines))

        # text in screen pixels so it stays legible at every zoom
        painter.save()
        painter.resetTransform()
        font = QFont()
        font.setPointSize(9)
        font.setBold(True)
        painter.setFont(font)
        fm = painter.fontMetrics()
        for cx, cy, lines in labels:
            wx, wy = to_widget(cx, cy)
            width = max(fm.horizontalAdvance(t) for t in lines) + 6
            height = fm.height() * len(lines) + 2
            box = QRectF(wx - width / 2, wy - height / 2, width, height)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(255, 255, 255, 170))
            painter.drawRoundedRect(box, 3, 3)
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QColor(20, 40, 220))
            for i, text in enumerate(lines):
                painter.drawText(int(wx - fm.horizontalAdvance(text) / 2),
                                 int(box.top() + 1 + fm.ascent() + i * fm.height()), text)
        painter.setPen(QColor(110, 110, 110))
        for cx, cy, text in placeholders:
            wx, wy = to_widget(cx, cy)
            painter.drawText(int(wx - fm.horizontalAdvance(text) / 2), int(wy + fm.ascent() / 2), text)
        painter.restore()
