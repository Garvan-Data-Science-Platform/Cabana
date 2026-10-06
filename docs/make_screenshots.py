"""Regenerate the annotated GUI screenshots used in the documentation.

Drives the real Cabana GUI on Qt's offscreen platform, runs the analysis
steps on the bundled Picrosirius sample image and (when the slide is present)
fits the APGI TMA 4 slide, then grabs every page and draws the numbered red
markers referenced by the text in ``source/workflow.md`` and ``source/tma.md``.

    conda activate cabana
    python docs/make_screenshots.py            # all pages
    python docs/make_screenshots.py start tma  # a subset

Outputs are written to ``docs/source/media/``. Marker numbers must stay in
step with the numbered lists in the Markdown pages.
"""
import os
import sys
import tempfile
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from cabana.log import Log  # noqa: E402
Log.init_log_path(tempfile.mkdtemp())

from PyQt5.QtCore import QEventLoop, QPoint, QRect, QTimer, Qt  # noqa: E402
from PyQt5.QtGui import QColor, QFont, QPainter, QPen  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402

MEDIA = os.path.join(REPO, "docs", "source", "media")
IMG = os.path.join(REPO, "data", "Picrocirius",
                   "540.vsi - 20x_BF multi-band_01Annotation (Ellipse) (Tumor)_0.tif")
SLIDE = os.path.join(REPO, "large", "APGI TMA", "APGI TMA 4 PicRed.vsi")
RED = QColor(220, 30, 30)


# ----------------------------------------------------------------- helpers --
def settle(app, ms=300):
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec_()
    app.processEvents()


def wait_until(app, predicate, timeout_s=900, label=""):
    t0 = time.time()
    while not predicate():
        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"timed out waiting for {label}")
        settle(app, 100)
    settle(app, 400)
    print(f"  {label} done in {time.time() - t0:.1f} s")


def rect_of(win, widget):
    return QRect(widget.mapTo(win, QPoint(0, 0)), widget.size())


def union(win, widgets):
    r = QRect(rect_of(win, widgets[0]))
    for w in widgets[1:]:
        r = r.united(rect_of(win, w))
    return r


def menu_bar_rect(win):
    bar = win.menuBar()
    acts = bar.actions()
    r = bar.actionGeometry(acts[0]).united(bar.actionGeometry(acts[-1]))
    return QRect(bar.mapTo(win, r.topLeft()), r.size())


def grab(win, name, marks):
    """Grab the window and draw numbered red ellipses.

    ``marks``: (number, QRect[, side]); side places the number badge at the
    ellipse's top-left corner ("tl", default), "left", "right", "above" or
    "below".
    """
    pix = win.grab()
    p = QPainter(pix)
    p.setRenderHint(QPainter.Antialiasing)
    pen = QPen(RED)
    pen.setWidth(3)
    p.setFont(QFont("Helvetica", 20, QFont.Bold))
    d = 32
    for mark in marks:
        number, rect = mark[0], mark[1]
        side = mark[2] if len(mark) > 2 else "tl"
        ell = rect.adjusted(-8, -5, 8, 5)
        p.setBrush(Qt.NoBrush)
        p.setPen(pen)
        p.drawEllipse(ell)
        if side == "left":
            c = QRect(ell.left() - d - 4, ell.center().y() - d // 2, d, d)
        elif side == "right":
            c = QRect(ell.right() + 4, ell.center().y() - d // 2, d, d)
        elif side == "above":
            c = QRect(ell.center().x() - d // 2, ell.top() - d - 2, d, d)
        elif side == "below":
            c = QRect(ell.center().x() - d // 2, ell.bottom() + 2, d, d)
        else:
            c = QRect(ell.left() + ell.width() // 8 - d // 2, ell.top() - d // 2 - 2, d, d)
        p.setBrush(RED)
        p.setPen(QPen(QColor(255, 255, 255), 2))
        p.drawEllipse(c)
        p.drawText(c, Qt.AlignCenter, str(number))
    p.end()
    os.makedirs(MEDIA, exist_ok=True)
    path = os.path.join(MEDIA, name)
    pix.save(path)
    print("saved", os.path.relpath(path, REPO))


# ------------------------------------------------------------------- pages --
def main(argv):
    only = set(argv)
    want = lambda name: not only or name in only  # noqa: E731

    app = QApplication.instance() or QApplication([])
    from cabana.cabana_gui import MainWindow
    win = MainWindow()          # maximises itself to the 800x600 offscreen screen
    win.showNormal()
    win.resize(1500, 980)
    if win.theme_combo.currentText() != "Light":
        win.theme_combo.setCurrentText("Light")
    settle(app, 500)

    if want("start"):
        grab(win, "start.png", [
            (1, menu_bar_rect(win), "right"),
            (2, rect_of(win, win.start_slide_btn)),
            (3, rect_of(win, win.start_open_btn)),
            (4, rect_of(win, win.start_params_btn)),
        ])

    analysis = want("segmentation") or want("detection") or want("gap") or want("batch")
    if analysis:
        win.load_original_image(IMG)
        win.image_panel.setImage(win.ori_img)
        settle(app, 500)
        win.run_segmentation()
        wait_until(app, lambda: win.seg_img is not None, label="segmentation")
    if want("segmentation"):
        grab(win, "segmentation.png", [
            (1, rect_of(win, win.toggle_seg_btn), "left"),
            (2, rect_of(win, win.color_btn), "right"),
            (3, union(win, (win.color_thresh_slider, win.num_labels_slider, win.max_iters_slider))),
            (4, rect_of(win, win.patch_size_spinner), "left"),
            (5, rect_of(win, win.white_bg_cb), "right"),
            (6, rect_of(win, win.segment_btn)),
        ])

    if analysis:
        win.show_page(win.pages.indexOf(win.det_tab))
        settle(app)
        win.run_detection()
        wait_until(app, lambda: win.wdt_img is not None, label="detection")
    if want("detection"):
        grab(win, "detection.png", [
            (1, rect_of(win, win.line_width_range)),
            (2, rect_of(win, win.line_step_slider)),
            (3, rect_of(win, win.contrast_range)),
            (4, rect_of(win, win.min_length_slider)),
            (5, union(win, (win.dark_line_cb, win.extend_line_cb, win.overlay_fibres_cb)), "right"),
            (6, rect_of(win, win.detect_btn)),
        ])

    if analysis:
        win.show_page(win.pages.indexOf(win.gap_tab))
        settle(app)
        win.run_gap_analysis()
        wait_until(app, lambda: win.gap_img is not None, label="gap analysis")
    if want("gap"):
        grab(win, "gap_analysis.png", [
            (1, rect_of(win, win.toggle_gap_btn), "above"),
            (2, rect_of(win, win.min_gap_slider)),
            (3, rect_of(win, win.max_hdm_slider)),
            (4, rect_of(win, win.analyze_btn)),
        ])

    if want("batch"):
        win.show_page(win.pages.indexOf(win.bat_tab))
        settle(app)
        win.param_file = "/data/TMA4/Parameters.yml"
        win.param_file_path.setText(win.param_file)
        win.input_folder = "/data/TMA4_cores/BF/Patients/Images"
        win.input_folder_path.setText(win.input_folder)
        win.output_folder = "/data/TMA4_cores/BF/Patients/Output"
        win.output_folder_path.setText(win.output_folder)
        win.set_mask_folder("/data/TMA4_cores/BF/Patients/Masks")
        win._check_batch_processing_ready()
        win.stats_cb.setChecked(True)
        win.scores_cb.setChecked(True)
        # the status line and bar as they look during a run
        win.progress_label.setText("Batch 2/9: Segmenting 8010718.vsi - TMA4_BF_A3Annotation (Tumor)_1.png (3/5)")
        win.progress_label.setVisible(True)
        win.progress_bar.setVisible(True)
        win.progress_bar.setValue(14)
        settle(app)
        grab(win, "batch_processing.png", [
            (1, rect_of(win, win.param_btn), "right"),
            (2, rect_of(win, win.input_btn), "right"),
            (3, rect_of(win, win.output_btn), "right"),
            (4, rect_of(win, win.mask_btn), "right"),
            (5, rect_of(win, win.batch_size_spinner), "left"),
            (6, union(win, (win.stats_cb, win.scores_cb)), "right"),
            (7, rect_of(win, win.process_batch_btn)),
            (8, union(win, (win.progress_label, win.progress_bar)), "above"),
        ])
        win.progress_label.setVisible(False)
        win.progress_bar.setVisible(False)

    if want("tma"):
        if not os.path.exists(SLIDE):
            print("TMA slide not found, skipping tma.png:", SLIDE)
        else:
            win.show_page(win.pages.indexOf(win.tma_tab))
            settle(app)
            win.tma_slide = SLIDE
            win.tma_slide_path.setText(SLIDE)
            out = os.path.join(os.path.dirname(SLIDE), "APGI_TMA_4_PicRed_cores")
            win.tma_output = out
            win.tma_output_path.setText(out)
            win._start_tma_preview()
            wait_until(app, lambda: win.tma_reader is not None, label="slide preview")
            win.tma_array_combo.setCurrentIndex(win.tma_array_combo.findData(4))
            settle(app)
            win.run_tma_fit()
            wait_until(app, lambda: win.tma_pre is not None and win.tma_fit_btn.text() == "Fit Cores",
                       label="core fitting")
            grab(win, "tma.png", [
                (1, rect_of(win, win.tma_slide_btn), "right"),
                (2, rect_of(win, win.tma_array_combo), "left"),
                (3, union(win, (win.tma_core_diameter_spin, win.tma_sat_spin, win.tma_recover_cb))),
                (4, rect_of(win, win.tma_fit_btn)),
                (5, union(win, (win.tma_offset_spin, win.tma_stain_spin, win.tma_dmin_spin, win.tma_dmax_spin))),
                (6, union(win, (win.tma_edit_cb, win.tma_links_cb, win.tma_undo_btn))),
                (7, union(win, (win.tma_margin_spin, win.tma_erode_spin, *win.tma_channel_cbs.values()))),
                (8, rect_of(win, win.tma_output_btn), "right"),
                (9, rect_of(win, win.tma_export_btn)),
            ])


if __name__ == "__main__":
    main(sys.argv[1:])
