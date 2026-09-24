"""Tissue micro-array (TMA) preprocessing.

Fits a circle to every core on a whole-slide scan, snaps the circles to the
array grid, matches the grid to the printed ICGC/APGI array map, and exports
one image and one binary mask per core and channel:

    <out>/BF/Images/TMA1_A3_8010718_1734_BF.png
    <out>/BF/Masks/TMA1_A3_8010718_1734_BF.png
    <out>/POL/Images/TMA1_A3_8010718_1734_POL.png
    <out>/POL/Masks/TMA1_A3_8010718_1734_POL.png
    <out>/cores.csv
    <out>/overlay.png

The mask is white inside the fitted circle (shrunk by ``erode_px``) and
black in the corners, so downstream analysis can ignore the space outside
the core. ``Images`` and ``Masks`` are drop-in inputs for ``BatchCabana``
(input folder and ROI-mask folder respectively).
"""

import csv
import os
from dataclasses import dataclass, asdict

import cv2
import numpy as np
from scipy import ndimage as ndi

from .tma_maps import (MAP_COLS, MAP_ROWS, ROW_LETTERS, core_stem, load_array_map,
                       occupancy_grid)
from .wsi import open_slide

def write_png_with_resolution(path, bgr, pixel_size_um):
    """Write a BGR image as PNG carrying EXIF X/YResolution (pixels per µm),
    the field :func:`cabana.io.split2batches` reads to recover µm/pixel."""
    from PIL import Image
    img = Image.fromarray(bgr[:, :, ::-1]) if bgr.ndim == 3 else Image.fromarray(bgr)
    if pixel_size_um:
        exif = Image.Exif()
        exif[282] = exif[283] = 1.0 / float(pixel_size_um)   # XResolution, YResolution
        exif[296] = 1                                          # ResolutionUnit: none (per µm)
        img.save(path, exif=exif.tobytes())
    else:
        img.save(path)


# Objects further than this (in grid pitches) from their lattice position are
# not cores and are ignored by the orientation match. Fixed on purpose so that
# patient-ID assignment never depends on user-tunable QC filter values.
MAP_MAX_GRID_OFFSET = 0.5

# The eight rigid transforms that map the printed array onto the scan.
ORIENTATIONS = ("auto", "0", "90", "180", "270", "0+flip", "90+flip", "180+flip", "270+flip")


@dataclass
class Core:
    """One fitted core. Coordinates are level-0 pixels of the slide."""
    index: int
    cx: float
    cy: float
    radius: float
    fill: float           # fraction of the circle covered by tissue footprint
    row: int = -1         # grid row index in the scan (0-based)
    col: int = -1         # grid column index in the scan (0-based)
    map_row: int = -1     # row index in the printed map (0-based, A=0)
    map_col: int = -1     # column index in the printed map (0-based)
    outside_map: bool = False   # grid cell has no counterpart in the printed map
    recovered: bool = False     # found by the faint-core pass at a predicted grid position
    grid_offset: float = 0.0    # distance to the lattice-predicted centre, in grid pitches
    diameter_um: float = 0.0    # fitted diameter
    stain_frac: float = 0.0     # fraction of the disc above the stain saturation threshold
    excluded: bool = False      # rejected by the QC filters (see TMAPreprocessor.apply_filters)
    reason: str = ""            # why it was excluded, e.g. "off_grid", "diameter", "stain"

    @property
    def flag(self):
        if self.outside_map:
            return "outside_map"
        if self.excluded:
            return "excluded"
        if self.recovered:
            return "recovered"
        return "ok"


# ---------------------------------------------------------------------------
# Circle fitting
# ---------------------------------------------------------------------------

def tissue_mask(img_bgr, sat_thresh=15, val_ratio=0.965):
    """Foreground mask from HSV saturation and a per-column background level.

    The per-column background compensates for the vertical banding that slide
    scanners produce in the illumination.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)
    col_bg = np.median(val, axis=0, keepdims=True)
    return ((sat > sat_thresh) | (val < val_ratio * col_bg)).astype(np.uint8)


def _ellipse(diameter_px):
    d = max(3, int(round(diameter_px)) | 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))


def fit_cores(img_bgr, pixel_size_um, core_diameter_um=1250.0,
              min_component_frac=0.05, debris_frac=0.05, sat_thresh=15, val_ratio=0.965,
              open_frac=0.1):
    """Fit a circle to each tissue core in a low-resolution slide image.

    Returns a list of ``(cx, cy, r, fill)`` in pixels of ``img_bgr``.

    Steps (ported from the original prototype): threshold tissue
    (``sat_thresh``, ``val_ratio``; see :func:`tissue_mask`), close gaps of
    ~30% core diameter, open with ~``open_frac`` core diameters to cut thin
    structures such as coverslip edges or scratches off the cores, drop
    components smaller than ``min_component_frac`` of the largest, then inside
    each component remove small detached debris blobs (< ``debris_frac`` of
    the main blob and not touching it) before taking the minimum enclosing
    circle of the remaining tissue.

    Because an enclosing circle is set by its outermost points, debris beside a
    core inflates it. Circles larger than 1.1 times the slide's median radius
    are therefore rebuilt: starting from the largest tissue piece (split at
    thin attachments when it is itself too large), neighbouring pieces are
    added nearest first only while the circle stays within that bound, so
    fragments of a broken core are kept and outlying debris is not. Circles
    larger than 1.5 or smaller than 0.2 nominal cores are rejected.
    """
    raw = tissue_mask(img_bgr, sat_thresh=sat_thresh, val_ratio=val_ratio)
    core_px = core_diameter_um / pixel_size_um
    closed = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, _ellipse(0.3 * core_px))
    if open_frac and open_frac > 0:
        closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, _ellipse(open_frac * core_px))
    lbl, n = ndi.label(closed)
    if n == 0:
        return []
    sizes = ndi.sum(closed, lbl, range(1, n + 1))
    keep = np.nonzero(sizes > min_component_frac * sizes.max())[0] + 1
    # a real core cannot be smaller than a quarter of the expected area
    min_area = 0.25 * np.pi * (core_px / 2) ** 2 * 0.15
    keep = [k for k in keep if sizes[k - 1] >= min_area]

    small = _ellipse(0.08 * core_px)
    split = _ellipse(0.12 * core_px)
    objects = ndi.find_objects(lbl)
    pad = int(0.15 * core_px) + 3
    comps = []          # (offset_x, offset_y, pieces[list of point arrays], footprint)
    for k in keep:
        sl = objects[k - 1]
        ys = slice(max(0, sl[0].start - pad), min(raw.shape[0], sl[0].stop + pad))
        xs = slice(max(0, sl[1].start - pad), min(raw.shape[1], sl[1].stop + pad))
        lbl_w = lbl[ys, xs]
        comp_raw = ((lbl_w == k) & (raw[ys, xs] > 0)).astype(np.uint8)
        blobs = cv2.morphologyEx(comp_raw, cv2.MORPH_CLOSE, small)
        blbl, bn = ndi.label(blobs)
        if bn == 0:
            continue
        bsz = ndi.sum(blobs, blbl, range(1, bn + 1))
        main = int(np.argmax(bsz)) + 1
        near_main = cv2.dilate((blbl == main).astype(np.uint8), small)
        pieces = []
        for b in range(1, bn + 1):
            if b != main and not (bsz[b - 1] >= debris_frac * bsz[main - 1]
                                  or ((blbl == b) & (near_main > 0)).any()):
                continue    # small detached speck
            piece = (comp_raw & (blbl == b)).astype(np.uint8)
            cnts, _ = cv2.findContours(piece, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if cnts:
                pieces.append((int(piece.sum()), np.vstack([c.reshape(-1, 2) for c in cnts]), piece))
        if not pieces:
            continue
        pieces.sort(key=lambda t: -t[0])
        comps.append((xs.start, ys.start, pieces, ndi.binary_fill_holes(lbl_w == k)))

    def enclosing(point_sets):
        (cx, cy), r = cv2.minEnclosingCircle(np.vstack(point_sets).astype(np.float32))
        return cx, cy, r

    # first pass: circle around all kept pieces (original behaviour)
    first = [enclosing([p[1] for p in pieces]) for _, _, pieces, _ in comps]
    radii = [r for _, _, r in first if 0.2 * core_px / 2 <= r <= 1.5 * core_px / 2]
    r_ref = float(np.median(radii)) if radii else core_px / 2
    bound = 1.1 * r_ref

    circles = []
    for (ox, oy, pieces, footprint), (cx, cy, r) in zip(comps, first):
        if r > bound:
            # Debris is inflating the circle. If the largest piece alone is
            # already too big, split thin attachments off it first.
            cand = [p[1] for p in pieces]
            if enclosing([pieces[0][1]])[2] > bound:
                opened = cv2.morphologyEx(pieces[0][2], cv2.MORPH_OPEN, split)
                sub_lbl, sub_n = ndi.label(opened)
                subs = []
                for j in range(1, sub_n + 1):
                    cnts, _ = cv2.findContours((sub_lbl == j).astype(np.uint8), cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_NONE)
                    if cnts:
                        subs.append((int((sub_lbl == j).sum()), np.vstack([c.reshape(-1, 2) for c in cnts])))
                if subs:
                    subs.sort(key=lambda t: -t[0])
                    cand = [q[1] for q in subs] + [p[1] for p in pieces[1:]]
            # greedy: start from the largest piece, add nearest pieces while
            # the enclosing circle stays within the typical core size
            chosen = [cand[0]]
            cx, cy, r = enclosing(chosen)
            rest = cand[1:]
            changed = True
            while changed and rest:
                changed = False
                rest.sort(key=lambda q: np.hypot(q[:, 0].mean() - cx, q[:, 1].mean() - cy))
                for q in list(rest):
                    ncx, ncy, nr = enclosing(chosen + [q])
                    if nr <= bound:
                        chosen.append(q)
                        rest.remove(q)
                        cx, cy, r = ncx, ncy, nr
                        changed = True
                        break
        if r > 1.5 * core_px / 2 or r < 0.2 * core_px / 2:
            continue   # merged neighbours or debris
        yy, xx = np.ogrid[:footprint.shape[0], :footprint.shape[1]]
        inside = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
        fill = float(footprint[inside].sum() / max(1, inside.sum()))
        circles.append((float(cx + ox), float(cy + oy), float(r), fill))
    return circles


# ---------------------------------------------------------------------------
# Grid inference and orientation
# ---------------------------------------------------------------------------

def fit_lattice(circles, rows, cols):
    """Least-squares lattice ``(x, y) = a + b*col + c*row`` through circle
    centres, refitted without outliers (residual above three times the median
    residual, at least 0.05 pitch) so displaced objects cannot drag the grid
    towards themselves. Returns ``(coef_x, coef_y, pitch)``; handles a single
    populated row or column."""
    pts = np.array([[c[0], c[1]] for c in circles], dtype=float)
    rows = np.asarray(rows, float)
    cols = np.asarray(cols, float)
    n = len(pts)
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    pitch = float(np.median(d.min(axis=1))) if n > 1 else 1.0
    keep = np.ones(n, bool)
    for _ in range(4):
        A = np.stack([np.ones(n), cols, rows], axis=1)[keep]
        coef_x, *_ = np.linalg.lstsq(A, pts[keep, 0], rcond=None)
        coef_y, *_ = np.linalg.lstsq(A, pts[keep, 1], rcond=None)
        if len(np.unique(rows[keep])) < 2:
            coef_x[2], coef_y[2] = 0.0, pitch
        if len(np.unique(cols[keep])) < 2:
            coef_x[1], coef_y[1] = pitch, 0.0
        pred = np.stack([coef_x[0] + coef_x[1] * cols + coef_x[2] * rows,
                         coef_y[0] + coef_y[1] * cols + coef_y[2] * rows], axis=1)
        res = np.sqrt(((pts - pred) ** 2).sum(1))
        new_keep = res < max(3.0 * float(np.median(res[keep])), 0.05 * pitch)
        if new_keep.sum() < 3 or np.array_equal(new_keep, keep):
            break
        keep = new_keep
    pitch = float(0.5 * (np.hypot(coef_x[1], coef_y[1]) + np.hypot(coef_x[2], coef_y[2])))
    return coef_x, coef_y, pitch


def stain_fraction(sat, cx, cy, r, sat_thresh):
    """Fraction of the disc (fit-level pixels) whose saturation exceeds ``sat_thresh``."""
    h, w = sat.shape
    y0, y1 = max(0, int(cy - r)), min(h, int(cy + r) + 1)
    x0, x1 = max(0, int(cx - r)), min(w, int(cx + r) + 1)
    if y1 <= y0 or x1 <= x0:
        return 0.0
    yy, xx = np.ogrid[y0:y1, x0:x1]
    disc = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
    return float((sat[y0:y1, x0:x1][disc] > sat_thresh).sum() / max(1, disc.sum()))


def recover_faint_cores(img_bgr, circles, rows, cols, n_rows, n_cols, pixel_size_um,
                        core_diameter_um=1250.0, sat_thresh=15, val_ratio=0.965, min_fill=0.03):
    """Look for pale cores at empty grid positions.

    A linear lattice model ``(x, y) = f(row, col)`` is fitted to the cores
    already found (this absorbs slide rotation). For every empty cell the
    expected disc is tested with a more permissive tissue threshold (half the
    saturation threshold, twice the brightness margin); when at least
    ``min_fill`` of the disc is tissue a core is added at the tissue centroid
    with the median radius. Returns ``(circles, rows, cols, recovered_flags)``.
    """
    n = len(circles)
    if n < 3 or n_rows * n_cols <= n:
        return circles, rows, cols, [False] * n
    pts = np.array([[c[0], c[1]] for c in circles], dtype=float)
    A = np.stack([np.ones(n), cols.astype(float), rows.astype(float)], axis=1)
    coef_x, *_ = np.linalg.lstsq(A, pts[:, 0], rcond=None)
    coef_y, *_ = np.linalg.lstsq(A, pts[:, 1], rcond=None)
    # With a single populated row (or column) the lattice slope along that
    # axis is undetermined; fall back to the pitch measured on the other axis.
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    pitch = float(np.median(d.min(axis=1)))
    if len(np.unique(rows)) < 2:
        coef_x[2], coef_y[2] = 0.0, pitch
    if len(np.unique(cols)) < 2:
        coef_x[1], coef_y[1] = pitch, 0.0
    r_med = float(np.median([c[2] for c in circles]))
    core_px = core_diameter_um / pixel_size_um
    if not (0.2 * core_px / 2 < r_med < 1.5 * core_px / 2):
        r_med = core_px / 2

    permissive = tissue_mask(img_bgr, sat_thresh=max(2, sat_thresh / 2),
                             val_ratio=1 - 2 * (1 - val_ratio))
    h, w = permissive.shape
    occupied = {(int(r), int(c)) for r, c in zip(rows, cols)}
    out_c, out_r, out_k, flags = list(circles), list(rows), list(cols), [False] * n
    for r in range(n_rows):
        for c in range(n_cols):
            if (r, c) in occupied:
                continue
            cx = coef_x[0] + coef_x[1] * c + coef_x[2] * r
            cy = coef_y[0] + coef_y[1] * c + coef_y[2] * r
            if not (r_med <= cx < w - r_med and r_med <= cy < h - r_med):
                continue
            y0, y1 = int(cy - r_med), int(cy + r_med) + 1
            x0, x1 = int(cx - r_med), int(cx + r_med) + 1
            win = permissive[y0:y1, x0:x1]
            yy, xx = np.ogrid[y0:y1, x0:x1]
            disc = (xx - cx) ** 2 + (yy - cy) ** 2 <= r_med ** 2
            tissue = (win > 0) & disc
            frac = float(tissue.sum() / max(1, disc.sum()))
            if frac < min_fill:
                continue
            ys, xs = np.nonzero(tissue)
            mx, my = float(xs.mean() + x0), float(ys.mean() + y0)
            # keep the centre near the lattice prediction (tissue may be off-centre)
            mx = cx + np.clip(mx - cx, -0.25 * r_med, 0.25 * r_med)
            my = cy + np.clip(my - cy, -0.25 * r_med, 0.25 * r_med)
            out_c.append((mx, my, r_med, frac))
            out_r.append(r)
            out_k.append(c)
            flags.append(True)
    return out_c, np.array(out_r, int), np.array(out_k, int), flags


def _lattice_indices(v, pitch):
    """Snap 1-D positions to a lattice ``x0 + k * pitch``.

    The phase ``x0`` is the circular mean of ``v mod pitch``; pitch and phase
    are then refined by least squares and the indices re-assigned. Returns
    0-based integer indices."""
    ang = 2 * np.pi * (v % pitch) / pitch
    x0 = pitch * (np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()) / (2 * np.pi)) % pitch
    k = np.round((v - x0) / pitch).astype(int)
    for _ in range(3):
        if k.max() == k.min():
            break
        A = np.stack([np.ones_like(v), k.astype(float)], axis=1)
        (x0, pitch), *_ = np.linalg.lstsq(A, v, rcond=None)
        k = np.round((v - x0) / pitch).astype(int)
    return k - k.min()


def infer_grid(circles, core_diameter_px=None):
    """Assign row and column indices to circle centres.

    The pitch is the median nearest-neighbour distance (ignoring fragments
    closer than 0.6 core diameters); rows and columns are then snapped to a
    lattice of that pitch independently in y and x. Returns
    ``(rows, cols, n_rows, n_cols)``.
    """
    pts = np.array([[c[0], c[1]] for c in circles], dtype=float)
    if len(pts) < 2:
        return np.zeros(len(pts), int), np.zeros(len(pts), int), 1, 1
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    nn = d.min(axis=1)
    if core_diameter_px:
        nn = nn[nn > 0.6 * core_diameter_px]
    pitch = float(np.median(nn)) if len(nn) else float(np.median(d.min(axis=1)))
    rows = _lattice_indices(pts[:, 1], pitch)
    cols = _lattice_indices(pts[:, 0], pitch)
    return rows, cols, int(rows.max()) + 1, int(cols.max()) + 1


def merge_grid_duplicates(circles, rows, cols, max_radius=None):
    """Merge circles that fell into the same grid cell (fragmented cores).

    Members are taken largest first; each further member is merged only if the
    minimum enclosing circle of the merged discs stays within ``max_radius``
    (when given). Members that would inflate the circle beyond it, typically
    debris next to a core, are dropped. The fill grade is the area-weighted
    mean of the merged members. Returns ``(circles, rows, cols)``.
    """
    def disc_points(c):
        cx, cy, rad, _ = c
        ang = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        return np.stack([cx + rad * np.cos(ang), cy + rad * np.sin(ang)], axis=1)

    groups = {}
    for i, (r, c) in enumerate(zip(rows, cols)):
        groups.setdefault((int(r), int(c)), []).append(i)
    out_c, out_r, out_k = [], [], []
    for (r, c), idx in sorted(groups.items()):
        if len(idx) == 1:
            out_c.append(circles[idx[0]])
        else:
            idx = sorted(idx, key=lambda i: -circles[i][2])
            members = [idx[0]]
            for i in idx[1:]:
                pts = np.vstack([disc_points(circles[j]) for j in members + [i]])
                (_, _), mr = cv2.minEnclosingCircle(pts.astype(np.float32))
                if max_radius is None or mr <= max_radius:
                    members.append(i)
            if len(members) == 1:
                out_c.append(circles[members[0]])
            else:
                pts = np.vstack([disc_points(circles[j]) for j in members])
                (mx, my), mr = cv2.minEnclosingCircle(pts.astype(np.float32))
                areas = np.array([circles[j][2] ** 2 for j in members])
                fill = float(sum(circles[j][3] * a for j, a in zip(members, areas)) / areas.sum())
                out_c.append((float(mx), float(my), float(mr), fill))
        out_r.append(r)
        out_k.append(c)
    return out_c, np.array(out_r, int), np.array(out_k, int)


def _transform_grid(grid, orientation):
    """Apply one of the eight orientations to a 2-D array."""
    rot, flip = (orientation.split("+") + [""])[:2]
    out = np.rot90(grid, k=int(rot) // 90)
    if flip:
        out = out[:, ::-1]
    return out


def match_orientation(observed, array_number, orientation="auto"):
    """Match the observed occupancy grid to the printed map.

    ``observed`` is a boolean ``(n_rows, n_cols)`` array of the scan. For each
    candidate orientation the transformed map is compared at every offset that
    keeps the observed grid inside it. The score rewards cores present in both,
    penalises cores present in the scan but absent from the map heavily (they
    cannot exist) and dropped cores lightly (cores fall off sections).

    Returns ``(orientation, index_lookup, score, ties)`` where ``index_lookup``
    is an ``(n_rows, n_cols, 2)`` array giving the map ``(row, col)`` of each
    scan cell (``-1`` where the scan cell lies outside the map) and ``ties``
    lists every orientation that reached the same best score.

    Occupancy alone cannot distinguish a transform from its mirror image when
    the array is fully populated, so ``ties`` is frequently non-empty for
    arrays 3 to 8 and the caller should let the user confirm the orientation.
    Ties are broken in the order of :data:`ORIENTATIONS`.
    """
    map_occ = occupancy_grid(array_number)
    map_idx = np.stack(np.meshgrid(np.arange(MAP_ROWS), np.arange(MAP_COLS), indexing="ij"), axis=-1)
    candidates = [o for o in ORIENTATIONS[1:]] if orientation == "auto" else [orientation]
    best = None
    for o in candidates:
        occ_t = _transform_grid(map_occ, o)
        idx_t = _transform_grid(map_idx, o)
        mh, mw = occ_t.shape
        oh, ow = observed.shape
        for oy in range(min(0, mh - oh), max(0, mh - oh) + 1):
            for ox in range(min(0, mw - ow), max(0, mw - ow) + 1):
                score = 0.0
                lookup = np.full((oh, ow, 2), -1, dtype=int)
                for r in range(oh):
                    for c in range(ow):
                        mr, mc = r + oy, c + ox
                        inside = 0 <= mr < mh and 0 <= mc < mw
                        has_map = inside and occ_t[mr, mc]
                        if inside:
                            lookup[r, c] = idx_t[mr, mc]
                        if observed[r, c] and has_map:
                            score += 1.0
                        elif observed[r, c] and not has_map:
                            score -= 3.0
                        elif has_map and not observed[r, c]:
                            score -= 0.5
                if best is None or score > best[2] + 1e-9:
                    best = (o, lookup, score)
                    ties = [o]
                elif abs(score - best[2]) <= 1e-9 and o not in ties:
                    ties.append(o)
    return best[0], best[1], best[2], ties


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------

class TMAPreprocessor:
    """Fit, map and export the cores of one TMA slide.

    Parameters
    ----------
    slide_path : str
        ``.vsi`` slide or a flat whole-slide image.
    array_number : int or None
        ICGC array number used to look up patient IDs. ``None`` skips ID
        mapping and names cores by grid position only.
    slide_name : str or None
        Prefix for output filenames. Defaults to the slide's own name.
    pixel_size_um : float or None
        Override for slides without calibration metadata.
    core_diameter_um : float
        Nominal core diameter.
    margin_um : float
        Extra border around the fitted circle in the crop.
    erode_px : int
        Shrink of the circle mask, in level-0 pixels, to keep the core edge
        out of the analysis.
    fit_pixel_size_um : float
        Resolution at which circle fitting is performed.
    reader : SlideReader, optional
        An already opened slide; ``slide_path`` is then informational only.
    sat_thresh : int
        HSV saturation above which a pixel counts as tissue (lower = more
        sensitive to pale cores, more debris).
    val_ratio : float
        Pixels darker than this fraction of the per-column background also
        count as tissue.
    recover_faint : bool
        After grid inference, test empty grid positions for pale tissue with
        a permissive threshold and add cores flagged ``recovered``.
    min_fill : float
        Minimum tissue fraction of the expected disc for a faint core.
    max_grid_offset : float
        QC: exclude cores whose centre is further than this many grid pitches
        from the lattice-predicted position (debris between cores).
    min_diameter_frac, max_diameter_frac : float
        QC: exclude cores whose fitted diameter is outside this range, as a
        fraction of ``core_diameter_um``.
    min_stain_frac, stain_sat : float, int
        QC: exclude cores where less than ``min_stain_frac`` of the disc has
        HSV saturation above ``stain_sat``. Control cores are exempt.

    Patient IDs never depend on the QC filter values: the orientation match
    ignores only objects more than :data:`MAP_MAX_GRID_OFFSET` pitches from
    any grid position (fixed, not user-tunable), and every QC filter is
    applied after IDs are assigned.
    """

    def __init__(self, slide_path, array_number=None, slide_name=None, pixel_size_um=None,
                 core_diameter_um=1250.0, margin_um=30.0, erode_px=8, fit_pixel_size_um=5.0,
                 orientation="auto", reader=None, sat_thresh=15, val_ratio=0.965,
                 recover_faint=True, min_fill=0.03, max_grid_offset=0.35,
                 min_diameter_frac=0.8, max_diameter_frac=1.2,
                 min_stain_frac=0.02, stain_sat=40):
        self.reader = reader if reader is not None else open_slide(slide_path, pixel_size_um=pixel_size_um)
        if reader is not None and pixel_size_um:
            self.reader.pixel_size_um = pixel_size_um
        if not self.reader.pixel_size_um:
            raise ValueError("Pixel size is unknown; pass pixel_size_um explicitly.")
        self.slide_path = slide_path
        self.array_number = array_number
        self.slide_name = slide_name or self.reader.slide_name
        self.core_diameter_um = core_diameter_um
        self.margin_um = margin_um
        self.erode_px = erode_px
        self.fit_pixel_size_um = fit_pixel_size_um
        self.orientation = orientation
        self.sat_thresh = sat_thresh
        self.val_ratio = val_ratio
        self.recover_faint = recover_faint
        self.min_fill = min_fill
        self.max_grid_offset = max_grid_offset
        self.min_diameter_frac = min_diameter_frac
        self.max_diameter_frac = max_diameter_frac
        self.min_stain_frac = min_stain_frac
        self.stain_sat = stain_sat
        self.cores = []
        self.grid_shape = (0, 0)
        self.matched_orientation = None
        self.match_score = None
        self.orientation_ties = []
        self._fit_level = None
        self._fit_image = None

    # -- stage 1: fit -------------------------------------------------------
    def fit(self):
        r = self.reader
        lv = r.best_level_for_pixel_size(self.fit_pixel_size_um)
        ds = r.level_downsample(lv)
        img = r.read_level(lv, channel=0)
        px_um = r.pixel_size_um * ds
        circles = fit_cores(img, px_um, self.core_diameter_um,
                            sat_thresh=self.sat_thresh, val_ratio=self.val_ratio)
        rows, cols, n_rows, n_cols = infer_grid(circles, self.core_diameter_um / px_um)
        # typical core radius on this slide, from cells holding a single circle
        cell_counts = {}
        for rr, cc in zip(rows, cols):
            cell_counts[(int(rr), int(cc))] = cell_counts.get((int(rr), int(cc)), 0) + 1
        single = [c[2] for c, rr, cc in zip(circles, rows, cols) if cell_counts[(int(rr), int(cc))] == 1]
        r_typ = float(np.median(single)) if single else self.core_diameter_um / px_um / 2
        circles, rows, cols = merge_grid_duplicates(circles, rows, cols, max_radius=1.1 * r_typ)
        recovered = [False] * len(circles)
        if self.recover_faint:
            circles, rows, cols, recovered = recover_faint_cores(
                img, circles, rows, cols, n_rows, n_cols, px_um, self.core_diameter_um,
                sat_thresh=self.sat_thresh, val_ratio=self.val_ratio, min_fill=self.min_fill)
        self.cores = [Core(index=i + 1, cx=c[0] * ds, cy=c[1] * ds, radius=c[2] * ds, fill=c[3],
                           row=int(rows[i]), col=int(cols[i]), recovered=bool(recovered[i]))
                      for i, c in enumerate(circles)]
        self.grid_shape = (n_rows, n_cols)
        self._fit_level, self._fit_image = lv, img
        self._sat = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1]
        self._pitch_px = None
        if len(circles) >= 3:
            coef_x, coef_y, pitch = fit_lattice(circles, rows, cols)
            self._pitch_px = pitch * ds
            for c in self.cores:
                px = coef_x[0] + coef_x[1] * c.col + coef_x[2] * c.row
                py = coef_y[0] + coef_y[1] * c.col + coef_y[2] * c.row
                c.grid_offset = float(np.hypot(c.cx / ds - px, c.cy / ds - py) / max(pitch, 1e-6))
        for c in self.cores:
            c.diameter_um = 2 * c.radius * r.pixel_size_um
        self.update_stain()
        # row-major numbering, matching the printed maps
        self.cores.sort(key=lambda c: (c.row, c.col))
        for i, c in enumerate(self.cores, 1):
            c.index = i
        return self.cores

    # -- stage 2: map -------------------------------------------------------
    def update_stain(self):
        """Recompute ``stain_frac`` for every core with the current ``stain_sat``."""
        if self._fit_image is None:
            return
        ds = self.reader.level_downsample(self._fit_level)
        for c in self.cores:
            c.stain_frac = stain_fraction(self._sat, c.cx / ds, c.cy / ds, c.radius / ds, self.stain_sat)

    def _geometric_reason(self, c):
        if c.grid_offset > self.max_grid_offset:
            return "off_grid"
        d_frac = c.diameter_um / self.core_diameter_um
        if d_frac < self.min_diameter_frac or d_frac > self.max_diameter_frac:
            return "diameter"
        return ""

    def map_to_array(self):
        """Assign map positions (patient IDs), then apply the QC filters.

        Only objects further than :data:`MAP_MAX_GRID_OFFSET` pitches from
        the lattice are left out of the occupancy used for the orientation
        match; the QC filters run afterwards and cannot change the IDs.
        """
        for c in self.cores:
            c.map_row = c.map_col = -1
            c.outside_map = False
        if self.array_number is None or not self.cores:
            self.matched_orientation = None
            self.apply_filters()
            return None
        observed = np.zeros(self.grid_shape, dtype=bool)
        for c in self.cores:
            if c.grid_offset <= MAP_MAX_GRID_OFFSET:
                observed[c.row, c.col] = True
        o, lookup, score, ties = match_orientation(observed, self.array_number, self.orientation)
        self.matched_orientation, self.match_score = o, score
        self.orientation_ties = ties
        amap = self.array_map()
        for c in self.cores:
            c.map_row, c.map_col = (int(v) for v in lookup[c.row, c.col])
            info = amap.get((c.map_row, c.map_col)) if c.map_row >= 0 else None
            # Debris outside the array lands in cells the map does not have.
            c.outside_map = info is None or info.empty
        self.apply_filters()
        return o

    def apply_filters(self):
        """Set ``excluded``/``reason`` on every core from the QC parameters.

        Cheap (no image access) and independent of the ID assignment, so
        callers can re-run it whenever a filter value changes.
        """
        for c in self.cores:
            info = self.core_info(c)
            reason = self._geometric_reason(c)
            is_control = info is not None and info.is_control
            if not reason and not is_control and c.stain_frac < self.min_stain_frac:
                reason = "stain"
            c.excluded = bool(reason)
            c.reason = reason
        return self.exclusion_summary()

    def exclusion_summary(self):
        """``{reason: count}`` over cores that lie on the map."""
        out = {}
        for c in self.cores:
            if c.excluded and not c.outside_map:
                out[c.reason] = out.get(c.reason, 0) + 1
        return out

    def replicate_counts(self):
        """``{patient_id: (kept, total)}`` over patient cores on the map."""
        counts = {}
        for c in self.cores:
            info = self.core_info(c)
            if info is None or not info.patient_id or c.outside_map:
                continue
            kept, total = counts.get(info.patient_id, (0, 0))
            counts[info.patient_id] = (kept + (not c.excluded), total + 1)
        return counts

    def exportable_cores(self):
        return [c for c in self.cores if not c.outside_map and not c.excluded]

    def array_map(self):
        return load_array_map(self.array_number) if self.array_number is not None else {}

    def core_info(self, core):
        if core.map_row < 0:
            return None
        return self.array_map().get((core.map_row, core.map_col))

    def core_stem(self, core, channel=None):
        info = self.core_info(core)
        if info is not None:
            return core_stem(self.slide_name, core.map_row, core.map_col, info, channel)
        return core_stem(self.slide_name, core.row, core.col, None, channel)

    # -- stage 3: export ----------------------------------------------------
    def export(self, out_dir, channels=None, progress=None, cancel=None):
        """Write per-core images and masks, ``cores.csv`` and ``overlay.png``.

        Each channel gets a self-contained folder ``<out>/<channel>/Images``
        and ``<out>/<channel>/Masks`` (see :meth:`channel_dirs`), usable
        directly as the input and ROI-mask folders of a batch run.
        ``progress(done, total, message)`` is called after each file;
        ``cancel()`` returning True stops the export early.
        """
        r = self.reader
        channels = list(channels or r.channels)
        dirs = {}
        for ch in channels:
            img_dir, mask_dir = self.channel_dirs(out_dir, ch)
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(mask_dir, exist_ok=True)
            dirs[ch] = (img_dir, mask_dir)
        margin_px = self.margin_um / r.pixel_size_um
        cores = self.exportable_cores()
        total = len(cores) * len(channels)
        done = 0
        for core in cores:
            side = int(round(2 * (core.radius + margin_px)))
            x0 = int(round(core.cx - side / 2))
            y0 = int(round(core.cy - side / 2))
            mask = np.zeros((side, side), dtype=np.uint8)
            cv2.circle(mask, (int(round(core.cx - x0)), int(round(core.cy - y0))),
                       max(1, int(round(core.radius - self.erode_px))), 255, -1)
            for ch in channels:
                if cancel is not None and cancel():
                    return False
                stem = self.core_stem(core, ch)
                img_dir, mask_dir = dirs[ch]
                crop = r.read_region(x0, y0, side, side, channel=ch, level=0)
                write_png_with_resolution(os.path.join(img_dir, stem + ".png"), crop, r.pixel_size_um)
                cv2.imwrite(os.path.join(mask_dir, stem + ".png"), mask)
                done += 1
                if progress is not None:
                    progress(done, total, stem)
        self.write_manifest(os.path.join(out_dir, "cores.csv"), channels[0])
        cv2.imwrite(os.path.join(out_dir, "overlay.png"), self.draw_overlay())
        return True

    @staticmethod
    def channel_dirs(out_dir, channel):
        """``(images_dir, masks_dir)`` for one channel of an export."""
        base = os.path.join(out_dir, str(channel))
        return os.path.join(base, "Images"), os.path.join(base, "Masks")

    def write_manifest(self, path, channel=None):
        fields = ["stem", "slide", "array", "scan_row", "scan_col", "map_position", "map_label",
                  "sector", "patient_id", "icgc_id", "tissue", "note", "cx", "cy", "radius_px",
                  "fill", "grid_offset", "diameter_um", "stain_frac", "flag", "excluded",
                  "reason", "orientation"]
        amap = self.array_map()
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for c in self.cores:
                info = amap.get((c.map_row, c.map_col)) if c.map_row >= 0 else None
                w.writerow({
                    "stem": self.core_stem(c, channel),
                    "slide": self.slide_name,
                    "array": self.array_number if self.array_number is not None else "",
                    "scan_row": c.row + 1, "scan_col": c.col + 1,
                    "map_position": info.position if info else "",
                    "map_label": info.label if info else "",
                    "sector": info.sector if info else "",
                    "patient_id": info.patient_id if info else "",
                    "icgc_id": info.icgc_id if info else "",
                    "tissue": info.tissue if info else "",
                    "note": info.note if info else "",
                    "cx": round(c.cx, 1), "cy": round(c.cy, 1), "radius_px": round(c.radius, 1),
                    "fill": round(c.fill, 3), "grid_offset": round(c.grid_offset, 3),
                    "diameter_um": round(c.diameter_um, 1), "stain_frac": round(c.stain_frac, 3),
                    "flag": c.flag, "excluded": int(c.excluded), "reason": c.reason,
                    "orientation": self.matched_orientation or "",
                })

    def draw_overlay(self):
        """Fit-level image with numbered, colour-graded circles and map labels."""
        if self._fit_image is None:
            raise RuntimeError("Call fit() first")
        ds = self.reader.level_downsample(self._fit_level)
        ov = self._fit_image.copy()
        scale = max(0.4, ov.shape[1] / 3000.0)
        for c in self.cores:
            colour = {"ok": (0, 180, 0), "recovered": (200, 0, 200), "outside_map": (128, 128, 128),
                      "excluded": (90, 90, 90)}[c.flag]
            centre = (int(c.cx / ds), int(c.cy / ds))
            rad = int(c.radius / ds)
            thick = max(1, int(2 * scale))
            cv2.circle(ov, centre, rad, colour, thick)
            if c.excluded:
                k = int(0.5 * rad)
                cv2.line(ov, (centre[0] - k, centre[1] - k), (centre[0] + k, centre[1] + k), colour, thick)
                cv2.line(ov, (centre[0] - k, centre[1] + k), (centre[0] + k, centre[1] - k), colour, thick)
            info = self.core_info(c)
            text = f"{c.index}" + (f" {info.position}" if info and not info.empty else "")
            cv2.putText(ov, text, (centre[0] - int(20 * scale), centre[1] + int(6 * scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, (255, 0, 0), max(1, int(2 * scale)), cv2.LINE_AA)
        return ov

    def run(self, out_dir, channels=None, progress=None, cancel=None):
        self.fit()
        self.map_to_array()
        return self.export(out_dir, channels=channels, progress=progress, cancel=cancel)

    def close(self):
        self.reader.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    import argparse
    from tqdm import tqdm
    p = argparse.ArgumentParser(prog="cabana-tma",
                                description="Fit TMA cores and export per-core images and masks.")
    p.add_argument("slide", help=".vsi slide or whole-slide TIFF/PNG")
    p.add_argument("out_dir", help="output folder (<channel>/Images, <channel>/Masks, cores.csv, overlay.png)")
    p.add_argument("--array", type=int, default=None, help="ICGC array number for patient-ID lookup")
    p.add_argument("--slide-name", default=None, help="filename prefix (default: slide name)")
    p.add_argument("--pixel-size", type=float, default=None, help="µm per pixel if not in metadata")
    p.add_argument("--core-diameter", type=float, default=1250.0, help="nominal core diameter in µm (default 1250)")
    p.add_argument("--margin", type=float, default=30.0, help="crop margin around the circle in µm (default 30)")
    p.add_argument("--erode", type=int, default=8, help="mask shrink in pixels")
    p.add_argument("--orientation", choices=ORIENTATIONS, default="auto")
    p.add_argument("--channels", nargs="*", default=None, help="channels to export (default: all)")
    p.add_argument("--sat-thresh", type=int, default=15,
                   help="tissue saturation threshold; lower finds paler cores (default 15)")
    p.add_argument("--val-ratio", type=float, default=0.965,
                   help="pixels darker than this fraction of the background are tissue (default 0.965)")
    p.add_argument("--no-recover", action="store_true", help="disable the faint-core pass at empty grid cells")
    p.add_argument("--min-fill", type=float, default=0.03,
                   help="minimum tissue fraction for a faint core (default 0.03)")
    p.add_argument("--max-grid-offset", type=float, default=0.35,
                   help="QC: max distance from the grid position, in pitches (default 0.35)")
    p.add_argument("--diameter-range", type=float, nargs=2, default=(0.8, 1.2), metavar=("MIN", "MAX"),
                   help="QC: allowed diameter as fractions of --core-diameter (default 0.8 1.2)")
    p.add_argument("--min-stain", type=float, default=0.02,
                   help="QC: minimum stained fraction; controls exempt (default 0.02)")
    p.add_argument("--stain-sat", type=int, default=40,
                   help="QC: saturation above which a pixel counts as stained (default 40)")
    p.add_argument("--fit-only", action="store_true", help="write cores.csv and overlay.png only")
    a = p.parse_args(argv)

    pre = TMAPreprocessor(a.slide, array_number=a.array, slide_name=a.slide_name,
                          pixel_size_um=a.pixel_size, core_diameter_um=a.core_diameter,
                          margin_um=a.margin, erode_px=a.erode, orientation=a.orientation,
                          sat_thresh=a.sat_thresh, val_ratio=a.val_ratio,
                          recover_faint=not a.no_recover, min_fill=a.min_fill,
                          max_grid_offset=a.max_grid_offset, min_diameter_frac=a.diameter_range[0],
                          max_diameter_frac=a.diameter_range[1],
                          min_stain_frac=a.min_stain, stain_sat=a.stain_sat)
    pre.fit()
    o = pre.map_to_array()
    print(f"{pre.slide_name}: {len(pre.cores)} cores on a {pre.grid_shape[0]}x{pre.grid_shape[1]} grid"
          + (f", orientation {o} (score {pre.match_score:.1f})" if o else ""))
    summ = pre.exclusion_summary()
    if summ:
        print("  excluded: " + ", ".join(f"{v} {k}" for k, v in sorted(summ.items())))
    lost = [p for p, (k, t) in pre.replicate_counts().items() if k == 0]
    if lost:
        print(f"  patients with no remaining core: {', '.join(lost)}")
    if len(pre.orientation_ties) > 1:
        print(f"  WARNING: orientations {pre.orientation_ties} fit equally well; "
              f"confirm against the printed map or pass --orientation.")
    os.makedirs(a.out_dir, exist_ok=True)
    if a.fit_only:
        pre.write_manifest(os.path.join(a.out_dir, "cores.csv"), (a.channels or pre.reader.channels)[0])
        cv2.imwrite(os.path.join(a.out_dir, "overlay.png"), pre.draw_overlay())
    else:
        bar = tqdm(total=len(pre.exportable_cores()) * len(a.channels or pre.reader.channels), unit="img")
        pre.export(a.out_dir, channels=a.channels,
                   progress=lambda d, t, s: (bar.update(1), bar.set_postfix_str(s)))
        bar.close()
    pre.close()


if __name__ == "__main__":
    main()
