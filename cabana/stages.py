"""Shared per-image (and per-batch) pipeline stages.

Each function here is a pure-ish unit invoked by both ``Cabana`` (single image)
and ``BatchCabana`` (folder of images). Keeping the science in one place
prevents the two pipelines from drifting apart.
"""

import os
from glob import glob
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import imageio.v3 as iio
import tifffile
from skimage.color import rgb2hed, hed2rgb, rgb2gray

from .hdm import HDM
from .detector import FibreDetector
from .analyzer import SkeletonAnalyzer
from .orientation import OrientationAnalyzer
from PIL import Image
from skimage.feature import peak_local_max
from sklearn.metrics.pairwise import euclidean_distances

from .utils import join_path
from .viz import (mask_color_map, overlay_colorbar, color_survey_with_colorbar, skeleton_value_map,
                  image_with_colorbar, DIRECTION_CMAP)



def write_map_tiff(path, arr):
    """Write an analysis map as a TIFF with lossless deflate compression.

    float64 maps are stored as float32: the relative rounding is below 1e-7,
    far below anything the maps resolve, and halves the file. Compression
    matters most for the skeleton maps (length, curvature), which are almost
    entirely zero."""
    arr = np.asarray(arr)
    if arr.dtype == np.float64:
        arr = arr.astype(np.float32)
    tifffile.imwrite(path, arr, compression="zlib")

def summary_stats(values, label, include_count=False, count_label=None):
    """Return mean/std/percentile5/median/percentile95 keyed by ``'<stat> ({label})'``.

    If ``values`` is empty, every entry is 0. When ``include_count`` is true,
    a count entry is added under ``count_label`` (defaulting to
    ``f'Count ({label})'``).
    """
    keys = ('Mean', 'Std', 'Percentile5', 'Median', 'Percentile95')
    if len(values) == 0:
        out = {f'{k} ({label})': 0 for k in keys}
    else:
        out = {
            f'Mean ({label})':         float(np.mean(values)),
            f'Std ({label})':          float(np.std(values)),
            f'Percentile5 ({label})':  float(np.percentile(values, 5)),
            f'Median ({label})':       float(np.median(values)),
            f'Percentile95 ({label})': float(np.percentile(values, 95)),
        }
    if include_count:
        out[count_label or f'Count ({label})'] = int(len(values))
    return out


def safe_div(stats, num_col, den_col, out_col, denom_transform=None):
    """Single-row stats DataFrame helper: write num/den (or 0) to out_col.

    No-op when either input column is missing (the output column is not
    created). When both are present and the (optionally transformed)
    denominator is non-positive, writes 0. Mutates and returns ``stats``.
    """
    if num_col not in stats.columns or den_col not in stats.columns:
        return stats
    d = stats.loc[0, den_col]
    if denom_transform is not None:
        d = denom_transform(d)
    stats.loc[0, out_col] = stats.loc[0, num_col] / d if d > 0 else 0
    return stats


ORIENT_METRICS = (
    'Orient. Alignment', 'Orient. Variance',
    'Orient. Alignment (ROI)', 'Orient. Variance (ROI)',
    'Orient. Alignment (HDM)', 'Orient. Variance (HDM)',
    'Orient. Alignment (WIDTH)', 'Orient. Variance (WIDTH)',
)


FIBRE_AREA_METRICS = (
    'Area (ROI)', '% ROI Area', 'Area (WIDTH)', '% WIDTH Area',
    'Mean Fibre Intensity (ROI)', 'Mean Fibre Intensity (WIDTH)',
    'Mean Fibre Intensity (HDM)',
)


def run_hdm(args, source_path, hdm_dir, ext='.png', mask_dir=None):
    """Run HDM quantification on a single image path or a directory of images.

    Parameters
    ----------
    args : dict
        Loaded parameters YAML — needs Quantification + Detection sections.
    source_path : str
        Either a single image path or a directory containing images.
    hdm_dir : str
        Output directory for HDM artifacts and ``ResultsHDM.csv``.
    ext : str
        Image extension(s) to process. Default ``.png``.
    mask_dir : str, optional
        Folder of prepared ROI masks (``<stem>.png``); HDM is restricted to
        the mask and ``% HDM Area`` is relative to the mask area.

    Returns
    -------
    pandas.DataFrame
        Per-image HDM quantification results (the same object stored on
        ``HDM.df_hdm`` after the call).
    """
    hdm = HDM(
        max_hdm=args["Quantification"]["Maximum Display HDM"],
        sat_ratio=args["Quantification"]["Contrast Enhancement"],
        dark_line=args["Detection"]["Dark Line"],
    )
    hdm.quantify_black_space(source_path, hdm_dir, ext=ext, mask_dir=mask_dir)
    return hdm.df_hdm


ROI_MASK_DIRNAME = 'ROIMasks'
_MASK_SUFFIXES = ('.png', '_mask.png', '.tif', '.tiff')


def find_external_mask(mask_dir, stem):
    """Locate the external ROI mask for an image stem inside ``mask_dir``.

    Accepts ``<stem>.png``, ``<stem>_mask.png``, ``<stem>.tif``/``.tiff``.
    Returns the path or ``None``.
    """
    if not mask_dir or not stem or not os.path.isdir(mask_dir):
        return None
    for suffix in _MASK_SUFFIXES:
        candidate = join_path(mask_dir, stem + suffix)
        if os.path.exists(candidate):
            return candidate
    return None


def prepare_roi_mask(mask_dir, stem, shape, out_dir, out_stem=None, crop_box=None):
    """Copy the external mask for ``stem`` into ``out_dir`` as a 0/255 PNG
    matching the analysed image.

    ``shape`` is the (h, w) of the *full* input image; the mask is resized to
    it (nearest neighbour) and then, when ``crop_box=(x0, y0, x1, y1)`` is
    given, cropped to the block that is actually analysed. Returns the written
    path or ``None`` when no mask exists.
    """
    from .segmenter import load_roi_mask
    src = find_external_mask(mask_dir, stem)
    if src is None:
        return None
    mask = load_roi_mask(src, shape)
    if mask is None:
        return None
    if crop_box is not None:
        x0, y0, x1, y1 = crop_box
        mask = mask[y0:y1, x0:x1]
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    dst = join_path(out_dir, (out_stem or stem) + '.png')
    cv2.imwrite(dst, mask)
    return dst


def roi_mask_for(roimask_dir, stem):
    """Load the prepared ROI mask (0/255 ``uint8``) for ``stem`` or ``None``."""
    path = join_path(roimask_dir, stem + '.png') if roimask_dir else None
    if not path or not os.path.exists(path):
        return None
    mask = cv2.imread(path, 0)
    return None if mask is None else ((mask > 128).astype(np.uint8) * 255)


def restrict_to_roi(binary_black_on_white, roi_mask, erode_px=0):
    """Blank a black-on-white fibre image outside ``roi_mask``.

    The mask is eroded by ``erode_px`` first so the artificial edge between
    tissue and background fill is never reported as a fibre.
    """
    if roi_mask is None:
        return binary_black_on_white
    keep = roi_mask
    if erode_px and erode_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(erode_px) + 1, 2 * int(erode_px) + 1))
        keep = cv2.erode(roi_mask, k)
    out = binary_black_on_white.copy()
    out[keep == 0] = 255
    return out


def compute_fibre_areas(img_mask_path, ori_img_path, width_mask_path, hdm_mask_path,
                        roi_mask_path=None):
    """Compute the seven fibre-area / mean-intensity metrics for one image.

    Returns a dict keyed by ``FIBRE_AREA_METRICS``. If the ROI mask is empty,
    all values are 0. When ``roi_mask_path`` (an external analysis-region
    mask, e.g. a TMA core circle) is given, the percentage metrics are taken
    relative to that region instead of the whole image and the WIDTH mask is
    restricted to it.
    """
    img_mask = cv2.imread(img_mask_path, 0)
    area_roi = float(np.sum(img_mask > 128))
    if area_roi == 0:
        return dict.fromkeys(FIBRE_AREA_METRICS, 0)

    analysis_area = float(img_mask.shape[0] * img_mask.shape[1])
    ext_mask = None
    if roi_mask_path and os.path.exists(roi_mask_path):
        ext_mask = cv2.imread(roi_mask_path, 0)
        if ext_mask is not None and ext_mask.shape[:2] == img_mask.shape[:2] and np.any(ext_mask > 128):
            analysis_area = float(np.sum(ext_mask > 128))
        else:
            ext_mask = None
    percent_roi = area_roi / analysis_area
    ori_img = np.asarray(iio.imread(ori_img_path))
    if ori_img.ndim == 2:
        ori_img = np.stack([ori_img] * 3, axis=-1)
    elif ori_img.ndim == 3 and ori_img.shape[2] == 4:
        ori_img = ori_img[..., :3]          # colour deconvolution needs exactly three channels

    # eosin channel only; intermediates are freed early (several full-size
    # float64 copies otherwise coexist on large images)
    eosin = rgb2hed(ori_img)[:, :, 1].copy()
    null = np.zeros_like(eosin)
    ihc_e = hed2rgb(np.stack((null, eosin, null), axis=-1))
    del eosin, null
    red_img = (rgb2gray(ihc_e) * 255).astype(np.uint8)
    del ihc_e

    width_mask = cv2.imread(width_mask_path, 0)
    hdm_mask = cv2.imread(hdm_mask_path, 0)
    if ext_mask is not None:
        width_mask = width_mask.copy()
        width_mask[ext_mask <= 128] = 255
        hdm_mask = hdm_mask.copy()
        hdm_mask[ext_mask <= 128] = 0
    area_width = float(np.sum(width_mask < 128))
    percent_width = area_width / analysis_area

    grayscale = (rgb2gray(ori_img) * 255).astype(np.uint8)
    if np.count_nonzero(red_img < 180):
        mean_intensity_roi = np.mean(red_img[(img_mask > 128) & (red_img < 180)])
        mean_intensity_width = np.mean(red_img[(width_mask < 128) & (red_img < 180)])
        mean_intensity_hdm = np.mean(red_img[(hdm_mask > 0) & (red_img < 180)])
    else:
        mean_intensity_roi = np.mean(grayscale[img_mask > 128])
        mean_intensity_width = np.mean(grayscale[width_mask < 128])
        mean_intensity_hdm = np.mean(grayscale[hdm_mask > 0])

    if np.isnan(mean_intensity_roi):
        mean_intensity_roi = np.mean(grayscale[img_mask > 128]) if np.any(img_mask > 128) else 0
    if np.isnan(mean_intensity_width):
        mean_intensity_width = np.mean(grayscale[width_mask < 128]) if np.any(width_mask < 128) else 0
    if np.isnan(mean_intensity_hdm):
        mean_intensity_hdm = np.mean(grayscale[hdm_mask > 0]) if np.any(hdm_mask > 0) else 0

    return {
        'Area (ROI)': area_roi,
        '% ROI Area': percent_roi,
        'Area (WIDTH)': area_width,
        '% WIDTH Area': percent_width,
        'Mean Fibre Intensity (ROI)': mean_intensity_roi,
        'Mean Fibre Intensity (WIDTH)': mean_intensity_width,
        'Mean Fibre Intensity (HDM)': mean_intensity_hdm,
    }


def hdm_reference_area(stats):
    """Area (µm²) that ``% HDM Area`` refers to, per row of a stats frame: the
    analysed ROI-mask area when an external mask was used (recovered from
    ``Fibre Area (ROI, µm²)`` / ``% ROI Area``), else the whole image."""
    total = stats['Total Image Area (µm²)'].astype(float)
    if 'Fibre Area (ROI, µm²)' in stats.columns and '% ROI Area' in stats.columns:
        pct = stats['% ROI Area'].astype(float)
        roi = stats['Fibre Area (ROI, µm²)'].astype(float)
        with np.errstate(divide='ignore', invalid='ignore'):
            analysed = np.where(pct > 0, roi / pct, total)
        return pd.Series(analysed, index=stats.index)
    return total


def _range_from_params(section, min_key, max_key, step_key, cast=float):
    """``np.arange(min, max + step, step)`` with the YAML values validated."""
    lo, hi, step = cast(section[min_key]), cast(section[max_key]), cast(section[step_key])
    if step <= 0 or lo > hi:
        raise ValueError(f"{step_key} must be > 0 and {min_key} <= {max_key} "
                         f"(got {min_key}={lo}, {max_key}={hi}, {step_key}={step})")
    return np.arange(lo, hi + step, step)


def build_fibre_detector(args):
    """Construct a FibreDetector from a parsed parameters dict."""
    d = args["Detection"]
    line_widths = _range_from_params(d, "Min Line Width", "Max Line Width", "Line Width Step")
    return FibreDetector(
        line_widths=line_widths,
        low_contrast=d["Low Contrast"],
        high_contrast=d["High Contrast"],
        dark_line=d["Dark Line"],
        extend_line=d["Extend Line"],
        correct_pos=False,
        min_len=d["Minimum Line Length"],
        max_len=d["Maximum Line Length"],
    )


def detect_one_image(det, roi_img_path, mask_dir, export_subdir, color_subdir,
                     mask_filename=None, roi_mask=None, roi_erode_px=0):
    """Run detection on one ROI image and write the standard artifacts.

    Parameters
    ----------
    det : FibreDetector
    roi_img_path : str
        Path to the input ROI image.
    mask_dir : str
        Top-level Masks directory; receives the binary contour image.
    export_subdir : str
        Per-image directory under Exports/ for Mask/Width PNGs.
    color_subdir : str
        Per-image directory under Colors/ for color and gray Width PNGs.
    mask_filename : str, optional
        Filename for the mask file written to ``mask_dir``. Defaults to the
        basename of ``roi_img_path``.
    roi_mask : ndarray, optional
        External 0/255 analysis-region mask; detected fibres outside it (after
        eroding by ``roi_erode_px``) are discarded so the region edge is not
        reported as a fibre.

    Returns
    -------
    dict
        ``{'contour_img', 'width_img', 'binary_contours', 'binary_widths',
        'int_width_img'}``.
    """
    det.detect_lines(roi_img_path)
    contour_img, width_img, binary_contours, binary_widths, int_width_img = det.get_results()
    if roi_mask is not None:
        binary_contours = restrict_to_roi(binary_contours, roi_mask, roi_erode_px)
        binary_widths = restrict_to_roi(binary_widths, roi_mask, roi_erode_px)

    base = mask_filename or os.path.basename(roi_img_path)

    Path(export_subdir).mkdir(parents=True, exist_ok=True)

    iio.imwrite(join_path(mask_dir, base), binary_contours)
    iio.imwrite(join_path(export_subdir, "Mask.png"), binary_contours)
    iio.imwrite(join_path(export_subdir, "Width.png"), binary_widths)
    if color_subdir is not None:            # None: visualisations are not saved
        Path(color_subdir).mkdir(parents=True, exist_ok=True)
        iio.imwrite(join_path(color_subdir, "color_mask.png"), contour_img)
        iio.imwrite(join_path(color_subdir, "color_width.png"), width_img)
        iio.imwrite(join_path(color_subdir, "gray_width.png"), int_width_img)

    return {
        'contour_img': contour_img,
        'width_img': width_img,
        'binary_contours': binary_contours,
        'binary_widths': binary_widths,
        'int_width_img': int_width_img,
    }


def build_skeleton_analyzer(args):
    """Construct a SkeletonAnalyzer from a parsed parameters dict.

    Note: dark_line=True because fibre detection writes black-on-white masks.
    """
    min_branch_len = int(args["Quantification"]["Minimum Branch Length"])
    return SkeletonAnalyzer(
        skel_thresh=min_branch_len,
        branch_thresh=min_branch_len,
        hole_threshold=8,
        dark_line=True,
    )


def curve_windows_from_args(args):
    """Curvature window sizes (pixels) from the Quantification section."""
    return _range_from_params(args["Quantification"], "Minimum Curvature Window",
                              "Maximum Curvature Window", "Curvature Window Step", cast=int)


def quantify_one_skeleton(skel_analyzer, mask_path, export_subdir,
                          ims_res, curve_windows):
    """Analyze one fibre-mask image; return metrics + curve maps + key images.

    The caller is responsible for resetting the analyzer between calls when
    reusing an instance.

    Returns
    -------
    metrics : dict
        Per-image scalar metrics (matches the columns Cabana / BatchCabana
        write into their stats frames).
    curve_maps : dict
        Mapping from window size to curvature map array.
    key_pts_image : ndarray
    length_map : ndarray
    """
    skel_analyzer.analyze_image(mask_path)

    metrics = {
        'Area of Fibre Spines (µm²)': skel_analyzer.proj_area * ims_res ** 2,
        'Lacunarity': skel_analyzer.lacunarity,
        'Total Length (µm)': skel_analyzer.total_length * ims_res,
        'Endpoints': skel_analyzer.num_tips,
        'Avg Length (µm)': skel_analyzer.growth_unit * ims_res,
        'Branchpoints': skel_analyzer.num_branches,
        'Box-Counting Fractal Dimension': skel_analyzer.frac_dim,
        'Total Image Area (µm²)': np.prod(skel_analyzer.raw_image.shape[:2]) * ims_res ** 2,
    }

    Path(export_subdir).mkdir(parents=True, exist_ok=True)
    iio.imwrite(join_path(export_subdir, "Skeleton.png"), skel_analyzer.key_pts_image)
    write_map_tiff(join_path(export_subdir, "Length_Map.tif"),
                   (skel_analyzer.length_map_all * ims_res).astype(np.float32))

    curve_maps = {}
    for win_sz in curve_windows:
        skel_analyzer.calc_curve_all(win_sz)
        metrics[f"Curvature (win_sz={win_sz})"] = skel_analyzer.avg_curve_all
        curve_maps[int(win_sz)] = skel_analyzer.curve_map_all
        write_map_tiff(join_path(export_subdir, f"Curve_Map_{win_sz}.tif"),
                       skel_analyzer.curve_map_all)

    return metrics, curve_maps, skel_analyzer.key_pts_image, skel_analyzer.length_map_all


def _detected_fibres(export_subdir, mask_roi):
    """Detected fibre pixels inside the ROI (``Mask.png`` from fibre detection,
    fibres = 0); the ROI itself when detection has not run."""
    path = join_path(export_subdir, "Mask.png")
    if not os.path.exists(path):
        return mask_roi > 0
    return (iio.imread(path) == 0) & (mask_roi > 0)


def analyze_one_orientation(orient_analyzer, roi_img_path, mask_roi_path,
                            mask_hdm_path, mask_width_path,
                            export_subdir, color_subdir):
    """Compute orientation metrics + write artifacts for one ROI image.

    Returns ``(metrics_dict, images_dict)``. When the ROI mask is empty,
    metrics are all zero and images_dict is empty (placeholder zero-images
    are still written so downstream globs find them).
    """
    Path(export_subdir).mkdir(parents=True, exist_ok=True)
    if color_subdir is not None:            # None: visualisations are not saved
        Path(color_subdir).mkdir(parents=True, exist_ok=True)
    mask_roi = iio.imread(mask_roi_path)

    if np.sum(mask_roi) == 0:
        empty = np.zeros_like(mask_roi)
        for name in ("Energy", "Coherency", "Orientation", "Color_Survey"):
            write_map_tiff(join_path(export_subdir, f"{name}.tif"), empty)
        if color_subdir is not None:
            for name in ("orient_vf", "angular_hist"):
                iio.imwrite(join_path(color_subdir, f"{name}.png"), empty)
        return dict.fromkeys(ORIENT_METRICS, 0), {}

    mask_hdm = (iio.imread(mask_hdm_path) > 0).astype(np.uint8) * 255
    mask_width = 255 - iio.imread(mask_width_path)

    orient_analyzer.compute_orient(roi_img_path)

    metrics = {
        'Orient. Alignment': orient_analyzer.mean_coherency(),
        'Orient. Variance': orient_analyzer.circular_variance(),
        'Orient. Alignment (ROI)': orient_analyzer.mean_coherency(mask=mask_roi),
        'Orient. Variance (ROI)': orient_analyzer.circular_variance(mask=mask_roi),
        'Orient. Alignment (HDM)': orient_analyzer.mean_coherency(mask=mask_hdm),
        'Orient. Variance (HDM)': orient_analyzer.circular_variance(mask=mask_hdm),
        'Orient. Alignment (WIDTH)': orient_analyzer.mean_coherency(mask=mask_width),
        'Orient. Variance (WIDTH)': orient_analyzer.circular_variance(mask=mask_width),
    }

    images = {
        'energy': orient_analyzer.get_energy_image(),
        'coherency': orient_analyzer.get_coherency_image(),
        'orientation': orient_analyzer.get_orientation_image(),
        'color_survey': orient_analyzer.draw_color_survey(),
        'vector_field': orient_analyzer.draw_vector_field(fibre_mask=_detected_fibres(export_subdir, mask_roi)),
        'angular_hist': orient_analyzer.draw_angular_hist(mask=mask_roi),
    }

    write_map_tiff(join_path(export_subdir, "Energy.tif"), images['energy'])
    write_map_tiff(join_path(export_subdir, "Coherency.tif"), images['coherency'])
    write_map_tiff(join_path(export_subdir, "Orientation.tif"), images['orientation'])
    write_map_tiff(join_path(export_subdir, "Color_Survey.tif"), images['color_survey'])
    if color_subdir is not None:
        image_with_colorbar(images['vector_field'], join_path(color_subdir, "orient_vf.png"),
                            DIRECTION_CMAP, 0, 180, "Direction (°)", ticks=[0, 45, 90, 135, 180])
        iio.imwrite(join_path(color_subdir, "angular_hist.png"), images['angular_hist'])

    return metrics, images

def write_color_maps(rgb_img, export_subdir, color_subdir):
    """Colour overlays of one image's analysis maps (from ``export_subdir``)
    into ``color_subdir``.

    ``rgb_img`` is the analysed image (RGB). Callers skip this when
    ``Configs: Save Visualisations`` is off; the gap overlays are written
    there by the gap analysis itself.
    """
    Path(color_subdir).mkdir(parents=True, exist_ok=True)
    curves = [(os.path.basename(p)[len("Curve_Map"):-4], p)
              for p in glob(join_path(export_subdir, "Curve_Map_*"))]

    mask_img = 255 - iio.imread(join_path(export_subdir, "Mask.png"))
    if np.sum(mask_img) == 0:
        blank = np.zeros_like(rgb_img)
        names = ["color_mask", "color_skeleton", "color_energy", "color_coherency",
                 "color_orientation", "color_length", "orient_color_survey"]
        for name in names + ["color_curve" + suffix for suffix, _ in curves]:
            iio.imwrite(join_path(color_subdir, name + ".png"), blank)
        return

    mask_img = np.repeat(mask_img[:, :, np.newaxis], 3, axis=2)
    Image.fromarray(mask_color_map(rgb_img, mask_img)).save(join_path(color_subdir, "color_mask.png"))

    skeleton = iio.imread(join_path(export_subdir, "Skeleton.png"))
    red_pos = np.where((skeleton[..., 0] == 255) & (skeleton[..., 1] == 0) & (skeleton[..., 2] == 0))
    skeleton[red_pos[0], red_pos[1], :] = [0, 255, 0]
    index_pos = np.where(cv2.cvtColor(skeleton, cv2.COLOR_BGR2GRAY) == 0)
    skeleton[index_pos[0], index_pos[1], :] = rgb_img[index_pos[0], index_pos[1], :]
    Image.fromarray(skeleton).save(join_path(color_subdir, "color_skeleton.png"))

    energy_map = iio.imread(join_path(export_subdir, "Energy.tif"))
    cohere_map = iio.imread(join_path(export_subdir, "Coherency.tif"))
    orient_map = iio.imread(join_path(export_subdir, "Orientation.tif"))
    length_map = iio.imread(join_path(export_subdir, "Length_Map.tif"))

    overlay_colorbar(rgb_img, energy_map, join_path(color_subdir, "color_energy.png"),
                     clabel="Normalized Energy", mode="overlay")
    color_survey_with_colorbar(orient_map, np.ones_like(orient_map), np.ones_like(orient_map),
                               join_path(color_subdir, "color_orientation.png"),
                               clabel="Orientation (rad)")
    overlay_colorbar(rgb_img, cohere_map, join_path(color_subdir, "color_coherency.png"),
                     clabel="Coherency", mode="overlay")
    skeleton_value_map(length_map, join_path(color_subdir, "color_length.png"), "Length (µm)", "viridis", rgb_img)
    for suffix, path in curves:
        skeleton_value_map(tifffile.imread(path) / 180, join_path(color_subdir, "color_curve" + suffix + ".png"),
                           "Curliness", "viridis", rgb_img)
    color_survey_with_colorbar(orient_map, cohere_map, energy_map,
                               join_path(color_subdir, "orient_color_survey.png"))


def fit_gap_circles(gap_mask, min_gap_diameter, roi_mask=None, progress=None):
    """Fill the gaps between fibres with non-overlapping circles.

    ``gap_mask`` is non-zero in the gaps (between fibres). Circles of at
    least ``min_gap_diameter`` px are placed largest-first on the distance
    map, repeatedly, until none fits. Space outside ``roi_mask`` (when given)
    and the image border are not gaps. ``progress(fraction)`` is called after
    each round with the share of the image filled. Returns an ``(N, 3)``
    float array of ``(row, col, radius)`` in pixels.
    """
    min_gap_radius = min_gap_diameter / 2
    min_dist = int(np.max([1, min_gap_radius]))
    mask = gap_mask.copy()
    if roi_mask is not None:
        mask[roi_mask == 0] = 0
    # set border pixels to zero to avoid partial circles
    mask[0, :] = mask[-1, :] = mask[:, :1] = mask[:, -1:] = 0

    circles = []
    downsample_factor = 2
    while True:
        dist_map = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

        # downsample distance map and upscale detected centers to original image size
        dist_map_downscaled = cv2.resize(dist_map, None, fx=1 / downsample_factor, fy=1 / downsample_factor)
        centers = peak_local_max(dist_map_downscaled, min_distance=min_dist,
                                 exclude_border=False) * downsample_factor
        radius = dist_map[centers[:, 0], centers[:, 1]]
        eligible = np.hstack([centers[radius > min_gap_radius, :], radius[radius > min_gap_radius][:, None]])
        if len(eligible) == 0:
            break

        while len(eligible) > 0:
            if eligible[1:, :].size > 0:
                dist = euclidean_distances(eligible[[0], :2], eligible[1:, :2])[0]
                overlapping = np.nonzero(dist < eligible[0, 2] + eligible[1:, 2])[0] + 1
                eligible = np.delete(eligible, overlapping, axis=0)
            circle = eligible[0, :]
            cv2.circle(mask, (int(circle[1]), int(circle[0])), int(circle[2]), 0, -1)
            circles.append(circle)
            eligible = np.delete(eligible, 0, axis=0)
        if progress is not None:
            progress(np.count_nonzero(mask == 0) / mask.size)
    return np.array(circles) if circles else np.zeros((0, 3))


# Gap display, shared by the GUI and the saved overlays: intra-fibre gaps
# (centre inside the segmentation mask) in orange, other gaps in blue with a
# dashed outline, both lightly filled
GAP_INTRA_RGB = (230, 120, 20)
GAP_OTHER_RGB = (40, 110, 210)
GAP_FILL_ALPHA = 35          # of 255


def write_rgb(path, rgb):
    """Write an RGB image (OpenCV writes BGR)."""
    cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def gap_intra(circles, seg_mask):
    """Per circle: centre inside ``seg_mask`` (all True without a mask)."""
    if seg_mask is None:
        return np.ones(len(circles), bool)
    return seg_mask[circles[:, 0].astype(int), circles[:, 1].astype(int)] > 0


def gap_base_image(gap_mask, tissue_rgb=None):
    """RGB image the gap circles are drawn on: the fibres (zero in
    ``gap_mask``) in dark grey on white, or over the faded tissue."""
    fibres = gap_mask == 0
    if tissue_rgb is None:
        return np.where(fibres[:, :, None], 70, 255).astype(np.uint8).repeat(3, axis=2)
    base = (tissue_rgb.astype(np.float32) * 0.45 + 255 * 0.55).astype(np.uint8)
    base[fibres] = (25, 25, 25)
    return base


def draw_gap_circles(base_rgb, circles, intra, thickness=2):
    """``base_rgb`` with the gap circles drawn in the shared gap colours."""
    out = base_rgb.copy()
    if len(circles) == 0:
        return out
    centres = [(int(round(c)), int(round(r))) for r, c in circles[:, :2]]
    radii = [int(round(r)) for r in circles[:, 2]]
    fill, covered = out.copy(), np.zeros(out.shape[:2], np.uint8)
    for centre, radius, is_intra in zip(centres, radii, intra):
        cv2.circle(fill, centre, radius, GAP_INTRA_RGB if is_intra else GAP_OTHER_RGB, -1)
        cv2.circle(covered, centre, radius, 255, -1)
    a = GAP_FILL_ALPHA / 255
    on = covered > 0
    out[on] = (out[on] * (1 - a) + fill[on] * a).astype(np.uint8)
    for centre, radius, is_intra in zip(centres, radii, intra):
        if is_intra:
            cv2.circle(out, centre, radius, GAP_INTRA_RGB, thickness, cv2.LINE_AA)
            continue
        step = max(1.0, np.degrees(6.0 / max(radius, 1)))          # ~6 px dashes and gaps
        for start in np.arange(0, 360, 2 * step):
            cv2.ellipse(out, centre, (radius, radius), 0, start, start + step,
                        GAP_OTHER_RGB, thickness, cv2.LINE_AA)
    return out

