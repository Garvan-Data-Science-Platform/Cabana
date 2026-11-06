"""Birefringence (BRF) colour bands of polarised-light Picrosirius Red images.

Collagen viewed under polarised light shines red-orange, yellow or green,
roughly from thick, densely packed fibres to thin, loosely packed ones. This
module reproduces the classification of the CMM Lab PR-BRF Analyser ImageJ
macro (www.github.com/tcox-lab): ImageJ's HSB stack is thresholded per band
on hue, keeping pixels whose saturation is at least ``Saturation Min`` and
whose brightness is at least ``Brightness Threshold``. The HSB values follow
Java's ``Color.RGBtoHSB`` in float32, truncated to 0-255 as ImageJ does, so
band pixel counts match the macro exactly (a float64 port moves ~0.4 % of
pixels by one hue level).

Counts are taken on the hard masks. The channel images add a mild soft edge
for display only; selected pixels keep their original colour and intensity.
"""

import cv2
import numpy as np

BANDS = ("Red-Orange", "Yellow", "Green")

# Defaults of the macro (hue on ImageJ's 0-255 HSB scale, ranges inclusive)
DEFAULTS = {
    "Brightness Threshold": 10,
    "Saturation Min": 0,
    "Red-Orange Min": 0,
    "Red-Orange Max": 27,
    "Yellow Min": 28,
    "Yellow Max": 47,
    "Green Min": 48,
    "Green Max": 140,
    "Sort Into Folders": True,
}

# Display colours of the class map (RGB); bright pixels in no band are grey
BAND_COLORS = ((255, 90, 0), (255, 220, 0), (0, 200, 60))
UNCLASSIFIED_COLOR = (110, 110, 110)

# Above this mean grey level an image is bright-field, not polarised light
# (the same test cabana.wsi uses to name a slide's layers).
BRIGHTFIELD_MEAN = 100

_CHUNK = 1 << 21          # pixels per chunk of the HSB conversion


def settings(overrides=None):
    """The macro defaults (:data:`DEFAULTS`) with ``overrides`` applied."""
    return dict(DEFAULTS, **(overrides or {}))


def band_ranges(params):
    """``[(min, max), ...]`` hue ranges of :data:`BANDS` from settings."""
    return [(int(params[f"{band} Min"]), int(params[f"{band} Max"])) for band in BANDS]


def hsb(rgb):
    """ImageJ's HSB stack of an 8-bit RGB image: ``(hue, saturation, brightness)``
    as uint8 arrays, computed in chunks so large images stay small in memory."""
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError("hsb expects an 8-bit RGB image")
    flat = rgb[..., :3].reshape(-1, 3)
    out = [np.empty(flat.shape[0], np.uint8) for _ in range(3)]
    for start in range(0, flat.shape[0], _CHUNK):
        part = slice(start, start + _CHUNK)
        for arr, values in zip(out, _hsb_chunk(flat[part])):
            arr[part] = values
    return tuple(arr.reshape(rgb.shape[:2]) for arr in out)


def _hsb_chunk(px):
    """Java ``Color.RGBtoHSB`` (float32) of ``N x 3`` RGB pixels, each value
    then multiplied by 255.0 in double precision and truncated (ImageJ's
    ``ColorProcessor.getHSB``)."""
    f32 = np.float32
    r, g, b = (px[:, i].astype(np.int32) for i in range(3))
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin
    grey = delta == 0
    span = np.where(grey, 1, delta).astype(f32)
    rc = (cmax - r).astype(f32) / span
    gc = (cmax - g).astype(f32) / span
    bc = (cmax - b).astype(f32) / span
    hue = np.where(r == cmax, bc - gc,
                   np.where(g == cmax, f32(2) + rc - bc, f32(4) + gc - rc)).astype(f32) / f32(6)
    hue = np.where(hue < 0, hue + f32(1), hue).astype(f32)
    hue[grey] = 0
    sat = delta.astype(f32) / np.where(cmax == 0, 1, cmax).astype(f32)
    to_byte = lambda v: (v.astype(np.float64) * 255.0).astype(np.uint8)     # noqa: E731
    # brightness = cmax / 255f * 255.0 truncates back to cmax for every 0-255
    return to_byte(hue), to_byte(sat), cmax.astype(np.uint8)


def band_masks(hue, sat, bright, ranges, brightness_min=10, saturation_min=0):
    """Hard selection masks, one per hue range, and the brightness/saturation
    gate shared by all bands. A range with ``min > max`` wraps past 255."""
    gate = (bright >= brightness_min) & (sat >= saturation_min)
    masks = []
    for lo, hi in ranges:
        in_range = (hue >= lo) & (hue <= hi) if lo <= hi else (hue >= lo) | (hue <= hi)
        masks.append(in_range & gate)
    return masks, gate


def separate(rgb, params):
    """``(masks, gate)`` of an RGB image under the settings ``params``."""
    hue, sat, bright = hsb(rgb)
    return band_masks(hue, sat, bright, band_ranges(params),
                      int(params["Brightness Threshold"]), int(params["Saturation Min"]))


def band_stats(masks, gate, rgb, region=None, pixel_size_um=1.0):
    """Statistics of one image as a flat ``{column: value}`` dict.

    ``region`` is the analysed area (bool mask: the ROI mask of the image, e.g.
    a TMA core or a stroma mask), ``None`` for the whole image. It is called
    the region, not the ROI, because Cabana's ROI columns mean the segmented
    fibre area. Columns per band:

    - ``BRF <band> Pixels`` / ``Area (µm²)``: selected pixels in the region.
    - ``(% Region)``: of the region; ``(% Image)``: band pixels anywhere in the
      image over all image pixels (the macro's measure of an unmasked image).
    - ``(% Signal)``: of the pixels in all bands within the region (the
      bands sum to 100, the macro's "% Area").
    - ``Sum Intensity`` / ``(% Intensity)``: summed grey level of the band and
      its share of the bands' total.

    plus the totals over all bands, ``BRF Unclassified`` (pixels passing the
    brightness and saturation gate in no band, e.g. deep red just below the
    hue wrap), and the pixel counts of the region and the image.
    """
    gray = _gray(rgb)
    region = np.ones(gray.shape, bool) if region is None else region.astype(bool)
    return _stats(_counts(masks, gate, gray, region), pixel_size_um)


def _gray(rgb):
    return cv2.cvtColor(np.ascontiguousarray(rgb[..., :3]), cv2.COLOR_RGB2GRAY)


def _counts(masks, gate, gray, region):
    """Additive counts of one image or one strip of it (see :func:`_stats`)."""
    in_any = np.logical_or.reduce(masks) & region
    return {
        "pixels": np.array([int((m & region).sum()) for m in masks], np.int64),
        "pixels_image": np.array([int(m.sum()) for m in masks], np.int64),
        "sums": np.array([float(gray[m & region].sum(dtype=np.float64)) for m in masks]),
        "any": int(in_any.sum()),
        "unclassified": int((gate & region & ~in_any).sum()),
        "region": int(region.sum()),
        "image": int(region.size),
    }


def _stats(counts, pixel_size_um):
    """The :func:`band_stats` columns from (summed) :func:`_counts`."""
    px_area = float(pixel_size_um) ** 2
    n_region, n_image, n_any = counts["region"], counts["image"], counts["any"]
    n_signal = int(counts["pixels"].sum())
    total_sum = float(counts["sums"].sum())

    def pct(num, den):
        return num / den * 100.0 if den > 0 else 0.0

    stats = {}
    for band, n, n_img, s in zip(BANDS, counts["pixels"], counts["pixels_image"], counts["sums"]):
        n, n_img, s = int(n), int(n_img), float(s)
        stats[f"BRF {band} Pixels"] = n
        stats[f"BRF {band} Area (µm²)"] = n * px_area
        stats[f"BRF {band} (% Region)"] = pct(n, n_region)
        stats[f"BRF {band} (% Image)"] = pct(n_img, n_image)
        stats[f"BRF {band} (% Signal)"] = pct(n, n_signal)
        stats[f"BRF {band} Sum Intensity"] = s
        stats[f"BRF {band} (% Intensity)"] = pct(s, total_sum)
    stats["BRF Total Pixels"] = n_any
    stats["BRF Total Area (µm²)"] = n_any * px_area
    stats["BRF Total (% Region)"] = pct(n_any, n_region)
    stats["BRF Unclassified Pixels"] = counts["unclassified"]
    stats["BRF Unclassified (% Region)"] = pct(counts["unclassified"], n_region)
    stats["Region Pixels"] = n_region
    stats["Image Pixels"] = n_image
    return stats


# ---- strip-wise processing, so large sections need little memory ----

STRIP_PIXELS = 1 << 20    # pixels per strip: the working arrays of a strip take
                          # about 70 bytes per pixel, so ~70 MB whatever the image size
HALO = 8                  # extra rows each side of a strip: the soft edge (3x3 close,
                          # 2x2 open, 3x3 blur) reaches fewer than 8 rows, so strips
                          # give exactly the whole-image result


def _strips(shape, rows=None, halo=0):
    """``(y0, y1, a, b)``: strip rows ``[y0, y1)`` inside the read rows ``[a, b)``;
    ``rows`` defaults to :data:`STRIP_PIXELS` worth of rows."""
    height, width = shape[:2]
    rows = rows or max(1, STRIP_PIXELS // max(1, width))
    for y0 in range(0, height, rows):
        y1 = min(height, y0 + rows)
        yield y0, y1, max(0, y0 - halo), min(height, y1 + halo)


def measure(rgb, params, region=None, pixel_size_um=1.0, rows=None):
    """:func:`band_stats` of an image under ``params``, computed in strips
    (same result, a fraction of the memory)."""
    total = None
    for y0, y1, _, _ in _strips(rgb.shape, rows):
        part = rgb[y0:y1]
        masks, gate = separate(part, params)
        sub = np.ones(part.shape[:2], bool) if region is None else region[y0:y1].astype(bool)
        counts = _counts(masks, gate, _gray(part), sub)
        total = counts if total is None else {k: total[k] + counts[k] for k in total}
    return _stats(total, pixel_size_um)


def band_image(rgb, params, key, rows=None):
    """The channel image of one band (``key`` in :data:`BANDS`) or of all bands
    (``"Total"``), computed in strips with :data:`HALO` rows of overlap;
    identical to :func:`channel_images` of the whole image."""
    index = None if key == "Total" else BANDS.index(key)
    out = np.empty(rgb.shape[:2] + (3,), np.uint8)
    for y0, y1, a, b in _strips(rgb.shape, rows, HALO):
        masks, _ = separate(rgb[a:b], params)
        mask = np.logical_or.reduce(masks) if index is None else masks[index]
        out[y0:y1] = channel_image(rgb[a:b], mask)[y0 - a:y1 - a]
    return out


def _soft_weight(mask):
    """Mask cleaned with a close/open and given a mild Gaussian edge (display only)."""
    weight = mask.astype(np.float32)
    weight = cv2.morphologyEx(weight, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    weight = cv2.morphologyEx(weight, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return np.clip(cv2.GaussianBlur(weight, (3, 3), 0), 0.0, 1.0)


def channel_image(rgb, mask):
    """The image where ``mask`` is set (original colour and intensity), black
    elsewhere, with a soft edge so it does not look pixelated."""
    weight = _soft_weight(mask)[:, :, np.newaxis]
    return np.clip(rgb[..., :3].astype(np.float32) * weight, 0, 255).astype(np.uint8)


def channel_images(rgb, masks):
    """``{band: image}`` for :data:`BANDS` plus ``"Total"`` (all bands)."""
    images = {band: channel_image(rgb, m) for band, m in zip(BANDS, masks)}
    images["Total"] = channel_image(rgb, np.logical_or.reduce(masks))
    return images


def class_map(masks, gate):
    """RGB image painting each band in its colour and bright pixels in no band
    grey, on black: the quickest view of where the band limits fall."""
    out = np.zeros(gate.shape + (3,), np.uint8)
    out[gate] = UNCLASSIFIED_COLOR
    for mask, color in zip(masks, BAND_COLORS):
        out[mask] = color
    return out


def looks_brightfield(rgb):
    """Whether an image looks bright-field rather than polarised light."""
    return float(np.asarray(rgb).mean()) > BRIGHTFIELD_MEAN


GRID_LABELS = ("Original", "Red-Orange", "Yellow", "Green")


def grid_view(rgb, masks, max_side=2048, gap=6):
    """2 x 2 preview: the original image and the red-orange, yellow and green
    channel images, each labelled; quadrants larger than ``max_side`` are
    shrunk first so a large section stays light to display."""
    h, w = rgb.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    if scale < 1.0:
        small = cv2.resize(np.ascontiguousarray(rgb[..., :3]), size, interpolation=cv2.INTER_AREA)
        masks = [cv2.resize(m.astype(np.float32), size, interpolation=cv2.INTER_AREA) >= 0.5 for m in masks]
    else:
        small = np.ascontiguousarray(rgb[..., :3])
    tiles = [small] + [channel_image(small, m) for m in masks]
    qh, qw = small.shape[:2]
    out = np.full((2 * qh + gap, 2 * qw + gap, 3), 90, np.uint8)
    font_scale = max(0.4, min(qw, qh) / 700)
    thickness = max(1, round(font_scale * 2))
    for i, (tile, label) in enumerate(zip(tiles, GRID_LABELS)):
        y, x = (i // 2) * (qh + gap), (i % 2) * (qw + gap)
        out[y:y + qh, x:x + qw] = tile
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        pad = max(3, th // 3)
        cv2.rectangle(out, (x, y), (x + tw + 2 * pad, y + th + base + 2 * pad), (0, 0, 0), -1)
        cv2.putText(out, label, (x + pad, y + pad + th), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255), thickness, cv2.LINE_AA)
    return out
