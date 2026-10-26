"""Visualisation helpers (colour-mapped overlays, colour bars, color-survey
images) extracted from the monolithic ``cabana.utils``. ``cabana.utils``
re-exports the public names below so existing
``from .utils import overlay_colorbar`` style imports keep working.
"""

import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt  # noqa: F401  (kept for callers that use plt module-level)
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import binary_erosion, gaussian_filter


def _normalize(x, pmin=2, pmax=98, axis=None, eps=1e-20, dtype=np.float32):
    """Local copy of utils.normalize to avoid a circular import."""
    mi = np.percentile(x, pmin, axis=axis, keepdims=True)
    ma = np.percentile(x, pmax, axis=axis, keepdims=True)
    if dtype is not None:
        x = x.astype(dtype, copy=False)
        mi = dtype(mi) if np.isscalar(mi) else mi.astype(dtype, copy=False)
        ma = dtype(ma) if np.isscalar(ma) else ma.astype(dtype, copy=False)
        eps = dtype(eps)
    x = (x - mi) / (ma - mi + eps)
    return np.clip(x, 0, 1)


def mask_color_map(img, mask, rgb_color=[0.224, 1.0, 0.0784], sigma=0.5):
    imarray = mask / np.max(mask)
    eroded = binary_erosion(imarray, iterations=2)
    outlines = imarray - eroded
    blur = gaussian_filter(outlines, sigma)
    outlines = outlines[:, :, None] * rgb_color
    blur = blur[:, :, None] * rgb_color
    glow = np.clip(outlines + blur, 0, 1)
    glow = (np.squeeze(glow) * 255).astype(np.uint8)
    index_pos = np.where(cv2.cvtColor(glow, cv2.COLOR_RGB2GRAY) == 0)
    glow[index_pos[0], index_pos[1], :] = img[index_pos[0], index_pos[1], :]
    return glow


def _save_figure(fig, image_ax, image_height, save_path):
    """Save a figure as an RGB PNG at the resolution that draws the image in
    ``image_ax`` at its own pixel size (no enlargement, no detail lost).

    The resolution and bounding box are passed explicitly so the output does
    not depend on global matplotlib settings (importing seaborn_image sets
    300 dpi, which enlarged these figures 2 to 3 times). The layout scales as
    a whole, so the colour bar keeps its position and size relative to the
    image."""
    import io
    from PIL import Image
    fig.canvas.draw()
    shown = image_ax.get_window_extent().height      # image height in pixels at fig.dpi
    dpi = fig.dpi * image_height / max(shown, 1.0)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', transparent=False, facecolor='white')
    buf.seek(0)
    Image.open(buf).convert('RGB').save(save_path)


def color_survey_with_colorbar(orient, coherency, energy, save_path,
                               clabel="Color Survey", dpi=100, font_size=12):
    """Color survey visualization with a colorbar (thread-safe)."""
    hue = (((orient + np.pi / 2) / np.pi) * 179).astype(np.uint8)
    saturation = (coherency * 255).astype(np.uint8)
    value = (energy * 255).astype(np.uint8)
    hsv_image = np.stack((hue, saturation, value), axis=-1)
    colored_img = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

    height, width = orient.shape[:2]
    fig = Figure(figsize=(width / dpi * 1.2, height / dpi), dpi=dpi)
    canvas = FigureCanvasAgg(fig)
    gs = fig.add_gridspec(1, 2, width_ratios=[20, 1], wspace=0.02)
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(colored_img)
    ax.axis('off')
    cax = fig.add_subplot(gs[0, 1])
    norm = matplotlib.colors.Normalize(vmin=-np.pi / 2, vmax=np.pi / 2)
    sm = matplotlib.cm.ScalarMappable(cmap="hsv", norm=norm)
    sm.set_array(orient)
    cbar = fig.colorbar(sm, cax=cax)
    cbar.ax.set_ylabel(clabel, fontsize=font_size)
    cbar.ax.tick_params(labelsize=font_size)
    cbar.set_ticks(ticks=[-np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2],
                   labels=[r'-$\pi$/2', r'-$\pi$/4', '0', r'$\pi$/4', r'$\pi$/2'])
    _save_figure(fig, ax, height, save_path)
    fig.clf()
    return colored_img


def overlay_colorbar(rgb_img, img, save_path,
                     clabel="Curvature (degrees)", cmap='plasma',
                     mode="overwrite", dpi=100, font_size=12):
    """Overlay a colorbar on an image (thread-safe)."""
    height, width = img.shape[:2]

    if mode == "overwrite":
        colormap = matplotlib.colormaps.get_cmap(cmap)
        colored_img = colormap(img / np.max(img))[:, :, :3]
        hsv_img = cv2.cvtColor((colored_img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        hsv_img[:, :, 2] = (_normalize(hsv_img[..., 2], 0, 100) * 255).astype(np.uint8)
        colored_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
        bg_index_pos = np.where(img <= 0)
        colored_img[bg_index_pos[0], bg_index_pos[1], :] = rgb_img[bg_index_pos[0], bg_index_pos[1], :]
    elif mode == "overlay":
        gray_img = np.repeat(cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)[..., None], 3, axis=2)
        n_colors = 256
        import seaborn as sns
        hues = sns.color_palette(cmap, n_colors + 1)
        hue_indices = (n_colors * (img / np.max(img))).astype(int)
        hues = np.array(hues)
        hue_color = np.take(hues, hue_indices, axis=0)
        colored_img = (gray_img * hue_color).astype(np.uint8)
    elif mode == "weighted":
        gray_img = np.repeat(cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)[..., None], 3, axis=2)
        colormap = matplotlib.colormaps.get_cmap(cmap)
        colored_img = colormap(img / np.max(img))[:, :, :3]
        hsv_img = cv2.cvtColor((colored_img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        hsv_img[:, :, 2] = (_normalize(hsv_img[..., 2], 0, 100) * 255).astype(np.uint8)
        colored_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
        colored_img = cv2.addWeighted(colored_img, 0.3, gray_img, 0.7, 50)

    fig = Figure(figsize=(width / dpi * 1.2, height / dpi), dpi=dpi)
    canvas = FigureCanvasAgg(fig)
    gs = fig.add_gridspec(1, 2, width_ratios=[20, 1], wspace=0.02)
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(colored_img)
    ax.axis('off')
    cax = fig.add_subplot(gs[0, 1])
    norm = matplotlib.colors.Normalize(vmin=np.min(img), vmax=np.max(img))
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(img)
    cbar = fig.colorbar(sm, cax=cax)
    cbar.ax.set_ylabel(clabel, fontsize=font_size)
    cbar.ax.tick_params(labelsize=font_size)
    _save_figure(fig, ax, height, save_path)
    fig.clf()
    return rgb_img


def add_colorbar(rgb_img, img, clabel="Curvature (degrees)", cmap='inferno',
                 dpi=100, font_size=12):
    height, width = img.shape[:2]
    colormap = matplotlib.colormaps.get_cmap(cmap)
    colored_img = colormap(img / np.max(img))[:, :, :3]
    hsv_img = cv2.cvtColor((colored_img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    hsv_img[:, :, 2] = (_normalize(hsv_img[..., 2], 0, 100) * 255).astype(np.uint8)
    colored_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB) / 255.0

    fig1 = Figure(figsize=(width / dpi * 1.2, height / dpi), dpi=dpi, frameon=False)
    ax1 = fig1.add_subplot()
    img_tmp = ax1.imshow(img, cmap=cmap)
    fig2 = Figure(figsize=(width / dpi * 1.2, height / dpi), dpi=dpi, frameon=False)
    ax2 = fig2.add_subplot()
    ax2.imshow(cv2.addWeighted(rgb_img, 0.3, (colored_img * 255).astype(np.uint8), 0.7, 20))
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cax.tick_params(labelsize=font_size)
    cbar = fig2.colorbar(img_tmp, cax=cax)
    cbar.ax.set_ylabel(clabel, fontsize=font_size)
    fig2.patch.set_visible(False)
    ax2.axis('off')
    canvas = FigureCanvasAgg(fig2)
    canvas.draw()
    s, (width, height) = canvas.print_to_buffer()
    full_frame = np.frombuffer(s, np.uint8).reshape((height, width, 4))[..., :3]
    return full_frame


# Fibre direction (0 to 180 degrees) as hue; saturation and value are kept
# below 1 so the colours stay readable on white and on faded tissue.
DIRECTION_CMAP = matplotlib.colors.ListedColormap(matplotlib.colors.hsv_to_rgb(
    np.stack([np.linspace(0, 1, 256), np.full(256, 0.85), np.full(256, 0.8)], axis=1)), name="direction")


def direction_rgb(theta):
    """RGB (0-255) of an axial direction ``theta`` in radians, from DIRECTION_CMAP."""
    hue = (theta % np.pi) / np.pi
    return tuple(int(255 * c) for c in matplotlib.colors.hsv_to_rgb([hue, 0.85, 0.8]))


def image_with_colorbar(img_rgb, save_path, cmap, vmin, vmax, clabel, ticks=None,
                        extend="neither", font_size=10):
    """Save ``img_rgb`` with a colour bar beside it, drawn at its own pixel size."""
    height, width = img_rgb.shape[:2]
    fig = Figure(figsize=(width / 100 * 1.25, height / 100), dpi=100)
    FigureCanvasAgg(fig)
    gs = fig.add_gridspec(1, 2, width_ratios=[20, 1], wspace=0.03)
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(img_rgb, interpolation="nearest")
    ax.axis("off")
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax))
    cbar = fig.colorbar(sm, cax=fig.add_subplot(gs[0, 1]), extend=extend, ticks=ticks)
    cbar.ax.set_ylabel(clabel, fontsize=font_size)
    cbar.ax.tick_params(labelsize=font_size)
    _save_figure(fig, ax, height, save_path)
    fig.clf()


def skeleton_value_map(values, save_path, clabel, cmap, rgb_img, percentile=99, fade=0.6):
    """Save a per-skeleton-pixel map (zero = no fibre) as 3 px wide coloured
    lines over the faded image ``rgb_img``. The colour scale runs from 0 to the
    ``percentile`` of the fibre values so a few extreme fibres do not wash out
    the rest; the colour bar marks values above it. The widening is for
    display only."""
    on = values > 0
    vmax = float(np.percentile(values[on], percentile)) if on.any() else 1.0
    vmax = vmax if vmax > 0 else 1.0
    wide = cv2.dilate(values.astype(np.float32), np.ones((3, 3), np.uint8))   # keeps the larger value
    out = (rgb_img[:, :, :3].astype(np.float32) * (1 - fade) + 255 * fade).astype(np.uint8)
    shown = wide > 0
    colours = matplotlib.colormaps[cmap](np.clip(wide[shown] / vmax, 0, 1))[:, :3]
    out[shown] = (colours * 255).astype(np.uint8)
    image_with_colorbar(out, save_path, cmap, 0, vmax, clabel, extend="max")

