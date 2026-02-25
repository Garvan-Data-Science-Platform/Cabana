import os
import re
import cv2
import math
import random
import shutil
import colorsys
import matplotlib
from glob import glob
import seaborn as sns
from enum import Enum
from numba import jit
from .constants import *
import scipy.ndimage as ndi
import seaborn_image as isns
from .correct import Correct
# matplotlib.use('Agg', force=True)
from PIL import Image, ExifTags
from matplotlib.figure import Figure
from scipy.ndimage import convolve, gaussian_filter1d
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.stats import multivariate_normal
from scipy.ndimage import binary_erosion
from scipy.ndimage import gaussian_filter
from skimage.segmentation import mark_boundaries
from skimage.morphology import dilation, disk
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backends.backend_agg import FigureCanvasAgg
import imageio_ffmpeg
matplotlib.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()

read_bar_format = "%s{l_bar}%s{bar}%s{r_bar}" % ("\033[0;34m", "\033[0;34m", "\033[0;34m")


def array_divide(a, b):
    return np.divide(a, b, out=np.zeros_like(a, dtype=np.float64), where=(b != 0), casting="unsafe")


def contains_oversized(img_paths, max_res=2048):
    max_size = max_res * max_res
    for img_path in img_paths:
        image = Image.open(img_path)
        resolution = image.size
        if resolution[0] * resolution[1] > max_size:
            return True
    return False


def normalize(x, pmin=2, pmax=98, axis=None, eps=1e-20, dtype=np.float32):
    """Percentile-based image normalization."""

    mi = np.percentile(x, pmin, axis=axis, keepdims=True)
    ma = np.percentile(x, pmax, axis=axis, keepdims=True)
    if dtype is not None:
        x = x.astype(dtype, copy=False)
        mi = dtype(mi) if np.isscalar(mi) else mi.astype(dtype, copy=False)
        ma = dtype(ma) if np.isscalar(ma) else ma.astype(dtype, copy=False)
        eps = dtype(eps)

    x = (x - mi) / (ma - mi + eps)

    return np.clip(x, 0, 1)


def info_color_map(img, info_map, cbar_label="Length", cmap="PiYG", radius=1):
    '''This function aims to overlay curve or length map onto the original image.
    Some black spots might be observed in the resultant images.
    These are caused by the missing pixels in the length map.'''
    height, width = info_map.shape[:2]

    if np.max(info_map) > 1:
        cbar_ticks = np.arange(np.min(info_map), np.max(info_map), (np.max(info_map) - np.min(info_map)) // 4)
    else:
        cbar_ticks = np.linspace(0, 1, 5)

    # info_map_normalized = (normalize(info_map) * 255).astype(np.uint8)
    fig, ax = plt.subplots(figsize=(width / 100.0, height / 100.0))
    ax = isns.imgplot(info_map, ax=ax, cmap=sns.color_palette(cmap, as_cmap=True),
                      cbar=False, dx=5, units="um")
    ax.set_xticks([])
    ax.set_yticks([])

    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    plt.margins(0, 0)
    user_home_dir = os.path.expanduser("~")
    fig.savefig(join_path(user_home_dir, ".tmp_image.png"), bbox_inches='tight', pad_inches=0)
    plt.close()
    image_data = cv2.imread(join_path(os.path.expanduser("~"), ".tmp_image.png"))
    X = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
    color_info_map = np.flipud(cv2.resize(X, (width, height)))

    fig, ax = plt.subplots(figsize=(width / 100.0, height / 100.0))
    ax = isns.imgplot(info_map, ax=ax, cmap=sns.color_palette(cmap, as_cmap=True),
                      cbar_label=cbar_label,
                      cbar_ticks=cbar_ticks)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_visible(False)

    fig.savefig(join_path(user_home_dir, ".tmp_colorbar.png"), bbox_inches='tight', pad_inches=0)
    plt.close()
    image_data = cv2.imread(join_path(user_home_dir, ".tmp_colorbar.png"))
    Y = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)

    color_info_map[:, :, 0] = dilation(color_info_map[:, :, 0], disk(radius))
    color_info_map[:, :, 1] = dilation(color_info_map[:, :, 1], disk(radius))
    color_info_map[:, :, 2] = dilation(color_info_map[:, :, 2], disk(radius))

    index_pos = np.where(dilation(info_map > 0, disk(radius)) == 0)
    color_info_map[index_pos[0], index_pos[1], :] = img[index_pos[0], index_pos[1], :]

    height_ratio = np.min([Y.shape[0] / X.shape[0], 1])
    Y_height = int(height * height_ratio)
    h, w = Y.shape[:2]
    r = Y_height / float(h)
    dim = (int(w * r), Y_height)
    colorbar_img = cv2.resize(Y, dim)
    cb_h, cb_w = colorbar_img.shape[:2]
    hor_gap_ratio = 0.5

    final_img = np.zeros((height, int(width + 3 * hor_gap_ratio * cb_w), 3), dtype=np.uint8) + 255
    final_img[:height, :width, :] = color_info_map
    final_img[(height - cb_h) // 2:(height - cb_h) // 2 + cb_h,
    width + int(hor_gap_ratio * cb_w):width + int(hor_gap_ratio * cb_w) + cb_w, :] = colorbar_img
    if os.path.exists(join_path(user_home_dir, ".tmp_image.png")):
        os.remove(join_path(user_home_dir, ".tmp_image.png"))
    if os.path.exists(join_path(user_home_dir, ".tmp_colorbar.png")):
        os.remove(join_path(user_home_dir, ".tmp_colorbar.png"))

    return final_img


def mask_color_map(img, mask, rgb_color=[0.224, 1.0, 0.0784], sigma=0.5):
    imarray = mask / np.max(mask)
    eroded = binary_erosion(imarray, iterations=2)

    outlines = imarray - eroded

    # Convolve with a Gaussian to effect a blur.
    blur = gaussian_filter(outlines, sigma)

    # Make binary images into neon green.
    outlines = outlines[:, :, None] * rgb_color
    blur = blur[:, :, None] * rgb_color

    # Combine the images and constraint to [0, 1].
    glow = np.clip(outlines + blur, 0, 1)
    glow = (np.squeeze(glow) * 255).astype(np.uint8)

    index_pos = np.where(cv2.cvtColor(glow, cv2.COLOR_RGB2GRAY) == 0)
    glow[index_pos[0], index_pos[1], :] = img[index_pos[0], index_pos[1], :]

    return glow


def orient_vf(img, orient_map, wgts_map=None, color=(255, 255, 0), thickness=1, size=15, scale=80):
    ny, nx = orient_map.shape[:2]
    xstart = (nx - (nx // size) * size) // 2
    ystart = (ny - (ny // size) * size) // 2

    x_blk_num = len(range(xstart, nx, size))
    y_blk_num = len(range(ystart, ny, size))

    size2 = size * size

    blk_stats = np.zeros((y_blk_num, x_blk_num, 4))
    blk_wgts = np.ones((y_blk_num, x_blk_num))

    for y in range(ystart, ny, size):
        for x in range(xstart, nx, size):
            blk_stats[(y - ystart) // size, (x - xstart) // size, 0] = y
            blk_stats[(y - ystart) // size, (x - xstart) // size, 1] = x

            top = y - size // 2 if y - size // 2 >= 0 else 0
            bot = y + size // 2 if y + size // 2 <= ny else ny
            lft = x - size // 2 if x - size // 2 >= 0 else 0
            rht = x + size // 2 if x + size // 2 <= nx else nx

            dx = np.mean(np.cos(orient_map[top:bot, lft:rht]))
            dy = np.mean(np.sin(orient_map[top:bot, lft:rht]))
            blk_stats[(y - ystart) // size, (x - xstart) // size, 2] = dy
            blk_stats[(y - ystart) // size, (x - xstart) // size, 3] = dx

            if wgts_map is not None:
                blk_wgts[(y - ystart) // size, (x - xstart) // size] = np.max(wgts_map[top:bot, lft:rht])

    min_val = np.min(blk_wgts)
    max_val = np.max(blk_wgts)

    if min_val != max_val:
        blk_wgts = normalize(blk_wgts, pmin=5, pmax=5, axis=[0, 1])

    vf = np.zeros((ny, nx, 3), dtype=np.uint8)

    for blk_yi in range(y_blk_num):
        for blk_xi in range(x_blk_num):
            r = blk_wgts[blk_yi, blk_xi] * scale / 100.0 * size * 0.5
            y1 = int(blk_stats[blk_yi, blk_xi, 0] + size // 2 - r * blk_stats[blk_yi, blk_xi, 2])
            x1 = int(blk_stats[blk_yi, blk_xi, 1] + size // 2 + r * blk_stats[blk_yi, blk_xi, 3])
            y2 = int(blk_stats[blk_yi, blk_xi, 0] + size // 2 + r * blk_stats[blk_yi, blk_xi, 2])
            x2 = int(blk_stats[blk_yi, blk_xi, 1] + size // 2 - r * blk_stats[blk_yi, blk_xi, 3])
            vf = cv2.line(vf, (x1, y1), (x2, y2), color, thickness)
    return cv2.addWeighted(img, 0.7, vf, 0.7, 10)


def width_color_map(img, width_img, mask_img, width_color=[0, 255, 255], mask_color=[255, 255, 0]):
    contour_img = img.copy()
    index_pos = np.where((mask_img[:, :, 0] == 255))
    contour_img[index_pos[0], index_pos[1], :] = mask_color

    contour_img = (mark_boundaries(contour_img, (width_img[:, :, 0] > 128).astype(np.uint8),
                                   color=[i / 255 if i > 1 else i for i in width_color]) * 255).astype(np.uint8)

    return contour_img


def sbs_color_map(img, info_map, save_name, cbar_label="Length", cmap="coolwarm"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 9))

    axes[0].imshow(img)
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    axes[1] = isns.imgplot(info_map, ax=axes[1],
                           cmap=sns.color_palette(cmap, as_cmap=True),
                           cbar_label=cbar_label)
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    fig.savefig(save_name, bbox_inches='tight', pad_inches=0)
    plt.close()


def color_survey_with_colorbar(orient, coherency, energy, save_path, clabel="Color Survey", dpi=100, font_size=12):
    """
    Create a color survey visualization with colorbar using a thread-safe approach.

    Parameters:
    -----------
    orient : numpy.ndarray
        Orientation data in the range [-pi/2, pi/2]
    coherency : numpy.ndarray
        Coherency data in the range [0, 1]
    energy : numpy.ndarray
        Energy data in the range [0, 1]
    save_path : str
        Path to save the output image
    clabel : str, optional
        Colorbar label, default is "Color Survey"
    dpi : int, optional
        Dots per inch for the figure, default is 100
    font_size : int, optional
        Font size for labels, default is 12
    """
    # Normalize orientation to [0, 1] then scale to [0, 179] for hue
    hue = (((orient + np.pi / 2) / np.pi) * 179).astype(np.uint8)

    # Scale coherency and energy to [0, 255] for saturation and value
    saturation = (coherency * 255).astype(np.uint8)
    value = (energy * 255).astype(np.uint8)

    # Stack the channels to create an HSV image
    hsv_image = np.stack((hue, saturation, value), axis=-1)

    # Convert the HSV image to an RGB image
    colored_img = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

    height, width = orient.shape[:2]

    # Create a single figure with GridSpec for layout control
    fig = Figure(figsize=(width / dpi * 1.2, height / dpi), dpi=dpi)
    canvas = FigureCanvasAgg(fig)

    # Create a 2x1 grid layout
    gs = fig.add_gridspec(1, 2, width_ratios=[20, 1], wspace=0.02)

    # Add the main image subplot
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(colored_img)
    ax.axis('off')

    # Add the colorbar subplot
    cax = fig.add_subplot(gs[0, 1])

    # Create a ScalarMappable with the hsv colormap for the colorbar
    norm = matplotlib.colors.Normalize(vmin=-np.pi / 2, vmax=np.pi / 2)
    sm = matplotlib.cm.ScalarMappable(cmap="hsv", norm=norm)
    sm.set_array(orient)

    # Create the colorbar using the ScalarMappable
    cbar = fig.colorbar(sm, cax=cax)
    cbar.ax.set_ylabel(clabel, fontsize=font_size)
    cbar.ax.tick_params(labelsize=font_size)
    cbar.set_ticks(ticks=[-np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2],
                   labels=[r'-$\pi$/2', r'-$\pi$/4', '0', r'$\pi$/4', r'$\pi$/2'])

    # Draw the canvas and save the figure
    canvas.draw()
    fig.savefig(save_path, format='png', transparent=False, facecolor='white')

    # Clean up resources
    fig.clf()

    return colored_img


def sbs_color_survey(img, info_map, save_name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 9))

    axes[0].imshow(img)
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    axes[1].imshow(info_map)
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    fig.savefig(save_name, bbox_inches='tight', pad_inches=0)
    plt.close()


def split2batches(img_paths, max_batch_size=5):
    pixel_res = []
    for img_path in img_paths:
        img_info = Image.open(img_path)
        img_exif = img_info.getexif()

        if img_exif is None:
            print('Sorry, image has no exif data. Setting to default 1.0.')
            pixel_res.append(1.0)
        else:
            xres, yres = 1.0, 1.0
            found = False
            for key, val in img_exif.items():
                if key in ExifTags.TAGS:
                    if ExifTags.TAGS[key] == "XResolution":
                        xres = round(1.0 / float(val), 2)
                        found = True
                    if ExifTags.TAGS[key] == "YResolution":
                        yres = round(1.0 / float(val), 2)
                        found = True
            if found:
                if xres != yres:
                    print('Warning: XResolution and YResolution in metadata are different! Using XResolution...')
                pixel_res.append(xres)
            else:
                print('Warning: No pixel resolution available in metadata! Setting to default 1.0.')
                pixel_res.append(1.0)
    assert len(pixel_res) == len(img_paths)

    # sort image path based on the pixel resolution
    img_paths = [x for _, x in sorted(zip(pixel_res, img_paths))]
    pixel_res = [y for y, _ in sorted(zip(pixel_res, img_paths))]
    path_batches = []
    res_batches = []

    pres_value = pixel_res[0]
    path_batch = [img_paths[0]]

    for res, img_path in zip(pixel_res[1:], img_paths[1:]):
        if pres_value == res:
            if len(path_batch) < max_batch_size:
                path_batch.append(img_path)
            else:
                path_batches.append(path_batch)
                res_batches.append(pres_value)
                path_batch = [img_path]
        else:
            path_batches.append(path_batch)
            res_batches.append(pres_value)
            path_batch = [img_path]
            pres_value = res

    if len(path_batch) > 0:
        path_batches.append(path_batch)
        res_batches.append(pres_value)

    return path_batches, res_batches


def join_path(*args):
    return os.path.join(*args).replace("\\", "/")


def create_folder(folder, overwrite=True):
    if os.path.exists(folder):
        if overwrite:
            shutil.rmtree(folder)
            os.mkdir(folder)
    else:
        os.makedirs(folder)


def mean_image(image, labels):
    im_rp = image.reshape(-1, image.shape[2])
    labels_1d = np.reshape(labels, -1)
    uni = np.unique(labels_1d)
    uu = np.zeros(im_rp.shape)
    for i in uni:
        loc = np.where(labels_1d == i)[0]
        mm = np.mean(im_rp[loc, :], axis=0)
        uu[loc, :] = mm
    return np.reshape(uu, [image.shape[0], image.shape[1], image.shape[2]]).astype('uint8')


def cal_color_dist(rgb_image, hue=1.0):
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV).astype(np.float64)
    hsv[:, :, 0] = hsv[:, :, 0] / 180.0
    hsv[:, :, 1] = hsv[:, :, 1] / 255.0
    hsv[:, :, 2] = hsv[:, :, 2] / 255.0

    # mu = np.array([60.0 / 180.0, 160.0 / 255.0, 200.0 / 255.0])
    if np.mean(hsv[:, :, :2]) == 0:  # grayscale
        mu = np.array([0, 0, 0.99])
    else:
        mu = np.array([hue, 160.0 / 255.0, 200.0 / 255.0])  # this is the mean value previously used for color images
    # mu = np.array([hue, sat, val])
    sigma = np.array([.1, .3, .5])
    covariance = np.diag(sigma ** 2)

    rv = multivariate_normal(mean=mu, cov=covariance)
    z = rv.pdf(hsv)
    ref = rv.pdf(mu)
    absolute_greenness = z / ref
    relative_greenness = (z - np.min(z)) / (np.max(z) - np.min(z) + np.finfo(float).eps)

    return absolute_greenness, relative_greenness


def crop_img_from_center(img, crop_size=(512, 512)):
    assert (img.shape[0] >= crop_size[0])
    assert (img.shape[1] >= crop_size[1])
    assert (len(img.shape) == 2 or len(img.shape) == 3)
    cw = img.shape[1] // 2
    ch = img.shape[0] // 2
    x = cw - crop_size[1] // 2
    y = ch - crop_size[0] // 2
    if len(img.shape) == 3:
        return img[y:y + crop_size[0], x:x + crop_size[1], :]
    else:
        return img[y:y + crop_size[0], x:x + crop_size[1]]


def crop_img_from_center(img, width=512):
    assert (img.shape[1] >= width)
    assert (len(img.shape) == 2 or len(img.shape) == 3)
    height = img.shape[0] * width // img.shape[1]
    cw = img.shape[1] // 2
    ch = img.shape[0] // 2
    x = cw - width // 2
    y = ch - height // 2
    if len(img.shape) == 3:
        return img[y:y + height, x:x + width, :]
    else:
        return img[y:y + height, x:x + width]


def save_result_img(save_path, rgb_img, img_labels, mean_img, absolute_greenness, relative_greenness, thresholded):
    # cv2.imwrite(os.path.join(save_path, 'rgb_4.png'), cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
    # # cv2.imwrite(os.path.join(save_path, 'labels_3.png'), cv2.cvtColor(img_labels, cv2.COLOR_RGB2BGR))
    # cv2.imwrite(os.path.join(save_path, 'mean_4.png'), cv2.cvtColor(mean_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(save_path, 'K684.vsi - 40x_BF_01Annotation (Polygon) (Malignant area)_0.png'), thresholded)
    # fig = plt.figure(figsize=(15, 10))
    # ax = fig.add_subplot(2, 3, 1)
    # ax.set_title('Original image')
    # plt.axis('off')
    # plt.imshow(rgb_img)

    # ax = fig.add_subplot(2, 3, 2)
    # ax.set_title('Semantic segmentation')
    # plt.axis('off')
    # plt.imshow(img_labels)

    # ax = fig.add_subplot(2, 3, 3)
    # ax.set_title('Mean image')
    # plt.axis('off')
    # plt.imshow(mean_img)

    # ax = fig.add_subplot(2, 3, 4)
    # ax.set_title('Binary mask')
    # plt.axis('off')
    # plt.imshow(thresholded, cmap='gray')

    # ax = fig.add_subplot(2, 3, 5)
    # ax.set_title('Relative greenness')
    # plt.axis('off')
    # plt.imshow(relative_greenness, cmap='gray', vmin=0, vmax=1)

    # ax = fig.add_subplot(2, 3, 6)
    # ax.set_title('Absolute greenness')
    # plt.axis('off')
    # plt.imshow(absolute_greenness, cmap='gray', vmin=0, vmax=1)

    # plt.tight_layout()
    # plt.savefig(save_path, bbox_inches='tight')
    # plt.show(block=False)
    # plt.close("all")


def add_scale_bar(image, scale_length_pixels, scale_length_units="μm",
                  position="bottom-right", thickness=3, color=(0, 0, 0),
                  font_scale=0.55):
    """
    Add a scale bar to an image.

    Args:
        image: Input image (numpy array or file path)
        scale_length_pixels: Length of scale bar in pixels
        scale_length_units: What the scale bar represents (e.g., "μm", "mm", "cm")
        position: Position of scale bar - "bottom-right", "bottom-left", "top-right", "top-left"
        thickness: Thickness of the scale bar line in pixels
        color: Color of scale bar (BGR tuple, default white)
        font_scale: Font size for text label

    Returns:
        Image with scale bar added
    """

    # Load image if path is provided
    if isinstance(image, str):
        img = cv2.imread(image)
    else:
        img = image.copy()

    if img is None:
        raise ValueError("Could not load image")

    height, width = img.shape[:2]

    # Define position coordinates based on string input
    margin = 10
    if position == "bottom-right":
        x_start = width - scale_length_pixels - margin
        y_pos = height - margin
    elif position == "bottom-left":
        x_start = margin
        y_pos = height - margin
    elif position == "top-right":
        x_start = width - scale_length_pixels - margin
        y_pos = margin + 15
    elif position == "top-left":
        x_start = margin
        y_pos = margin + 15
    else:
        raise ValueError("Position must be 'bottom-right', 'bottom-left', 'top-right', or 'top-left'")

    # Draw the scale bar line
    cv2.line(img, (x_start, y_pos), (x_start + scale_length_pixels, y_pos),
             color, thickness)

    # Add text label
    label = f"50 um"
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, font_scale, 1)[0]
    text_x = x_start + (scale_length_pixels - text_size[0]) // 2
    text_y = y_pos - 10

    cv2.putText(img, label, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX,
                font_scale, color, 1, cv2.LINE_AA)

    return img


def save_result_video(save_path, rgb_img, all_img_labels, all_mean_imgs, all_absolute_greenness, all_relative_greenness,
                      all_masks):
    imgs = []
    fig = plt.figure(figsize=(15, 10))
    ori_img = cv2.resize(rgb_img, (512, 512))
    for i in range(len(all_img_labels)):
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.set_title('Original image')
        ax1.axis('off')

        ori_img_sb = add_scale_bar(ori_img, 70)
        ax1.imshow(ori_img_sb)

        ax2 = fig.add_subplot(2, 2, 2)
        ax2.set_title('Semantic segmentation')
        ax2.axis('off')
        ax2.imshow(cv2.resize(all_img_labels[i], (512, 512)))

        ax3 = fig.add_subplot(2, 2, 3)
        ax3.set_title('Mean image')
        ax3.axis('off')
        ax3.imshow(cv2.resize(all_mean_imgs[i], (512, 512)))

        ax4 = fig.add_subplot(2, 2, 4)
        ax4.set_title('Segmented image')
        ax4.axis('off')
        binary_mask = cv2.resize(all_masks[i], (512, 512))
        rgb_img_copy = ori_img.copy()
        rgb_img_copy[binary_mask==0] = [224, 224, 224]  # Set background to a light color
        ax4.imshow(rgb_img_copy)

        plt.tight_layout()
        imgs.append([ax1, ax2, ax3, ax4])
        #
        # ax5 = fig.add_subplot(2, 3, 5)
        # ax5.set_title('Relative redness')
        # ax5.axis('off')
        # ax5.imshow(cv2.resize(all_relative_greenness[i], (512, 512)), cmap='gray', vmin=0, vmax=1)
        #
        # ax6 = fig.add_subplot(2, 3, 6)
        # ax6.set_title('Relative greenness')
        # ax6.axis('off')
        # ax6.imshow(cv2.resize(all_absolute_greenness[i], (512, 512)), cmap='gray', vmin=0, vmax=1)

        # plt.tight_layout()
        # imgs.append([ax1, ax2, ax3, ax4, ax5, ax6])

    ani = animation.ArtistAnimation(fig, imgs, interval=80, blit=False)
    ani.save(save_path, fps=3)


def save_result_video_old(save_path, rgb_img, gt_mask, all_img_labels, all_mean_imgs, all_greenness, all_masks):
    imgs = []
    fig = plt.figure(figsize=(10, 15))

    for i in range(len(all_img_labels)):
        ax1 = fig.add_subplot(2, 3, 1)
        ax1.set_title('Original image')
        ax1.axis('off')
        ax1.imshow(cv2.resize(rgb_img, (512, 512)))

        ax2 = fig.add_subplot(2, 3, 2)
        ax2.set_title('Semantic segmentation')
        ax2.axis('off')
        ax2.imshow(cv2.resize(all_img_labels[i], (512, 512)))

        ax5 = fig.add_subplot(2, 3, 3)
        ax5.set_title('Mean image')
        ax5.axis('off')
        ax5.imshow(cv2.resize(all_mean_imgs[i], (512, 512)))

        ax4 = fig.add_subplot(2, 3, 4)
        ax4.set_title('Ground truth')
        ax4.axis('off')
        ax4.imshow(cv2.resize(gt_mask, (512, 512)), cmap='gray')

        ax3 = fig.add_subplot(2, 3, 5)
        ax3.set_title('Binary mask')
        ax3.axis('off')
        ax3.imshow(cv2.resize(all_masks[i], (512, 512)), cmap='gray')

        ax6 = fig.add_subplot(2, 3, 6)
        ax6.set_title('Greenness')
        ax6.axis('off')
        ax6.imshow(cv2.resize(all_greenness[i], (512, 512)), cmap='gray', vmin=0, vmax=1)

        plt.tight_layout()
        imgs.append([ax1, ax2, ax3, ax4, ax5, ax6])

    ani = animation.ArtistAnimation(fig, imgs, interval=80, blit=False)
    ani.save(save_path)


def color_coded_map(gt, det):
    gt = gt.astype(bool)
    det = det.astype(bool)
    green_area = np.logical_and(det, gt)
    red_area = np.logical_and(det, np.logical_not(gt))
    blue_area = np.logical_and(np.logical_not(det), gt)

    color_map = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.uint8)
    tmp_map = np.zeros((gt.shape[0], gt.shape[1]), dtype=np.uint8)
    tmp_map[green_area] = 255
    color_map[:, :, 1] = tmp_map

    tmp_map = np.zeros((gt.shape[0], gt.shape[1]), dtype=np.uint8)
    tmp_map[red_area] = 255
    color_map[:, :, 2] = tmp_map

    tmp_map = np.zeros((gt.shape[0], gt.shape[1]), dtype=np.uint8)
    tmp_map[blue_area] = 255
    color_map[:, :, 0] = tmp_map
    return color_map


def sanitize_filename(filename):
    # Define the set of characters to be removed
    forbidden_chars = r"[ ,:?\/*]"

    # Use regular expressions to remove forbidden characters
    sanitized_filename = re.sub(forbidden_chars, '_', filename)

    return sanitized_filename


def export_parameters(param_path, out_file):
    if not os.path.exists(param_path):
        print("{} not exists.".format(param_path))
    else:
        with open(out_file, 'a+') as hf:
            if os.path.basename(param_path).endswith('.txt'):
                str_header = f"\n******{os.path.basename(param_path)}******\n"
                hf.write(str_header)
                with open(param_path) as f:
                    lines = f.readlines()
                    for line in lines:
                        param_pair = line.rstrip().split(",")
                        key = param_pair[0]
                        value = param_pair[1]
                        hf.write(f"{key}:   {value}\n")
                str_footer = '*' * ((len(str_header) - 3) // 2) + "End" + '*' * ((len(str_header) - 3) // 2) + "\n"
                hf.write(str_footer)
            # elif os.path.basename(param_path).endswith('.xml'):
            #     tree = ET.parse(param_path)
            #     root = tree.getroot()
            #
            #     for entry in root.iter('entry'):
            #         key = entry.attrib['key']
            #         text = entry.text.strip()
            #         hf.write(f"{key}: {text}\n")
            # else:
            #     pass


def get_img_paths(folder, image_types=['*.[Tt][Ii][Ff]*', '*.[Pp][Nn][Gg]', '*.[Jj][Pp][Gg]', '*.[Jj][Pp][Ee][Gg]']):
    img_paths = []
    for image_type in image_types:
        img_paths.extend(glob(join_path(folder, image_type)))
    return img_paths


def convert_parameters(param_file_in_micros, param_file_in_pixels, ims_res):
    with open(param_file_in_micros, 'r') as rf, open(param_file_in_pixels, 'w+') as wf:
        lines = rf.readlines()
        for line in lines:
            param_pair = line.rstrip().split(",")
            key = param_pair[0]
            value = param_pair[1]
            if key.lower().startswith("dark line"):
                wf.write(line)
            elif key.lower().startswith("contrast saturation"):
                wf.write(line)
            elif key.lower().startswith("min line width"):
                wf.write("Min Line Width,{:d}\n".format(int(float(value) / ims_res)))
            elif key.lower().startswith("max line width"):
                wf.write("Max Line Width,{:d}\n".format(int(float(value) / ims_res)))
            elif key.lower().startswith("line width step"):
                wf.write("Line Width Step,{:d}\n".format(int(float(value) / ims_res)))
            elif key.lower().startswith("low contrast") or key.lower().startswith("high contrast"):
                wf.write(line)
            elif key.lower().startswith("min curvature window"):
                wf.write("Min Curvature Window,{:d}\n".format(int(float(value) / ims_res)))
            elif key.lower().startswith("max curvature window"):
                wf.write("Max Curvature Window,{:d}\n".format(int(float(value) / ims_res)))
            elif key.lower().startswith("minimum branch length"):
                wf.write("Minimum Branch Length,{:d}\n".format(int(float(value) / ims_res)))
            elif key.lower().startswith("maximum display hdm"):
                wf.write(line)
            elif key.lower().startswith("minimum gap diameter"):
                wf.write("Minimum Gap Diameter,{:d}\n".format(int(float(value) / ims_res)))
            else:
                print('Invalid parameter {}'.format(key))


class Line:
    id_counter = 0  # Class variable for tracking ID

    def __init__(self, x=None, y=None):
        self.id = Line.id_counter
        Line.id_counter += 1

        self.num = 0 if x is None else len(x)
        self.row = [] if y is None else y
        self.col = [] if x is None else x
        self.angle = [0.0] * self.num
        self.response = [0.0] * self.num
        self.width_l = [0.0] * self.num
        self.width_r = [0.0] * self.num
        self.asymmetry = [0.0] * self.num
        self.intensity = [0.0] * self.num
        self.cont_class = None  # Placeholder for contour class

    def get_contour_class(self):
        return self.cont_class

    def set_contour_class(self, cont_class):
        self.cont_class = cont_class

    def get_x_coordinates(self):
        return self.col

    def get_y_coordinates(self):
        return self.row

    def get_response(self):
        return self.response

    def get_intensity(self):
        return self.intensity

    def get_angle(self):
        return self.angle

    def get_asymmetry(self):
        return self.asymmetry

    def get_line_width_l(self):
        return self.width_l

    def get_line_width_r(self):
        return self.width_r

    def get_number(self):
        return self.num

    def get_id(self):
        return self.id

    def get_line_class(self):
        return self.get_contour_class()

    def get_start_or_end_position(self, x, y):
        dist_start = ((self.col[0] - x) ** 2 + (self.row[0] - y) ** 2) ** 0.5
        dist_end = ((self.col[-1] - x) ** 2 + (self.row[-1] - y) ** 2) ** 0.5
        return 0 if dist_start < dist_end else self.num - 1

    def estimate_length(self):
        length = 0
        for i in range(1, self.num):
            length += ((self.col[i] - self.col[i - 1]) ** 2 + (self.row[i] - self.row[i - 1]) ** 2) ** 0.5
        return length

    @classmethod
    def reset_counter(cls):
        cls.id_counter = 0


class Crossref:
    def __init__(self, y=0, x=0, value=0.0, done=False):
        self.y = y
        self.x = x
        self.value = value
        self.done = done

    def __lt__(self, other):
        # Reverse the logic for descending order
        return self.value > other.value


class LinesUtil:
    DERIV_R = 1  # Derivative in row direction
    DERIV_C = 2  # Derivative in column direction
    DERIV_RR = 3  # Second derivative in row direction
    DERIV_RC = 4  # Second derivative in row and column direction
    DERIV_CC = 5  # Second derivative in column direction

    MODE_LIGHT = 1  # Extract bright lines
    MODE_DARK = 2  # Extract dark lines

    MAX_SIZE_MASK_0 = 3.09023230616781  # Size for Gaussian mask
    MAX_SIZE_MASK_1 = 3.46087178201605  # Size for 1st derivative mask
    MAX_SIZE_MASK_2 = 3.82922419517181  # Size for 2nd derivative mask

    @staticmethod
    def MASK_SIZE(MAX, sigma):
        return int(MAX * sigma + 0.5)  # Maximum mask index

    @staticmethod
    def LINCOOR(row, col, width):
        return row * width + col

    @staticmethod
    def BR(row, height):
        return np.abs(row) if row < 0 else (height - row + height - 2) if row >= height else row

    @staticmethod
    def BC(col, width):
        return np.abs(col) if col < 0 else (width - col + width - 2) if col >= width else col

    class ContourClass(Enum):
        # The cont no junc
        cont_no_junc = 1,
        # The cont start junc
        # no end point is a junction
        cont_start_junc = 2,
        # The cont end junc.
        # only the start point of the line is a junction
        cont_end_junc = 3,
        # The cont both junc.
        # only the end point of the line is a junction
        cont_both_junc = 4,
        # The cont closed.
        # both end points of the line are junctions
        cont_closed = 5  # the contour is closed


class Junction:
    def __init__(self, cont1=-1, cont2=-1, pos=0, y=0.0, x=0.0,
                 line_cont1=None, line_cont2=None, is_non_terminal=False):
        self.cont1 = cont1
        self.cont2 = cont2
        self.pos = pos
        self.y = y
        self.x = x
        self.line_cont1 = line_cont1
        self.line_cont2 = line_cont2
        self.is_non_terminal = is_non_terminal

    def __lt__(self, other):
        """Implements less than for sorting. Compares first by cont1, then by pos."""
        if self.cont1 != other.cont1:
            return self.cont1 < other.cont1
        else:
            return self.pos < other.pos


class Normal:
    SQRT_2_PI_INV = 0.39894228040143267793994605993
    SQRTPI = 1.772453850905516027
    UPPERLIMIT = 20.0

    P10 = 242.66795523053175
    P11 = 21.979261618294152
    P12 = 6.9963834886191355
    P13 = -.035609843701815385

    Q10 = 215.05887586986120
    Q11 = 91.164905404514901
    Q12 = 15.082797630407787
    Q13 = 1.0

    P20 = 300.4592610201616005
    P21 = 451.9189537118729422
    P22 = 339.3208167343436870
    P23 = 152.9892850469404039
    P24 = 43.16222722205673530
    P25 = 7.211758250883093659
    P26 = .5641955174789739711
    P27 = -.0000001368648573827167067

    Q20 = 300.4592609569832933
    Q21 = 790.9509253278980272
    Q22 = 931.3540948506096211
    Q23 = 638.9802644656311665
    Q24 = 277.5854447439876434
    Q25 = 77.00015293522947295
    Q26 = 12.78272731962942351
    Q27 = 1.0

    P30 = -.00299610707703542174
    P31 = -.0494730910623250734
    P32 = -.226956593539686930
    P33 = -.278661308609647788
    P34 = -.0223192459734184686

    Q30 = .0106209230528467918
    Q31 = .191308926107829841
    Q32 = 1.05167510706793207
    Q33 = 1.98733201817135256
    Q34 = 1.0

    SQRT2 = 1.41421356237309504880

    @staticmethod
    def getNormal(x):
        if x < -Normal.UPPERLIMIT:
            return 0.0
        if x > Normal.UPPERLIMIT:
            return 1.0

        y = x / Normal.SQRT2
        sn = 1
        if y < 0:
            y = -y
            sn = -1

        y2 = y * y
        y4 = y2 * y2
        y6 = y4 * y2

        if y < 0.46875:
            R1 = Normal.P10 + Normal.P11 * y2 + Normal.P12 * y4 + Normal.P13 * y6
            R2 = Normal.Q10 + Normal.Q11 * y2 + Normal.Q12 * y4 + Normal.Q13 * y6
            erf = y * R1 / R2
            if sn == 1:
                phi = 0.5 + 0.5 * erf
            else:
                phi = 0.5 - 0.5 * erf
        elif y < 4.0:
            y3 = y2 * y
            y5 = y4 * y
            y7 = y6 * y
            R1 = Normal.P20 + Normal.P21 * y + Normal.P22 * y2 + Normal.P23 * y3 + Normal.P24 * y4 + Normal.P25 * y5 + Normal.P26 * y6 + Normal.P27 * y7
            R2 = Normal.Q20 + Normal.Q21 * y + Normal.Q22 * y2 + Normal.Q23 * y3 + Normal.Q24 * y4 + Normal.Q25 * y5 + Normal.Q26 * y6 + Normal.Q27 * y7
            erfc = np.exp(-y2) * R1 / R2
            if sn == 1:
                phi = 1.0 - 0.5 * erfc
            else:
                phi = 0.5 * erfc
        else:
            z = y4
            z2 = z * z
            z3 = z2 * z
            z4 = z2 * z2
            R1 = Normal.P30 + Normal.P31 * z + Normal.P32 * z2 + Normal.P33 * z3 + Normal.P34 * z4
            R2 = Normal.Q30 + Normal.Q31 * z + Normal.Q32 * z2 + Normal.Q33 * z3 + Normal.Q34 * z4
            erfc = (np.exp(-y2) / y) * (1.0 / Normal.SQRTPI + R1 / (R2 * y2))
            if sn == 1:
                phi = 1.0 - 0.5 * erfc
            else:
                phi = 0.5 * erfc

        return phi


def phi0(x, sigma):
    return Normal.getNormal(x / sigma)


def phi1(x, sigma):
    t = x / sigma
    return Normal.SQRT_2_PI_INV / sigma * np.exp(-0.5 * t * t)


def phi2(x, sigma):
    t = x / sigma
    return -x * Normal.SQRT_2_PI_INV / (sigma ** 3.0) * np.exp(-0.5 * t * t)


def compute_gauss_mask_0(sigma):
    limit = LinesUtil.MASK_SIZE(LinesUtil.MAX_SIZE_MASK_0, sigma)
    n = int(limit)
    h = np.zeros(2 * n + 1, dtype=float)
    for i in range(-n + 1, n):
        h[n + i] = phi0(-i + 0.5, sigma) - phi0(-i - 0.5, sigma)
    h[0] = 1.0 - phi0(n - 0.5, sigma)
    h[2 * n] = phi0(-n + 0.5, sigma)
    return h, n


def compute_gauss_mask_1(sigma):
    limit = LinesUtil.MASK_SIZE(LinesUtil.MAX_SIZE_MASK_1, sigma)
    n = int(limit)
    h = np.zeros(2 * n + 1, dtype=float)
    for i in range(-n + 1, n):
        h[n + i] = phi1(-i + 0.5, sigma) - phi1(-i - 0.5, sigma)
    h[0] = -phi1(n - 0.5, sigma)
    h[2 * n] = phi1(-n + 0.5, sigma)
    return h, n


def compute_gauss_mask_2(sigma):
    limit = LinesUtil.MASK_SIZE(LinesUtil.MAX_SIZE_MASK_2, sigma)
    n = int(limit)
    h = np.zeros(2 * n + 1, dtype=float)
    for i in range(-n + 1, n):
        h[n + i] = phi2(-i + 0.5, sigma) - phi2(-i - 0.5, sigma)
    h[0] = -phi2(n - 0.5, sigma)
    h[2 * n] = phi2(-n + 0.5, sigma)
    return h, n


def convolve_gauss(image, sigma, deriv_type):
    if deriv_type == LinesUtil.DERIV_R:
        hr, nr = compute_gauss_mask_1(sigma)
        hc, nc = compute_gauss_mask_0(sigma)
    elif deriv_type == LinesUtil.DERIV_C:
        hr, nr = compute_gauss_mask_0(sigma)
        hc, nc = compute_gauss_mask_1(sigma)
    elif deriv_type == LinesUtil.DERIV_RR:
        hr, nr = compute_gauss_mask_2(sigma)
        hc, nc = compute_gauss_mask_0(sigma)
    elif deriv_type == LinesUtil.DERIV_RC:
        hr, nr = compute_gauss_mask_1(sigma)
        hc, nc = compute_gauss_mask_1(sigma)
    elif LinesUtil.DERIV_CC:
        hr, nr = compute_gauss_mask_0(sigma)
        hc, nc = compute_gauss_mask_2(sigma)

    return convolve(convolve(image, hr.reshape(-1, 1), mode='nearest'), hc.reshape(1, -1), mode='nearest')


@jit(nopython=True)
def compute_eigenvals(dfdrr, dfdrc, dfdcc):
    eigval = np.zeros(2, dtype=float)
    eigvec = np.zeros((2, 2), dtype=float)
    if dfdrc != 0.0:
        theta = 0.5 * (dfdcc - dfdrr) / dfdrc
        t = 1.0 / (np.abs(theta) + (theta ** 2 + 1.0) ** 0.5)
        if theta < 0.0:
            t = -t
        c = 1.0 / (t ** 2 + 1.0) ** 0.5
        s = t * c
        e1 = dfdrr - t * dfdrc
        e2 = dfdcc + t * dfdrc
    else:
        c = 1.0
        s = 0.0
        e1 = dfdrr
        e2 = dfdcc

    n1 = c
    n2 = -s

    if np.abs(e1) > np.abs(e2):
        eigval[0] = e1
        eigval[1] = e2
        eigvec[0][0] = n1
        eigvec[0][1] = n2
        eigvec[1][0] = -n2
        eigvec[1][1] = n1
    elif np.abs(e1) < np.abs(e2):
        eigval[0] = e2
        eigval[1] = e1
        eigvec[0][0] = -n2
        eigvec[0][1] = n1
        eigvec[1][0] = n1
        eigvec[1][1] = n2
    else:
        if e1 < e2:
            eigval[0] = e1
            eigval[1] = e2
            eigvec[0][0] = n1
            eigvec[0][1] = n2
            eigvec[1][0] = -n2
            eigvec[1][1] = n1
        else:
            eigval[0] = e2
            eigval[1] = e1
            eigvec[0][0] = -n2
            eigvec[0][1] = n1
            eigvec[1][0] = n1
            eigvec[1][1] = n2
    return eigval, eigvec


def normalize(x, pmin=2, pmax=98, axis=None, eps=1e-20, dtype=np.float32):
    """Percentile-based image normalization."""

    mi = np.percentile(x, pmin, axis=axis, keepdims=True)
    ma = np.percentile(x, pmax, axis=axis, keepdims=True)
    if dtype is not None:
        x = x.astype(dtype, copy=False)
        mi = dtype(mi) if np.isscalar(mi) else mi.astype(dtype, copy=False)
        ma = dtype(ma) if np.isscalar(ma) else ma.astype(dtype, copy=False)
        eps = dtype(eps)

    x = (x - mi) / (ma - mi + eps)

    return np.clip(x, 0, 1)


@jit(nopython=True)
def closest_point(ly, lx, dy, dx, py, px):
    my = py - ly
    mx = px - lx
    den = dy * dy + dx * dx
    nom = my * dy + mx * dx
    tt = nom / den if den != 0 else 0
    return ly + tt * dy, lx + tt * dx, tt


@jit(nopython=True)
def bresenham(ny, nx, length, py=0.0, px=0.0):
    points = []
    x, y = 0, 0
    dx, dy = abs(nx), abs(ny)
    s1 = 1 if nx > 0 else -1
    s2 = 1 if ny > 0 else -1
    px *= s1
    py *= s2
    xchg = False
    if dy > dx:
        dx, dy = dy, dx
        px, py = py, px
        xchg = True

    maxit = int(np.ceil(length * dx))
    d_err = dy / dx
    e = (0.5 - px) * dy / dx - (0.5 - py)
    for i in range(maxit + 1):
        points.append([y, x])
        while e >= -1e-8:
            if xchg:
                x += s1
            else:
                y += s2
            e -= 1
            if e > -1:
                points.append([y, x])
        if xchg:
            y += s2
        else:
            x += s1
        e += d_err
    return np.array(points)


@jit(nopython=True)
def interpolate_gradient_test(grady, gradx, py, px):
    giy, gix = math.floor(py), math.floor(px)
    gfy, gfx = py % 1.0, px % 1.0

    gy1, gx1 = grady[giy, gix], gradx[giy, gix]
    gy2, gx2 = grady[giy + 1, gix], gradx[giy + 1, gix]
    gy3, gx3 = grady[giy, gix + 1], grady[giy, gix + 1]
    gy4, gx4 = grady[giy + 1, gix + 1], gradx[giy + 1, gix + 1]

    gy = (1 - gfy) * ((1 - gfx) * gy1 + gfx * gy2) + gfy * ((1 - gfx) * gy3 + gfx * gy4)
    gx = (1 - gfy) * ((1 - gfx) * gx1 + gfx * gx2) + gfy * ((1 - gfx) * gx3 + gfx * gx4)

    return gy, gx


def fill_gaps(master, slave1, slave2, cont, min_value=0.0):
    num_points = cont.num
    i = 0
    while i < num_points:
        if master[i] == 0:
            j = i + 1
            while j < num_points and master[j] == 0:
                j += 1

            s = i
            e = j - 1

            has_start = s > 0
            has_end = j < num_points

            # Use min_value if one end is missing
            m_s = master[s - 1] if has_start else min_value
            m_e = master[j] if has_end else min_value

            if slave1 is not None:
                s1_s = slave1[s - 1] if has_start else min_value
                s1_e = slave1[j] if has_end else min_value
            if slave2 is not None:
                s2_s = slave2[s - 1] if has_start else min_value
                s2_e = slave2[j] if has_end else min_value

            # Clamp j to avoid out-of-bounds on cont arrays
            j_clamped = min(j, num_points - 1)

            # Arc length for the gap segment
            arc_len = np.sum(np.sqrt(
                np.diff(cont.row[s - 1 if has_start else s: j_clamped + 1])**2 +
                np.diff(cont.col[s - 1 if has_start else s: j_clamped + 1])**2
            ))
            if arc_len == 0:
                arc_len = j - s  # fallback to index distance

            len_ = 0
            for k in range(s, j):  # fill s to j-1
                d_r = cont.row[k] - cont.row[k - 1] if k > 0 else 0
                d_c = cont.col[k] - cont.col[k - 1] if k > 0 else 0
                len_ += np.sqrt(d_r ** 2 + d_c ** 2)
                w = len_ / arc_len if arc_len > 0 else 0

                master[k] = max((1 - w) * m_s + w * m_e, min_value)

                if slave1 is not None and slave1[k] == 0:
                    slave1[k] = max((1 - w) * s1_s + w * s1_e, min_value)
                if slave2 is not None and slave2[k] == 0:
                    slave2[k] = max((1 - w) * s2_s + w * s2_e, min_value)

            # Also fill j (the right endpoint) if it was zero
            if j < num_points:
                if master[j] == 0:
                    master[j] = max(m_e, min_value)
                if slave1 is not None and slave1[j] == 0:
                    slave1[j] = max(s1_e, min_value)
                if slave2 is not None and slave2[j] == 0:
                    slave2[j] = max(s2_e, min_value)

            i = j
        else:
            i += 1

    return master


def detect_branches(skeleton):
    selems = list()
    selems.append(np.array([[0, 1, 0], [1, 1, 1], [0, 0, 0]]))
    selems.append(np.array([[1, 0, 1], [0, 1, 0], [1, 0, 0]]))
    selems.append(np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0]]))
    selems.append(np.array([[0, 1, 0], [1, 1, 0], [0, 0, 1]]))
    selems.append(np.array([[0, 0, 1], [1, 1, 1], [0, 1, 0]]))
    selems = [np.rot90(selems[i], k=j) for i in range(5) for j in range(4)]

    selems.append(np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
    selems.append(np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]]))

    branches = np.zeros_like(skeleton, dtype=bool)
    for selem in selems:
        branches |= ndi.binary_hit_or_miss(skeleton, selem)

    return branches


def detect_ends(skeleton):
    selems = list()
    selems.append(np.array([[0, 1, 0], [0, 1, 0], [0, 0, 0]]))
    selems.append(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]]))
    selems = [np.rot90(selems[i], k=j) for i in range(2) for j in range(4)]
    ends = np.zeros_like(skeleton, dtype=bool)
    for selem in selems:
        ends |= ndi.binary_hit_or_miss(skeleton, selem)
    return ends


def normalize_to_half_circle(angle):
    if angle < 0.0:
        angle += 2.0 * np.pi
    if angle >= np.pi:
        angle -= np.pi
    return angle


def interpolate_response(resp, x, y, px, py, width, height):
    i1 = resp[LinesUtil.LINCOOR(LinesUtil.BR(x - 1, height), LinesUtil.BC(y - 1, width), width)]
    i2 = resp[LinesUtil.LINCOOR(LinesUtil.BR(x - 1, height), y, width)]
    i3 = resp[LinesUtil.LINCOOR(LinesUtil.BR(x - 1, height), LinesUtil.BC(y + 1, width), width)]
    i4 = resp[LinesUtil.LINCOOR(x, LinesUtil.BC(y - 1, width), width)]
    i5 = resp[LinesUtil.LINCOOR(x, y, width)]
    i6 = resp[LinesUtil.LINCOOR(x, LinesUtil.BC(y + 1, width), width)]
    i7 = resp[LinesUtil.LINCOOR(LinesUtil.BR(x + 1, height), LinesUtil.BC(y - 1, width), width)]
    i8 = resp[LinesUtil.LINCOOR(LinesUtil.BR(x + 1, height), y, width)]
    i9 = resp[LinesUtil.LINCOOR(LinesUtil.BR(x + 1, height), LinesUtil.BC(y + 1, width), width)]
    t1 = i1 + i2 + i3
    t2 = i4 + i5 + i6
    t3 = i7 + i8 + i9
    t4 = i1 + i4 + i7
    t5 = i2 + i5 + i8
    t6 = i3 + i6 + i9
    d = (-i1 + 2 * i2 - i3 + 2 * i4 + 5 * i5 + 2 * i6 - i7 + 2 * i8 - i9) / 9
    dr = (t3 - t1) / 6
    dc = (t6 - t4) / 6
    drr = (t1 - 2 * t2 + t3) / 6
    dcc = (t4 - 2 * t5 + t6) / 6
    drc = (i1 - i3 - i7 + i9) / 4
    xx = px - x
    yy = py - y
    return d + xx * dr + yy * dc + xx ** 2 * drr + xx * yy * drc + yy ** 2 * dcc


def interpolate_gradient(gradx, grady, px, py, width):
    gix = int(px // 1)
    giy = int(py // 1)
    gfx = px % 1.0
    gfy = py % 1.0

    gpos = LinesUtil.LINCOOR(gix, giy, width)
    gx1, gy1 = gradx[gpos], grady[gpos]
    gpos = LinesUtil.LINCOOR(gix + 1, giy, width)
    gx2, gy2 = gradx[gpos], grady[gpos]
    gpos = LinesUtil.LINCOOR(gix, giy + 1, width)
    gx3, gy3 = gradx[gpos], grady[gpos]
    gpos = LinesUtil.LINCOOR(gix + 1, giy + 1, width)
    gx4, gy4 = gradx[gpos], grady[gpos]

    gx = (1 - gfy) * ((1 - gfx) * gx1 + gfx * gx2) + gfy * ((1 - gfx) * gx3 + gfx * gx4)
    gy = (1 - gfy) * ((1 - gfx) * gy1 + gfx * gy2) + gfy * ((1 - gfx) * gy3 + gfx * gy4)
    return gx, gy


def fix_locations(cont, width_l, width_r, grad_l, grad_r, pos_y, pos_x, sigma_map,
                  correct_pos=True, mode=LinesUtil.MODE_DARK):
    num_points = cont.num
    correction = np.zeros(num_points, dtype=float)
    # contr = np.zeros(num_points, dtype=float)
    asymm = np.zeros(num_points, dtype=float)

    # Fill gaps in width_l and width_r
    fill_gaps(width_l, grad_l, None, cont)
    fill_gaps(width_r, grad_r, None, cont)

    # Correct positions if required
    if correct_pos:
        correct_start = (
                (cont.cont_class in [
                    LinesUtil.ContourClass.cont_no_junc,
                    LinesUtil.ContourClass.cont_end_junc,
                    LinesUtil.ContourClass.cont_closed
                ]) and (width_r[0] > 0 and width_l[0] > 0)
        )
        correct_end = (
                (cont.cont_class in [
                    LinesUtil.ContourClass.cont_no_junc,
                    LinesUtil.ContourClass.cont_start_junc,
                    LinesUtil.ContourClass.cont_closed
                ]) and (width_r[-1] > 0 and width_l[-1] > 0)
        )

        for i in range(num_points):
            if width_r[i] > 0 and width_l[i] > 0:
                w_est = (width_r[i] + width_l[i]) * LINE_WIDTH_COMPENSATION
                if grad_r[i] <= grad_l[i]:
                    r_est = grad_r[i] / grad_l[i]
                    weak_is_r = True
                else:
                    r_est = grad_l[i] / grad_r[i]
                    weak_is_r = False
                sigma = sigma_map[int(cont.row[i]), int(cont.col[i])]
                w_real, h_real, corr, w_strong, w_weak = Correct.line_corrections(
                    sigma, w_est, r_est)
                w_real /= LINE_WIDTH_COMPENSATION
                corr /= LINE_WIDTH_COMPENSATION
                width_r[i], width_l[i] = w_real, w_real
                if weak_is_r:
                    asymm[i] = h_real
                    correction[i] = -corr
                else:
                    asymm[i] = -h_real
                    correction[i] = corr

        fill_gaps(width_l, correction, asymm, cont)
        width_r = width_l[:]

        if not correct_start:
            correction[0] = 0
        if not correct_end:
            correction[-1] = 0

        for i in range(num_points):
            py, px = pos_y[i], pos_x[i]
            ny, nx = np.sin(cont.angle[i]), np.cos(cont.angle[i])
            px += correction[i] * nx
            py += correction[i] * ny
            pos_y[i], pos_x[i] = py, px

    # Update position of line and add extracted width
    width_l = gaussian_filter1d(width_l, 3.0, mode='mirror')
    width_r = gaussian_filter1d(width_r, 3.0, mode='mirror')
    cont.width_l = np.array([float(w) for w in width_l])
    cont.width_r = np.array([float(w) for w in width_r])
    cont.row = np.array([float(y) for y in pos_y])
    cont.col = np.array([float(x) for x in pos_x])

    # # Calculate true contrast if required
    # if correct_pos:
    #     cont.asymmetry = np.zeros(num_points, dtype=float)
    #     cont.intensity = np.zeros(num_points, dtype=float)
    #
    #     for i in range(num_points):
    #         response = cont.response[i]
    #         asymmetry = np.abs(asymm[i])
    #         correct = np.abs(correction[i])
    #         width = cont.width_l[i]
    #         sigma = sigma_map[round(cont.row[i]), round(cont.col[i])]
    #
    #         if width < MIN_LINE_WIDTH:
    #             contrast = 0
    #         else:
    #             contrast = (response / np.abs(phi2(correct + width, sigma) +
    #                                           (asymmetry - 1) * phi2(correct - width, sigma)))
    #
    #         if contrast > MAX_CONTRAST:  # Define MAX_CONTRAST
    #             contrast = 0
    #
    #         contr[i] = contrast
    #
    #     fill_gaps(contr, None, None, cont)
    #
    #     for i in range(num_points):
    #         cont.asymmetry[i] = asymm[i]
    #         if mode == LinesUtil.MODE_LIGHT:
    #             cont.intensity[i] = contr[i]
    #         else:
    #             cont.intensity[i] = -contr[i]

    return cont


def color_line_segments(image, conts):
    for cont in conts:
        hue = random.random()
        saturation = 1
        brightness = 1
        # Convert HSV color to RGB color
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, brightness)
        # Scale RGB values to the range [0, 255]
        random_color = [int(r * 255), int(g * 255), int(b * 255)]
        for j in range(cont.num):
            image[round(cont.row[j]), round(cont.col[j]), :] = random_color

    return image


def visualize(gray, mag, ny, nx, saliency, gd=5):
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    h00 = axes[0, 0].imshow(gray)
    h01 = axes[0, 1].imshow(mag)
    h10 = axes[1, 0].imshow(saliency * 127)
    axes[1, 1].imshow(gray)
    rows, cols = saliency.shape
    indices = np.argwhere(saliency)
    row_idx, col_idx = indices[:, 0], indices[:, 1]
    Y, X = np.mgrid[0:rows, 0:cols]
    S = np.zeros_like(mag)
    S[row_idx, col_idx] = mag[row_idx, col_idx]

    axes[1, 1].quiver(X[::gd, ::gd], Y[::gd, ::gd],
                      (S * nx)[::gd, ::gd],
                      (S * ny)[::gd, ::gd], angles='xy', scale=100)

    fig.colorbar(h00, ax=axes[0, 0])
    fig.colorbar(h01, ax=axes[0, 1])
    fig.colorbar(h10, ax=axes[1, 0])
    axes[0, 0].set_title("Gray image")
    axes[0, 1].set_title("Eigen magnitude")
    axes[1, 0].set_title("Saliency map")
    axes[1, 1].set_title("Eigen vectors")
    plt.show()


def overlay_colorbar(rgb_img,
                     img,
                     save_path,
                     clabel="Curvature (degrees)",
                     cmap='plasma',
                     mode="overwrite",
                     dpi=100,
                     font_size=12):
    """
    Overlay a colorbar on an image using a thread-safe approach.
    """
    height, width = img.shape[:2]

    if mode == "overwrite":
        colormap = matplotlib.colormaps.get_cmap(cmap)
        colored_img = colormap(img / np.max(img))[:, :, :3]
        hsv_img = cv2.cvtColor((colored_img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        hsv_img[:, :, 2] = (normalize(hsv_img[..., 2], 0, 100) * 255).astype(np.uint8)
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
        hsv_img[:, :, 2] = (normalize(hsv_img[..., 2], 0, 100) * 255).astype(np.uint8)
        colored_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
        colored_img = cv2.addWeighted(colored_img, 0.3, gray_img, 0.7, 50)

    # Create the figure for the result with colorbar - using Figure instead of plt.figure
    fig = Figure(figsize=(width / dpi * 1.2, height / dpi), dpi=dpi)
    canvas = FigureCanvasAgg(fig)

    # Create a 2x1 grid layout
    gs = fig.add_gridspec(1, 2, width_ratios=[20, 1], wspace=0.02)

    # Add the main image subplot
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(colored_img)
    ax.axis('off')

    # Add the colorbar subplot
    cax = fig.add_subplot(gs[0, 1])

    # Create a ScalarMappable with the same colormap for the colorbar
    # This is the key change - creating a new mappable directly on the target figure
    norm = matplotlib.colors.Normalize(vmin=np.min(img), vmax=np.max(img))
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(img)

    # Create the colorbar using the ScalarMappable
    cbar = fig.colorbar(sm, cax=cax)
    cbar.ax.set_ylabel(clabel, fontsize=font_size)
    cbar.ax.tick_params(labelsize=font_size)

    # Save the figure using the canvas directly
    canvas.draw()
    fig.savefig(save_path, format='png', transparent=False, facecolor='white')

    # Clean up resources
    fig.clf()

    return rgb_img

def add_colorbar(rgb_img, img, clabel="Curvature (degrees)", cmap='inferno', dpi=100, font_size=12):
    height, width = img.shape[:2]
    colormap = matplotlib.colormaps.get_cmap(cmap)
    colored_img = colormap(img / np.max(img))[:, :, :3]
    hsv_img = cv2.cvtColor((colored_img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    hsv_img[:, :, 2] = (normalize(hsv_img[..., 2], 0, 100) * 255).astype(np.uint8)
    colored_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB) / 255.0

    fig1 = Figure(figsize=(width / dpi * 1.2, height / dpi), dpi=dpi, frameon=False)
    ax1 = fig1.add_subplot()
    img_tmp = ax1.imshow(img, cmap=cmap)
    fig2 = Figure(figsize=(width / dpi * 1.2, height / dpi), dpi=dpi, frameon=False)
    ax2 = fig2.add_subplot()
    ax2.imshow(cv2.addWeighted(rgb_img, 0.3, (colored_img*255).astype(np.uint8), 0.7, 20))
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


if __name__ == "__main__":
    path = join_path("C:\\Users\\Dcouments", "TWOMBLI", "")
    print(path)
