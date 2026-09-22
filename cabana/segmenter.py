"""
Self-Supervised Semantic Segmentation for Image Analysis

This module provides functionality for semantic segmentation of images, particularly
focused on isolating objects of interest based on color properties. It uses a 
combination of neural networks and conditional random fields (CRF) to perform 
unsupervised segmentation.

Main components:
- Single image segmentation with CNN + CRF
- ROI (Region of Interest) generation
- Batch processing capability for multiple images
- Optional visualization of the segmentation process

Example usage:
    python script.py --input image.png --hue_value 0.3 --rt 0.25
"""

import os
import cv2
import imutils
from . import convcrf
import argparse
import numpy as np
import torch.nn.init
from glob import glob
from pathlib import Path
from skimage import measure
import torch.optim as optim
from .log import Log
from torch.autograd import Variable
from skimage.morphology import remove_small_objects, remove_small_holes
from .models import BackBone, LightConv3x3
from .utils import mean_image, cal_color_dist

# Set fixed seed for reproducible results
SEED = 0
torch.use_deterministic_algorithms(True)


def parse_args():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='Self-Supervised Semantic Segmentation')
    parser.add_argument('--num_channels', default=32, type=int,
                        help='Number of channels in the segmentation model')
    parser.add_argument('--max_iter', default=30, type=int,
                        help='Maximum number of training iterations')
    parser.add_argument('--min_labels', default=2, type=int,
                        help='Minimum number of labels to stop training early')
    parser.add_argument('--hue_value', default=1.0, type=float,
                        help='Hue value of the color of interest (0-1 range)')
    parser.add_argument('--lr', default=0.1, type=float,
                        help='Learning rate for the segmentation model')
    parser.add_argument('--sz_filter', default=5, type=int,
                        help='CRF filter size')
    parser.add_argument('--rt', default=0.2, type=float,
                        help='Relative color threshold for object detection')
    parser.add_argument('--mode', type=str, default="both",
                        help='Processing mode')
    parser.add_argument('--min_size', default=64, type=int,
                        help='The smallest allowable object size in pixels')
    parser.add_argument('--max_size', default=2048, type=int,
                        help='The maximal allowable image size')
    parser.add_argument('--white_background', default=True, type=bool,
                        help='Set background color to white in output images')
    parser.add_argument('--roi_dir', type=str, default="./output/ROIs",
                        help='Directory to save ROI images')
    parser.add_argument('--bin_dir', type=str, default="./output/Bins",
                        help='Directory to save binary mask images')
    parser.add_argument('--input', type=str, help='Input image path', required=False)
    parser.add_argument('--patch_size', default=0, type=int,
                        help='Segment in square patches of this many pixels (0 = whole image)')
    parser.add_argument('--patch_overlap', default=0.125, type=float,
                        help='Overlap between patches as a fraction of the patch size')
    parser.add_argument('--roi_mask_path', type=str, default=None,
                        help='Optional binary ROI mask intersected with the segmentation')
    args, _ = parser.parse_known_args()
    return args


def tile_windows(length, patch, overlap):
    """Edge-anchored 1-D tiling: window starts so that every window is exactly
    ``patch`` long and the last window ends on the image edge.

    Returns ``[0]`` when ``length <= patch``.
    """
    if length <= patch:
        return [0]
    step = max(1, patch - overlap)
    starts = list(range(0, length - patch, step))
    starts.append(length - patch)
    return starts


def _feather_weights(h, w, margin):
    """Weight map that ramps linearly from 0 at the patch border to 1 at
    ``margin`` pixels inside, used to blend overlapping patches."""
    if margin <= 0:
        return np.ones((h, w), dtype=np.float32)
    ramp_y = np.minimum(np.arange(h) + 1, np.arange(h)[::-1] + 1) / float(margin)
    ramp_x = np.minimum(np.arange(w) + 1, np.arange(w)[::-1] + 1) / float(margin)
    return (np.clip(ramp_y, 0, 1)[:, None] * np.clip(ramp_x, 0, 1)[None, :]).astype(np.float32)


def segment_color_distance(img_bgr, args, iter_callback=None, cnn_size=512):
    """Train the self-supervised CNN + CRF on one image and return the relative
    colour-distance map to the hue of interest.

    The image is resized so that its longer side is ``cnn_size`` before
    training (no rotation). The returned map has that reduced size; callers
    threshold it with ``args.rt`` and resize as needed. ``iter_callback(it,
    max_iter)`` is invoked after every iteration.
    """
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    h, w = img_bgr.shape[:2]
    if w >= h:
        img = imutils.resize(img_bgr, width=cnn_size)
    else:
        img = imutils.resize(img_bgr, height=cnn_size)
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_size = img.shape[:2]

    chw = img.transpose(2, 0, 1)
    data = torch.from_numpy(np.array([chw.astype('float32') / 255.]))
    img_var = torch.Tensor(chw.reshape([1, 3, *img_size]))

    config = convcrf.default_conf
    config['filter_size'] = args.sz_filter
    gausscrf = convcrf.GaussCRF(conf=config, shape=img_size, nclasses=args.num_channels,
                                use_gpu=torch.cuda.is_available())
    model = BackBone([LightConv3x3], [2], [args.num_channels // 2, args.num_channels])
    if torch.cuda.is_available():
        data, img_var = data.cuda(), img_var.cuda()
        gausscrf, model = gausscrf.cuda(), model.cuda()
    data = Variable(data)
    img_var = Variable(img_var)

    model.train()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    image_labels = None
    for batch_idx in range(args.max_iter):
        optimizer.zero_grad()
        output = model(data)[0]
        unary = output.unsqueeze(0)
        prediction = gausscrf.forward(unary=unary, img=img_var)
        target = torch.argmax(prediction.squeeze(0), axis=0).reshape(img_size[0] * img_size[1], )
        output = output.permute(1, 2, 0).contiguous().view(-1, args.num_channels)
        im_target = target.data.cpu().numpy()
        image_labels = im_target.reshape(img_size[0], img_size[1]).astype("uint8")
        num_labels = len(np.unique(im_target))

        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        if iter_callback is not None:
            iter_callback(batch_idx, args.max_iter)
        if num_labels <= args.min_labels:
            Log.logger.debug(f"nLabels {num_labels} reached minLabels {args.min_labels}")
            break

    labels = measure.label(image_labels)
    mean_img = mean_image(rgb_image, labels)
    _, rel_color_dist = cal_color_dist(mean_img, args.hue_value)
    return rel_color_dist.astype(np.float32)


def _clean_mask(thresholded, min_size):
    thresholded = remove_small_holes(thresholded, max_size=min_size)
    thresholded = remove_small_objects(thresholded, min_size)
    return thresholded


def segment_image(img_bgr, args, iter_callback=None, cnn_size=512):
    """Segment one image and return a binary ``uint8`` mask (0/255) at the
    image's own resolution.

    When ``args.patch_size`` is 0 or the image fits inside one patch, the
    whole image is reduced to ``cnn_size`` on its longer side, segmented, and
    the thresholded mask is up-sampled with nearest-neighbour interpolation
    (the original behaviour). Otherwise the image is tiled with edge-anchored
    square patches of ``args.patch_size`` pixels overlapping by
    ``args.patch_overlap`` (fraction of the patch); every patch is reduced to
    ``cnn_size`` and segmented independently, the colour-distance maps are
    up-sampled to patch resolution and blended with linear feathering, and the
    merged map is thresholded once. ``min_size`` is scaled by the square of the
    patch reduction factor so it keeps meaning "objects smaller than N pixels
    at CNN resolution".
    """
    h, w = img_bgr.shape[:2]
    patch = int(getattr(args, 'patch_size', 0) or 0)
    if patch <= 0 or max(h, w) <= patch:
        dist = segment_color_distance(img_bgr, args, iter_callback, cnn_size)
        thresholded = _clean_mask(dist > args.rt, args.min_size)
        mask = 255 * thresholded.astype("uint8")
        return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    overlap = int(round(patch * float(getattr(args, 'patch_overlap', 0.125) or 0)))
    ys = tile_windows(h, patch, overlap)
    xs = tile_windows(w, patch, overlap)
    n_patches = len(ys) * len(xs)
    acc = np.zeros((h, w), dtype=np.float32)
    wsum = np.zeros((h, w), dtype=np.float32)
    weights = _feather_weights(patch, patch, overlap)
    k = 0
    for y0 in ys:
        for x0 in xs:
            ph, pw = min(patch, h - y0), min(patch, w - x0)
            sub = img_bgr[y0:y0 + ph, x0:x0 + pw]

            def _cb(it, max_iter, _k=k):
                if iter_callback is not None:
                    iter_callback(_k * max_iter + it, n_patches * max_iter)

            dist = segment_color_distance(sub, args, _cb, cnn_size)
            dist = cv2.resize(dist, (pw, ph), interpolation=cv2.INTER_LINEAR)
            wgt = weights[:ph, :pw]
            acc[y0:y0 + ph, x0:x0 + pw] += dist * wgt
            wsum[y0:y0 + ph, x0:x0 + pw] += wgt
            k += 1
    merged = acc / np.maximum(wsum, 1e-6)
    scale = (patch / float(cnn_size)) ** 2
    thresholded = _clean_mask(merged > args.rt, max(1, int(round(args.min_size * scale))))
    return 255 * thresholded.astype("uint8")


def load_roi_mask(mask_path, shape):
    """Read an external ROI mask as 0/255 ``uint8`` matching ``shape`` (h, w).

    Returns ``None`` when ``mask_path`` is falsy or unreadable. Masks of a
    different size are resized with nearest-neighbour interpolation.
    """
    if not mask_path or not os.path.exists(mask_path):
        return None
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    if mask.shape[:2] != tuple(shape[:2]):
        Log.logger.warning(f"ROI mask {os.path.basename(mask_path)} is {mask.shape[1]}x{mask.shape[0]}; "
                           f"resizing to the image size {shape[1]}x{shape[0]}.")
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return ((mask > 128).astype(np.uint8)) * 255


def segment_single_image(args, iter_callback=None):
    """
    Segment a single image using a CNN + CRF approach and write the ROI image
    and binary mask.

    Steps:
    1. Load the input image (``args.input``)
    2. Segment it (whole-image or patch-wise, see :func:`segment_image`)
    3. Intersect the result with an external ROI mask when
       ``args.roi_mask_path`` is set (e.g. a TMA core circle)
    4. Save ``<name>_roi.png`` (background masked out) and ``<name>_mask.png``

    Args:
        args: Segmentation arguments (see :func:`parse_args`)
        iter_callback: Optional callable ``(iteration, max_iter)`` invoked after
            every training iteration so hosts can display progress.

    Returns:
        tuple: (mask area in pixels, mask area fraction of the image)
    """
    ori_img = cv2.imread(args.input)
    img_name = os.path.splitext(os.path.basename(args.input))[0]
    ori_height, ori_width = ori_img.shape[:2]

    mask = segment_image(ori_img, args, iter_callback)

    roi_mask = load_roi_mask(getattr(args, 'roi_mask_path', None), ori_img.shape)
    if roi_mask is not None:
        mask = cv2.bitwise_and(mask, roi_mask)

    roi_img = generate_rois(ori_img, mask, args.white_background)
    cv2.imwrite(os.path.join(args.roi_dir, img_name + '_roi.png'), roi_img)
    cv2.imwrite(os.path.join(args.bin_dir, img_name + '_mask.png'), mask)

    return np.sum(mask > 128), np.sum(mask > 128) / (ori_width * ori_height)


def visualize_fibres(img, mask, result_path, thickness=3, border_color=[255, 255, 0]):
    """
    Visualize detected fibres by adding colored borders.

    Args:
        img (numpy.ndarray): Original image
        mask (numpy.ndarray): Binary mask of the detected objects
        result_path (str): Path to save the visualization result
        thickness (int): Thickness of the border
        border_color (list): RGB color for the border
    """
    # Convert mask to grayscale and invert
    mask_gray = cv2.bitwise_not(cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY))

    # Dilate the mask to create borders
    kernel = np.ones((thickness, thickness), np.uint8)
    dilated_mask = cv2.dilate(mask_gray, kernel)

    # Get coordinates of border pixels
    border_color = np.array(border_color)
    (x_idx, y_idx) = np.where(dilated_mask == 255)

    # Apply border color to the image
    img_with_border = img.copy()
    for row, col in zip(list(x_idx), list(y_idx)):
        img_with_border[row, col, :] = border_color

    # Save the result
    cv2.imwrite(result_path, img_with_border)


def generate_rois(img, roi, white_background=True, thickness=3):
    """
    Generate ROI by masking the background of an image.

    Args:
        img (numpy.ndarray): Original image
        roi (numpy.ndarray): Binary mask defining the ROI
        white_background (bool): If True, use white background, else black
        thickness (int): Thickness of the border

    Returns:
        numpy.ndarray: Image with background masked out
    """
    # Ensure roi is a grayscale image
    if roi.ndim > 2 and roi.shape[2] > 1:
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Resize roi to match image dimensions if needed
    if img.shape[:2] != roi.shape:
        roi = cv2.resize(roi, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Set background color based on preference
    background_color = [228, 228, 228] if white_background else [0, 0, 0]

    # Invert roi for processing
    roi = cv2.bitwise_not(roi)

    # Create a dilated mask for border pixels
    kernel = np.ones((thickness, thickness), np.uint8)
    eroded_roi = cv2.dilate(roi, kernel, iterations=1)

    # Apply background color to masked regions
    img_roi = img.copy()
    img_roi[eroded_roi == 255] = np.array(background_color, dtype=img_roi.dtype)

    return img_roi


if __name__ == "__main__":
    # # CSV header for results
    # header = ['Image', 'Area', '% Black']
    args = parse_args()
    segment_single_image(args)
    #
    # # Process images with specific number of channels
    # for num_labels in [48]:
    #     setattr(args, 'num_channels', num_labels)
    #     dst_folder = '/Users/lxfhfut/Dropbox/Garvan/Cabana/Test_ROI/'
    #     src_folder = '/Users/lxfhfut/Dropbox/Garvan/Cabana/Compressed images/'
    #
    #     # Find all images in the source folder
    #     img_names = glob(os.path.join(src_folder, '*.tif')) \
    #                 + glob(os.path.join(src_folder, '.tiff')) \
    #                 + glob(os.path.join(src_folder, '*.TIF')) \
    #                 + glob(os.path.join(src_folder, '*.TIFF')) \
    #                 + glob(os.path.join(src_folder, '*.png')) \
    #                 + glob(os.path.join(src_folder, '*.PNG'))
    #
    #     # Process with specific iteration numbers
    #     for iter_num in [30]:
    #         setattr(args, 'max_iter', iter_num)
    #         output_dir = os.path.join(dst_folder, str(iter_num))
    #         setattr(args, 'save_dir', output_dir)
    #
    #         # Ensure output directories exist
    #         os.makedirs(args.roi_dir, exist_ok=True)
    #         os.makedirs(args.bin_dir, exist_ok=True)
    #
    #         # Process each image and write results to CSV
    #         with open(os.path.join(output_dir, 'Results_ROI.csv'), 'w', encoding='UTF8') as f:
    #             writer = csv.writer(f)
    #             writer.writerow(header)
    #             for img_name in img_names:
    #                 print(f'Processing {img_name}')
    #                 setattr(args, 'input', img_name)
    #
    #                 if not os.path.exists(args.save_dir):
    #                     os.makedirs(args.save_dir)
    #
    #                 # Segment the image and get metrics
    #                 area, percent_black = segment_single_image(args)
    #                 data = [os.path.basename(img_name), area, percent_black]
    #                 writer.writerow(data)
    #
    #             print(f'Result has been saved in {args.save_dir}')