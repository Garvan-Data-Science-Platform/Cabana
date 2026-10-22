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
import sys
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
from skimage.morphology import remove_small_objects, remove_small_holes
from .models import BackBone, LightConv3x3
from .utils import mean_image, cal_color_dist, is_greyscale

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
                        help='Colour threshold: minimum absolute colour score of a region (1 = the target hue)')
    parser.add_argument('--mode', type=str, default="both",
                        help='Processing mode')
    parser.add_argument('--min_size', default=64, type=int,
                        help='The smallest allowable object size in pixels')
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


def select_device():
    """Torch device for the segmentation network: CUDA when available, else
    Apple's Metal backend (MPS) on macOS, else CPU. Set ``CABANA_DEVICE=cpu``
    (or ``cuda``/``mps``) to override."""
    forced = os.environ.get("CABANA_DEVICE", "").strip().lower()
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if sys.platform == "darwin" and mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _reduce(img_bgr, cnn_size):
    """Resize so that the longer side is ``cnn_size`` (no rotation)."""
    h, w = img_bgr.shape[:2]
    return imutils.resize(img_bgr, width=cnn_size) if w >= h else imutils.resize(img_bgr, height=cnn_size)


def _tensors(img, device):
    """Network input (0-1) and CRF image (0-255) tensors of a reduced BGR image."""
    chw = np.ascontiguousarray(img.transpose(2, 0, 1)).astype("float32")
    return torch.from_numpy(chw[None] / 255.).to(device), torch.from_numpy(chw[None]).to(device)


def _make_crf(shape, args, device):
    config = convcrf.default_conf
    config['filter_size'] = args.sz_filter
    return convcrf.GaussCRF(conf=config, shape=shape, nclasses=args.num_channels,
                            use_gpu=(device.type == "cuda")).to(device)


def _train_network(img, args, step, device):
    """Train the self-supervised CNN + CRF on the reduced image ``img``.

    The network is trained to agree with its own CRF-smoothed labels for up to
    ``args.max_iter`` iterations (fewer once only ``args.min_labels`` labels
    are left). ``step(i)`` is called after iteration ``i``. Returns the trained
    model, the CRF and the label map of the last iteration.
    """
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    img_size = img.shape[:2]
    data, img_var = _tensors(img, device)
    crf = _make_crf(img_size, args, device)
    model = BackBone([LightConv3x3], [2], [args.num_channels // 2, args.num_channels]).to(device)
    model.train()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    labels = None
    for it in range(args.max_iter):
        optimizer.zero_grad()
        output = model(data)[0]
        prediction = crf.forward(unary=output.unsqueeze(0), img=img_var)
        target = torch.argmax(prediction.squeeze(0), axis=0).reshape(img_size[0] * img_size[1], )
        output = output.permute(1, 2, 0).contiguous().view(-1, args.num_channels)
        im_target = target.data.cpu().numpy()
        labels = im_target.reshape(img_size).astype("uint8")
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        step(it)
        if len(np.unique(im_target)) <= args.min_labels:
            Log.logger.debug(f"nLabels reached minLabels {args.min_labels}")
            break
    # The model stays in training mode for labelling patches: its BatchNorm
    # layers then normalise each patch with the patch's own statistics. The
    # running statistics of eval mode are not settled after so few iterations
    # and lose most collagen on weakly stained images.
    return model, crf, labels


def _label_image(model, crfs, img, args, device):
    """Labels of the reduced image ``img`` from a trained model, without training."""
    shape = img.shape[:2]
    if shape not in crfs:
        crfs[shape] = _make_crf(shape, args, device)
    data, img_var = _tensors(img, device)
    with torch.no_grad():
        output = model(data)[0]
        prediction = crfs[shape].forward(unary=output.unsqueeze(0), img=img_var)
    return torch.argmax(prediction.squeeze(0), axis=0).cpu().numpy().astype("uint8")


def _colour_score(img, labels, hue_value, greyscale):
    """Absolute colour score (0-1) of every pixel of the reduced image ``img``:
    each connected region of one label takes its mean colour, scored by its
    closeness to the target hue (1 = exactly the target colour; bright white
    for a greyscale image)."""
    mean_img = mean_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), measure.label(labels))
    absolute, _ = cal_color_dist(mean_img, hue_value, greyscale=greyscale)
    return absolute.astype(np.float32)


def _per_image_scale(score):
    """Min-max normalise a greyscale image's score over the whole image."""
    lo, hi = float(score.min()), float(score.max())
    return (score - lo) / (hi - lo + np.finfo(float).eps)


def _patch_grid(h, w, args):
    """``(patch, overlap, ys, xs)`` of the patch tiling, or None when the image
    is segmented whole (Patch Size off, or the image fits in one patch)."""
    patch = int(getattr(args, 'patch_size', 0) or 0)
    if patch <= 0 or max(h, w) <= patch:
        return None
    overlap = int(round(patch * float(getattr(args, 'patch_overlap', 0.125) or 0)))
    return patch, overlap, tile_windows(h, patch, overlap), tile_windows(w, patch, overlap)


def segmentation_steps(h, w, args):
    """Progress steps of segmenting an ``h`` x ``w`` image: the training
    iterations plus one labelling pass per patch."""
    grid = _patch_grid(h, w, args)
    return int(args.max_iter) + (len(grid[2]) * len(grid[3]) if grid else 0)


def colour_score_map(img_bgr, args, iter_callback=None, cnn_size=512):
    """Absolute colour-score map of an image, see :func:`segment_image`.

    Returns ``(score, min_size)``: the score map (at the reduced size when the
    image is segmented whole, else at full size) and the smallest object size
    in pixels of that map. Falls back to the CPU when the accelerator rejects
    an operation (e.g. a kernel missing on MPS)."""
    device = select_device()
    try:
        return _colour_score_map(img_bgr, args, iter_callback, cnn_size, device)
    except (RuntimeError, NotImplementedError) as e:
        if device.type == "cpu":
            raise
        Log.logger.warning(f"Segmentation on {device} failed ({e}); retrying on CPU.")
        return _colour_score_map(img_bgr, args, iter_callback, cnn_size, torch.device("cpu"))


def _colour_score_map(img_bgr, args, iter_callback, cnn_size, device):
    h, w = img_bgr.shape[:2]
    total = segmentation_steps(h, w, args)

    def step(i):
        if iter_callback is not None:
            iter_callback(i, total)

    # Greyscale images (SHG, fluorescence) are scored by brightness, relative
    # to the image: the scores are min-max normalised once over the whole
    # image, as in earlier versions. Colour images use the absolute score.
    grey = is_greyscale(img_bgr)
    small = _reduce(img_bgr, cnn_size)
    model, crf, labels = _train_network(small, args, step, device)
    grid = _patch_grid(h, w, args)
    if grid is None:
        score = _colour_score(small, labels, args.hue_value, grey)
        return (_per_image_scale(score) if grey else score), int(args.min_size)

    # Train once, label every patch: the network trained on the whole image
    # labels each patch (reduced to cnn_size) without further training, so the
    # labels mean the same in every patch. The absolute scores are blended
    # across the overlaps and the merged map is thresholded once.
    patch, overlap, ys, xs = grid
    crfs = {small.shape[:2]: crf}
    acc = np.zeros((h, w), dtype=np.float32)
    wsum = np.zeros((h, w), dtype=np.float32)
    weights = _feather_weights(patch, patch, overlap)
    k = 0
    for y0 in ys:
        for x0 in xs:
            ph, pw = min(patch, h - y0), min(patch, w - x0)
            sub = _reduce(img_bgr[y0:y0 + ph, x0:x0 + pw], cnn_size)
            score = _colour_score(sub, _label_image(model, crfs, sub, args, device), args.hue_value, grey)
            wgt = weights[:ph, :pw]
            acc[y0:y0 + ph, x0:x0 + pw] += cv2.resize(score, (pw, ph), interpolation=cv2.INTER_LINEAR) * wgt
            wsum[y0:y0 + ph, x0:x0 + pw] += wgt
            step(int(args.max_iter) + k)
            k += 1
    # min_size keeps meaning "objects smaller than N pixels at CNN resolution"
    scale = (patch / float(cnn_size)) ** 2
    merged = acc / np.maximum(wsum, 1e-6)
    return (_per_image_scale(merged) if grey else merged), max(1, int(round(args.min_size * scale)))


def _clean_mask(thresholded, min_size):
    thresholded = remove_small_holes(thresholded, max_size=min_size)
    # skimage >= 0.26: ``max_size`` removes objects of size <= max_size, so
    # ``min_size - 1`` keeps the old "smaller than min_size" rule exactly.
    thresholded = remove_small_objects(thresholded, max_size=max(0, int(min_size) - 1))
    return thresholded


def mask_from_score(score, min_size, threshold, shape):
    """Binary ``uint8`` mask (0/255) of size ``shape`` from a colour-score map:
    regions scoring above ``threshold`` (absolute, 1 = the target colour),
    cleaned of objects and holes smaller than ``min_size`` pixels of the map,
    then up-sampled (nearest neighbour) when the map is smaller. Cheap, so a
    host can re-threshold without segmenting again."""
    mask = 255 * _clean_mask(score > threshold, min_size).astype("uint8")
    if mask.shape[:2] != tuple(shape[:2]):
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask


def segment_image(img_bgr, args, iter_callback=None, cnn_size=512, return_score=False):
    """Segment one image and return a binary ``uint8`` mask (0/255) at the
    image's own resolution.

    The self-supervised CNN + CRF is trained once on the image reduced to
    ``cnn_size`` on its longer side; each connected region of one label takes
    its mean colour, and pixels whose region scores above ``args.rt`` on the
    absolute colour score (1 = exactly the target hue ``args.hue_value``) are
    kept. For colour images the threshold is the same on every image: it is
    not rescaled to the image's own score range. Greyscale images (no colour
    at all, e.g. SHG) target bright white and keep the earlier behaviour: the
    scores are min-max normalised over the whole image (never per patch)
    before thresholding.

    When ``args.patch_size`` is set and the image is larger than one patch,
    the image is tiled with edge-anchored square patches overlapping by
    ``args.patch_overlap`` (fraction of the patch). The trained network labels
    each patch (reduced to ``cnn_size``) without further training, the scores
    are blended across the overlaps with linear feathering, and the merged map
    is thresholded once at full resolution. ``min_size`` is scaled by the
    square of the patch reduction factor so it keeps meaning "objects smaller
    than N pixels at CNN resolution".

    ``iter_callback(step, total)`` is called after every training iteration
    and every patch (see :func:`segmentation_steps`). With ``return_score``
    the result is ``(mask, (score, min_size))`` so a host can re-threshold
    with :func:`mask_from_score`.
    """
    score, min_size = colour_score_map(img_bgr, args, iter_callback, cnn_size)
    mask = mask_from_score(score, min_size, args.rt, img_bgr.shape)
    return (mask, (score, min_size)) if return_score else mask


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