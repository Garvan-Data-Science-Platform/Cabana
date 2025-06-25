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
import csv
import imutils
from . import convcrf
import argparse
import numpy as np
import torch.nn.init
from glob import glob
from tqdm import tqdm
from skimage import measure
import torch.optim as optim
from .log import Log
from torch.autograd import Variable
from skimage.morphology import remove_small_objects, remove_small_holes
from .models import BackBone, LightConv3x3
from .utils import mean_image, cal_color_dist, save_result_video, read_bar_format

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
    parser.add_argument('--save_video', default=False, action='store_true',
                        help='Save intermediate results as video')
    parser.add_argument('--save_frame_interval', default=2, type=int,
                        help='Save frame every N iterations when saving video')
    parser.add_argument('--roi_dir', type=str, default="./output/ROIs",
                        help='Directory to save ROI images')
    parser.add_argument('--bin_dir', type=str, default="./output/Bins",
                        help='Directory to save binary mask images')
    parser.add_argument('--input', type=str, help='Input image path', required=False)
    args, _ = parser.parse_known_args()
    return args


def segment_single_image(args):
    """
    Segment a single image using a CNN + CRF approach.

    This function performs the following steps:
    1. Load and preprocess the input image
    2. Initialize the model and CRF
    3. Train the model iteratively on this single image
    4. Apply thresholding based on color properties
    5. Generate and save ROI images and masks

    Args:
        args (argparse.Namespace): Configuration parameters

    Returns:
        tuple: (area, percentage) where area is the segmented area in pixels and
               percentage is the ratio of segmented area to the total image area
    """
    # Set seeds for reproducibility
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    # Load and preprocess the input image
    ori_img = cv2.imread(args.input)
    img_name = os.path.splitext(os.path.basename(args.input))[0]

    # Rotate image if portrait orientation is detected
    rotated = False
    if ori_img.shape[0] > ori_img.shape[1]:
        ori_img = cv2.rotate(ori_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        rotated = True

    ori_width, ori_height = ori_img.shape[::-1][1:]

    # Resize the image to a standard width while maintaining aspect ratio
    img = imutils.resize(ori_img, width=512)
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_size = img.shape[:2]

    # Prepare image for PyTorch processing
    img = img.transpose(2, 0, 1)  # Convert to channels-first format
    data = torch.from_numpy(np.array([img.astype('float32') / 255.]))
    img_var = torch.Tensor(img.reshape([1, 3, *img_size]))  # 1, 3, h, w

    # Initialize CRF with configuration
    config = convcrf.default_conf
    config['filter_size'] = args.sz_filter

    gausscrf = convcrf.GaussCRF(conf=config,
                                shape=img_size,
                                nclasses=args.num_channels,
                                use_gpu=torch.cuda.is_available())

    # Initialize the segmentation model
    model = BackBone([LightConv3x3], [2], [args.num_channels // 2, args.num_channels])

    # Move to GPU if available
    if torch.cuda.is_available():
        data = data.cuda()
        img_var = img_var.cuda()
        gausscrf = gausscrf.cuda()
        model = model.cuda()

    data = Variable(data)
    img_var = Variable(img_var)

    # Set up model for training
    model.train()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    # Initialize arrays for storing intermediate results if video saving is enabled
    label_colours = np.random.randint(255, size=(100, 3))
    all_image_labels = []
    all_mean_images = []
    all_absolute_greenness = []
    all_relative_greenness = []
    all_thresholded = []

    # Training loop with progress bar
    pbar = tqdm(range(args.max_iter), bar_format=read_bar_format)
    for batch_idx in pbar:
        # Forward pass
        optimizer.zero_grad()
        output = model(data)[0]
        unary = output.unsqueeze(0)
        prediction = gausscrf.forward(unary=unary, img=img_var)
        target = torch.argmax(prediction.squeeze(0), axis=0).reshape(img_size[0] * img_size[1], )
        output = output.permute(1, 2, 0).contiguous().view(-1, args.num_channels)

        # Process prediction
        im_target = target.data.cpu().numpy()
        image_labels = im_target.reshape(img_size[0], img_size[1]).astype("uint8")
        num_labels = len(np.unique(im_target))

        # Save intermediate results if video saving is enabled
        if args.save_video and not (batch_idx % args.save_frame_interval):
            im_target_rgb = np.array([label_colours[c % 100] for c in im_target])
            im_target_rgb = im_target_rgb.reshape(img_size[0], img_size[1], 3).astype("uint8")
            mean_img = mean_image(rgb_image, measure.label(image_labels))
            abs_color_dist, rel_color_dist = cal_color_dist(mean_img, args.hue_value)
            thresholded = 255 * ((rel_color_dist > args.rt).astype("uint8"))
            all_mean_images.append(mean_img)
            all_absolute_greenness.append(abs_color_dist)
            all_relative_greenness.append(rel_color_dist)
            all_thresholded.append(thresholded)
            all_image_labels.append(im_target_rgb)

        # Backward pass
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        # Update progress bar
        pbar.set_description(
            f"Iteration {batch_idx}/{args.max_iter}: {num_labels} labels, loss: {loss.item():.2f}")

        # Early stopping condition
        if num_labels <= args.min_labels:
            Log.logger.debug(f"nLabels {num_labels} reached minLabels {args.min_labels}: {args.input}")
            break

    # Save results
    if args.save_video:
        # Save video with intermediate results
        save_result_path = os.path.join(args.bin_dir, img_name + "_result.mp4")
        save_result_video(save_result_path, rgb_image, all_image_labels, all_mean_images,
                          all_absolute_greenness, all_relative_greenness, all_thresholded)
    else:
        # Process and save the final segmentation result
        labels = measure.label(image_labels)
        mean_img = mean_image(rgb_image, labels)
        abs_color_dist, rel_color_dist = cal_color_dist(mean_img, args.hue_value)

        # Apply threshold and clean up small objects/holes
        thresholded = rel_color_dist > args.rt
        thresholded = remove_small_holes(thresholded, args.min_size)
        thresholded = remove_small_objects(thresholded, args.min_size)
        mask = 255 * (thresholded.astype("uint8"))

        # Resize mask to original image size
        mask = cv2.resize(mask, (ori_width, ori_height), cv2.INTER_NEAREST)

        # Generate ROI with masked background
        roi_img = generate_rois(ori_img, (mask > 128).astype("uint8") * 255, args.white_background)

        # Save results, with rotation if needed
        if rotated:
            cv2.imwrite(os.path.join(args.roi_dir, img_name + '_roi.png'),
                        cv2.rotate(roi_img, cv2.ROTATE_90_CLOCKWISE))
            cv2.imwrite(os.path.join(args.bin_dir, img_name + '_mask.png'),
                        (cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE) > 128).astype("uint8") * 255)
        else:
            cv2.imwrite(os.path.join(args.roi_dir, img_name + '_roi.png'), roi_img)
            cv2.imwrite(os.path.join(args.bin_dir, img_name + '_mask.png'), (mask > 128).astype("uint8") * 255)

        # Return area and percentage metrics
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
    (x_idx, y_idx) = np.where(eroded_roi == 255)

    # Apply background color to masked regions
    img_roi = img.copy()
    for row, col in zip(list(x_idx), list(y_idx)):
        img_roi[row, col, :] = np.array(background_color)

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