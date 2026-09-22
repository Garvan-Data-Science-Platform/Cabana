import os
import cv2
import yaml
import shutil
import warnings
import numpy as np
import pandas as pd
from .log import Log
from .hdm import HDM
from glob import glob
from PIL import Image
import tifffile as tiff
import imageio.v3 as iio
from pathlib import Path
from .detector import FibreDetector
from .analyzer import SkeletonAnalyzer
from skimage.feature import peak_local_max
from .orientation import OrientationAnalyzer
from skimage.color import rgb2hed, hed2rgb, rgb2gray
from sklearn.metrics.pairwise import euclidean_distances
from .utils import create_folder, join_path, mask_color_map
from .segmenter import parse_args, segment_single_image, visualize_fibres
from .utils import overlay_colorbar, color_survey_with_colorbar


class Cabana:
    def __init__(self, param_file, input_image_path, out_folder, ignore_large=True, mask_dir=None):
        self.param_file = param_file
        self.input_image_path = input_image_path
        self.output_folder = out_folder
        self.ignore_large = ignore_large
        # Optional folder of external ROI masks (e.g. TMA core circles), one
        # ``<image stem>.png`` per image. Falls back to Configs: ROI Masks.
        self.ext_mask_dir = mask_dir
        self.roi_mask_path = None

        # Store image name and extension
        self.img_name = os.path.basename(input_image_path)
        self.name_wo_ext = os.path.splitext(self.img_name)[0]

        # Initialize parameters
        self.args = None  # args for Cabana program
        self.seg_args = parse_args()  # args for segmentation
        self.ims_res = 1.0  # µm/pixel

        # Create dataframe to store statistics
        self.stats = pd.DataFrame()

        # Store images
        self.original_img = None
        self.roi_img = None
        self.mask_img = None  # Binary mask
        self.hdm_img = None  # High density matrix
        self.fibre_img = None  # Visualized fibres
        self.skeleton_img = None
        self.contour_img = None
        self.width_img = None
        self.energy_img = None
        self.coherency_img = None
        self.orientation_img = None
        self.length_map = None
        self.curve_maps = {}  # Dictionary to store curvature maps
        self.gap_img = None  # All gaps
        self.intra_gap_img = None  # Intra-gaps

        # Store color-mapped visualizations
        self.color_mask = None
        self.color_skeleton = None
        self.color_energy = None
        self.color_coherency = None
        self.color_orientation = None
        self.color_length = None
        self.color_curves = {}
        self.orient_vector_field = None
        self.angular_hist = None
        self.orient_color_survey = None

        # Create sub-folders in output directory
        self.roi_dir = join_path(self.output_folder, 'ROIs', "")
        self.bin_dir = join_path(self.output_folder, 'Bins', "")
        self.mask_dir = join_path(self.output_folder, 'Masks', "")
        self.hdm_dir = join_path(self.output_folder, 'HDM', "")
        self.export_dir = join_path(self.output_folder, 'Exports', "")
        self.color_dir = join_path(self.output_folder, 'Colors', "")
        self.fibre_dir = join_path(self.output_folder, 'Fibres', "")
        self.eligible_dir = join_path(self.output_folder, 'Eligible')
        self.roimask_dir = join_path(self.output_folder, 'ROIMasks', "")

        # Create folders
        create_folder(self.eligible_dir)
        create_folder(self.roimask_dir)
        create_folder(self.fibre_dir)
        create_folder(self.mask_dir)
        create_folder(self.hdm_dir)
        create_folder(self.export_dir)
        create_folder(self.color_dir)
        create_folder(self.roi_dir)
        create_folder(self.bin_dir)

        # Create export sub-folders
        self.export_img_dir = join_path(self.export_dir, self.name_wo_ext)
        self.color_img_dir = join_path(self.color_dir, self.name_wo_ext)
        create_folder(self.export_img_dir)
        create_folder(self.color_img_dir)

        # Set segmentation arguments
        setattr(self.seg_args, 'roi_dir', self.roi_dir)
        setattr(self.seg_args, 'bin_dir', self.bin_dir)

    def initialize_params(self):
        """Initialize parameters from the parameter file"""
        with open(self.param_file) as pf:
            try:
                self.args = yaml.safe_load(pf)
            except yaml.YAMLError as exc:
                print(exc)

        # overwrite specific fields of seg_args with those in the parameter file
        setattr(self.seg_args, 'num_channels', int(self.args['Segmentation']["Number of Labels"]))
        setattr(self.seg_args, 'max_iter', int(self.args['Segmentation']["Max Iterations"]))
        setattr(self.seg_args, 'hue_value', float(self.args['Segmentation']["Normalized Hue Value"]))
        setattr(self.seg_args, 'rt', float(self.args['Segmentation']["Color Threshold"]))
        setattr(self.seg_args, 'max_size', int(self.args['Segmentation']["Max Size"]))
        setattr(self.seg_args, 'min_size', int(self.args['Segmentation']["Min Size"]))
        setattr(self.seg_args, 'patch_size', int(self.args['Segmentation'].get("Patch Size", 0) or 0))
        setattr(self.seg_args, 'white_background', self.args['Detection']["Dark Line"])
        if not self.ext_mask_dir:
            self.ext_mask_dir = self.args.get('Configs', {}).get('ROI Masks') or None

    def prepare_image(self):
        """Prepare the input image for analysis"""
        img = cv2.imread(self.input_image_path, cv2.IMREAD_UNCHANGED)

        # Convert 16-bit to 8-bit if needed
        if img.dtype == np.uint16:
            warnings.warn(f"Image {self.input_image_path} is 16-bit. Converting to 8-bit.")
            lower = np.percentile(img, 2)
            upper = np.percentile(img, 98)
            if upper <= lower:
                warnings.warn(f"Image {self.input_image_path} has no dynamic range. Skipping.")
                return
            img = np.clip(img, lower, upper)  # clip to 2nd and 98th percentile to remove outliers
            # Scale to full 8-bit range
            img = (((img - lower) / (upper - lower)) * 255.0).astype(np.uint8)

        # Convert grayscale to RGB if needed
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2) if len(img.shape) < 3 else img

        # Check if image is valid
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = img.shape[:2]
        bright_percent = np.sum(gray > 5) / width / height
        max_line_width = self.args['Detection']["Max Line Width"]
        max_size = self.args["Segmentation"]["Max Size"]

        # Check for black or small images
        if np.min([height, width]) < 2 * max_line_width or bright_percent <= 0.01:
            warnings.warn('Image is too dark or too small. No analysis will be performed.')
            return False

        # Check if image is too large
        crop_box = None
        if height * width > max_size ** 2 and bright_percent > 0.01:
            if self.ignore_large:
                warnings.warn('Image is too large. No analysis will be performed.')
                return False
            else:
                # Split large image into smaller blocks
                row_blk_sz = int(np.ceil(height / int(np.ceil(height / max_size))))
                col_blk_sz = int(np.ceil(width / int(np.ceil(width / max_size))))
                warnings.warn('Image is oversized. Splitting into smaller blocks.')
                # Use only the first block for simplicity
                crop_box = (0, 0, col_blk_sz, row_blk_sz)
                img = img[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]]

        # Store original image
        self.original_img = img
        cv2.imwrite(join_path(self.eligible_dir, self.name_wo_ext + ".png"), img)

        # Bring the external ROI mask (if any) alongside the eligible image
        from .stages import prepare_roi_mask
        self.roi_mask_path = prepare_roi_mask(self.ext_mask_dir, self.name_wo_ext, (height, width),
                                              self.roimask_dir, crop_box=crop_box)
        if self.roi_mask_path:
            print('Using external ROI mask {}'.format(self.roi_mask_path))
        return True

    def generate_roi(self):
        """Generate region of interest"""
        img_path = join_path(self.eligible_dir, self.name_wo_ext + ".png")

        from .segmenter import generate_rois, load_roi_mask
        setattr(self.seg_args, 'input', img_path)
        setattr(self.seg_args, 'roi_mask_path', self.roi_mask_path)
        if self.args["Configs"]["Segmentation"]:
            print('Segmenting image')
            segment_single_image(self.seg_args)
        else:
            print("No segmentation is applied prior to image analysis.")
            img = cv2.imread(self.seg_args.input)
            # The external ROI mask (if any) is the analysis region even when
            # segmentation is off; otherwise the whole image is analysed.
            mask = load_roi_mask(self.roi_mask_path, img.shape)
            if mask is None:
                mask = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8) * 255
                roi_img = img
            else:
                roi_img = generate_rois(img, mask, self.seg_args.white_background)
            cv2.imwrite(join_path(self.seg_args.roi_dir, self.name_wo_ext + '_roi.png'), roi_img)
            cv2.imwrite(join_path(self.seg_args.bin_dir, self.name_wo_ext + '_mask.png'), mask)

        # Load segmented images
        self.roi_img = cv2.imread(join_path(self.roi_dir, self.name_wo_ext + '_roi.png'))
        self.mask_img = cv2.imread(join_path(self.bin_dir, self.name_wo_ext + '_mask.png'), cv2.IMREAD_GRAYSCALE)

        print('ROIs have been saved in {}'.format(self.seg_args.roi_dir))
        print('Masks have been saved in {}'.format(self.seg_args.bin_dir))

    def detect_fibres(self):
        """Detect fibres in the ROI image"""
        from .stages import build_fibre_detector, detect_one_image, roi_mask_for
        d = self.args["Detection"]
        print(f"Detecting fibres with line widths in "
              f"[{d['Min Line Width']}, {d['Max Line Width']}] pixels")
        det = build_fibre_detector(self.args)
        img_path = join_path(self.roi_dir, self.name_wo_ext + '_roi.png')
        result = detect_one_image(det, img_path,
                                  mask_dir=self.mask_dir,
                                  export_subdir=self.export_img_dir,
                                  color_subdir=self.color_img_dir,
                                  roi_mask=roi_mask_for(self.roimask_dir, self.name_wo_ext),
                                  roi_erode_px=int(self.args['Detection']['Max Line Width']))
        self.contour_img = result['contour_img']
        self.width_img = result['width_img']

    def analyze_orientation(self):
        """Analyze orientation of fibres"""
        from .stages import analyze_one_orientation
        metrics, images = analyze_one_orientation(
            OrientationAnalyzer(2.0),
            roi_img_path=join_path(self.roi_dir, self.name_wo_ext + '_roi.png'),
            mask_roi_path=join_path(self.bin_dir, self.name_wo_ext + "_mask.png"),
            mask_hdm_path=join_path(self.hdm_dir, self.name_wo_ext + "_roi.png"),
            mask_width_path=join_path(self.export_img_dir, "Width.png"),
            export_subdir=self.export_img_dir,
            color_subdir=self.color_img_dir,
        )
        for k, v in metrics.items():
            self.stats.loc[0, k] = v
        if images:
            self.energy_img = images['energy']
            self.coherency_img = images['coherency']
            self.orientation_img = images['orientation']
            self.orient_color_survey = images['color_survey']
            self.orient_vector_field = images['vector_field']
            self.angular_hist = images['angular_hist']

    def quantify_skeleton(self):
        """Quantify skeleton metrics"""
        from .stages import (build_skeleton_analyzer, curve_windows_from_args,
                             quantify_one_skeleton)
        skel_analyzer = build_skeleton_analyzer(self.args)
        print(f"Min branch length = {skel_analyzer.branch_thresh} px, "
              f"Min hole area = {skel_analyzer.hole_thresh} px²")

        metrics, curve_maps, key_pts_image, length_map = quantify_one_skeleton(
            skel_analyzer,
            mask_path=join_path(self.mask_dir, self.name_wo_ext + '_roi.png'),
            export_subdir=self.export_img_dir,
            ims_res=self.ims_res,
            curve_windows=curve_windows_from_args(self.args),
        )
        for k, v in metrics.items():
            self.stats.loc[0, k] = v
        self.skeleton_img = key_pts_image
        self.length_map = length_map
        self.curve_maps = curve_maps

    def quantify_hdm(self):
        """Quantify high density matrix areas"""
        from .stages import run_hdm
        img_path = join_path(self.eligible_dir, self.name_wo_ext + ".png")
        df_hdm = run_hdm(self.args, img_path, self.hdm_dir, ext=".png", mask_dir=self.roimask_dir)
        self.stats.loc[0, 'Image'] = self.name_wo_ext + '_roi.png'
        self.stats.loc[0, '% HDM Area'] = df_hdm["% HDM Area"].iloc[0]

    def analyze_gaps(self):
        """Analyze gaps in the image"""
        min_gap_diameter = self.args["Gap Analysis"]["Minimum Gap Diameter"]
        if min_gap_diameter == 0:
            warnings.warn("minimum gap diameter = 0 pixels. Skipping gap analysis.")
            return

        print(f"Performing gap analysis with "
              f"minimum gap diameter = {min_gap_diameter * self.ims_res:.1f}µm "
              f"({min_gap_diameter} pixels).")

        # Create gap analysis directory
        gap_result_dir = join_path(self.mask_dir, 'GapAnalysis')
        Path(gap_result_dir).mkdir(parents=True, exist_ok=True)

        # Read images
        img_path = join_path(self.mask_dir, self.name_wo_ext + '_roi.png')
        min_gap_radius = min_gap_diameter / 2
        min_dist = int(np.max([1, min_gap_radius]))

        img = cv2.imread(img_path, 0)
        color_img = cv2.imread(join_path(self.eligible_dir, self.name_wo_ext + ".png"))
        mask = img.copy()

        # Space outside the external ROI mask (e.g. TMA core corners) is not a gap
        from .stages import roi_mask_for
        roi = roi_mask_for(self.roimask_dir, self.name_wo_ext)
        if roi is not None:
            mask[roi == 0] = 0

        # Set border pixels to zero to avoid partial circles
        mask[0, :] = mask[-1, :] = mask[:, :1] = mask[:, -1:] = 0

        final_circles = []
        downsample_factor = 2

        while True:
            dist_map = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

            # Downsample distance map and upscale detected centers to original image size
            dist_map_downscaled = cv2.resize(dist_map, None, fx=1 / downsample_factor, fy=1 / downsample_factor)
            centers_downscaled = peak_local_max(dist_map_downscaled, min_distance=min_dist, exclude_border=False)
            centers = centers_downscaled * downsample_factor

            radius = dist_map[centers[:, 0], centers[:, 1]]

            eligible_centers = centers[radius > min_gap_radius, :]
            eligible_radius = radius[radius > min_gap_radius]
            eligible_circles = np.hstack([eligible_centers, eligible_radius[:, None]])

            if len(eligible_circles) == 0:
                break

            result = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            while len(eligible_circles) > 0:
                if eligible_circles[1:, :].size > 0:
                    pw_euclidean_dist = \
                        euclidean_distances(eligible_circles[[0], :2], eligible_circles[1:, :2])[0]
                    pw_radius_sum = eligible_circles[0, 2] + eligible_circles[1:, 2]
                    neighbor_idx = np.nonzero(pw_euclidean_dist < pw_radius_sum)[0] + 1
                    eligible_circles = np.delete(eligible_circles, neighbor_idx, axis=0)

                circle = eligible_circles[0, :]
                result = cv2.circle(result, (int(circle[1]), int(circle[0])), int(circle[2]), (0, 0, 0), -1)
                final_circles.append(eligible_circles[0, :])
                eligible_circles = np.delete(eligible_circles, 0, axis=0)

            mask = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

        # Create visualizations
        final_result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        color_result = color_img.copy()

        for circle in final_circles:
            final_result = cv2.circle(final_result, (int(circle[1]), int(circle[0])),
                                      int(circle[2]), (0, 255, 255), 2)
            color_result = cv2.circle(color_result, (int(circle[1]), int(circle[0])),
                                      int(circle[2]), (0, 255, 255), 2)

        # Store gap image
        self.gap_img = color_result

        # Save images
        cv2.imwrite(join_path(gap_result_dir, self.name_wo_ext + "_GapImage.png"), final_result)
        cv2.imwrite(join_path(self.export_img_dir, "GapImage.png"), color_result)

        # Calculate gap metrics
        from .stages import summary_stats
        areas = np.pi * (np.array(final_circles)[:, 2] ** 2) * self.ims_res ** 2 if len(final_circles) else np.array([])
        radius_values = np.sqrt(areas / np.pi) if len(areas) else np.array([])

        for k, v in summary_stats(areas, 'All gaps area in µm²',
                                  include_count=True,
                                  count_label='Gap Circles Count (All)').items():
            self.stats.loc[0, k] = v
        for k, v in summary_stats(radius_values, 'All gaps radius in µm').items():
            self.stats.loc[0, k] = v

        # Save gap data to CSV
        if len(final_circles) > 0:
            final_circles = np.array(final_circles)
            data = {
                'Area (µm²)': areas,
                'X': final_circles[:, 1],
                'Y': final_circles[:, 0]
            }
            df = pd.DataFrame(data)
            df.to_csv(join_path(gap_result_dir, f"IndividualGaps_{self.name_wo_ext}.csv"), index=False)

    def analyze_intra_gaps(self):
        """Analyze gaps within the ROI"""
        if not self.args["Configs"]["Gap Analysis"] or self.args["Gap Analysis"]["Minimum Gap Diameter"] == 0:
            return

        # Create gap analysis directory
        gap_result_dir = join_path(self.mask_dir, 'GapAnalysis')
        Path(gap_result_dir).mkdir(parents=True, exist_ok=True)

        # Check if gap analysis was performed
        csv_file_path = join_path(gap_result_dir, f'IndividualGaps_{self.name_wo_ext}.csv')
        if not os.path.exists(csv_file_path):
            return

        print('Performing intra gap analysis')

        # Load gap data
        df_circles = pd.read_csv(csv_file_path)
        binary_mask = cv2.imread(join_path(self.bin_dir, self.name_wo_ext + '_mask.png'), 0)
        img_fibre = cv2.imread(join_path(self.mask_dir, self.name_wo_ext + '_roi.png'), 0)
        color_img_fibre = cv2.cvtColor(img_fibre, cv2.COLOR_GRAY2BGR)
        color_img = cv2.imread(join_path(self.eligible_dir, self.name_wo_ext + ".png"))

        areas = []
        circle_cnt = 0

        for index, row in df_circles.iterrows():
            area, x, y = row['Area (µm²)'], int(row['X']), int(row['Y'])
            radius = int(np.sqrt(area / np.pi) / self.ims_res)  # convert back to measurements in pixels
            if binary_mask[y, x] > 0:
                color_img_fibre = cv2.circle(color_img_fibre, (x, y), radius, (0, 255, 255), 1)
                color_img = cv2.circle(color_img, (x, y), radius, (0, 255, 255), 1)
                areas.append(area)
                circle_cnt += 1

        # Store intra gap image
        self.intra_gap_img = color_img

        # Save images
        cv2.imwrite(join_path(gap_result_dir, self.name_wo_ext + "_GapImage_intra_gaps.png"), color_img_fibre)
        cv2.imwrite(join_path(self.export_img_dir, "GapImage_intra_gaps.png"), color_img)

        from .stages import summary_stats
        areas = np.array(areas)
        radius = np.sqrt(areas / np.pi) if len(areas) else np.array([])

        for k, v in summary_stats(areas, 'ROI gaps area in µm²',
                                  include_count=True,
                                  count_label='Gap Circles Count (ROI)').items():
            self.stats.loc[0, k] = v
        for k, v in summary_stats(radius, 'ROI gaps radius in µm').items():
            self.stats.loc[0, k] = v

    def calc_fibre_areas(self):
        """Calculate fibre areas and related metrics"""
        from .stages import compute_fibre_areas
        metrics = compute_fibre_areas(
            img_mask_path=join_path(self.bin_dir, self.name_wo_ext + '_mask.png'),
            ori_img_path=join_path(self.eligible_dir, self.name_wo_ext + ".png"),
            width_mask_path=join_path(self.export_img_dir, "Width.png"),
            hdm_mask_path=join_path(self.hdm_dir, self.name_wo_ext + '_roi.png'),
            roi_mask_path=self.roi_mask_path,
        )
        for k, v in metrics.items():
            self.stats.loc[0, k] = v

    def visualize_fibres(self, thickness=3):
        """Generate visualizations of detected fibres"""
        print('Generating fibre visualization results')

        img_path = join_path(self.eligible_dir, self.name_wo_ext + ".png")
        mask_path = join_path(self.mask_dir, self.name_wo_ext + '_roi.png')

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path)

        # Generate visualization
        fibre_img_path = join_path(self.fibre_dir, self.name_wo_ext + '_fibres.png')
        visualize_fibres(img, mask, fibre_img_path, thickness)

        # Store fibre image
        self.fibre_img = cv2.imread(fibre_img_path)

    def combine_statistics(self):
        """Combine all statistics into a single dataframe"""
        print('Combining statistics')

        # Initialize image name if not set
        if 'Image' not in self.stats.columns:
            self.stats.loc[0, 'Image'] = self.name_wo_ext + '_roi.png'

        # Convert pixel areas to physical areas
        if 'Area (ROI)' in self.stats.columns:
            self.stats.loc[0, 'Fibre Area (ROI, µm²)'] = self.stats.loc[0, 'Area (ROI)'] * self.ims_res ** 2
            self.stats.loc[0, 'Fibre Area (WIDTH, µm²)'] = self.stats.loc[0, 'Area (WIDTH)'] * self.ims_res ** 2

        # Calculate HDM area
        if '% HDM Area' in self.stats.columns:
            self.stats.loc[0, 'Fibre Area (HDM, µm²)'] = self.stats.loc[0, '% HDM Area'] * self.stats.loc[
                0, 'Total Image Area (µm²)']

        # Calculate average thickness
        if 'Total Length (µm)' in self.stats.columns and self.stats.loc[0, 'Total Length (µm)'] > 0:
            total_length = self.stats.loc[0, 'Total Length (µm)']

            # HDM thickness
            if 'Fibre Area (HDM, µm²)' in self.stats.columns:
                self.stats.loc[0, 'Avg Thickness (HDM, µm)'] = self.stats.loc[0, 'Fibre Area (HDM, µm²)'] / total_length

            # ROI thickness
            if 'Fibre Area (ROI, µm²)' in self.stats.columns:
                self.stats.loc[0, 'Avg Thickness (ROI, µm)'] = self.stats.loc[0, 'Fibre Area (ROI, µm²)'] / total_length

            # WIDTH thickness
            if 'Fibre Area (WIDTH, µm²)' in self.stats.columns:
                self.stats.loc[0, 'Avg Thickness (WIDTH, µm)'] = self.stats.loc[
                                                                     0, 'Fibre Area (WIDTH, µm²)'] / total_length
        else:
            # Set thicknesses to 0 if no length
            thickness_metrics = ['Avg Thickness (HDM, µm)', 'Avg Thickness (ROI, µm)', 'Avg Thickness (WIDTH, µm)']
            for metric in thickness_metrics:
                self.stats.loc[0, metric] = 0

        # Add image resolution
        self.stats.loc[0, 'Image Res. (µm/pixel)'] = self.ims_res

    def normalize_statistics(self):
        """Normalize various statistics"""
        from .stages import safe_div
        print('Normalizing statistics')

        safe_div(self.stats, 'Fibre Area (WIDTH, µm²)', 'Fibre Area (ROI, µm²)',
                 'Fibre Coverage (WIDTH/ROI)')
        safe_div(self.stats, 'Branchpoints', 'Total Length (µm)',
                 'Branchpoints Density (µm⁻¹)')
        safe_div(self.stats, 'Endpoints', 'Total Length (µm)',
                 'Endpoints Density (µm⁻¹)')

        if (self.args["Configs"]["Gap Analysis"]
                and self.args["Gap Analysis"]["Minimum Gap Diameter"] > 0):
            # (numerator, denominator, output, denom_transform)
            ratios = [
                ('Mean (All gaps area in µm²)',  'Total Image Area (µm²)', 'Normalized Mean (All gaps area)', None),
                ('Std (All gaps area in µm²)',   'Total Image Area (µm²)', 'Normalized Std (All gaps area)',  None),
                ('Mean (ROI gaps area in µm²)',  'Fibre Area (ROI, µm²)',  'Normalized Mean (ROI gaps area)', None),
                ('Std (ROI gaps area in µm²)',   'Fibre Area (ROI, µm²)',  'Normalized Std (ROI gaps area)',  None),
                ('Mean (All gaps radius in µm)', 'Total Image Area (µm²)', 'Normalized Mean (All gaps radius)', np.sqrt),
                ('Std (All gaps radius in µm)',  'Total Image Area (µm²)', 'Normalized Std (All gaps radius)',  np.sqrt),
                ('Mean (ROI gaps radius in µm)', 'Fibre Area (ROI, µm²)',  'Normalized Mean (ROI gaps radius)', np.sqrt),
                ('Std (ROI gaps radius in µm)',  'Fibre Area (ROI, µm²)',  'Normalized Std (ROI gaps radius)',  np.sqrt),
                ('Gap Circles Count (All)',      'Total Image Area (µm²)', 'Gap density (All, µm⁻²)',          None),
                ('Gap Circles Count (ROI)',      'Fibre Area (ROI, µm²)',  'Gap density (ROI, µm⁻²)',          None),
            ]
            for num, den, out, t in ratios:
                safe_div(self.stats, num, den, out, denom_transform=t)

        # Normalize lacunarity
        if 'Lacunarity' in self.stats.columns and 'Total Length (µm)' in self.stats.columns:
            if self.stats.loc[0, 'Total Length (µm)'] > 0:
                ratio = self.stats.loc[0, 'Total Length (µm)'] * self.stats.loc[0, 'Image Res. (µm/pixel)'] / \
                        self.stats.loc[0, 'Total Image Area (µm²)']
                if ratio > 0 and ratio < 1:
                    self.stats.loc[0, 'Normalized Lacunarity'] = (self.stats.loc[0, 'Lacunarity'] - 1) / (
                                (1.0 - ratio) / ratio)
                else:
                    self.stats.loc[0, 'Normalized Lacunarity'] = 0
            else:
                self.stats.loc[0, 'Normalized Lacunarity'] = 0

        # If no segmentation is performed, remove ROI columns
        if not self.args["Configs"]["Segmentation"]:
            roi_columns = [col for col in self.stats.columns if 'ROI' in col]
            self.stats = self.stats.drop(columns=roi_columns)

    def generate_color_maps(self):
        """Generate color maps for visualization"""
        print('Generating color maps')

        # Load original image
        rgb_img = iio.imread(join_path(self.eligible_dir, self.name_wo_ext + ".png"))
        mask_img = 255 - iio.imread(join_path(self.export_img_dir, self.name_wo_ext + "_Mask.png"))

        if np.sum(mask_img) == 0:
            # Create empty placeholder images if no mask
            empty_img = np.zeros_like(rgb_img)
            color_maps = ["color_mask", "color_skeleton", "color_energy", "color_coherency",
                          "color_orientation", "color_length", "orient_color_survey"]

            for map_name in color_maps:
                iio.imwrite(join_path(self.color_img_dir, f"{self.name_wo_ext}_{map_name}.png"), empty_img)

            # Handle curve maps
            if hasattr(self, 'curve_maps') and self.curve_maps:
                for win_sz in self.curve_maps.keys():
                    iio.imwrite(join_path(self.color_img_dir, f"{self.name_wo_ext}_color_curve_{win_sz}.png"),
                                empty_img)

            return

        # Create mask color map
        mask_img = np.repeat(mask_img[:, :, np.newaxis], 3, axis=2)
        mask_glow = mask_color_map(rgb_img, mask_img)
        self.color_mask = mask_glow
        Image.fromarray(mask_glow).save(join_path(self.color_img_dir, f"{self.name_wo_ext}_color_mask.png"))

        # Create skeleton color map
        if os.path.exists(join_path(self.export_img_dir, f"{self.name_wo_ext}_Skeleton.png")):
            skeleton = iio.imread(join_path(self.export_img_dir, f"{self.name_wo_ext}_Skeleton.png"))
            red_pos = np.where((skeleton[..., 0] == 255) & (skeleton[..., 1] == 0) & (skeleton[..., 2] == 0))
            skeleton[red_pos[0], red_pos[1], :] = [0, 255, 0]
            index_pos = np.where(cv2.cvtColor(skeleton, cv2.COLOR_BGR2GRAY) == 0)
            skeleton[index_pos[0], index_pos[1], :] = rgb_img[index_pos[0], index_pos[1], :]
            self.color_skeleton = skeleton
            Image.fromarray(skeleton).save(join_path(self.color_img_dir, f"{self.name_wo_ext}_color_skeleton.png"))

        # Create energy, coherency, orientation, and length color maps
        if os.path.exists(join_path(self.export_img_dir, f"{self.name_wo_ext}_Energy.tif")):
            energy_map = iio.imread(join_path(self.export_img_dir, f"{self.name_wo_ext}_Energy.tif"))
            overlay_colorbar(rgb_img, energy_map,
                             join_path(self.color_img_dir, f"{self.name_wo_ext}_color_energy.png"),
                             clabel="Normalized Energy", mode="overlay")

        if os.path.exists(join_path(self.export_img_dir, f"{self.name_wo_ext}_Orientation.tif")):
            orient_map = iio.imread(join_path(self.export_img_dir, f"{self.name_wo_ext}_Orientation.tif"))
            color_survey_with_colorbar(orient_map, np.ones_like(orient_map), np.ones_like(orient_map),
                                       join_path(self.color_img_dir, f"{self.name_wo_ext}_color_orientation.png"),
                                       clabel="Orientation (rad)")

        if os.path.exists(join_path(self.export_img_dir, f"{self.name_wo_ext}_Coherency.tif")):
            cohere_map = iio.imread(join_path(self.export_img_dir, f"{self.name_wo_ext}_Coherency.tif"))
            overlay_colorbar(rgb_img, cohere_map,
                             join_path(self.color_img_dir, f"{self.name_wo_ext}_color_coherency.png"),
                             clabel="Coherency", mode="overlay")

        if os.path.exists(join_path(self.export_img_dir, f"{self.name_wo_ext}_Length_Map.tif")):
            length_map = iio.imread(join_path(self.export_img_dir, f"{self.name_wo_ext}_Length_Map.tif"))
            overlay_colorbar(rgb_img, length_map,
                             join_path(self.color_img_dir, f"{self.name_wo_ext}_color_length.png"),
                             clabel="Length (µm)", cmap='plasma', dpi=200, font_size=10)

        # Create curvature color maps
        curve_paths = glob(join_path(self.export_img_dir, f"{self.name_wo_ext}_Curve_Map_*"))
        for curve_path in curve_paths:
            curve_name_wo_ext = os.path.basename(curve_path)[:-4]
            suffix = curve_name_wo_ext[len(f"{self.name_wo_ext}_Curve_Map"):]
            curve_map = tiff.imread(curve_path) / 180
            save_path = join_path(self.color_img_dir, f"{self.name_wo_ext}_color_curve{suffix}.png")
            overlay_colorbar(rgb_img, curve_map, save_path,
                             clabel="Curliness", cmap='plasma', dpi=200, font_size=10)

        # Create orientation color survey
        if os.path.exists(join_path(self.export_img_dir, f"{self.name_wo_ext}_Orientation.tif")) and \
                os.path.exists(join_path(self.export_img_dir, f"{self.name_wo_ext}_Coherency.tif")):
            orient_map = iio.imread(join_path(self.export_img_dir, f"{self.name_wo_ext}_Orientation.tif"))
            cohere_map = iio.imread(join_path(self.export_img_dir, f"{self.name_wo_ext}_Coherency.tif"))
            energy_map = iio.imread(join_path(self.export_img_dir, f"{self.name_wo_ext}_Energy.tif"))
            color_survey_with_colorbar(orient_map, cohere_map, energy_map,
                                       join_path(self.color_img_dir, f"{self.name_wo_ext}_orient_color_survey.png"))

        # Copy gap images if they exist
        if os.path.exists(join_path(self.export_img_dir, f"{self.name_wo_ext}_GapImage.png")):
            shutil.copy(join_path(self.export_img_dir, f"{self.name_wo_ext}_GapImage.png"),
                        join_path(self.color_img_dir, f"{self.name_wo_ext}_all_gaps.png"))

        if os.path.exists(join_path(self.export_img_dir, f"{self.name_wo_ext}_GapImage_intra_gaps.png")):
            shutil.copy(join_path(self.export_img_dir, f"{self.name_wo_ext}_GapImage_intra_gaps.png"),
                        join_path(self.color_img_dir, f"{self.name_wo_ext}_intra_gaps.png"))
        print(f"Color maps have been saved to {self.color_img_dir}")

    def export_results(self):
        """Export all results to files"""
        # Export stats to CSV
        stats_csv = join_path(self.output_folder, f"{self.name_wo_ext}_QuantificationResults.csv")
        self.stats.to_csv(stats_csv, index=False, float_format='%.3f')
        print(f'Statistics have been saved to {stats_csv}')

        # Export stats as formatted text
        stats_txt = join_path(self.output_folder, f"{self.name_wo_ext}_QuantificationResults.txt")
        with open(stats_txt, 'w') as f:
            f.write(f"Quantification Results for {self.name_wo_ext}\n")
            f.write("=" * 50 + "\n\n")

            # Group metrics by category
            hdm_metrics = [col for col in self.stats.columns if 'HDM' in col]
            roi_metrics = [col for col in self.stats.columns if 'ROI' in col]
            width_metrics = [col for col in self.stats.columns if 'WIDTH' in col]
            skeleton_metrics = ['Area of Fibre Spines (µm²)', 'Lacunarity', 'Normalized Lacunarity',
                                'Total Length (µm)', 'Endpoints', 'Endpoints Density (µm⁻¹)',
                                'Avg Length (µm)', 'Branchpoints', 'Branchpoints Density (µm⁻¹)',
                                'Box-Counting Fractal Dimension']
            gap_metrics = [col for col in self.stats.columns if 'gap' in col.lower()]

            # Write metrics by category
            categories = [
                ('General', ['Image', 'Total Image Area (µm²)', 'Image Res. (µm/pixel)']),
                ('HDM Metrics', hdm_metrics),
                ('ROI Metrics', roi_metrics),
                ('WIDTH Metrics', width_metrics),
                ('Skeleton Metrics', skeleton_metrics),
                ('Gap Metrics', gap_metrics)
            ]

            for category, metrics in categories:
                relevant_metrics = [m for m in metrics if m in self.stats.columns]
                if relevant_metrics:
                    f.write(f"\n{category}:\n")
                    f.write("-" * 30 + "\n")
                    for metric in relevant_metrics:
                        value = self.stats.loc[0, metric]
                        if isinstance(value, (int, float)):
                            f.write(f"{metric}: {value:.3f}\n")
                        else:
                            f.write(f"{metric}: {value}\n")

        # Optional: Create a visualization summary image
        self.create_summary_visualization()

    def create_summary_visualization(self):
        """Create a summary visualization with multiple panels"""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.gridspec import GridSpec

            # Create figure
            fig = plt.figure(figsize=(16, 12))
            gs = GridSpec(3, 3, figure=fig)

            # Original image
            ax1 = fig.add_subplot(gs[0, 0])
            if self.original_img is not None:
                ax1.imshow(cv2.cvtColor(self.original_img, cv2.COLOR_BGR2RGB))
            ax1.set_title('Original Image')
            ax1.axis('off')

            # ROI image
            ax2 = fig.add_subplot(gs[0, 1])
            if self.roi_img is not None:
                ax2.imshow(cv2.cvtColor(self.roi_img, cv2.COLOR_BGR2RGB))
            ax2.set_title('ROI')
            ax2.axis('off')

            # Fibre detection
            ax3 = fig.add_subplot(gs[0, 2])
            if hasattr(self, 'fibre_img') and self.fibre_img is not None:
                ax3.imshow(cv2.cvtColor(self.fibre_img, cv2.COLOR_BGR2RGB))
            ax3.set_title('Detected Fibres')
            ax3.axis('off')

            # Skeleton
            ax4 = fig.add_subplot(gs[1, 0])
            if hasattr(self, 'skeleton_img') and self.skeleton_img is not None:
                ax4.imshow(cv2.cvtColor(self.skeleton_img, cv2.COLOR_BGR2RGB))
            ax4.set_title('Skeleton')
            ax4.axis('off')

            # Orientation
            ax5 = fig.add_subplot(gs[1, 1])
            orient_survey_path = join_path(self.color_img_dir, f"{self.name_wo_ext}_orient_color_survey.png")
            if os.path.exists(orient_survey_path):
                orient_survey = cv2.imread(orient_survey_path)
                ax5.imshow(cv2.cvtColor(orient_survey, cv2.COLOR_BGR2RGB))
            ax5.set_title('Orientation')
            ax5.axis('off')

            # Gap analysis
            ax6 = fig.add_subplot(gs[1, 2])
            if hasattr(self, 'gap_img') and self.gap_img is not None:
                ax6.imshow(cv2.cvtColor(self.gap_img, cv2.COLOR_BGR2RGB))
            ax6.set_title('Gap Analysis')
            ax6.axis('off')

            # Key metrics text
            ax7 = fig.add_subplot(gs[2, :])
            key_metrics = []

            if 'Total Length (µm)' in self.stats.columns:
                key_metrics.append(f"Total Length: {self.stats.loc[0, 'Total Length (µm)']:.1f} µm")

            if 'Avg Thickness (WIDTH, µm)' in self.stats.columns:
                key_metrics.append(f"Avg Thickness: {self.stats.loc[0, 'Avg Thickness (WIDTH, µm)']:.2f} µm")

            if 'Lacunarity' in self.stats.columns:
                key_metrics.append(f"Lacunarity: {self.stats.loc[0, 'Lacunarity']:.3f}")

            if 'Branchpoints' in self.stats.columns:
                key_metrics.append(f"Branchpoints: {int(self.stats.loc[0, 'Branchpoints'])}")

            if 'Endpoints' in self.stats.columns:
                key_metrics.append(f"Endpoints: {int(self.stats.loc[0, 'Endpoints'])}")

            if 'Orient. Alignment' in self.stats.columns:
                key_metrics.append(f"Orient. Alignment: {self.stats.loc[0, 'Orient. Alignment']:.3f}")

            if 'Gap Circles Count (All)' in self.stats.columns:
                key_metrics.append(f"Gaps: {int(self.stats.loc[0, 'Gap Circles Count (All)'])}")

            ax7.text(0.1, 0.5, '\n'.join(key_metrics), fontsize=12,
                     va='center', ha='left', transform=ax7.transAxes)
            ax7.set_title('Key Metrics')
            ax7.axis('off')

            # Adjust layout and save
            plt.tight_layout()
            plt.savefig(join_path(self.output_folder, f"{self.name_wo_ext}_summary.png"), dpi=150)
            plt.close()

        except Exception as e:
            warnings.warn(f"Could not create summary visualization: {str(e)}")

    def run(self):
        """Run the full analysis pipeline"""
        self.initialize_params()

        # Prepare image
        if not self.prepare_image():
            Log.logger.warning("Image preparation failed. Skipping analysis.")
            return False

        # Generate ROI
        self.generate_roi()

        if self.args["Configs"]["Quantification"]:
            # Perform quantification
            self.quantify_hdm()
            self.detect_fibres()
            self.quantify_skeleton()
            self.analyze_orientation()

            # Visualize results
            self.visualize_fibres()

            # Calculate areas
            self.calc_fibre_areas()

            # Gap analysis
            if self.args["Configs"]["Gap Analysis"]:
                self.analyze_gaps()
                self.analyze_intra_gaps()

            # Combine and normalize statistics
            self.combine_statistics()
            self.normalize_statistics()

            return True
        else:
            print('Segmentation is done. No further analysis will be conducted.')
            return False