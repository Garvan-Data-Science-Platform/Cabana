import os
import cv2
import yaml
import tempfile
import numpy as np
import pandas as pd
from .log import Log, LOGS_DIR
from .hdm import HDM
import imageio.v3 as iio
from pathlib import Path
from .detector import FibreDetector
from .analyzer import SkeletonAnalyzer
from .orientation import OrientationAnalyzer
from skimage.color import rgb2hed, hed2rgb, rgb2gray
from .utils import create_folder, is_oversized, join_path, max_image_size
from .io import save_visualisations
from .segmenter import parse_args, segment_single_image, visualize_fibres


class Cabana:
    def __init__(self, param_file, input_image_path, out_folder, ignore_large=True, mask_dir=None):
        self.param_file = param_file
        self.input_image_path = input_image_path
        self.output_folder = out_folder
        self.ignore_large = ignore_large
        # Optional folder of external ROI masks (e.g. TMA core circles), one
        # ``<image stem>.png`` per image.
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
        # Working copy of the input (8-bit, cropped) lives in a temporary
        # folder made by prepare_image, not in the output: run() removes it at
        # the end, otherwise it goes when this object is discarded
        self.eligible_dir = None
        self._work_dir = None
        self.roimask_dir = join_path(self.output_folder, 'ROIMasks', "")

        # Create folders
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
                raise ValueError(f"Could not parse the parameter file {self.param_file}: {exc}") from exc

        # overwrite specific fields of seg_args with those in the parameter file
        setattr(self.seg_args, 'num_channels', int(self.args['Segmentation']["Number of Labels"]))
        setattr(self.seg_args, 'max_iter', int(self.args['Segmentation']["Max Iterations"]))
        setattr(self.seg_args, 'hue_value', float(self.args['Segmentation']["Normalized Hue Value"]))
        setattr(self.seg_args, 'rt', float(self.args['Segmentation']["Color Threshold"]))
        setattr(self.seg_args, 'min_size', int(self.args['Segmentation']["Min Size"]))
        setattr(self.seg_args, 'patch_size', int(self.args['Segmentation'].get("Patch Size", 0) or 0))
        setattr(self.seg_args, 'white_background', self.args['Detection']["Dark Line"])

    def prepare_image(self):
        """Prepare the input image for analysis"""
        # pixel size from the image metadata, as the batch pipeline does; 1.0 when absent
        from .io import split2batches
        _, res = split2batches([self.input_image_path])      # one batch, one resolution
        self.ims_res = float(res[0]) if len(res) else 1.0
        img = cv2.imread(self.input_image_path, cv2.IMREAD_UNCHANGED)
        if img is not None and img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # Convert 16-bit to 8-bit if needed
        if img.dtype == np.uint16:
            Log.logger.warning(f"Image {self.input_image_path} is 16-bit. Converting to 8-bit.")
            lower = np.percentile(img, 2)
            upper = np.percentile(img, 98)
            if upper <= lower:
                Log.logger.warning(f"Image {self.input_image_path} has no dynamic range. Skipping.")
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
        max_size = max_image_size(self.args)

        # Check for black or small images
        if np.min([height, width]) < 2 * max_line_width or bright_percent <= 0.01:
            Log.logger.warning('Image is too dark or too small. No analysis will be performed.')
            return False

        # Check if image is too large
        crop_box = None
        if is_oversized(height, width, max_size):
            if self.ignore_large:
                Log.logger.warning('Image is too large. No analysis will be performed.')
                return False
            else:
                # The single-image pipeline analyses the top-left block only; BatchProcessor
                # writes and analyses every block of an oversized image.
                row_blk_sz = int(np.ceil(height / int(np.ceil(height / max_size))))
                col_blk_sz = int(np.ceil(width / int(np.ceil(width / max_size))))
                Log.logger.warning(f'Image is oversized ({width}x{height} > Max Size {max_size}); only its '
                                   f'top-left {col_blk_sz}x{row_blk_sz} block is analysed. Use BatchProcessor '
                                   f'to analyse every block.')
                crop_box = (0, 0, col_blk_sz, row_blk_sz)
                img = img[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]]
                bright_percent = np.sum(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) > 5) / img.shape[0] / img.shape[1]
                if bright_percent <= 0.01:
                    Log.logger.warning('The analysed block is too dark. No analysis will be performed.')
                    return False

        # Store original image
        self.original_img = img
        if self._work_dir is None:
            self._work_dir = tempfile.TemporaryDirectory(prefix='cabana_', ignore_cleanup_errors=True)
            self.eligible_dir = self._work_dir.name
        cv2.imwrite(join_path(self.eligible_dir, self.name_wo_ext + ".png"), img)

        # Bring the external ROI mask (if any) alongside the eligible image
        from .stages import prepare_roi_mask
        self.roi_mask_path = prepare_roi_mask(self.ext_mask_dir, self.name_wo_ext, (height, width),
                                              self.roimask_dir, crop_box=crop_box)
        if self.roi_mask_path:
            Log.logger.info('Using external ROI mask {}'.format(self.roi_mask_path))
        return True

    def generate_roi(self):
        """Generate region of interest"""
        img_path = join_path(self.eligible_dir, self.name_wo_ext + ".png")

        from .segmenter import generate_rois, load_roi_mask
        setattr(self.seg_args, 'input', img_path)
        setattr(self.seg_args, 'roi_mask_path', self.roi_mask_path)
        if self.args["Configs"]["Segmentation"]:
            Log.logger.info('Segmenting image')
            segment_single_image(self.seg_args)
        else:
            Log.logger.info("No segmentation is applied prior to image analysis.")
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

        Log.logger.debug('ROIs have been saved in {}'.format(self.seg_args.roi_dir))
        Log.logger.debug('Masks have been saved in {}'.format(self.seg_args.bin_dir))

    def detect_fibres(self):
        """Detect fibres in the ROI image"""
        from .stages import build_fibre_detector, detect_one_image, roi_mask_for
        d = self.args["Detection"]
        Log.logger.info(f"Detecting fibres with line widths in "
                        f"[{d['Min Line Width']}, {d['Max Line Width']}] pixels")
        det = build_fibre_detector(self.args)
        img_path = join_path(self.roi_dir, self.name_wo_ext + '_roi.png')
        result = detect_one_image(det, img_path,
                                  mask_dir=self.mask_dir,
                                  export_subdir=self.export_img_dir,
                                  color_subdir=self._color_subdir(),
                                  roi_mask=roi_mask_for(self.roimask_dir, self.name_wo_ext),
                                  roi_erode_px=int(self.args['Detection']['Max Line Width']))
        self.contour_img = result['contour_img']
        self.width_img = result['width_img']

    def _visuals(self):
        """Whether visualisations are saved (Configs: Save Visualisations)."""
        return save_visualisations(self.args)

    def _color_subdir(self):
        """Colour-image folder, or None when visualisations are off."""
        return self.color_img_dir if self._visuals() else None

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
            color_subdir=self._color_subdir(),
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
        Log.logger.debug(f"Min branch length = {skel_analyzer.branch_thresh} px, "
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
            Log.logger.warning("minimum gap diameter = 0 pixels. Skipping gap analysis.")
            return

        Log.logger.info(f"Performing gap analysis with "
                        f"minimum gap diameter = {min_gap_diameter * self.ims_res:.1f}µm "
                        f"({min_gap_diameter} pixels).")

        # Create gap analysis directory
        gap_result_dir = join_path(self.mask_dir, 'GapAnalysis')
        Path(gap_result_dir).mkdir(parents=True, exist_ok=True)

        # Read images
        img_path = join_path(self.mask_dir, self.name_wo_ext + '_roi.png')
        img = cv2.imread(img_path, 0)
        color_img = cv2.imread(join_path(self.eligible_dir, self.name_wo_ext + ".png"))

        # Space outside the external ROI mask (e.g. TMA core corners) is not a gap
        from .stages import fit_gap_circles, roi_mask_for
        final_circles = fit_gap_circles(img, min_gap_diameter, roi_mask_for(self.roimask_dir, self.name_wo_ext))

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
        if self._visuals():
            cv2.imwrite(join_path(gap_result_dir, self.name_wo_ext + "_GapImage.png"), final_result)
            cv2.imwrite(join_path(self.color_img_dir, "all_gaps.png"), color_result)

        # Calculate gap metrics
        from .stages import summary_stats
        areas = np.pi * (final_circles[:, 2] ** 2) * self.ims_res ** 2
        radius_values = np.sqrt(areas / np.pi) if len(areas) else np.array([])

        for k, v in summary_stats(areas, 'All gaps area in µm²',
                                  include_count=True,
                                  count_label='Gap Circles Count (All)').items():
            self.stats.loc[0, k] = v
        for k, v in summary_stats(radius_values, 'All gaps radius in µm').items():
            self.stats.loc[0, k] = v

        # Save gap data to CSV, even without gaps (intra-gap analysis reads it)
        pd.DataFrame({'Area (µm²)': areas, 'X': final_circles[:, 1], 'Y': final_circles[:, 0]}).to_csv(
            join_path(gap_result_dir, f"IndividualGaps_{self.name_wo_ext}.csv"), index=False)

    def analyze_intra_gaps(self):
        """Analyze gaps within the ROI"""
        if not self.args["Configs"]["Gap Analysis"] or self.args["Gap Analysis"]["Minimum Gap Diameter"] == 0:
            return

        # Create gap analysis directory
        gap_result_dir = join_path(self.mask_dir, 'GapAnalysis')
        Path(gap_result_dir).mkdir(parents=True, exist_ok=True)

        from .stages import summary_stats

        # Check if gap analysis was performed
        csv_file_path = join_path(gap_result_dir, f'IndividualGaps_{self.name_wo_ext}.csv')
        if not os.path.exists(csv_file_path):
            return

        Log.logger.info('Performing intra gap analysis')

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
        if self._visuals():
            cv2.imwrite(join_path(gap_result_dir, self.name_wo_ext + "_GapImage_intra_gaps.png"), color_img_fibre)
            cv2.imwrite(join_path(self.color_img_dir, "intra_gaps.png"), color_img)

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
        Log.logger.info('Generating fibre visualization results')

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
        Log.logger.info('Combining statistics')

        # Initialize image name if not set
        if 'Image' not in self.stats.columns:
            self.stats.loc[0, 'Image'] = self.name_wo_ext + '_roi.png'

        # Convert pixel areas to physical areas
        if 'Area (ROI)' in self.stats.columns:
            self.stats.loc[0, 'Fibre Area (ROI, µm²)'] = self.stats.loc[0, 'Area (ROI)'] * self.ims_res ** 2
            self.stats.loc[0, 'Fibre Area (WIDTH, µm²)'] = self.stats.loc[0, 'Area (WIDTH)'] * self.ims_res ** 2

        # Calculate HDM area (relative to the analysed ROI area when a mask was used)
        if '% HDM Area' in self.stats.columns:
            from .stages import hdm_reference_area
            self.stats.loc[0, 'Fibre Area (HDM, µm²)'] = (
                self.stats.loc[0, '% HDM Area'] * float(hdm_reference_area(self.stats).iloc[0]))

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
        Log.logger.info('Normalizing statistics')

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

    def export_results(self):
        """Export all results to files"""
        # Export stats to CSV
        stats_csv = join_path(self.output_folder, f"{self.name_wo_ext}_QuantificationResults.csv")
        self.stats.to_csv(stats_csv, index=False, float_format='%.3f')
        Log.logger.info(f'Statistics have been saved to {stats_csv}')

        # Export stats as formatted text
        stats_txt = join_path(self.output_folder, f"{self.name_wo_ext}_QuantificationResults.txt")
        with open(stats_txt, 'w', encoding='utf-8') as f:     # metric names use µm⁻¹
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
            orient_survey_path = join_path(self.color_img_dir, "orient_color_survey.png")
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
            Log.logger.warning(f"Could not create summary visualization: {str(e)}")

    def run(self):
        """Run the full analysis pipeline, logging to ``<output>/Logs/``."""
        with Log.run_log(join_path(self.output_folder, LOGS_DIR), self.param_file):
            try:
                return self._run()
            finally:
                if self._work_dir is not None:
                    self._work_dir.cleanup()
                    self._work_dir = self.eligible_dir = None

    def _run(self):
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

            # Colour maps, as in batch runs
            if self._visuals():
                from .stages import write_color_maps
                write_color_maps(cv2.cvtColor(self.original_img, cv2.COLOR_BGR2RGB),
                                 self.export_img_dir, self.color_img_dir)

            return True
        else:
            Log.logger.info('Segmentation is done. No further analysis will be conducted.')
            return False