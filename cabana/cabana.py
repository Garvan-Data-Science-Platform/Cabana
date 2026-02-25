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
    def __init__(self, param_file, input_image_path, out_folder, ignore_large=True):
        self.param_file = param_file
        self.input_image_path = input_image_path
        self.output_folder = out_folder
        self.ignore_large = ignore_large

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

        # Create folders
        create_folder(self.eligible_dir)
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
        setattr(self.seg_args, 'white_background', self.args['Detection']["Dark Line"])

    def prepare_image(self):
        """Prepare the input image for analysis"""
        img = cv2.imread(self.input_image_path, cv2.IMREAD_UNCHANGED)

        # Convert 16-bit to 8-bit if needed
        if img.dtype == np.uint16:
            warnings.warn(f"Image {self.input_image_path} is 16-bit. Converting to 8-bit.")
            lower = np.percentile(img, 2)
            upper = np.percentile(img, 98)
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
                box = (0, 0, col_blk_sz, row_blk_sz)
                img = img[box[1]:box[3], box[0]:box[2]]

        # Store original image
        self.original_img = img
        cv2.imwrite(join_path(self.eligible_dir, self.name_wo_ext + ".png"), img)
        return True

    def generate_roi(self):
        """Generate region of interest"""
        img_path = join_path(self.eligible_dir, self.name_wo_ext + ".png")

        if self.args["Configs"]["Segmentation"]:
            print('Segmenting image')
            setattr(self.seg_args, 'input', img_path)
            segment_single_image(self.seg_args)
        else:
            print("No segmentation is applied prior to image analysis.")
            setattr(self.seg_args, 'input', img_path)
            img = cv2.imread(self.seg_args.input)
            mask = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8) * 255
            cv2.imwrite(join_path(self.seg_args.roi_dir, self.name_wo_ext + '_roi.png'), img)
            cv2.imwrite(join_path(self.seg_args.bin_dir, self.name_wo_ext + '_mask.png'), mask)

        # Load segmented images
        self.roi_img = cv2.imread(join_path(self.roi_dir, self.name_wo_ext + '_roi.png'))
        self.mask_img = cv2.imread(join_path(self.bin_dir, self.name_wo_ext + '_mask.png'), cv2.IMREAD_GRAYSCALE)

        print('ROIs have been saved in {}'.format(self.seg_args.roi_dir))
        print('Masks have been saved in {}'.format(self.seg_args.bin_dir))

    def detect_fibres(self):
        """Detect fibres in the ROI image"""
        dark_line = self.args["Detection"]["Dark Line"]
        extend_line = self.args["Detection"]["Extend Line"]
        min_line_width = self.args["Detection"]["Min Line Width"]
        max_line_width = self.args["Detection"]["Max Line Width"]
        line_width_step = self.args["Detection"]["Line Width Step"]
        line_widths = np.arange(min_line_width, max_line_width + line_width_step, line_width_step)
        low_contrast = self.args["Detection"]["Low Contrast"]
        high_contrast = self.args["Detection"]["High Contrast"]
        min_len = self.args["Detection"]["Minimum Line Length"]
        max_len = self.args["Detection"]["Maximum Line Length"]

        print(f"Detecting fibres with line widths in "
                        f"[{min_line_width}, {max_line_width}] pixels")

        det = FibreDetector(line_widths=line_widths,
                            low_contrast=low_contrast,
                            high_contrast=high_contrast,
                            dark_line=dark_line,
                            extend_line=extend_line,
                            correct_pos=False,
                            min_len=min_len,
                            max_len=max_len)

        img_path = join_path(self.roi_dir, self.name_wo_ext + '_roi.png')
        det.detect_lines(img_path)
        contour_img, width_img, binary_contours, binary_widths, int_width_img = det.get_results()

        # Store images
        self.contour_img = contour_img
        self.width_img = width_img

        # Save images
        iio.imwrite(join_path(self.mask_dir, self.name_wo_ext + '_roi.png'), binary_contours)
        iio.imwrite(join_path(self.export_img_dir, self.name_wo_ext + "_Mask.png"), binary_contours)
        iio.imwrite(join_path(self.export_img_dir, self.name_wo_ext + "_Width.png"), binary_widths)
        iio.imwrite(join_path(self.color_img_dir, self.name_wo_ext + "_color_mask.png"), contour_img)
        iio.imwrite(join_path(self.color_img_dir, self.name_wo_ext + "_color_width.png"), width_img)
        iio.imwrite(join_path(self.color_img_dir, self.name_wo_ext + "_gray_width.png"), int_width_img)

    def analyze_orientation(self):
        """Analyze orientation of fibres"""
        orient_analyzer = OrientationAnalyzer(2.0)
        img_path = join_path(self.roi_dir, self.name_wo_ext + '_roi.png')
        mask_roi = iio.imread(join_path(self.bin_dir, self.name_wo_ext + "_mask.png"))

        if np.sum(mask_roi) == 0:
            # Create empty placeholder images and store zero values for metrics
            empty_img = np.zeros_like(mask_roi)

            self.stats.loc[0, 'Orient. Alignment'] = 0
            self.stats.loc[0, 'Orient. Variance'] = 0
            self.stats.loc[0, 'Orient. Alignment (ROI)'] = 0
            self.stats.loc[0, 'Orient. Variance (ROI)'] = 0
            self.stats.loc[0, 'Orient. Alignment (HDM)'] = 0
            self.stats.loc[0, 'Orient. Variance (HDM)'] = 0
            self.stats.loc[0, 'Orient. Alignment (WIDTH)'] = 0
            self.stats.loc[0, 'Orient. Variance (WIDTH)'] = 0

            iio.imwrite(join_path(self.export_img_dir, self.name_wo_ext + "_Energy.tif"), empty_img)
            iio.imwrite(join_path(self.export_img_dir, self.name_wo_ext + "_Coherency.tif"), empty_img)
            iio.imwrite(join_path(self.export_img_dir, self.name_wo_ext + "_Orientation.tif"), empty_img)
            iio.imwrite(join_path(self.export_img_dir, self.name_wo_ext + "_Color_Survey.tif"), empty_img)
            iio.imwrite(join_path(self.color_img_dir, self.name_wo_ext + "_orient_vf.png"), empty_img)
            iio.imwrite(join_path(self.color_img_dir, self.name_wo_ext + "_angular_hist.png"), empty_img)
            return

        mask_hdm = (iio.imread(join_path(self.hdm_dir, self.name_wo_ext + "_roi.png")) > 0).astype(np.uint8) * 255
        mask_width = 255 - iio.imread(join_path(self.export_img_dir, self.name_wo_ext + "_Width.png"))

        orient_analyzer.compute_orient(img_path)

        # Store metrics in stats dataframe
        self.stats.loc[0, 'Orient. Alignment'] = orient_analyzer.mean_coherency()
        self.stats.loc[0, 'Orient. Variance'] = orient_analyzer.circular_variance()
        self.stats.loc[0, 'Orient. Alignment (ROI)'] = orient_analyzer.mean_coherency(mask=mask_roi)
        self.stats.loc[0, 'Orient. Variance (ROI)'] = orient_analyzer.circular_variance(mask=mask_roi)
        self.stats.loc[0, 'Orient. Alignment (HDM)'] = orient_analyzer.mean_coherency(mask=mask_hdm)
        self.stats.loc[0, 'Orient. Variance (HDM)'] = orient_analyzer.circular_variance(mask=mask_hdm)
        self.stats.loc[0, 'Orient. Alignment (WIDTH)'] = orient_analyzer.mean_coherency(mask=mask_width)
        self.stats.loc[0, 'Orient. Variance (WIDTH)'] = orient_analyzer.circular_variance(mask=mask_width)

        # Store images
        self.energy_img = orient_analyzer.get_energy_image()
        self.coherency_img = orient_analyzer.get_coherency_image()
        self.orientation_img = orient_analyzer.get_orientation_image()
        self.orient_color_survey = orient_analyzer.draw_color_survey()
        self.orient_vector_field = orient_analyzer.draw_vector_field(mask_roi / 255.0)
        self.angular_hist = orient_analyzer.draw_angular_hist(mask=mask_roi)

        # Save images
        iio.imwrite(join_path(self.export_img_dir, self.name_wo_ext + "_Energy.tif"),
                    self.energy_img)
        iio.imwrite(join_path(self.export_img_dir, self.name_wo_ext + "_Coherency.tif"),
                    self.coherency_img)
        iio.imwrite(join_path(self.export_img_dir, self.name_wo_ext + "_Orientation.tif"),
                    self.orientation_img)
        iio.imwrite(join_path(self.export_img_dir, self.name_wo_ext + "_Color_Survey.tif"),
                    self.orient_color_survey)
        iio.imwrite(join_path(self.color_img_dir, self.name_wo_ext + "_orient_vf.png"),
                    self.orient_vector_field)
        iio.imwrite(join_path(self.color_img_dir, self.name_wo_ext + "_angular_hist.png"),
                    self.angular_hist)

    def quantify_skeleton(self):
        """Quantify skeleton metrics"""
        min_skel_size = int(self.args["Quantification"]["Minimum Branch Length"])
        min_branch_len = int(self.args["Quantification"]["Minimum Branch Length"])
        min_hole_area = 8
        min_curve_win = int(self.args["Quantification"]["Minimum Curvature Window"])
        max_curve_win = int(self.args["Quantification"]["Maximum Curvature Window"])
        curve_win_step = int(self.args["Quantification"]["Curvature Window Step"])

        print(f"Min skeleton size = {min_skel_size} px, "
              f"Min branch length = {min_branch_len} px, "
              f"Min hole area = {min_hole_area} px²")

        # Fibre detection returns fibres in black color,
        # so dark_line is set to "True" for skeleton analysis
        skel_analyzer = SkeletonAnalyzer(skel_thresh=min_skel_size,
                                         branch_thresh=min_branch_len,
                                         hole_threshold=min_hole_area,
                                         dark_line=True)

        img_path = join_path(self.mask_dir, self.name_wo_ext + '_roi.png')
        skel_analyzer.analyze_image(img_path)

        # Store metrics in stats dataframe
        self.stats.loc[0, 'Area of Fibre Spines (µm²)'] = skel_analyzer.proj_area * self.ims_res ** 2
        self.stats.loc[0, 'Lacunarity'] = skel_analyzer.lacunarity
        self.stats.loc[0, 'Total Length (µm)'] = skel_analyzer.total_length * self.ims_res
        self.stats.loc[0, 'Endpoints'] = skel_analyzer.num_tips
        self.stats.loc[0, 'Avg Length (µm)'] = skel_analyzer.growth_unit * self.ims_res
        self.stats.loc[0, 'Branchpoints'] = skel_analyzer.num_branches
        self.stats.loc[0, 'Box-Counting Fractal Dimension'] = skel_analyzer.frac_dim
        self.stats.loc[0, 'Total Image Area (µm²)'] = np.prod(skel_analyzer.raw_image.shape[:2]) * self.ims_res ** 2

        # Store skeleton image
        self.skeleton_img = skel_analyzer.key_pts_image
        self.length_map = skel_analyzer.length_map_all

        # Save images
        iio.imwrite(join_path(self.export_img_dir, self.name_wo_ext + "_Skeleton.png"),
                    self.skeleton_img)
        iio.imwrite(join_path(self.export_img_dir, self.name_wo_ext + "_Length_Map.tif"),
                    self.length_map)

        # Calculate curvatures for various window sizes
        for win_sz in np.arange(min_curve_win, max_curve_win + curve_win_step, curve_win_step):
            skel_analyzer.calc_curve_all(win_sz)
            curve_key = f"Curvature (win_sz={win_sz})"
            self.stats.loc[0, curve_key] = skel_analyzer.avg_curve_all
            self.curve_maps[win_sz] = skel_analyzer.curve_map_all

            # Save curve map
            iio.imwrite(join_path(self.export_img_dir, f"{self.name_wo_ext}_Curve_Map_{win_sz}.tif"),
                        skel_analyzer.curve_map_all)

    def quantify_hdm(self):
        """Quantify high density matrix areas"""
        max_hdm = self.args["Quantification"]["Maximum Display HDM"]
        sat_ratio = self.args["Quantification"]["Contrast Enhancement"]
        dark_line = self.args["Detection"]["Dark Line"]

        hdm = HDM(max_hdm=max_hdm, sat_ratio=sat_ratio, dark_line=dark_line)

        # Analyze single image
        img_path = join_path(self.eligible_dir, self.name_wo_ext + ".png")

        # HDM analysis
        hdm.quantify_black_space(img_path, self.hdm_dir, ext=".png")

        # Store HDM image and metrics
        self.stats.loc[0, 'Image'] = self.name_wo_ext + '_roi.png'
        self.stats.loc[0, '% HDM Area'] = hdm.df_hdm["% HDM Area"].iloc[0]

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
        cv2.imwrite(join_path(self.export_img_dir, self.name_wo_ext + "_GapImage.png"), color_result)

        # Calculate gap metrics
        areas = np.pi * (np.array(final_circles)[:, 2] ** 2) * self.ims_res ** 2

        if len(areas) > 0:
            # Store gap metrics
            self.stats.loc[0, 'Mean (All gaps area in µm²)'] = np.mean(areas)
            self.stats.loc[0, 'Std (All gaps area in µm²)'] = np.std(areas)
            self.stats.loc[0, 'Percentile5 (All gaps area in µm²)'] = np.percentile(areas, 5)
            self.stats.loc[0, 'Median (All gaps area in µm²)'] = np.median(areas)
            self.stats.loc[0, 'Percentile95 (All gaps area in µm²)'] = np.percentile(areas, 95)
            self.stats.loc[0, 'Gap Circles Count (All)'] = areas.size

            # Convert to radius metrics
            radius_values = np.sqrt(areas / np.pi)
            self.stats.loc[0, 'Mean (All gaps radius in µm)'] = np.mean(radius_values)
            self.stats.loc[0, 'Std (All gaps radius in µm)'] = np.std(radius_values)
            self.stats.loc[0, 'Median (All gaps radius in µm)'] = np.median(radius_values)
            self.stats.loc[0, 'Percentile5 (All gaps radius in µm)'] = np.percentile(radius_values, 5)
            self.stats.loc[0, 'Percentile95 (All gaps radius in µm)'] = np.percentile(radius_values, 95)
        else:
            # Fill with zeros if no gaps found
            gap_metrics = ['Mean (All gaps area in µm²)', 'Std (All gaps area in µm²)',
                           'Percentile5 (All gaps area in µm²)', 'Median (All gaps area in µm²)',
                           'Percentile95 (All gaps area in µm²)', 'Gap Circles Count (All)',
                           'Mean (All gaps radius in µm)', 'Std (All gaps radius in µm)',
                           'Median (All gaps radius in µm)', 'Percentile5 (All gaps radius in µm)',
                           'Percentile95 (All gaps radius in µm)']

            for metric in gap_metrics:
                self.stats.loc[0, metric] = 0

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
        cv2.imwrite(join_path(self.export_img_dir, self.name_wo_ext + "_GapImage_intra_gaps.png"), color_img)

        areas = np.array(areas)

        if len(areas) > 0:
            radius = np.sqrt(areas / np.pi)

            # Store metrics
            self.stats.loc[0, 'Mean (ROI gaps area in µm²)'] = np.mean(areas)
            self.stats.loc[0, 'Std (ROI gaps area in µm²)'] = np.std(areas)
            self.stats.loc[0, 'Percentile5 (ROI gaps area in µm²)'] = np.percentile(areas, 5)
            self.stats.loc[0, 'Median (ROI gaps area in µm²)'] = np.median(areas)
            self.stats.loc[0, 'Percentile95 (ROI gaps area in µm²)'] = np.percentile(areas, 95)
            self.stats.loc[0, 'Mean (ROI gaps radius in µm)'] = np.mean(radius)
            self.stats.loc[0, 'Std (ROI gaps radius in µm)'] = np.std(radius)
            self.stats.loc[0, 'Percentile5 (ROI gaps radius in µm)'] = np.percentile(radius, 5)
            self.stats.loc[0, 'Median (ROI gaps radius in µm)'] = np.median(radius)
            self.stats.loc[0, 'Percentile95 (ROI gaps radius in µm)'] = np.percentile(radius, 95)
            self.stats.loc[0, 'Gap Circles Count (ROI)'] = circle_cnt
        else:
            # Fill with zeros if no gaps found
            intra_gap_metrics = ['Mean (ROI gaps area in µm²)', 'Std (ROI gaps area in µm²)',
                                 'Percentile5 (ROI gaps area in µm²)', 'Median (ROI gaps area in µm²)',
                                 'Percentile95 (ROI gaps area in µm²)', 'Gap Circles Count (ROI)',
                                 'Mean (ROI gaps radius in µm)', 'Std (ROI gaps radius in µm)',
                                 'Median (ROI gaps radius in µm)', 'Percentile5 (ROI gaps radius in µm)',
                                 'Percentile95 (ROI gaps radius in µm)']

            for metric in intra_gap_metrics:
                self.stats.loc[0, metric] = 0

    def calc_fibre_areas(self):
        """Calculate fibre areas and related metrics"""
        img_mask = cv2.imread(join_path(self.bin_dir, self.name_wo_ext + '_mask.png'), 0)
        area_roi = np.sum(img_mask > 128).astype(float)  # ROI area

        if area_roi == 0:
            # Fill with zeros if no fibres found
            fibre_metrics = ['Area (ROI)', '% ROI Area', 'Area (WIDTH)', '% WIDTH Area',
                             'Mean Fibre Intensity (ROI)', 'Mean Fibre Intensity (WIDTH)',
                             'Mean Fibre Intensity (HDM)']
            for metric in fibre_metrics:
                self.stats.loc[0, metric] = 0
            return

        percent_roi = area_roi / img_mask.shape[0] / img_mask.shape[1]  # % ROI area
        ori_img = iio.imread(join_path(self.eligible_dir, self.name_wo_ext + ".png"))

        # Calculate mean intensities
        hed = rgb2hed(ori_img)
        null = np.zeros_like(hed[:, :, 0])
        ihc_e = hed2rgb(np.stack((null, hed[:, :, 1], null), axis=-1))
        red_img = (rgb2gray(ihc_e) * 255).astype(np.uint8)

        width_mask = cv2.imread(join_path(self.export_img_dir, self.name_wo_ext + "_Width.png"), 0)
        hdm_mask = cv2.imread(join_path(self.hdm_dir, self.name_wo_ext + '_roi.png'), 0)
        area_width = np.sum(width_mask < 128).astype(float)  # WIDTH area
        percent_width = area_width / np.product(width_mask.shape[:2])  # % WIDTH area

        if np.count_nonzero(red_img < 180):
            mean_intensity_roi = np.mean(red_img[(img_mask > 128) & (red_img < 180)])
            mean_intensity_width = np.mean(red_img[(width_mask < 128) & (red_img < 180)])
            mean_intensity_hdm = np.mean(red_img[(hdm_mask > 0) & (red_img < 180)])
        else:
            grayscale = (rgb2gray(ori_img) * 255).astype(np.uint8)
            mean_intensity_roi = np.mean(grayscale[img_mask > 128])
            mean_intensity_width = np.mean(grayscale[width_mask < 128])
            mean_intensity_hdm = np.mean(grayscale[hdm_mask > 0])

        # Store metrics
        self.stats.loc[0, 'Area (ROI)'] = area_roi
        self.stats.loc[0, '% ROI Area'] = percent_roi
        self.stats.loc[0, 'Area (WIDTH)'] = area_width
        self.stats.loc[0, '% WIDTH Area'] = percent_width
        self.stats.loc[0, 'Mean Fibre Intensity (ROI)'] = mean_intensity_roi
        self.stats.loc[0, 'Mean Fibre Intensity (WIDTH)'] = mean_intensity_width
        self.stats.loc[0, 'Mean Fibre Intensity (HDM)'] = mean_intensity_hdm

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
        print('Normalizing statistics')

        # Normalize fibre area
        if 'Fibre Area (WIDTH, µm²)' in self.stats.columns and 'Fibre Area (ROI, µm²)' in self.stats.columns:
            if self.stats.loc[0, 'Fibre Area (ROI, µm²)'] > 0:
                self.stats.loc[0, 'Fibre Coverage (WIDTH/ROI)'] = self.stats.loc[0, 'Fibre Area (WIDTH, µm²)'] / \
                                                                  self.stats.loc[0, 'Fibre Area (ROI, µm²)']
            else:
                self.stats.loc[0, 'Fibre Coverage (WIDTH/ROI)'] = 0

        # Normalize branch and end points
        if 'Total Length (µm)' in self.stats.columns and self.stats.loc[0, 'Total Length (µm)'] > 0:
            if 'Branchpoints' in self.stats.columns:
                self.stats.loc[0, 'Branchpoints Density (µm⁻¹)'] = self.stats.loc[0, 'Branchpoints'] / self.stats.loc[
                    0, 'Total Length (µm)']

            if 'Endpoints' in self.stats.columns:
                self.stats.loc[0, 'Endpoints Density (µm⁻¹)'] = self.stats.loc[0, 'Endpoints'] / self.stats.loc[
                    0, 'Total Length (µm)']
        else:
            density_metrics = ['Branchpoints Density (µm⁻¹)', 'Endpoints Density (µm⁻¹)']
            for metric in density_metrics:
                self.stats.loc[0, metric] = 0

        # Normalize gap area
        if self.args["Configs"]["Gap Analysis"] and self.args["Gap Analysis"]["Minimum Gap Diameter"] > 0:
            # Normalize all gaps area
            if 'Mean (All gaps area in µm²)' in self.stats.columns and 'Total Image Area (µm²)' in self.stats.columns:
                if self.stats.loc[0, 'Total Image Area (µm²)'] > 0:
                    self.stats.loc[0, 'Normalized Mean (All gaps area)'] = self.stats.loc[
                                                                               0, 'Mean (All gaps area in µm²)'] / \
                                                                           self.stats.loc[0, 'Total Image Area (µm²)']
                    self.stats.loc[0, 'Normalized Std (All gaps area)'] = self.stats.loc[
                                                                              0, 'Std (All gaps area in µm²)'] / \
                                                                          self.stats.loc[0, 'Total Image Area (µm²)']
                else:
                    self.stats.loc[0, 'Normalized Mean (All gaps area)'] = 0
                    self.stats.loc[0, 'Normalized Std (All gaps area)'] = 0

            # Normalize ROI gaps area
            if 'Mean (ROI gaps area in µm²)' in self.stats.columns and 'Fibre Area (ROI, µm²)' in self.stats.columns:
                if self.stats.loc[0, 'Fibre Area (ROI, µm²)'] > 0:
                    self.stats.loc[0, 'Normalized Mean (ROI gaps area)'] = self.stats.loc[
                                                                               0, 'Mean (ROI gaps area in µm²)'] / \
                                                                           self.stats.loc[0, 'Fibre Area (ROI, µm²)']
                    self.stats.loc[0, 'Normalized Std (ROI gaps area)'] = self.stats.loc[
                                                                              0, 'Std (ROI gaps area in µm²)'] / \
                                                                          self.stats.loc[0, 'Fibre Area (ROI, µm²)']
                else:
                    self.stats.loc[0, 'Normalized Mean (ROI gaps area)'] = 0
                    self.stats.loc[0, 'Normalized Std (ROI gaps area)'] = 0

            # Normalize gap radius
            if 'Mean (All gaps radius in µm)' in self.stats.columns and 'Total Image Area (µm²)' in self.stats.columns:
                if self.stats.loc[0, 'Total Image Area (µm²)'] > 0:
                    self.stats.loc[0, 'Normalized Mean (All gaps radius)'] = self.stats.loc[
                                                                                 0, 'Mean (All gaps radius in µm)'] / np.sqrt(
                        self.stats.loc[0, 'Total Image Area (µm²)'])
                    self.stats.loc[0, 'Normalized Std (All gaps radius)'] = self.stats.loc[
                                                                                0, 'Std (All gaps radius in µm)'] / np.sqrt(
                        self.stats.loc[0, 'Total Image Area (µm²)'])
                else:
                    self.stats.loc[0, 'Normalized Mean (All gaps radius)'] = 0
                    self.stats.loc[0, 'Normalized Std (All gaps radius)'] = 0

            if 'Mean (ROI gaps radius in µm)' in self.stats.columns and 'Fibre Area (ROI, µm²)' in self.stats.columns:
                if self.stats.loc[0, 'Fibre Area (ROI, µm²)'] > 0:
                    self.stats.loc[0, 'Normalized Mean (ROI gaps radius)'] = self.stats.loc[
                                                                                 0, 'Mean (ROI gaps radius in µm)'] / np.sqrt(
                        self.stats.loc[0, 'Fibre Area (ROI, µm²)'])
                    self.stats.loc[0, 'Normalized Std (ROI gaps radius)'] = self.stats.loc[
                                                                                0, 'Std (ROI gaps radius in µm)'] / np.sqrt(
                        self.stats.loc[0, 'Fibre Area (ROI, µm²)'])
                else:
                    self.stats.loc[0, 'Normalized Mean (ROI gaps radius)'] = 0
                    self.stats.loc[0, 'Normalized Std (ROI gaps radius)'] = 0

            # Normalize gap number
            if 'Gap Circles Count (All)' in self.stats.columns and 'Total Image Area (µm²)' in self.stats.columns:
                if self.stats.loc[0, 'Total Image Area (µm²)'] > 0:
                    self.stats.loc[0, 'Gap density (All, µm⁻²)'] = self.stats.loc[0, 'Gap Circles Count (All)'] / \
                                                                   self.stats.loc[0, 'Total Image Area (µm²)']
                else:
                    self.stats.loc[0, 'Gap density (All, µm⁻²)'] = 0

            if 'Gap Circles Count (ROI)' in self.stats.columns and 'Fibre Area (ROI, µm²)' in self.stats.columns:
                if self.stats.loc[0, 'Fibre Area (ROI, µm²)'] > 0:
                    self.stats.loc[0, 'Gap density (ROI, µm⁻²)'] = self.stats.loc[0, 'Gap Circles Count (ROI)'] / \
                                                                   self.stats.loc[0, 'Fibre Area (ROI, µm²)']
                else:
                    self.stats.loc[0, 'Gap density (ROI, µm⁻²)'] = 0

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