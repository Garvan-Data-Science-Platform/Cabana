import os
import cv2
import numpy as np
import pandas as pd
import imageio.v3 as iio
from skimage import exposure
from .utils import join_path


class HDM:
    """
    Hematoxylin Density Map (HDM) processing class.

    This class provides functionality to quantify dark areas in histological images,
    typically used for measuring hematoxylin-stained regions in histopathology.

    Parameters
    ----------
    max_hdm : int, optional
        Maximum intensity value to consider for HDM calculation. Default is 220.
    sat_ratio : float, optional
        Saturation ratio for contrast enhancement. Default is 0.
    dark_line : bool, optional
        Whether to invert the image (to treat dark lines as positive). Default is False.

    Attributes
    ----------
    max_hdm : int
        Maximum intensity value for HDM calculation.
    sat_ratio : float
        Saturation ratio used for contrast enhancement.
    dark_line : bool
        Flag indicating whether dark regions should be treated as positive.
    df_hdm : pandas.DataFrame or None
        DataFrame containing HDM quantification results.
    """

    def __init__(self, max_hdm=220, sat_ratio=0, dark_line=False):
        """
        Initialize the HDM processor with the given parameters.

        Parameters
        ----------
        max_hdm : int, optional
            Maximum intensity threshold. Default is 220.
        sat_ratio : float, optional
            Saturation ratio for contrast enhancement (0-1). Default is 0.
        dark_line : bool, optional
            If True, inverts the image to treat dark lines as positive. Default is False.
        """
        self.max_hdm = max_hdm
        self.sat_ratio = sat_ratio
        self.dark_line = dark_line
        self.df_hdm = None

    def quantify_black_space(self, image_path, save_dir=None, ext=".png", mask_dir=None):
        """
        Quantify the dark (hematoxylin-stained) regions in histological images.

        This method processes one or multiple images, enhances their contrast,
        calculates the percentage of dark regions, and saves the results.

        Parameters
        ----------
        image_path : str
            Path to an image file or directory containing images.
        save_dir : str
            Directory where processed images and results will be saved.
        ext : str or list, optional
            File extension(s) to process. Default is ".png".
        mask_dir : str, optional
            Folder holding a 0/255 ROI mask ``<image stem>.png`` per image.
            When present, HDM pixels outside the mask are discarded and the
            area percentage is taken relative to the mask area.

        Returns
        -------
        None
            Results are saved to a CSV file and stored in the df_hdm attribute.
        """
        # Convert extension to list if it's a string
        ext_list = [ext.lower()] if isinstance(ext, str) else [e.lower() for e in ext]

        # Check if the path is a directory or a single file
        img_paths = []
        if os.path.isdir(image_path):
            # Process all matching files in directory
            for f in os.listdir(image_path):
                if any(f.lower().endswith(e) for e in ext_list):
                    img_paths.append(os.path.join(image_path, f))
        elif os.path.isfile(image_path) and any(image_path.lower().endswith(e) for e in ext_list):
            # Process single file
            img_paths.append(image_path)

        # Store results for each image
        img_names = []
        hdm = []
        hdm_imgs = []

        # Process each image
        for img_path in img_paths:
            # Enhance image contrast
            enhanced_image = self.enhance_contrast(img_path)

            # Restrict to the external ROI mask when one is available
            denominator = np.prod(enhanced_image.shape[:2])
            stem = os.path.basename(img_path)[:-4]
            mask_path = join_path(mask_dir, stem + ".png") if mask_dir else None
            if mask_path and os.path.exists(mask_path):
                roi = cv2.imread(mask_path, 0)
                if roi is not None and roi.shape[:2] == enhanced_image.shape[:2] and np.any(roi > 128):
                    enhanced_image = enhanced_image.copy()
                    enhanced_image[roi <= 128] = 0
                    denominator = np.count_nonzero(roi > 128)
            hdm_imgs.append(enhanced_image)

            # Store image name and calculate HDM area percentage
            output_filename = stem + "_roi.png"
            img_names.append(output_filename)

            # Calculate ratio of non-zero pixels to analysed pixels (HDM area percentage)
            hdm.append(np.count_nonzero(enhanced_image > 0) / denominator)

        # Save results to CSV
        if save_dir is not None:
            # Save the enhanced image
            for output_filename in img_names:
                enhanced_image = hdm_imgs[img_names.index(output_filename)]
                cv2.imwrite(join_path(save_dir, output_filename), enhanced_image)

            result_csv = join_path(save_dir, "ResultsHDM.csv")
            data = {'Image': img_names, '% HDM Area': hdm}
            self.df_hdm = pd.DataFrame(data)
            self.df_hdm.to_csv(result_csv, index=False)
        return pd.DataFrame({'Image': img_names, '% HDM Area': hdm})

    def enhance_contrast(self, image_path):
        """
        Enhance image contrast for better HDM calculation.

        This method loads an image, normalizes it, applies intensity clipping,
        and enhances contrast using percentile-based intensity rescaling.

        Parameters
        ----------
        image_path : str
            Path to the image file.

        Returns
        -------
        ndarray
            Enhanced image as an 8-bit grayscale array.
        """
        # Load image using imageio
        raw_image = np.asarray(iio.imread(image_path))

        # Normalize to 8-bit if needed
        image = raw_image if raw_image.dtype == np.uint8 else cv2.normalize(raw_image, None, 0, 255, cv2.NORM_MINMAX)

        # Convert to grayscale if the image is RGB
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image

        # Clip intensity values to max_hdm and normalize to 0-255 range
        image = np.clip(image, 0, self.max_hdm).astype(float)
        image = ((image - image.min()) / (image.max() - image.min() + np.finfo(float).eps) * 255).astype(np.uint8)

        # Apply percentile-based contrast enhancement if sat_ratio > 0
        percent_saturation = self.sat_ratio * 100
        pl, pu = np.percentile(image, (percent_saturation / 2.0, 100 - percent_saturation / 2.0))
        enhanced_image = exposure.rescale_intensity(image, in_range=(pl, pu))

        # Invert image if dark_line is True (to highlight dark features)
        if self.dark_line:
            enhanced_image = 255 - enhanced_image

        return enhanced_image.astype(np.uint8)