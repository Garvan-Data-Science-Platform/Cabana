import cv2
import numpy as np
from scipy.stats import circvar, kurtosis
import imageio.v3 as iio
from skimage.util import img_as_bool
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from .utils import normalize


class OrientationAnalyzer:
    """
    A class for analyzing the orientation, coherency, and energy of image structures.

    This class processes an image to extract orientation information based on structure
    tensors and provides methods to visualize and analyze the orientations.

    Parameters
    ----------
    sigma : float, optional
        Standard deviation for the Gaussian filter used in structure tensor computation.
        Default is 2.0.

    Attributes
    ----------
    sigma : float
        Standard deviation for Gaussian filtering.
    image : ndarray
        The input image.
    gray : ndarray
        Grayscale version of the input image.
    orient : ndarray
        Orientation angle map in radians, ranging from -π/2 to π/2.
    coherency : ndarray
        Coherency map ranging from 0 to 1, where higher values indicate more consistent orientation.
    energy : ndarray
        Energy map of the gradient, normalized to range from 0 to 1.
    dxx, dxy, dyy : ndarray
        Components of the structure tensor.
    """

    def __init__(self, sigma=2.0):
        """
        Initialize the OrientationAnalyzer with a given sigma value.

        Parameters
        ----------
        sigma : float, optional
            Standard deviation for the Gaussian filter. Default is 2.0.
        """
        self.sigma = sigma
        self.image = None
        self.gray = None
        self.orient = None
        self.coherency = None
        self.energy = None
        self.dxx = None
        self.dxy = None
        self.dyy = None

    def compute_orient(self, image):
        """
        Compute the orientation, coherency, and energy maps of the input image.

        This method calculates the structure tensor of the image and derives
        orientation, coherency, and energy maps from it.

        Parameters
        ----------
        image : str or ndarray
            Input image path or image array.

        Returns
        -------
        None
            Updates the class attributes with computed values.
        """
        # Load image if a path is provided, otherwise use the array
        self.image = iio.imread(image) if isinstance(image, str) else image

        # Normalize to uint8 if needed
        if self.image.dtype != np.uint8:
            self.image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)

        # Convert to grayscale if image is RGB
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY) if self.image.ndim == 3 else self.image

        # Convert to float for processing
        gray = self.gray.astype(float)

        # Apply Gaussian blur to reduce noise
        blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=self.sigma, sigmaY=self.sigma)

        # Compute gradients using Sobel operators
        grad_x = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT)
        grad_y = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT)

        # Compute structure tensor components with Gaussian smoothing
        dxx = cv2.GaussianBlur(grad_x * grad_x, (0, 0), sigmaX=self.sigma, sigmaY=self.sigma)
        dxy = cv2.GaussianBlur(grad_x * grad_y, (0, 0), sigmaX=self.sigma, sigmaY=self.sigma)
        dyy = cv2.GaussianBlur(grad_y * grad_y, (0, 0), sigmaX=self.sigma, sigmaY=self.sigma)

        # Compute energy (trace of the structure tensor)
        energy = dxx + dyy
        # Normalize energy to range [0, 1]
        self.energy = normalize(energy, 2, 98)

        # Compute orientation angle from structure tensor
        # The orientation is perpendicular to the gradient direction
        self.orient = 0.5 * np.arctan2(2.0 * dxy, dyy - dxx)

        # Ensure orientation is in range [-π/2, π/2] by wrapping
        self.orient[self.orient > np.pi / 2] = self.orient[self.orient > np.pi / 2] - np.pi
        self.orient[self.orient < -np.pi / 2] = self.orient[self.orient < -np.pi / 2] + np.pi

        # Compute coherency from eigenvalues of the structure tensor
        # Coherency measures how consistently oriented the structures are
        self.coherency = np.sqrt((dyy - dxx) ** 2 + 4.0 * dxy ** 2) / (dxx + dyy + np.finfo(float).eps)

        # Store structure tensor components for later use
        self.dxx = dxx
        self.dxy = dxy
        self.dyy = dyy

    def get_orientation_image(self, mask=None):
        """
        Get the orientation image, optionally masked.

        Parameters
        ----------
        mask : ndarray, optional
            Boolean mask to apply. Default is None (no mask).

        Returns
        -------
        ndarray
            Orientation image with values in radians from -π/2 to π/2.
        """
        # Create a full mask if none is provided
        mask = np.ones_like(self.gray, dtype=bool) if mask is None else img_as_bool(mask)

        # Create empty image and fill with orientation values where mask is True
        orient_image = np.zeros_like(self.gray, dtype=float)
        orient_image[mask] = self.orient[mask]
        return orient_image

    def get_coherency_image(self, mask=None):
        """
        Get the coherency image, optionally masked.

        Parameters
        ----------
        mask : ndarray, optional
            Boolean mask to apply. Default is None (no mask).

        Returns
        -------
        ndarray
            Coherency image with values from 0 to 1.
        """
        mask = np.ones_like(self.gray, dtype=bool) if mask is None else img_as_bool(mask)
        coherency_image = np.zeros_like(self.gray, dtype=float)
        coherency_image[mask] = self.coherency[mask]
        return coherency_image

    def get_energy_image(self, mask=None):
        """
        Get the energy image, optionally masked.

        Parameters
        ----------
        mask : ndarray, optional
            Boolean mask to apply. Default is None (no mask).

        Returns
        -------
        ndarray
            Energy image with values from 0 to 1.
        """
        mask = np.ones_like(self.gray, dtype=bool) if mask is None else img_as_bool(mask)
        energy_image = np.zeros_like(self.gray, dtype=float)
        energy_image[mask] = self.energy[mask]
        return energy_image

    def mean_orientation(self, mask=None):
        """
        Calculate the mean orientation within a mask.

        Parameters
        ----------
        mask : ndarray, optional
            Boolean mask to apply. Default is None (no mask).

        Returns
        -------
        float
            Mean orientation in degrees.
        """
        mask = np.ones_like(self.gray, dtype=bool) if mask is None else img_as_bool(mask)

        # Calculate mean of structure tensor components
        vxy = np.mean(self.dxy[mask])
        vxx = np.mean(self.dxx[mask])
        vyy = np.mean(self.dyy[mask])

        # Calculate orientation from mean tensor components and convert to degrees
        return np.rad2deg(0.5 * np.arctan2(2.0 * vxy, vyy - vxx))

    def mean_coherency(self, mask=None):
        """
        Calculate the mean coherency within a mask.

        Parameters
        ----------
        mask : ndarray, optional
            Boolean mask to apply. Default is None (no mask).

        Returns
        -------
        float
            Mean coherency, ranging from 0 to 1.
        """
        mask = np.ones_like(self.gray, dtype=bool) if mask is None else img_as_bool(mask)

        # Calculate mean of structure tensor components
        vxy = np.mean(self.dxy[mask])
        vxx = np.mean(self.dxx[mask])
        vyy = np.mean(self.dyy[mask])

        # Calculate coherency from mean tensor components
        return np.sqrt((vyy - vxx) ** 2 + 4.0 * vxy ** 2) / (vxx + vyy + np.finfo(float).eps)

    def circular_variance(self, mask=None):
        """
        Calculate the circular variance of orientations within a mask.

        Circular variance is a measure of the spread of angles and ranges from 0 to 1,
        where 0 indicates perfect alignment and 1 indicates random orientations.

        Parameters
        ----------
        mask : ndarray, optional
            Boolean mask to apply. Default is None (no mask).

        Returns
        -------
        float
            Circular variance, ranging from 0 to 1.
        """
        mask = np.ones_like(self.gray, dtype=bool) if mask is None else img_as_bool(mask)

        # Add π/2 to shift range to [0, π] before calculating circular variance
        return circvar(self.orient[mask] + np.pi / 2.0, high=np.pi)

    def randomness_orientation(self, bins=180, mask=None):
        """
        Calculate a measure of orientation randomness using entropy.

        This method computes the Kullback-Leibler divergence between the orientation
        distribution and a uniform distribution, then transforms it into a randomness measure.

        Parameters
        ----------
        bins : int, optional
            Number of bins for histogram. Default is 180.
        mask : ndarray, optional
            Boolean mask to apply. Default is None (no mask).

        Returns
        -------
        float
            Randomness measure, with higher values indicating more random orientations.
        """
        mask = np.ones_like(self.gray, dtype=bool) if mask is None else img_as_bool(mask)

        # Create histogram of orientations
        hist, _ = np.histogram((self.orient[mask] + np.pi / 2) / np.pi * 180, bins=bins, range=(0, 180), density=True)

        # Calculate non-zero probabilities
        probabilities = hist[hist > 0] / np.sum(hist[hist > 0])

        # Create a uniform distribution for comparison
        uniform_probabilities = np.full(bins, 1.0 / bins)

        # Calculate Kullback-Leibler divergence
        kl_divergence = np.sum(probabilities * np.log(probabilities / uniform_probabilities))

        # Transform KL divergence into a randomness measure (higher = more random)
        return 1.0 / (np.sqrt(kl_divergence) + 1.0 + np.finfo(float).eps)

    def draw_angular_hist(self, N=8, mask=None):
        """
        Draw a polar histogram of orientations.

        Parameters
        ----------
        N : int, optional
            Number of bins in the histogram. Default is 8.
        mask : ndarray, optional
            Boolean mask to apply. Default is None (no mask).

        Returns
        -------
        ndarray
            RGB image of the polar histogram.
        """
        mask = np.ones_like(self.gray, dtype=bool) if mask is None else img_as_bool(mask)
        if mask.sum() == 0:
            return np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)

        # Create figure and polar axis
        fig = Figure(figsize=(4, 4), dpi=200)
        ax = fig.add_subplot(polar=True)

        # Draw right half in [-π/2, π/2]
        angles = self.orient[mask]
        distribution = np.histogram(angles, bins=N, range=(-0.5 * np.pi, 0.5 * np.pi), density=True)[0]
        theta = (np.arange(N) + 0.5) * np.pi / N - np.pi / 2.0
        width = np.pi / N  # Width of bars
        colors = plt.cm.hsv((theta + np.pi / 2.0) / np.pi)

        # Draw first half of the histogram
        ax.bar(theta, distribution, width=width, color=colors)

        # Draw symmetric half in [π/2, 1.5*π] to handle orientation ambiguity
        distribution = np.histogram(angles + np.pi, bins=N, range=(0.5 * np.pi, 1.5 * np.pi), density=True)[0]
        theta = (np.arange(N) + 0.5) * np.pi / N + np.pi / 2.0
        colors = plt.cm.hsv((theta - np.pi / 2.0) / np.pi)

        # Draw second half of the histogram
        ax.bar(theta, distribution, width=width, color=colors)

        # Configure axis appearance
        ax.set_yticklabels([])
        ax.set_xticks([i / 4.0 * np.pi for i in range(8)])
        ax.set_xticklabels([r'$0$', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$',
                            r'$\pi$', r'$\frac{5\pi}{4}$', r'$\frac{3\pi}{2}$', r'$\frac{7\pi}{4}$'])
        ax.tick_params(labelsize=8)
        ax.set_title('Angular Distribution', pad=-5, fontsize=8)

        # Render figure to numpy array
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        s, (width, height) = canvas.print_to_buffer()
        X = np.frombuffer(s, np.uint8).reshape((height, width, 4))
        return X[..., :3]  # Return RGB image

    def draw_vector_field(self, wgts_map=None, color=(255, 255, 0), thickness=1, size=15, scale=80):
        """
        Draw a vector field visualization of the orientations.

        Parameters
        ----------
        wgts_map : ndarray, optional
            Weights map to scale vector lengths. Default is None.
        color : tuple, optional
            RGB color for vectors. Default is (255, 255, 0).
        thickness : int, optional
            Line thickness for vectors. Default is 1.
        size : int, optional
            Grid size for vector placement. Default is 15.
        scale : int, optional
            Scaling factor for vector lengths. Default is 80.

        Returns
        -------
        ndarray
            RGB image with vector field overlay.
        """
        ny, nx = self.orient.shape[:2]

        # Calculate starting points to center the grid
        xstart = (nx - (nx // size) * size) // 2
        ystart = (ny - (ny // size) * size) // 2

        # Calculate number of blocks in each dimension
        x_blk_num = len(range(xstart, nx, size))
        y_blk_num = len(range(ystart, ny, size))

        # Initialize arrays for block statistics and weights
        blk_stats = np.zeros((y_blk_num, x_blk_num, 4))  # [y, x, dy, dx]
        blk_wgts = np.ones((y_blk_num, x_blk_num))

        # Calculate average orientation for each block
        for y in range(ystart, ny, size):
            for x in range(xstart, nx, size):
                # Store block center coordinates
                blk_stats[(y - ystart) // size, (x - xstart) // size, 0] = y
                blk_stats[(y - ystart) // size, (x - xstart) // size, 1] = x

                # Calculate block boundaries with bounds checking
                top = y - size // 2 if y - size // 2 >= 0 else 0
                bot = y + size // 2 if y + size // 2 <= ny else ny
                lft = x - size // 2 if x - size // 2 >= 0 else 0
                rht = x + size // 2 if x + size // 2 <= nx else nx

                # Calculate average orientation components
                dx = np.mean(np.cos(self.orient[top:bot, lft:rht]))
                dy = np.mean(np.sin(self.orient[top:bot, lft:rht]))
                blk_stats[(y - ystart) // size, (x - xstart) // size, 2] = dy
                blk_stats[(y - ystart) // size, (x - xstart) // size, 3] = dx

                # If weights map provided, calculate maximum weight for this block
                if wgts_map is not None:
                    blk_wgts[(y - ystart) // size, (x - xstart) // size] = np.max(
                        wgts_map[top:bot, lft:rht]
                    )

        # Normalize weights if they vary
        min_val = np.min(blk_wgts)
        max_val = np.max(blk_wgts)

        if min_val != max_val:
            blk_wgts = normalize(blk_wgts, pmin=0, pmax=100, axis=[0, 1])

        # Prepare image to draw on (convert to RGB if grayscale)
        if self.image.ndim == 2 or self.image.shape[2] == 1:
            img_draw = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
        else:
            img_draw = self.image.copy()

        # Draw vectors directly on the image
        for blk_yi in range(y_blk_num):
            for blk_xi in range(x_blk_num):
                r = blk_wgts[blk_yi, blk_xi] * scale / 100.0 * size * 0.5

                y0 = int(blk_stats[blk_yi, blk_xi, 0])
                x0 = int(blk_stats[blk_yi, blk_xi, 1])
                dy = blk_stats[blk_yi, blk_xi, 2]
                dx = blk_stats[blk_yi, blk_xi, 3]

                y1 = int(y0 - r * dy)
                x1 = int(x0 + r * dx)
                y2 = int(y0 + r * dy)
                x2 = int(x0 - r * dx)

                # Draw line
                cv2.line(img_draw, (x1, y1), (x2, y2), color, thickness)

        return img_draw
    def draw_color_survey(self, mask=None):
        """
        Create a colored visualization of orientations using HSV color space.

        The hue represents orientation, saturation represents coherency,
        and value represents energy.

        Parameters
        ----------
        mask : ndarray, optional
            Boolean mask to apply. Default is None (no mask).

        Returns
        -------
        ndarray
            RGB image representing the orientation survey.
        """
        mask = np.ones_like(self.gray, dtype=bool) if mask is None else img_as_bool(mask)

        # Normalize orientation to [0, 1] then scale to [0, 179] for hue
        # Adding π/2 shifts range from [-π/2, π/2] to [0, π]
        hue = (((self.orient + np.pi / 2) / np.pi) * 179).astype(np.uint8)

        # Scale coherency and energy to [0, 255] for saturation and value
        saturation = (self.coherency * 255).astype(np.uint8)
        value = (self.energy * 255).astype(np.uint8)

        # Zero out masked regions
        hue[~mask] = 0
        saturation[~mask] = 0
        value[~mask] = 0

        # Stack the channels to create an HSV image
        hsv_image = np.stack((hue, saturation, value), axis=-1)

        # Convert HSV to RGB for display
        rgb_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

        return rgb_image


if __name__ == "__main__":
    # Example usage
    dir_analyzer = OrientationAnalyzer(3.0)
    dir_analyzer.compute_orient(
        '/Users/lxfhfut/Dropbox/Garvan/Cabana/539.vsi - 20x_BF multi-band_01Annotation (Ellipse) (Tumor)_0.tif')
    print(dir_analyzer.circular_variance())
    print(kurtosis(dir_analyzer.orient.flatten()))

    # Generate and display circular histogram
    circular_hist = dir_analyzer.draw_angular_hist()
    fig, ax = plt.subplots()
    ax.imshow(circular_hist)
    fig.patch.set_visible(False)
    ax.axis('off')
    plt.show()

    # Commented out visualization code below can be uncommented for additional visualizations
    # dir_analyzer.draw_angular_hist()
    # fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    # axes[0, 0].imshow(rgb)
    # axes[0, 1].imshow(energy, cmap='gray')
    # axes[0, 2].imshow(orient, cmap='gray')
    # axes[1, 0].imshow(coherency, cmap='gray')
    # axes[1, 1].imshow(vector_field)
    # axes[1, 2].imshow(color_survey)
    # plt.show()