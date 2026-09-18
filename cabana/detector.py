import os
import cv2
import time
import imageio.v3 as iio
import skimage as ski
from .constants import *
from .log import Log
import matplotlib.pyplot as plt
from scipy.ndimage import convolve, gaussian_filter, gaussian_filter1d
from .utils import (LinesUtil, Junction, Crossref, Line, convolve_gauss,
                   bresenham, fix_locations, regularize_normals, interpolate_gradient_test,
                   closest_point, normalize_to_half_circle)


def _eigh_2x2_symmetric(a, b, c):
    """Closed-form eigendecomposition of the 2x2 symmetric matrix [[a, b], [b, c]].

    For a 2x2 symmetric matrix the eigenvalues and an orthonormal eigenvector
    pair have closed-form expressions, removing the per-pixel LAPACK overhead
    of np.linalg.eigh on a (..., 2, 2) tensor:

        T = (a + c) / 2,  D = sqrt(((a - c) / 2)^2 + b^2)
        lam_min = T - D,  lam_max = T + D
        phi = 0.5 * atan2(2 b, a - c)         # principal direction for lam_max
        v(lam_max) = (cos phi, sin phi)
        v(lam_min) = (-sin phi, cos phi)

    Returned arrays match the np.linalg.eigh contract for the trailing
    (2, 2) view of the input tensor: eigvals are sorted ascending and
    eigvecs[..., :, k] is the unit eigenvector corresponding to eigvals[..., k].
    Eigenvector signs may differ from LAPACK's choice; this is harmless for
    downstream uses that only consume eigenvectors quadratically.

    Parameters
    ----------
    a, b, c : ndarray
        Matrix entries [0, 0], [0, 1] (= [1, 0]) and [1, 1] respectively.
        Must be broadcastable to a common shape.

    Returns
    -------
    eigvals : ndarray, shape (..., 2)
        Ascending eigenvalues.
    eigvecs : ndarray, shape (..., 2, 2)
        Orthonormal eigenvectors as columns.
    """
    half_trace = 0.5 * (a + c)
    half_diff = 0.5 * (a - c)
    disc = np.sqrt(half_diff * half_diff + b * b)
    lam_min = half_trace - disc
    lam_max = half_trace + disc
    eigvals = np.stack([lam_min, lam_max], axis=-1)

    phi = 0.5 * np.arctan2(2.0 * b, a - c)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    eigvecs = np.empty(eigvals.shape[:-1] + (2, 2), dtype=eigvals.dtype)
    # column 0: eigenvector for the smaller eigenvalue
    eigvecs[..., 0, 0] = -sin_phi
    eigvecs[..., 1, 0] = cos_phi
    # column 1: eigenvector for the larger eigenvalue
    eigvecs[..., 0, 1] = cos_phi
    eigvecs[..., 1, 1] = sin_phi
    return eigvals, eigvecs


# Minimum fraction of the ridge-to-background intensity range that must have
# been covered at an accepted fibre edge. A genuine edge lies on the intensity
# ramp between the ridge and the background; a gradient-magnitude maximum
# caused by texture inside the fibre plateau has a rise near zero. See
# FibreDetector.compute_line_width.
EDGE_MIN_RISE = 0.25


class FibreDetector:
    """
    FibreDetector class to detect fibres (as ridges) in images.
    """
    def __init__(self,
                 line_widths=[3, 5],
                 low_contrast=100,
                 high_contrast=200,
                 gamma=2.0,
                 min_len=5,
                 max_len=0,
                 dark_line=True,
                 correct_pos=False,
                 estimate_width=True,
                 extend_line=False):

        '''
        :param line_widths (int or array): range of widths of fibres to be detected.
        :param low_contrast (int): lower contrast threshold for ridge detection.
                                   Contrast is measured by the difference between the fibre pixels and background.
                                   Reduce this threshold if you see many miss-detected fibres or the fibres are not
                                   too bright compared to background pixels.
        :param high_contrast (int): high contrast threshold for ridge detection.
        :param gamma: gamma value
        :param min_len:
        :param max_len:
        :param dark_line:
        :param correct_pos:
        :param estimate_width:
        :param extend_line:
        '''

        self.line_widths = np.array([line_widths]) if np.isscalar(line_widths) else np.array(line_widths)
        self.low_contrast = low_contrast
        self.high_contrast = high_contrast
        self.min_len = min_len
        self.max_len = max_len
        self.dark_line = dark_line
        self.correct_pos = correct_pos
        self.estimate_width = estimate_width
        self.extend_line = extend_line

        self.sigmas = self.line_widths / (2 * np.sqrt(3)) + 0.5
        self.clow = self.low_contrast
        self.chigh = self.high_contrast
        if self.dark_line:
            self.clow = 255 - self.high_contrast
            self.chigh = 255 - self.low_contrast

        self.gamma = gamma
        self.image = None
        self.gray = None
        self.derivatives = None
        self.lower_thresh = None
        self.upper_thresh = None
        self.contours = None
        self.junctions = None
        self.eigvals = None
        self.eigvecs = None
        self.eigval = None
        self.gradx = None
        self.grady = None
        self.sigma_map = None
        self.normx = None
        self.normy = None
        self.posx = None
        self.posy = None
        self.ismax = None
        self.mode = LinesUtil.MODE_DARK if self.dark_line else LinesUtil.MODE_LIGHT

    def apply_filtering(self):
        height, width = self.gray.shape[:2]
        num_scales = len(self.sigmas)
        saliency = np.zeros((height, width, num_scales), dtype=float)
        orientation = np.zeros((height, width, 2, num_scales), dtype=float)
        rys = np.zeros((height, width, num_scales), dtype=float)
        rxs = np.zeros((height, width, num_scales), dtype=float)
        ryys = np.zeros((height, width, num_scales), dtype=float)
        rxys = np.zeros((height, width, num_scales), dtype=float)
        rxxs = np.zeros((height, width, num_scales), dtype=float)

        low_threshs = np.zeros((height, width, num_scales), dtype=float)
        high_threshs = np.zeros((height, width, num_scales), dtype=float)
        sigma_maps = np.zeros((height, width, num_scales), dtype=float)
        # Gradient magnitude at every scale, kept for width estimation: each
        # contour's edges are located in the gradient of ITS OWN scale rather
        # than in the per-pixel patchwork of selected scales.
        grad_scales = np.zeros((num_scales, height, width), dtype=float)

        # filtering at different scales
        gray = self.gray.astype(float)
        for scale_idx, sigma in enumerate(self.sigmas):
            ry = convolve_gauss(gray, sigma, LinesUtil.DERIV_R)
            rx = convolve_gauss(gray, sigma, LinesUtil.DERIV_C)
            ryy = convolve_gauss(gray, sigma, LinesUtil.DERIV_RR)
            rxy = convolve_gauss(gray, sigma, LinesUtil.DERIV_RC)
            rxx = convolve_gauss(gray, sigma, LinesUtil.DERIV_CC)

            # Closed-form eigendecomposition of the 2x2 Hessian per pixel.
            # Equivalent to np.linalg.eigh on a (H, W, 2, 2) tensor with
            # M[0,0]=ryy, M[0,1]=M[1,0]=rxy, M[1,1]=rxx, but ~10x faster.
            eigvals, eigvecs = _eigh_2x2_symmetric(ryy, rxy, rxx)

            # maximum absolute eigen as the saliency of lines
            idx = np.absolute(eigvals).argsort()[..., ::-1]
            eigvals_tmp = np.take_along_axis(eigvals, idx, axis=-1)
            eigvecs_tmp = np.take_along_axis(eigvecs, idx[:, :, None, :], axis=-1)

            # Polarity-aware saliency. The principal Hessian eigenvalue at a
            # ridge pixel is positive for a dark line on a bright background
            # (intensity has a local minimum) and negative for a bright line
            # on a dark background. argmax across scales picks the largest
            # value, so we must build saliency with the sign that makes a
            # genuine ridge response register as a large positive number for
            # the requested polarity. Without this, MODE_LIGHT runs select
            # the LEAST-negative saliency (the over-smoothed scale where the
            # line is least sharp) instead of the strongest response.
            sign = 1.0 if self.mode == LinesUtil.MODE_DARK else -1.0
            saliency[:, :, scale_idx] = sigma ** self.gamma * sign * eigvals_tmp[:, :, 0]
            orientation[:, :, :, scale_idx] = eigvecs_tmp[:, :, :, 0]

            # store intermediate results
            rys[..., scale_idx] = ry
            rxs[..., scale_idx] = rx
            grad_scales[scale_idx] = np.hypot(ry, rx)
            ryys[..., scale_idx] = ryy
            rxys[..., scale_idx] = rxy
            rxxs[..., scale_idx] = rxx

            # Continuous Steger ridge-saliency threshold (Steger 1996, Eq. 12,
            # scaled by 0.17 * sigma^gamma to match this module's normalised
            # saliency = sigma^gamma * eigval). The historical ImageJ port
            # used np.floor here because its response array was integer-
            # quantised; nothing in this Python pipeline is, so flooring
            # discarded sub-unit precision without producing an
            # integer-aligned threshold (the outer 0.17*sigma**gamma factor
            # turned the floored value back into a non-integer). Removing
            # the floor restores the continuous behaviour described in the
            # paper and makes detection sensitivity move smoothly with the
            # user-set Low/High Contrast values instead of in discrete jumps.
            line_width = 2 * np.sqrt(3) * (sigma - 0.5)
            _gauss = (line_width / (np.sqrt(2 * np.pi) * sigma ** 3) *
                      np.exp(-line_width ** 2 / (8 * sigma ** 2)))
            low_thresh = 0.17 * sigma ** self.gamma * self.clow * _gauss
            high_thresh = 0.17 * sigma ** self.gamma * self.chigh * _gauss
            low_threshs[..., scale_idx] = low_thresh
            high_threshs[..., scale_idx] = high_thresh
            sigma_maps[..., scale_idx] = sigma

        # get the scale index of the maximum saliency and the corresponding derivatives and thresholds
        global_max_idx = saliency.argsort()[..., -1]
        self.lower_thresh = np.squeeze(np.take_along_axis(low_threshs, global_max_idx[:, :, None], axis=-1))
        self.upper_thresh = np.squeeze(np.take_along_axis(high_threshs, global_max_idx[:, :, None], axis=-1))
        self.sigma_map = np.squeeze(np.take_along_axis(sigma_maps, global_max_idx[:, :, None], axis=-1))

        self.derivatives = np.zeros((5, height, width), dtype=float)

        self.derivatives[0, ...] = np.squeeze(np.take_along_axis(rys, global_max_idx[:, :, None], axis=-1))
        self.derivatives[1, ...] = np.squeeze(np.take_along_axis(rxs, global_max_idx[:, :, None], axis=-1))
        self.derivatives[2, ...] = np.squeeze(np.take_along_axis(ryys, global_max_idx[:, :, None], axis=-1))
        self.derivatives[3, ...] = np.squeeze(np.take_along_axis(rxys, global_max_idx[:, :, None], axis=-1))
        self.derivatives[4, ...] = np.squeeze(np.take_along_axis(rxxs, global_max_idx[:, :, None], axis=-1))

        self.grady = self.derivatives[0, ...]
        self.gradx = self.derivatives[1, ...]
        self.grad_scales = grad_scales

        self.eigvals = np.take_along_axis(saliency, global_max_idx[:, :, None], axis=-1)
        self.eigvecs = np.take_along_axis(orientation, global_max_idx[:, :, None, None], axis=-1)

    def compute_line_points(self):
        height, width = self.gray.shape[:2]
        self.ismax = np.zeros((height, width), dtype=int)
        self.eigval = np.zeros((height, width), dtype=float)
        self.normx = np.zeros((height, width), dtype=float)
        self.normy = np.zeros((height, width), dtype=float)
        self.posx = np.zeros((height, width), dtype=float)
        self.posy = np.zeros((height, width), dtype=float)

        ry = self.derivatives[0, ...]
        rx = self.derivatives[1, ...]
        ryy = self.derivatives[2, ...]
        rxy = self.derivatives[3, ...]
        rxx = self.derivatives[4, ...]

        # self.eigvals stores the polarity-aware saliency at the per-pixel
        # selected scale (apply_filtering bakes the mode sign in), so it is
        # already positive for genuine ridges in BOTH dark- and light-line
        # modes. No mode-dependent negation needed here.
        val = self.eigvals[:, :, 0]
        val_mask = val > 0.0
        self.eigval[val_mask] = val[val_mask]

        nx_ = self.eigvecs[..., 1, 0]
        ny_ = self.eigvecs[..., 0, 0]
        numerator = ry * ny_ + rx * nx_
        denominator = ryy * ny_ ** 2 + 2.0 * rxy * nx_ * ny_ + rxx * nx_ ** 2

        t = numerator / (denominator + np.finfo(float).eps)
        py_ = t * ny_
        px_ = t * nx_

        bnd_mask = (abs(py_) <= PIXEL_BOUNDARY) & (abs(px_) <= PIXEL_BOUNDARY)
        base_mask = val_mask & bnd_mask
        upper_mask = base_mask & (val >= self.upper_thresh)
        lower_mask = base_mask & (val >= self.lower_thresh) & (val < self.upper_thresh)
        self.ismax[upper_mask] = 2
        self.ismax[lower_mask] = 1

        self.normy[base_mask] = ny_[base_mask]
        self.normx[base_mask] = nx_[base_mask]
        Y, X = np.mgrid[0:height, 0:width]
        self.posy[base_mask] = Y[base_mask] + py_[base_mask]
        self.posx[base_mask] = X[base_mask] + px_[base_mask]

    def extend_lines(self, label):
        height, width = label.shape[:2]
        num_junc = len(self.junctions)
        s = 1 if self.mode == LinesUtil.MODE_DARK else -1
        length = 2.5 * self.sigma_map
        max_line = np.ceil(length * 1.2).astype(int)
        num_cont = len(self.contours)
        for idx_cont in range(num_cont):
            tmp_cont = self.contours[idx_cont]
            num_pnt = tmp_cont.num
            if num_pnt == 1 or tmp_cont.get_contour_class() == LinesUtil.ContourClass.cont_closed:
                continue

            # Check both ends of the line (it==-1: start, it==1: end).
            for it in [-1, 1]:
                if it == -1:
                    trow = tmp_cont.row
                    tcol = tmp_cont.col
                    tangle = tmp_cont.angle
                    tresp = tmp_cont.response

                    # Start point of the line.
                    if tmp_cont.get_contour_class() in [LinesUtil.ContourClass.cont_start_junc,
                                                        LinesUtil.ContourClass.cont_both_junc]:
                        continue
                    dy, dx = trow[1] - trow[0], tcol[1] - tcol[0]
                    alpha = tangle[0]
                    ny, nx = np.sin(alpha), np.cos(alpha)
                    if ny * dx - nx * dy < 0:
                        # Turn the normal by +90 degrees.
                        my, mx = -nx, ny
                    else:
                        # Turn the normal by -90 degrees.
                        my, mx = nx, -ny
                    py, px = trow[0], tcol[0]
                    response = tresp[0]
                else:
                    trow = tmp_cont.row
                    tcol = tmp_cont.col
                    tangle = tmp_cont.angle
                    tresp = tmp_cont.response

                    # End point of the line.
                    if tmp_cont.get_contour_class() in [LinesUtil.ContourClass.cont_end_junc,
                                                        LinesUtil.ContourClass.cont_both_junc]:
                        continue
                    dy, dx = trow[num_pnt - 1] - trow[num_pnt - 2], tcol[num_pnt - 1] - tcol[num_pnt - 2]
                    alpha = tangle[num_pnt - 1]
                    ny, nx = np.sin(alpha), np.cos(alpha)
                    if ny * dx - nx * dy < 0:
                        # Turn the normal by -90 degrees.
                        my, mx = nx, -ny
                    else:
                        # Turn the normal by +90 degrees.
                        my, mx = -nx, ny
                    py, px = trow[num_pnt - 1], tcol[num_pnt - 1]
                    response = tresp[num_pnt - 1]

                # Determine the current pixel and calculate the pixels on the search line.
                y, x = int(py + 0.5), int(px + 0.5)
                dy, dx = py - y, px - x
                line = bresenham(my, mx, max_line[LinesUtil.BR(y, height), LinesUtil.BC(x, width)], dy, dx)
                num_line = line.shape[0]
                exty, extx = np.zeros(num_line, dtype=float), np.zeros(num_line, dtype=float)

                # Now determine whether we can go only uphill (bright lines)
                # or downhill (dark lines) until we hit another line.
                num_add = 0
                add_ext = False
                for k in range(num_line):
                    nexty, nextx = y + line[k, 0], x + line[k, 1]
                    nextpy, nextpx, t = closest_point(py, px, my, mx, nexty, nextx)

                    # Ignore points before or less than half a pixel away from the true end point of the line.
                    if t <= 0.5:
                        continue
                    # Stop if the gradient can't be interpolated any more or if the next point lies outside the image.
                    if (nextpx < 0 or nextpy < 0 or nextpy >= height - 1 or nextpx >= width - 1 or
                            nextx < 0 or nexty < 0 or nexty >= height or nextx >= width):
                        break
                    gy, gx = interpolate_gradient_test(self.grady, self.gradx, nextpy, nextpx)

                    # Stop if we can't go uphill anymore.
                    # This is determined by the dot product of the line direction and the gradient.
                    # If it is smaller than 0 we go downhill (reverse for dark lines).
                    if s * (mx * gx + my * gy) < 0 and label[nexty, nextx] == 0:
                        break
                    # Have we hit another line?
                    if label[nexty, nextx] > 0:
                        m = label[nexty, nextx] - 1
                        # Search for the junction point on the other line.
                        dist = np.sqrt((nextpy - self.contours[m].row) ** 2 + (nextpx - self.contours[m].col) ** 2)
                        j = np.argmin(dist)

                        exty[num_add] = self.contours[m].row[j]
                        extx[num_add] = self.contours[m].col[j]
                        end_resp = self.contours[m].response[j]
                        end_angle = self.contours[m].angle[j]
                        beta = end_angle
                        if beta >= np.pi:
                            beta -= np.pi
                        diff1 = abs(beta - alpha)
                        if diff1 >= np.pi:
                            diff1 = 2.0 * np.pi - diff1
                        diff2 = abs(beta + np.pi - alpha)
                        if diff2 >= np.pi:
                            diff2 = 2.0 * np.pi - diff2
                        if diff1 < diff2:
                            end_angle = beta
                        else:
                            end_angle = beta + np.pi
                        num_add += 1
                        add_ext = True
                        break
                    else:
                        exty[num_add], extx[num_add] = nextpy, nextpx
                        num_add += 1

                if add_ext:
                    # Make room for the new points.
                    num_pnt += num_add
                    new_row = np.zeros(num_pnt, dtype=float)
                    new_col = np.zeros(num_pnt, dtype=float)
                    new_angle = np.zeros(num_pnt, dtype=float)
                    new_resp = np.zeros(num_pnt, dtype=float)

                    tmp_cont.row = new_row
                    tmp_cont.col = new_col
                    tmp_cont.angle = new_angle
                    tmp_cont.response = new_resp
                    tmp_cont.num = num_pnt
                    if it == -1:
                        tmp_cont.row[num_add:] = trow
                        tmp_cont.row[:num_add] = exty[:num_add][::-1]
                        tmp_cont.col[num_add:] = tcol
                        tmp_cont.col[:num_add] = extx[:num_add][::-1]
                        tmp_cont.angle[num_add:] = tangle
                        tmp_cont.angle[:num_add] = float(alpha)
                        tmp_cont.response[num_add:] = tresp
                        tmp_cont.response[:num_add] = float(response)
                        tmp_cont.angle[0] = end_angle
                        tmp_cont.response[0] = end_resp
                        # Adapt indices of the previously found junctions.
                        for k in range(num_junc):
                            if self.junctions[k].cont1 == idx_cont:
                                self.junctions[k].pos += num_add
                    else:
                        # Insert points at the end of the line.
                        tmp_cont.row[:num_pnt - num_add] = trow
                        tmp_cont.row[num_pnt - num_add:] = exty[:num_add]
                        tmp_cont.col[:num_pnt - num_add] = tcol
                        tmp_cont.col[num_pnt - num_add:] = extx[:num_add]
                        tmp_cont.angle[:num_pnt - num_add] = tangle
                        tmp_cont.angle[num_pnt - num_add:] = float(alpha)
                        tmp_cont.response[:num_pnt - num_add] = tresp
                        tmp_cont.response[num_pnt - num_add:] = float(response)
                        tmp_cont.angle[-1] = end_angle
                        tmp_cont.response[-1] = end_resp

                    # Add the junction point only if it is not one of the other line's endpoints.
                    if 0 < j < self.contours[m].num - 1:
                        if it == -1:
                            if tmp_cont.get_contour_class() == LinesUtil.ContourClass.cont_end_junc:
                                tmp_cont.set_contour_class(LinesUtil.ContourClass.cont_both_junc)
                            else:
                                tmp_cont.set_contour_class(LinesUtil.ContourClass.cont_start_junc)
                        else:
                            if tmp_cont.get_contour_class() == LinesUtil.ContourClass.cont_start_junc:
                                tmp_cont.set_contour_class(LinesUtil.ContourClass.cont_both_junc)
                            else:
                                tmp_cont.set_contour_class(LinesUtil.ContourClass.cont_end_junc)

                        if it == -1:
                            self.junctions.append(Junction(m, idx_cont, j, tmp_cont.row[0], tmp_cont.col[0]))
                        else:
                            self.junctions.append(
                                Junction(m, idx_cont, j, tmp_cont.row[num_pnt - 1], tmp_cont.col[num_pnt - 1]))

                        num_junc += 1

    def compute_contours(self):
        height, width = self.eigval.shape[:2]
        label = np.zeros((height, width), dtype=int)
        indx = np.zeros((height, width), dtype=int)

        num_cont, num_junc = 0, 0
        self.junctions, self.contours = [], []

        cross = []
        area = 0
        for r_idx in range(height):
            for c_idx in range(width):
                if self.ismax[r_idx, c_idx] >= 2:
                    area += 1
                    cross.append(Crossref(r_idx, c_idx, self.eigval[r_idx, c_idx], False))

        response_2d = self.eigval.reshape(height, width)
        resp_dr = convolve(response_2d, kernel_r, mode='mirror')
        resp_dc = convolve(response_2d, kernel_c, mode='mirror')
        resp_dd = convolve(response_2d, kernel_d, mode='mirror')
        resp_drr = convolve(response_2d, kernel_rr, mode='mirror')
        resp_drc = convolve(response_2d, kernel_rc, mode='mirror')
        resp_dcc = convolve(response_2d, kernel_cc, mode='mirror')

        # Sorting cross list in ascending order by value
        cross.sort()

        # Updating indx based on the sorted cross list
        for ci, cref in enumerate(cross):
            indx[cref.y, cref.x] = ci + 1

        indx_max = 0
        while True:
            cls = LinesUtil.ContourClass.cont_no_junc
            while indx_max < area and cross[indx_max].done:
                indx_max += 1

            if indx_max == area:
                break

            max_val = cross[indx_max].value
            maxy, maxx = cross[indx_max].y, cross[indx_max].x
            if max_val == 0.0:
                break

            # Initialize line data
            row, col, angle, resp = [], [], [], []

            # Add starting point to the line.
            num_pnt = 0
            label[maxy, maxx] = num_cont + 1
            if indx[maxy, maxx] != 0:
                cross[indx[maxy, maxx] - 1].done = True

            # Select line direction
            row.append(maxy)
            col.append(maxx)
            nx = -self.normx[maxy, maxx]
            ny = self.normy[maxy, maxx]
            alpha = normalize_to_half_circle(np.arctan2(ny, nx))
            octant = int(np.floor(4.0 / np.pi * alpha + 0.5)) % 4

            ''' * Select normal to the line. The normal points to the right of the line as the
                * line is traversed from 0 to num-1. Since the points are sorted in reverse
                * order before the second iteration, the first beta actually has to point to
                * the left of the line!
            '''

            beta = alpha + np.pi / 2.0
            if beta >= 2.0 * np.pi:
                beta -= 2.0 * np.pi
            angle.append(beta)
            yy = self.posy[maxy, maxx] - maxy
            xx = self.posx[maxy, maxx] - maxx
            interpolated_response = (resp_dd[maxy, maxx] + yy * resp_dr[maxy, maxx] + xx * resp_dc[maxy, maxx] +
                                     yy ** 2 * resp_drr[maxy, maxx] + xx * yy * resp_drc[maxy, maxx] +
                                     xx ** 2 * resp_dcc[maxy, maxx])
            resp.append(interpolated_response)
            num_pnt += 1

            # Mark double responses as processed.
            for ni in range(2):
                nexty = maxy + cleartab[octant][ni][0]
                nextx = maxx + cleartab[octant][ni][1]
                if nexty < 0 or nexty >= height or nextx < 0 or nextx >= width:
                    continue
                if self.ismax[nexty, nextx] > 0:
                    nx = -self.normx[nexty, nextx]
                    ny = self.normy[nexty, nextx]
                    nextalpha = normalize_to_half_circle(np.arctan2(ny, nx))
                    diff = abs(alpha - nextalpha)
                    if diff >= np.pi / 2.0:
                        diff = np.pi - diff
                    if diff < MAX_ANGLE_DIFFERENCE:
                        label[nexty, nextx] = num_cont + 1
                        if indx[nexty, nextx] != 0:
                            cross[indx[nexty, nextx] - 1].done = True

            for it in range(1, 3):
                y, x = maxy, maxx
                ny, nx = self.normy[y, x], -self.normx[y, x]

                alpha = normalize_to_half_circle(np.arctan2(ny, nx))
                last_octant = int(np.floor(4.0 / np.pi * alpha + 0.5)) % 4 if it == 1 else \
                    int(np.floor(4.0 / np.pi * alpha + 0.5)) % 4 + 4
                last_beta = alpha + np.pi / 2.0
                if last_beta >= 2.0 * np.pi:
                    last_beta -= 2.0 * np.pi

                if it == 2:
                    # Sort the points found in the first iteration in reverse.
                    row.reverse()
                    col.reverse()
                    angle.reverse()
                    resp.reverse()

                while True:
                    ny, nx = self.normy[y, x], -self.normx[y, x]
                    py, px = self.posy[y, x], self.posx[y, x]

                    # Orient line direction with respect to the last line direction
                    alpha = normalize_to_half_circle(np.arctan2(ny, nx))
                    octant = int(np.floor(4.0 / np.pi * alpha + 0.5)) % 4

                    if octant == 0 and 3 <= last_octant <= 5:
                        octant = 4
                    elif octant == 1 and 4 <= last_octant <= 6:
                        octant = 5
                    elif octant == 2 and 4 <= last_octant <= 7:
                        octant = 6
                    elif octant == 3 and (last_octant == 0 or last_octant >= 6):
                        octant = 7
                    last_octant = octant

                    # Determine appropriate neighbor
                    nextismax = False
                    nexti = 1
                    mindiff = float('inf')
                    for ti in range(3):
                        nexty, nextx = y + dirtab[octant][ti][0], x + dirtab[octant][ti][1]
                        if nexty < 0 or nexty >= height or nextx < 0 or nextx >= width:
                            continue
                        if self.ismax[nexty, nextx] == 0:
                            continue
                        nextpy, nextpx = self.posy[nexty, nextx], self.posx[nexty, nextx]
                        dy, dx = nextpy - py, nextpx - px
                        dist = np.sqrt(dx ** 2 + dy ** 2)
                        ny, nx = self.normy[nexty, nextx], -self.normx[nexty, nextx]
                        nextalpha = normalize_to_half_circle(np.arctan2(ny, nx))
                        diff = abs(alpha - nextalpha)
                        if diff >= np.pi / 2.0:
                            diff = np.pi - diff
                        diff = dist + diff
                        if diff < mindiff:
                            mindiff = diff
                            nexti = ti
                        if not (self.ismax[nexty, nextx] == 0):
                            nextismax = True

                    # Mark double responses as processed
                    for ni in range(2):
                        nexty, nextx = y + cleartab[octant][ni][0], x + cleartab[octant][ni][1]
                        if nexty < 0 or nexty >= height or nextx < 0 or nextx >= width:
                            continue
                        if self.ismax[nexty, nextx] > 0:
                            ny, nx = self.normy[nexty, nextx], -self.normx[nexty, nextx]
                            nextalpha = normalize_to_half_circle(np.arctan2(ny, nx))
                            diff = abs(alpha - nextalpha)
                            if diff >= np.pi / 2.0:
                                diff = np.pi - diff
                            if diff < MAX_ANGLE_DIFFERENCE:
                                label[nexty, nextx] = num_cont + 1
                                if not (indx[nexty, nextx] == 0):
                                    cross[indx[nexty, nextx] - 1].done = True

                    # Have we found the end of the line?
                    if not nextismax:
                        break  # Exit the loop if the end of the line is found

                    # Add the neighbor to the line if not at the end
                    y += dirtab[octant][nexti][0]
                    x += dirtab[octant][nexti][1]

                    row.append(self.posy[y, x])
                    col.append(self.posx[y, x])
                    # Orient normal to the line direction with respect to the last normal
                    ny, nx = self.normy[y, x], self.normx[y, x]

                    beta = normalize_to_half_circle(np.arctan2(ny, nx))
                    diff1 = min(abs(beta - last_beta), 2.0 * np.pi - abs(beta - last_beta))
                    alt_beta = (beta + np.pi) % (2.0 * np.pi)  # Normalize alternative beta
                    diff2 = min(abs(alt_beta - last_beta), 2.0 * np.pi - abs(alt_beta - last_beta))
                    # Choose the angle with the smallest difference and update
                    chosen_beta = beta if diff1 < diff2 else alt_beta
                    angle.append(chosen_beta)
                    last_beta = chosen_beta

                    yy, xx = self.posy[y, x] - maxy, self.posx[y, x] - maxx
                    interpolated_response = (resp_dd[y, x] + yy * resp_dr[y, x] + xx * resp_dc[y, x] +
                                             yy ** 2 * resp_drr[y, x] + xx * yy * resp_drc[y, x] +
                                             xx ** 2 * resp_dcc[y, x])
                    resp.append(interpolated_response)
                    num_pnt += 1

                    # If the appropriate neighbor is already processed a junction point is found
                    if label[y, x] > 0:
                        k = label[y, x] - 1
                        if k == num_cont:
                            # Line intersects itself
                            for j in range(num_pnt - 1):
                                if row[j] == self.posy[y, x] and col[j] == self.posx[y, x]:
                                    if j == 0:
                                        # Contour is closed
                                        cls = LinesUtil.ContourClass.cont_closed
                                        row.reverse()
                                        col.reverse()
                                        angle.reverse()
                                        resp.reverse()
                                        it = 2
                                    else:
                                        # Determine contour class
                                        if it == 2:
                                            if cls == LinesUtil.ContourClass.cont_start_junc:
                                                cls = LinesUtil.ContourClass.cont_both_junc
                                            else:
                                                cls = LinesUtil.ContourClass.cont_end_junc
                                            # Index j is correct
                                            self.junctions.append(
                                                Junction(num_cont, num_cont, j, self.posy[y, x], self.posx[y, x]))
                                            num_junc += 1
                                        else:
                                            cls = LinesUtil.ContourClass.cont_start_junc
                                            # Index num_pnt-1-j is correct since the line will be sorted in reverse
                                            self.junctions.append(Junction(num_cont, num_cont,
                                                                      num_pnt - 1 - j, self.posy[y, x],
                                                                      self.posx[y, x]))
                                            num_junc += 1
                                    break
                            j = -1
                        else:
                            for j in range(self.contours[k].num):
                                if self.contours[k].row[j] == self.posy[y, x] and self.contours[k].col[j] == self.posx[y, x]:
                                    break
                            if j == self.contours[k].num:
                                # No point found on the other line, a double response occurred
                                dist = np.sqrt(
                                    (self.posy[y, x] - self.contours[k].row) ** 2 + (self.posx[y, x] - self.contours[k].col) ** 2)
                                j = np.argmin(dist)
                                row.append(self.contours[k].row[j])
                                col.append(self.contours[k].col[j])
                                beta = self.contours[k].angle[j]
                                if beta >= np.pi:
                                    beta -= np.pi
                                diff1 = abs(beta - last_beta)
                                if diff1 >= np.pi:
                                    diff1 = 2.0 * np.pi - diff1
                                diff2 = abs(beta + np.pi - last_beta)
                                if diff2 >= np.pi:
                                    diff2 = 2.0 * np.pi - diff2
                                angle.append(beta) if diff1 < diff2 else angle.append(beta + np.pi)
                                resp.append(self.contours[k].response[j])
                                num_pnt += 1
                        if 0 < j < self.contours[k].num - 1:
                            # Determine contour class
                            if it == 1:
                                cls = LinesUtil.ContourClass.cont_start_junc
                            elif cls == LinesUtil.ContourClass.cont_start_junc:
                                cls = LinesUtil.ContourClass.cont_both_junc
                            else:
                                cls = LinesUtil.ContourClass.cont_end_junc

                            # Add the new junction
                            self.junctions.append(Junction(k, num_cont, j, row[num_pnt - 1], col[num_pnt - 1]))
                            num_junc += 1
                        break

                    label[y, x] = num_cont + 1
                    if indx[y, x] != 0:
                        cross[indx[y, x] - 1].done = True

            if num_pnt > 1:
                # Create a new Line object and copy the current line's attributes
                new_line = Line()
                new_line.row = np.array(row)
                new_line.col = np.array(col)
                new_line.angle = np.array(angle)
                new_line.response = np.array(resp)
                new_line.width_r = None
                new_line.width_l = None
                new_line.asymmetry = None
                new_line.intensity = None
                new_line.num = num_pnt
                new_line.set_contour_class(cls)

                # Add the new line to the list of contours
                self.contours.append(new_line)
                num_cont += 1
            else:
                # Delete the point from the label image; using maxx and maxy as coordinates in the label image
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        if label[LinesUtil.BR(maxy + i, height), LinesUtil.BC(maxx + j, width)] == num_cont + 1:
                            label[LinesUtil.BR(maxy + i, height), LinesUtil.BC(maxx + j, width)] = 0

        if self.extend_line:
            self.extend_lines(label)

        # Adjust angles to point to the right of the line
        for i in range(num_cont):
            tmp_cont = self.contours[i]
            num_pnt = tmp_cont.num
            if num_pnt > 1:
                k = (num_pnt - 1) // 2
                dy, dx = tmp_cont.row[k + 1] - tmp_cont.row[k], tmp_cont.col[k + 1] - tmp_cont.col[k]
                ny, nx = np.sin(tmp_cont.angle[k]), np.cos(tmp_cont.angle[k])

                # If angles point to the left of the line, they have to be adapted
                if ny * dx - nx * dy < 0:
                    tmp_cont.angle = np.array([(ang + np.pi) % (2 * np.pi) for ang in tmp_cont.angle])

    def compute_line_width(self):
        height, width = self.grady.shape[:2]
        # Scan (and cap) using the LARGEST detection scale, not the per-pixel
        # sigma_map. The per-scale saliency argmax is noisy and frequently
        # picks the smallest scale on a wide, flat-topped fibre. With that
        # sigma the scan reaches only ~3*sigma from the ridge and the true
        # edge is out of range, so the only candidates are weak noise ripples
        # inside the fibre plateau and the boundary collapses onto the centre
        # line. The user's Max Line Width is the intended upper bound on
        # fibre width, so the largest sigma is the right scan length.
        sigma_max = float(np.max(self.sigmas))
        max_length = int(np.ceil(2.5 * sigma_max * 1.2))
        max_half_width = 2.5 * sigma_max
        # Lightly smoothed intensity for the edge "rise" test below. For a
        # dark line the background is brighter than the ridge, so flip the
        # sign in MODE_LIGHT and treat both cases as "intensity increases
        # away from the ridge".
        gray_s = gaussian_filter(self.gray.astype(float), 1.0)
        if self.mode == LinesUtil.MODE_LIGHT:
            gray_s = -gray_s

        # One scale per contour: the median of the per-pixel selected scale
        # over the contour's points. The per-pixel argmax is noisy (it can
        # flip between the smallest and largest scale from point to point on
        # textured tissue); the median gives a stable, fibre-matched scale.
        # Contours are grouped by scale so the edge-detection arrays are
        # computed once per scale actually used.
        scale_of_cont = np.zeros(len(self.contours), dtype=int)
        for i, cont in enumerate(self.contours):
            rr = np.clip(np.round(cont.row).astype(int), 0, height - 1)
            cc = np.clip(np.round(cont.col).astype(int), 0, width - 1)
            sig_med = np.median(self.sigma_map[rr, cc])
            scale_of_cont[i] = int(np.argmin(np.abs(self.sigmas - sig_med)))

        for scale_idx in np.unique(scale_of_cont):
            edge = self._edge_arrays(self.grad_scales[scale_idx])
            sigma_c = float(self.sigmas[scale_idx])
            for i in np.flatnonzero(scale_of_cont == scale_idx):
                self._measure_contour_width(self.contours[i], edge, sigma_c,
                                            gray_s, max_length, max_half_width)

    @staticmethod
    def _edge_arrays(grad):
        """Edge-detection arrays (Steger) from one gradient-magnitude image."""
        grad_dr = convolve(grad, kernel_r, mode='mirror')
        grad_dc = convolve(grad, kernel_c, mode='mirror')
        grad_dd = convolve(grad, kernel_d, mode='mirror')
        grad_drr = convolve(grad, kernel_rr, mode='mirror')
        grad_drc = convolve(grad, kernel_rc, mode='mirror')
        grad_dcc = convolve(grad, kernel_cc, mode='mirror')

        # Closed-form eigendecomposition of the 2x2 gradient-magnitude Hessian
        # per pixel; same shapes and contract as np.linalg.eigh on a
        # (H, W, 2, 2) tensor, faster.
        eigvals, eigvecs = _eigh_2x2_symmetric(2 * grad_drr, grad_drc, 2 * grad_dcc)
        idx = np.absolute(eigvals).argsort()[..., ::-1]
        eigvals = np.take_along_axis(eigvals, idx, axis=-1)
        eigvecs = np.take_along_axis(eigvecs, idx[:, :, None, :], axis=-1)

        bb = grad_dr * eigvecs[:, :, 0, 0] + grad_dc * eigvecs[:, :, 1, 0]
        aa = 2.0 * (grad_drr * eigvecs[:, :, 0, 0] ** 2 + grad_drc * eigvecs[:, :, 0, 0] * eigvecs[:, :, 1, 0] +
                    grad_dcc * eigvecs[:, :, 1, 0] ** 2)
        tt = bb / (aa + np.finfo(float).eps)
        pp1, pp2 = tt * eigvecs[:, :, 0, 0], tt * eigvecs[:, :, 1, 0]
        grad_rl = (grad_dd + pp1 * grad_dr + pp2 * grad_dc +
                   pp1 * pp1 * grad_drr + pp1 * pp2 * grad_drc + pp2 * pp2 * grad_dcc)
        return eigvals, eigvecs, pp1, pp2, grad_rl

    def _measure_contour_width(self, cont, edge, sigma_c, gray_s, max_length, max_half_width):
        eigvals, eigvecs, pp1, pp2, grad_rl = edge
        height, width = gray_s.shape[:2]
        regularize_normals(cont)
        num_points = cont.num
        width_l = np.zeros(num_points, dtype=float)
        width_r = np.zeros(num_points, dtype=float)
        grad_l = np.zeros(num_points, dtype=float)
        grad_r = np.zeros(num_points, dtype=float)
        pos_x = np.zeros(num_points, dtype=float)
        pos_y = np.zeros(num_points, dtype=float)

        for j in range(num_points):
            py, px = cont.row[j], cont.col[j]
            pos_y[j], pos_x[j] = py, px
            r, c = LinesUtil.BR(round(py), height), LinesUtil.BC(round(px), width)
            ny, nx = np.sin(cont.angle[j]), np.cos(cont.angle[j])

            line = bresenham(ny, nx, max_length)
            num_line = line.shape[0]
            width_r[j] = width_l[j] = 0

            for direct in [-1, 1]:
                # Local background on this side: the brightest smoothed
                # intensity along the scan (for a dark line). Candidates
                # are accepted only if the intensity at the candidate has
                # risen at least EDGE_MIN_RISE of the way from the ridge
                # to that background. The reference implementation accepts
                # the first gradient-magnitude maximum of any kind, but
                # on textured tissue a maximum one or two pixels from the
                # ridge, still inside the fibre plateau, is common and
                # collapses the boundary onto the centre line.
                scan_y = np.clip(r + direct * line[:, 0], 0, height - 1)
                scan_x = np.clip(c + direct * line[:, 1], 0, width - 1)
                scan_i = gray_s[scan_y, scan_x]
                i_ridge = scan_i[0]
                i_range = scan_i.max() - i_ridge
                if i_range <= 0.0:
                    continue
                for k in range(num_line):
                    y, x = scan_y[k], scan_x[k]
                    val = -eigvals[y, x, 0]
                    if val > 0.0:
                        p1, p2 = pp1[y, x], pp2[y, x]

                        if abs(p1) <= 0.5 and abs(p2) <= 0.5:
                            # Require the |grad| Hessian's principal
                            # eigenvector at this pixel to be roughly
                            # parallel to the line normal (ny, nx). At a
                            # genuine line edge the gradient is
                            # perpendicular to the line, so the eigenvector
                            # of the largest |grad|-Hessian eigenvalue
                            # aligns with the line normal. A scan that has
                            # walked into a neighbouring fibre tends to
                            # land on that fibre's edge, whose eigenvector
                            # points along OUR line direction (~90 deg off
                            # our normal). Rejecting candidates more than
                            # 60 deg off the normal kills those "spurs"
                            # while still tolerating moderate curvature and
                            # smoothing-induced misalignment.
                            ev_dot = eigvecs[y, x, 0, 0] * ny + eigvecs[y, x, 1, 0] * nx
                            if abs(ev_dot) < 0.5:
                                continue
                            # Edge must sit on the ridge-to-background
                            # intensity ramp, not inside the plateau.
                            if (scan_i[k] - i_ridge) < EDGE_MIN_RISE * i_range:
                                continue
                            t = ny * (py - (r + direct * line[k, 0] + p1)) + nx * (px - (c + direct * line[k, 1] + p2))
                            # Cap by the largest detection scale: a half-
                            # width beyond ~2.5*sigma_max cannot belong to
                            # a line of the requested width range, so it
                            # is almost certainly a neighbouring fibre's
                            # edge. Drop it; fill_gaps interpolates from
                            # neighbouring good points.
                            if abs(t) > max_half_width:
                                continue
                            if direct == 1:
                                grad_r[j] = grad_rl[y, x]
                                width_r[j] = abs(t)
                            else:
                                grad_l[j] = grad_rl[y, x]
                                width_l[j] = abs(t)
                            break
        fix_locations(cont, width_l, width_r, grad_l, grad_r, pos_y, pos_x,
                      sigma_c, self.correct_pos, self.mode)

    def prune_contours(self):
        if self.min_len <= 0:
            return

        id_remove = set()
        conts = []
        for i, cont in enumerate(self.contours):
            cont_len = cont.estimate_length()
            if cont_len < self.min_len or (0 < self.max_len < cont_len):
                id_remove.add(i)
            else:
                conts.append(cont)

        juncs = [j for j in self.junctions
                 if j.cont1 not in id_remove and j.cont2 not in id_remove]

        self.contours = conts
        self.junctions = juncs

    def detect_lines(self, image):
        self.image = iio.imread(image) if isinstance(image, str) else image

        # Normalize non-uint8 inputs (e.g. 16-bit microscopy, float TIFFs) to
        # uint8 using a percentile range. Min/max scaling is dominated by
        # outliers (hot pixels, dead pixels, saturated reflections), which
        # silently re-scales every image differently and breaks the meaning
        # of the user-facing Low/High Contrast thresholds across acquisitions.
        # Using the (0.5, 99.5) percentiles caps the influence of a small
        # fraction of extreme pixels at each end while preserving the bulk of
        # the dynamic range.
        if self.image.dtype != np.uint8:
            img_arr = self.image.astype(np.float64)
            lo, hi = np.percentile(img_arr, (0.5, 99.5))
            dyn = hi - lo
            if dyn < np.finfo(float).eps:
                Log.logger.warning(
                    f"Detector input has zero dynamic range "
                    f"(p0.5={lo:.3g}, p99.5={hi:.3g}); producing a blank image."
                )
                self.image = np.zeros(img_arr.shape, dtype=np.uint8)
            else:
                if dyn < 50.0 * (np.iinfo(self.image.dtype).max / 255.0
                                 if np.issubdtype(self.image.dtype, np.integer)
                                 else 1.0):
                    Log.logger.warning(
                        f"Detector input has very low dynamic range "
                        f"(p99.5 - p0.5 = {dyn:.3g} in original units); "
                        f"contrast thresholds may not behave as expected."
                    )
                self.image = np.clip(
                    (img_arr - lo) / dyn * 255.0, 0, 255
                ).astype(np.uint8)

        # Convert to grayscale
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY) if self.image.ndim == 3 else self.image

        self.apply_filtering()
        self.compute_line_points()
        self.compute_contours()
        self.compute_line_width()
        self.prune_contours()

    def get_results(self):
        all_contour_points, all_width_left, all_width_right = [], [], []
        int_width_img = np.zeros(self.image.shape[:2], dtype=np.uint8)
        for cont in self.contours:
            num_points = cont.num
            # last_w_r, last_w_l = 0, 0
            contour_points, width_left, width_right = [], [], []

            for j in range(num_points):
                px, py = cont.col[j], cont.row[j]
                nx, ny = np.cos(cont.angle[j]), np.sin(cont.angle[j])
                contour_points.append([round(px), round(py)])

                px_r, py_r = px + cont.width_r[j] * nx, py + cont.width_r[j] * ny
                px_l, py_l = px - cont.width_l[j] * nx, py - cont.width_l[j] * ny
                int_width_img[int(py), int(px)] = int(cont.width_r[j] + cont.width_l[j])

                # if last_w_r > 0 and cont.width_r[j] > 0:
                width_right.append([round(px_r), round(py_r)])

                # if last_w_l > 0 and cont.width_l[j] > 0:
                width_left.append([round(px_l), round(py_l)])

                # last_w_r, last_w_l = cont.width_r[j], cont.width_l[j]

            all_contour_points.append(np.array(contour_points))
            all_width_right.append(np.array(width_right))
            all_width_left.append(np.array(width_left))

        result_img = self.image.copy() if self.image.ndim > 2 else np.repeat(self.image[:, :, None], 3, axis=2)
        contour_img = cv2.polylines(result_img, all_contour_points, False, (255, 255, 0))

        width_img = contour_img.copy()
        width_img = cv2.polylines(width_img, all_width_right, False, (0, 255, 0))
        width_img = cv2.polylines(width_img, all_width_left, False, (0, 255, 0))

        height, width = self.image.shape[:2]
        binary_contours = np.ones((height, width), dtype=np.uint8) * 255

        for contour_points in all_contour_points:
            for points in contour_points:
                binary_contours[min([points[1], height - 1]), min([points[0], width - 1])] = 0

        binary_widths = np.ones((height, width), dtype=np.uint8) * 255
        for width_left, width_right in zip(all_width_left, all_width_right):
            poly_points = np.concatenate((width_left, width_right[::-1, :]), axis=0)
            mask = ski.draw.polygon2mask((height, width), poly_points[:, [1, 0]])
            binary_widths[mask] = 0

        return contour_img, width_img, binary_contours, binary_widths, int_width_img

    def save_results(self, save_dir=None, make_binary=True, draw_junc=False, draw_width=False):
        all_contour_points, all_width_left, all_width_right = [], [], []
        for cont in self.contours:
            num_points = cont.num
            # last_w_r, last_w_l = 0, 0
            contour_points, width_left, width_right = [], [], []

            for j in range(num_points):
                px, py = cont.col[j], cont.row[j]
                nx, ny = np.cos(cont.angle[j]), np.sin(cont.angle[j])
                contour_points.append([round(px), round(py)])

                if draw_width:
                    px_r, py_r = px + cont.width_r[j] * nx, py + cont.width_r[j] * ny
                    px_l, py_l = px - cont.width_l[j] * nx, py - cont.width_l[j] * ny

                    # if last_w_r > 0 and cont.width_r[j] > 0:
                    width_right.append([round(px_r), round(py_r)])

                    # if last_w_l > 0 and cont.width_l[j] > 0:
                    width_left.append([round(px_l), round(py_l)])

                    # last_w_r, last_w_l = cont.width_r[j], cont.width_l[j]

            all_contour_points.append(np.array(contour_points))
            if draw_width:
                all_width_right.append(np.array(width_right))
                all_width_left.append(np.array(width_left))

        if save_dir is None:
            save_dir = os.getcwd()

        result_img = self.image.copy() if self.image.ndim > 2 else np.repeat(self.image[:, :, None], 3, axis=2)

        img = cv2.polylines(result_img, all_contour_points, False, (255, 0, 0))
        iio.imwrite(os.path.join(save_dir, "contours.png"), img)

        if draw_width:
            img = cv2.polylines(img, all_width_right, False, (0, 255, 0))
            img = cv2.polylines(img, all_width_left, False, (0, 255, 0))
            iio.imwrite(os.path.join(save_dir, "contours_widths.png"), img)

        if draw_junc:
            for junc in self.junctions:
                img = cv2.circle(img, (round(junc.x), round(junc.y)), 2, (0, 255, 255), -1)
            iio.imwrite(os.path.join(save_dir, "contours_widths_junctions.png"), img)

        if make_binary:
            height, width = self.image.shape[:2]
            binary_contours = np.ones((height, width), dtype=np.uint8) * 255

            for contour_points in all_contour_points:
                for points in contour_points:
                    binary_contours[min([points[1], height-1]), min([points[0], width-1])] = 0
            iio.imwrite(os.path.join(save_dir, "binary_contours.png"), binary_contours)

            binary_widths = np.ones((height, width), dtype=np.uint8) * 255
            for width_left, width_right in zip(all_width_left, all_width_right):
                poly_points = np.concatenate((width_left, width_right[::-1, :]), axis=0)
                mask = ski.draw.polygon2mask((height, width), poly_points[:, [1, 0]])
                binary_widths[mask] = 0
            iio.imwrite(os.path.join(save_dir, "binary_widths.png"), binary_widths)

    def show_results(self):
        all_contour_points, all_width_left, all_width_right = [], [], []
        height, width = self.image.shape[:2]
        for cont in self.contours:
            num_points = cont.num
            last_w_r, last_w_l = 0, 0
            contour_points, width_left, width_right = [], [], []

            for j in range(num_points):
                px, py = cont.col[j], cont.row[j]
                nx, ny = np.cos(cont.angle[j]), np.sin(cont.angle[j])
                contour_points.append([LinesUtil.BC(round(px), width), LinesUtil.BR(round(py), height)])

                px_r, py_r = px + cont.width_r[j] * nx, py + cont.width_r[j] * ny
                px_l, py_l = px - cont.width_l[j] * nx, py - cont.width_l[j] * ny

                if last_w_r > 0 and cont.width_r[j] > 0:
                    width_right.append([round(px_r), round(py_r)])

                if last_w_l > 0 and cont.width_l[j] > 0:
                    width_left.append([round(px_l), round(py_l)])

                last_w_r, last_w_l = cont.width_r[j], cont.width_l[j]

            all_contour_points.append(np.array(contour_points))
            all_width_right.append(np.array(width_right))
            all_width_left.append(np.array(width_left))

        result_img = self.image.copy() if self.image.ndim > 2 else np.repeat(self.image[:, :, None], 3, axis=2)
        img_contours = cv2.polylines(result_img, all_contour_points, False, (255, 0, 0))
        img_cont_width = cv2.polylines(img_contours.copy(), all_width_right, False, (0, 255, 0))
        img_cont_width = cv2.polylines(img_cont_width, all_width_left, False, (0, 255, 0))

        height, width = self.image.shape[:2]
        binary_contours = np.ones((height, width), dtype=np.uint8) * 255

        for contour_points in all_contour_points:
            for points in contour_points:
                binary_contours[min([points[1], height - 1]), min([points[0], width - 1])] = 0

        binary_widths = np.ones((height, width), dtype=np.uint8) * 255
        for width_left, width_right in zip(all_width_left, all_width_right):
            if width_left.size == 0 or width_right.size == 0:
                continue
            poly_points = np.concatenate((width_left, width_right[::-1, :]), axis=0)
            mask = ski.draw.polygon2mask((height, width), poly_points[:, [1, 0]])
            binary_widths[mask] = 0

        fig, axes = plt.subplots(2, 2, figsize=(16, 16))
        axes[0, 0].imshow(img_contours)
        axes[0, 0].set_title("contours")
        axes[0, 1].imshow(img_cont_width)
        axes[0, 1].set_title("contours and widths")
        axes[1, 0].imshow(binary_contours, cmap='gray')
        axes[1, 0].set_title("binary contours")
        axes[1, 1].imshow(binary_widths, cmap='gray')
        axes[1, 1].set_title("binary widths")
        plt.show()



