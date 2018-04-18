from __future__ import print_function
import numpy as np
from numpy import linalg as LA
from scipy.signal import convolve
from scipy import ndimage as ndi
from sklearn import cluster
from skimage import morphology as _morph
from . import kernels
from .operators import convolve2d, laplacian as _laplacian
from .utils import zero_crossings, normalize
from ._dictlearn import hessian


class Index(object):
    def __init__(self, shape):
        self.shape = shape
        n = len(self.shape)

        if n not in [2, 3, 4]:
            raise ValueError('Invalid dimensions')

    def __call__(self, i, axis=-1):
        n = len(self.shape)
        return getattr(self, '_%dd' % n)(i, axis)

    @staticmethod
    def _2d(i, axis):
        if axis == 0:
            return np.index_exp[i, :]
        else:
            return np.index_exp[:, i]

    @staticmethod
    def _3d(i, axis):
        if axis == 0:
            return np.index_exp[i, :, :]
        elif axis == 1:
            return np.index_exp[:, i, :]
        else:
            return np.index_exp[:, :, i]

    @staticmethod
    def _4d(i, axis):
        if axis == 0:
            return np.index_exp[i, :, :, :]
        elif axis == 1:
            return np.index_exp[:, i, :, :]
        elif axis == 2:
            return np.index_exp[:, :, i, :]
        else:
            return np.index_exp[:, :, :, i]


def log_scales(scale_min, scale_max, n_scales):
    """
    """
    stepsize = max((np.log(scale_max) - np.log(scale_min)) / (n_scales - 1),
                   1e-6)
    scales = np.exp(scale_min + np.arange(n_scales) * stepsize)
    return scale_min / scales.min() * scales


def multiscale_tophat(volume, selem, scales=tuple(range(1, 12, 2))):
    """
    """
    if isinstance(scales, (int, float)):
        scales = (scales,)

    responses = np.zeros(volume.shape + (len(scales),))
    index = Index(volume.shape)

    for i, scale in enumerate(scales):
        responses[index(i)] = _morph.white_tophat(volume, selem)

    return np.max(responses, axis=3)


def smallest_cluster(features, n_clusters, verbose=False):
    """

    Extract the smallest cluster of samples for the data 'features'.

    See examples/vessel_enhancement.py

    :param features:
        array of features, shape (n_samples, n_features)

    :param n_clusters:
        Number of features

    :param verbose:

    :return:
        Prediction vector of shape (n_samples,) where prediction[i] == True
        if features i belongs to the smallest cluster and
        prediction[i] == False if belongs to any other cluster
    """
    k_means = cluster.KMeans(n_clusters=n_clusters, verbose=verbose)
    labels = k_means.fit_predict(features)
    unique = np.unique(labels)
    count = {i: labels[labels == i].size for i in unique}
    smallest = min(count, key=lambda k: count[k])
    return labels == smallest


def line(image, kernel_size, sigma):
    """
        Carsten Steger, An Unbiased Detector of Curvilinear Structures
         For now this just finds the edges corresponding to size sigma

    :param image:
        Source image for detecting lines

    :param kernel_size:
        Size of filter kernel, has  to be odd

    :param sigma:
        Variance of gaussian, eqv to width of the lines to detect

    :return:
        lines, same shapes as image. 1 denotes a line point, 0 not a line
    """
    height, width = image.shape
    kerns = line_kernels(kernel_size, sigma)

    rx = convolve2d(image, kerns[0], boundary='symm')
    ry = convolve2d(image, kerns[1], boundary='symm')
    rxx = convolve2d(image, kerns[2], boundary='symm')
    rxy = convolve2d(image, kerns[3], boundary='symm')
    ryy = convolve2d(image, kerns[4], boundary='symm')

    lines = np.zeros_like(image)

    for x in range(height):
        for y in range(width):
            hessian = np.array([[rxx[x, y], rxy[x, y]],
                                [rxy[x, y], ryy[x, y]]])

            vals, vecs = LA.eig(hessian)
            i = np.argmax(np.abs(vals))
            val = vals[i]

            n = vecs[:, i]
            nx, ny = n

            t = -(rx[x, y] * nx + ry[x, y] * ny)
            t /= rxx[x, y] * nx ** 2 + 2 * rxy[x, y] * nx * ny + ryy[x, y] * ny ** 2
            px, py = t * n

            if -0.5 <= px <= 0.5 and -0.5 <= py <= 0.5:
                lines[x, y] = 1

            # if (px < -0.5 or px > 0.5) and (py < -0.5 or py > 0.5):
            #    lines[x, y] = 1

    return lines


def frangi_filter(volume, size=3, alpha=5.0, beta=5.0, c=10.0, gamma=1,
                  scale_min=1.0, scale_max=5.0, n_scales=10):
    scales = log_scales(scale_min, scale_max, n_scales)
    x, y, z = volume.shape
    responses = np.zeros((n_scales, x, y, z), dtype=np.float64)

    for i in range(n_scales):
        scale = scales[i]
        sigma = scale
        t = 4.0
        ddx = ndi.gaussian_filter1d(volume * sigma ** gamma, sigma=sigma, order=2,
                                    axis=0, truncate=t)
        ddy = ndi.gaussian_filter1d(volume * sigma ** gamma, sigma=sigma, order=2,
                                    axis=1, truncate=t)
        ddz = ndi.gaussian_filter1d(volume * sigma ** gamma, sigma=sigma, order=2,
                                    axis=2, truncate=t)
        dx = ndi.gaussian_filter1d(volume * sigma ** gamma, sigma=sigma, order=1,
                                   axis=0, truncate=t)
        dy = ndi.gaussian_filter1d(volume * sigma ** gamma, sigma=sigma, order=1,
                                   axis=1, truncate=t)
        dxdy = ndi.gaussian_filter1d(dx * sigma ** gamma, sigma=sigma, order=1,
                                     axis=1, truncate=t)
        dxdz = ndi.gaussian_filter1d(dx * sigma ** gamma, sigma=sigma, order=1,
                                     axis=2, truncate=t)
        dydz = ndi.gaussian_filter1d(dy * sigma ** gamma, sigma=sigma, order=1,
                                     axis=2, truncate=t)

        r = hessian.vesselness_single_scale(alpha, beta, c, x, y, z, ddx, ddy, ddz, dxdy, dxdz, dydz)
        responses[i] = r

    return responses


def tubular_candidates(volume, size=5, sigma=1.5):
    """
        Pre-selection of possible tubular points
    """
    x, y, z = volume.shape

    kernel_yy, kernel_zz, kernel_xx = derivatives(size, sigma, derivative=2)
    ky, kz, kx = derivatives(size, sigma, derivative=1)
    kernel_xy = kx * ky
    kernel_xz = kx * kz
    kernel_yz = ky * kz
    ddx = convolve(volume, kernel_xx, mode='same', method='fft')
    ddy = convolve(volume, kernel_yy, mode='same', method='fft')
    ddz = convolve(volume, kernel_zz, mode='same', method='fft')
    dxdy = convolve(volume, kernel_xy, mode='same', method='fft')
    dxdz = convolve(volume, kernel_xz, mode='same', method='fft')
    dydz = convolve(volume, kernel_yz, mode='same', method='fft')

    return hessian.tubular_candidate_points(x, y, z, ddx, ddy, ddz,
                                            dxdy, dxdz, dydz)


def tube(volume, candidates, size, scale_min, scale_max, n_scales,
         grad_func=None):
    """
    Karl Krissian, Model Based Detection of Tubular Structures 
    in 3D Images.

    """
    x, y, z = volume.shape
    n_scales = int(n_scales)

    if True:
        stepsize = max((np.log(scale_max) - np.log(scale_min)) / (n_scales - 1),
                       1e-6)
        scales = np.exp(scale_min + np.arange(n_scales) * stepsize)
        scales = scale_min / scales.min() * scales
    else:
        scales = np.linspace(scale_min, scale_max, n_scales, endpoint=True)

    responses = np.zeros((n_scales, x, y, z))
    gradient = np.zeros((x, y, z, 3), dtype=np.float64)

    for scale in range(n_scales):
        print('Scale', scale, 'of', n_scales, 'scales')
        kernel_yy, kernel_zz, kernel_xx = derivatives(size, scales[scale], 2)
        ky, kz, kx = derivatives(size, scales[scale], 1)
        kernel_xy = kx * ky
        kernel_xz = kx * kz
        kernel_yz = ky * kz
        ddx = convolve(volume, kernel_xx, mode='same', method='fft')
        ddy = convolve(volume, kernel_yy, mode='same', method='fft')
        ddz = convolve(volume, kernel_zz, mode='same', method='fft')
        dxdy = convolve(volume, kernel_xy, mode='same', method='fft')
        dxdz = convolve(volume, kernel_xz, mode='same', method='fft')
        dydz = convolve(volume, kernel_yz, mode='same', method='fft')

        sigma = scales[scale]
        if grad_func is None:
            gradient[:, :, :, 0] = -ndi.gaussian_filter1d(
                volume * sigma ** (5.0 / 4), sigma=sigma, order=1, axis=0
            )

            gradient[:, :, :, 1] = -ndi.gaussian_filter1d(
                volume * sigma ** (5.0 / 4), sigma=sigma, order=1, axis=1
            )

            gradient[:, :, :, 2] = -ndi.gaussian_filter1d(
                volume * sigma ** (5.0 / 4), sigma=sigma, order=1, axis=2
            )
        else:
            gradient = grad_func(volume, sigma)

        responses[scale] = hessian.single_scale_hessian_response(
            x, y, z, scales[scale] ** 2, gradient, candidates,
            ddx, ddy, ddz, dxdy, dxdz, dydz
        )

    return responses


def marr_hildreth(img, kernel_size=3, sigma=1.0):
    """
    Marr-Hildreth edge detection. Not very good
    http://www.hms.harvard.edu/bss/neuro/bornlab/qmbc/beta/day4/
    marr-hildreth-edge-prsl1980.pdf
    :param img:
        Source image

    :param kernel_size:
        Size of filter kernel, has to be odd

    :param sigma:
        stdev of gaussian kernel

    :return:
        binary image. Edges = 1, rest is 0
    """
    img = normalize(img)
    kernel = kernels.gaussian(kernel_size, sigma)
    img_smoothed = convolve2d(img, kernel)
    laplacian = convolve2d(img_smoothed, _laplacian)
    laplacian = convolve2d(laplacian, kernel)
    edges = zero_crossings(laplacian)
    return edges


def derivatives(size, sigma, derivative):
    mid = size // 2
    nhood = np.arange(-mid, mid + 1).reshape(-1, 1)
    plane = np.tile(nhood, size)
    xx, yy = plane.shape
    z = np.tile(plane.reshape(xx, yy, 1), size)
    y = z.T
    x = np.swapaxes(z, 0, 1)

    dx = kernels.gaussian1d(x, sigma, derivative)
    dy = kernels.gaussian1d(y, sigma, derivative)
    dz = kernels.gaussian1d(z, sigma, derivative)
    #      Y,  Z,  X            
    return dx, dy, dz


def line_kernels(size, sigma):
    if size % 2 == 0:
        raise ValueError('Need kernel size odd, not {}'.format(size))

    filters = np.zeros((5, size, size), dtype=np.float64)
    mid = size // 2

    nhood = np.arange(-mid, mid + 1).reshape(-1, 1)
    x = np.tile(nhood, size)
    y = x.T

    gx_d0 = kernels.gaussian1d(x, sigma)
    gy_d0 = kernels.gaussian1d(y, sigma)
    gx_d1 = kernels.gaussian1d(x, sigma, derivative=1)
    gy_d1 = kernels.gaussian1d(y, sigma, derivative=1)
    gx_d2 = kernels.gaussian1d(x, sigma, derivative=2)
    gy_d2 = kernels.gaussian1d(y, sigma, derivative=2)

    filters[0] = gx_d1 * gy_d0
    filters[1] = gx_d0 * gy_d1
    filters[2] = gx_d2 * gy_d0
    filters[3] = gx_d1 * gy_d1
    filters[4] = gx_d0 * gy_d2

    return filters
