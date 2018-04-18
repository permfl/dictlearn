from __future__ import print_function
import math
import numpy as np

from skimage import filters as _filters
from skimage import restoration

try:
    threshold_minimum = _filters.threshold_minimum
except AttributeError:
    # not available for py34
    pass

from . import wavelet


def arithmetic_mean(img, m, n, inplace=False, pad_with='zero'):
    """
        Arithmetic mean smoothing

        For each pixel (i, j) find the average pixel value in the nhood
        (m, n) and put back at pixel (i, j)

        :param img:
            Image, ndarray

        :param n:
            Height of nhood

        :param n:
            Width of nhood

        :param inplace: Default False.
            Overwrite values in img of True

        :param pad_with:
            Padding for boundary

        :returns:
            Image if not inplace, else None
    """
    if not inplace:
        img = img.copy()

    if pad_with == 'ones':
        pad_img = np.ones
    else:
        pad_img = np.zeros

    mask = np.ones((2*m+1, 2*n+1))
    padded = pad_img((img.shape[0] + 2*m, img.shape[1] + 2*n))
    padded[m:-m, n:-n] = img

    for i in range(m, padded.shape[0] - m):
        for j in range(n, padded.shape[1] - n):
            part = padded[i-m:i+m+1, j-n:j+n+1]
            img[i-m, j-n] = (part*mask).sum()/mask.size

    if not inplace:
        return img


def threshold(image, min_val, max_val=np.inf, type='hard'):
    """
        Threshold image

        :param image:
            Image to threshold

        :param min_val:
            Set all values in image less then min_val to zero

        :param max_val:
            Set all values in image greater than max_val to zero. 
            Only applicable if type == 'hard'

        :param type:
            Thresholding type. 'hard' sets to zero everything outside
            [min_val, max_val]. 'soft' will set to zero everything less 
            than min_val and move every value larger than min_val towards
            zero with distance min_val.
    """
    # todo test
    if type == 'hard':
        image[np.where(image <= min_val)] = 0
        image[np.where(image >= max_val)] = 0
    elif type == 'soft':
        if max_val != np.inf:
            raise ValueError('Cannot do soft threshold when '
                             'max value is supplied')
        image[np.where(np.abs(image) <= min_val)] = 0
        image[np.where(image < -min_val)] += min_val
        image[np.where(image > min_val)] -= min_val
    else:
        raise ValueError('{} is not a valid thresholding type'
                         .format(type))

    return image


def _hist_stats(image, n_bins):
    hist, bins = np.histogram(image.ravel(), bins=n_bins)
    count = np.arange(n_bins)
    a = np.cumsum(hist)
    b = np.cumsum(count*hist)
    c = np.cumsum(count**2*hist)

    return hist, bins, a, b, c


def threshold_entropy(image, bins=256, kind='max'):
    stats = _hist_stats(image, bins)
    non_zero = stats[0] > 0
    hist = stats[0][non_zero]
    bins = stats[1][1:][non_zero]
    A = stats[2][non_zero]
    entropy = np.cumsum(hist*np.log(hist))

    if kind == 'min':
        _idx = np.argmin
    else:
        _idx = np.argmax

    idx = _idx(
        entropy[:-1]/A[:-1] - np.log(A[:-1]) +
        (entropy[-1] - entropy[:-1])/(A[-1] - A[:-1]) - np.log(A[-1] - A[:-1])
    )

    return bins[idx]


def threshold_median(image):
    return np.median(image)


def threshold_mean(image):
    return np.mean(image)


def threshold_maxlink(image, n_bins=256):
    """
    :param image:
    :param n_bins:
    :return:
    """
    hist, bins, a, b, c = _hist_stats(image, n_bins)

    try:
        thresh = threshold_minimum(image, n_bins)
    except RuntimeError:
        thresh = threshold_mean(image)

    idx = np.argmin(np.abs(bins[1:] - thresh))
    run = 0
    stability = 5
    max_runs = 1000

    mean = np.zeros(stability)
    rho = np.zeros(stability)
    p = np.zeros(stability)
    q = np.zeros(stability)
    sigmasq = np.zeros(stability)
    tausq = np.zeros(stability)

    mean[0] = b[idx] / a[idx]
    rho[0] = (b[-1] - b[idx]) / (a[-1] - a[idx])
    p[0] = a[idx] / a[-1]
    q[0] = (a[-1] - a[idx]) / a[-1]
    sigmasq[0] = c[idx] / a[idx] - mean[0] * mean[0]
    tausq[0] = (c[-1] - c[idx]) / (a[-1] - a[idx]) - rho[0] * rho[0]
    iset = np.arange(hist.size)
    parameters = [p, q, mean, rho, sigmasq, tausq]

    while True:
        i = run % stability

        phi = p[i]/math.sqrt(sigmasq[i]) * \
            np.exp(-(iset - mean[i])**2 / (2*sigmasq[i]))/\
            (p[i]/math.sqrt(sigmasq[i]) * np.exp(-(iset - mean[i])**2 /
                                                 (2*sigmasq[i])) +
            q[i]/math.sqrt(tausq[i]) * np.exp(-(iset - rho[i])**2 /
                                              (2*tausq[i])))

        gamma = 1 - phi
        F = np.sum(phi * hist)
        G = np.sum(gamma * hist)

        ii = (run + 1) % stability
        p[ii] = F/a[-1]
        q[ii] = G/a[-1]
        mean[ii] = np.sum(iset*phi*hist/F)
        rho[ii] = np.sum(iset*gamma*hist/G)
        sigmasq[ii] = np.sum(iset**2*phi*hist/F) - mean[ii]**2
        tausq[ii] = np.sum(iset**2*gamma*hist/G) - rho[ii]**2

        run += 1

        if any(abs(thing[ii]) < 1e-13 for thing in parameters):
            break

        if sum(param.std() for param in parameters) < 6e-3:
            break

        if run >= max_runs:
            break

    # Threshold is largest root of w0*x^2 + 2*w1*x + w2 = 0
    w0 = 1/sigmasq[i] - 1/tausq[i]
    w1 = mean[i]/sigmasq[i] - rho[i]/tausq[i]
    w2 = w1 + math.log(sigmasq[i]*q[i]/tausq[i]/q[i])

    t = (w1 + math.sqrt(w1*w1 - w0*w2)) / w0
    return bins[int(t)]


threshold_otsu = _filters.threshold_otsu


def estimate_sigma(image):
    """
        Estimate variance of noise in image
    """
    if image.ndim == 2:
        coeffs = wavelet.Wavelet(image).detail[0]
        return wavelet.optimal_threshold(coeffs[np.nonzero(coeffs)])
    else:
        return restoration.estimate_sigma(image)
