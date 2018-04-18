from __future__ import division, print_function
import numpy as np
import pywt
import math


class Contourlet(object):
    pass


class Wavelet(object):
    def __init__(self, image, level=1, wavelet='db2'):
        self._image = image
        self._rimage = None
        self.level = level
        self.wavelet = wavelet
        self._approx = None
        self._detail = None

    def transform(self, mode='symmetric'):
        coeffs = pywt.wavedec2(self._image, self.wavelet, mode, self.level)
        self._approx = coeffs[0]
        self._detail = coeffs[1:][0]
        return self.approx, self._detail

    def reconstruct(self, mode='symmetric'):
        self._rimage = pywt.waverec2((self._approx, self._detail), self.wavelet, mode)
        return self._rimage

    @property
    def approx(self):
        if self._approx is None:
            self.transform()

        return self._approx

    @property
    def detail(self):
        if self._detail is None:
            self.transform()

        return self._detail[::-1]

    @property
    def image(self):
        return self._image


def transform(image, wavelet='db4'):
    approx, detail = pywt.dwt2(image, wavelet)
    return approx, detail


def reconstruct(approx, detail, wavelet):
    return pywt.idwt2((approx, detail), wavelet)


def threshold(coeffs, thresh=None, mode='hard'):
    """

    :param coeffs:
    :param thresh: Number or threshold type
                   ss: Sure Shrink
    :param mode: 'hard' or 'soft'
    :return:
    """
    if thresh is None:
        thresh = optimal_threshold(coeffs)
    elif isinstance(thresh, str):
        if thresh == 'ss':
            thresh = sure_shrink(coeffs)

    coeffs[np.where(abs(coeffs) <= thresh)] = 0

    if mode == 'soft':
        idx_pos = np.where(coeffs > thresh)
        idx_neg = np.where(coeffs < -thresh)
        coeffs[idx_pos] -= thresh
        coeffs[idx_neg] += thresh

    return coeffs


def optimal_threshold(coeffs):
    """
    http://biomet.oxfordjournals.org/content/81/3/425.full.pdf
    http://tx.technion.ac.il/~rc/SignalDenoisingUsingWavelets_RamiCohen.pdf
    :param coeffs:
    :return:
    """
    magic = 0.6745
    return abs(np.median(coeffs))/magic


def sure_shrink(coeffs):
    """
    http://www.csee.wvu.edu/~xinl/library/papers/math/statistics/donoho1995.pdf (1)
    :param coeffs: wavelet coeffs
    :return: optimal threshold
    """

    coeffs = coeffs.copy()
    coeffs = np.abs(coeffs.ravel())
    coeffs.sort()
    s = coeffs.size

    # noinspection PyTypeChecker
    sparse = 1/math.sqrt(s)*np.sum(coeffs**2 - 1)/math.log(s, 2)**(3/2)
    if sparse <= 1:
        return universal_threshold(coeffs)

    sure_min = np.inf
    optimal_thresh = 0
    num_smaller = 0
    sum_smaller = 0
    for thresh in coeffs:
        # Since thresh is increasing #coeffs < thresh will increase and can
        # keep track of these in coeffs_smaller and the bigger ones stay in coeffs array
        # For each iter we just look at the coeffs bigger than prev thresh. ca 100x faster
        # cardinality = coeffs[np.where(coeffs < thresh)].size
        # sure = s + sum(np.minimum(coeffs, thresh) ** 2) - 2 * cardinality

        coeffs_smaller = coeffs[np.where(coeffs < thresh)]
        coeffs = coeffs[np.where(coeffs >= thresh)]

        num_smaller += coeffs_smaller.size
        sum_smaller += np.sum(coeffs_smaller)
        sum_thresh = coeffs.size*thresh

        sure = s + (sum_smaller + sum_thresh)**2 - 2*num_smaller

        if sure < sure_min:
            sure_min = sure
            optimal_thresh = thresh

    return optimal_thresh


def universal_threshold(coeffs, sigma=1):
    if isinstance(coeffs, (int, float)):
        size = coeffs
    else:
        size = coeffs.size

    return np.sqrt(2*np.log(size))*sigma
