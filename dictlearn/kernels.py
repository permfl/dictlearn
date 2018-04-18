import math
import numpy as np


def gaussian(size, sigma=1.4, lowpass=True):
    """
    Create gaussian lowpass (or high) of size (size, size)

    :param size: Size of kernel
    :param sigma: Std deviation
    :param lowpass: True for lowpass, returns highpass filter of False
    :return: ndarray, shape(size, size)
    """
    kernel = np.zeros((size, size))
    n = size//2

    for r in range(size):
        for c in range(size):
            # indices centered at size//2, size//2
            y = n - r
            x = c - n
            kernel[r, c] = np.exp(-(x*x + y*y)/(2*sigma*sigma))

    scale = 2*np.pi*sigma*sigma

    if lowpass:
        return kernel/scale
    else:
        return 1 - kernel/scale


def gaussian3d(size, sigma):
    """
        3D Gaussian filter kernel

    :param size:
        int, filter size. size = x gives kernel size (x, x, x)

    :param sigma:
        stddev

    :return:
        Filter kernel, shape (size, size, size)
    """
    mid = size // 2
    nhood = np.arange(-mid, mid + 1).reshape(-1, 1)
    plane = np.tile(nhood, size)
    xx, yy = plane.shape
    z = np.tile(plane.reshape(xx, yy, 1), size)
    y = z.T
    x = np.swapaxes(z, 0, 1)

    gx = gaussian1d(x, sigma)
    gy = gaussian1d(y, sigma)
    gz = gaussian1d(z, sigma)
    return gx * gy * gz


def get_neighbourhood(size):
    """
    Create list of 2D indices of all pixels in its (size, size) nhood

    If size = 3:
        [(-1, -1), (-1, 0), (-1, 1),
         (0, -1 ), (0, 0),  (0, 1),
         (1, -1 ), (1, 0),  (1, 1)]

    :param size: Size of index set
    :return: list(tuple(x, y))
    """
    if isinstance(size, (int, float)):
        if size % 2 == 0:
            raise ValueError('Size has to be odd not %d' % size)

        center_x = center_y = size // 2
    else:
        if len(size) == 2:
            if size[0] % 2 == 0 or size[1] % 2 == 0:
                raise ValueError('Size had to be odd, not (%d, %d)'
                                 % (size[0], size[1]))
            center_x = size[0]//2
            center_y = size[1]//2
        elif len(size) == 1:
            center_x = center_y = size // 2
        else:
            raise ValueError('Only 2D neighbourhood')

    indices = []

    for i in range(-center_x, center_x + 1):
        for j in range(-center_y, center_y + 1):
            indices.append((i, j))

    return indices


def gaussian1d(x, sigma, derivative=0):
    """
        Compute gaussian at x, or its first/second derivative

    :param x:
        Neighbourhood indices, len(x) = kernel size. For a size 3 kernel
        this should be [-1, 0, 1]

    :param sigma:
        Standard deviation

    :param derivative:
        derivative=0 for normal, =1 for first deriv, =2 for second

    :return:
    """
    exp = np.exp(-0.5 * (x / sigma)**2)
    root = math.sqrt(2 * math.pi)

    if derivative == 0:
        return 1 / (root * sigma) * exp
    elif derivative == 1:
        return -x / (root * sigma**3) * exp
    elif derivative == 2:
        return (x**2 - sigma**2) / (root * sigma**5) * exp
    else:
        raise ValueError('Unsupported derivative {}. '
                         'Choose derivative in (0, 1, 2)'.format(derivative))

