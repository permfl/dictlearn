import numpy as np 
from scipy import signal


# 2nd derivative
laplacian = np.asarray(
    [[1,  1, 1],
     [1, -8, 1],
     [1,  1, 1]]
)


# Scharr Gradient Operators
scharr_x = np.asarray(
    [[3, 0,  -3],
    [10, 0, -10],
    [ 3, 0,  -3]]
)


scharr_y = np.asarray(
    [[ 3,  10,  3],
     [ 0,   0,  0],
     [-3, -10, -3]]
) 


sobel_x = np.asarray(
    [[1, 0, -1],
     [2, 0, -2],
     [1, 0, -1]]
)


sobel_y = np.asarray(
    [[1,  2,  1],
     [0,  0,  0],
     [-1, -2, -1]]
)


def convolve2d(image, kernel, mode='same', boundary='fill', fillvalue=0):
    return signal.convolve2d(image, kernel, mode, boundary, fillvalue)


convolve2d.__doc__ = signal.convolve2d.__doc__


def sobel(image):
    """
        Applies sobel operator in height and width direction
    """
    image = image.astype(float)
    img_x = convolve2d(image, sobel_x)
    img_y = convolve2d(image, sobel_y)
    return np.sqrt(img_x**2 + img_y**2), np.arctan2(img_y, img_x)
