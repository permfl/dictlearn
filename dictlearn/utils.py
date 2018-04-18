import math
import numpy as np
import warnings
from PIL import Image


class VisibleDeprecationWarning(RuntimeWarning):
    pass


def zero_crossings(image, threshold=0):
    """
    Marks zero-crossings.
    If p[i-1] > thresh and p[i+1] < thresh then p[i] = 1

    :param image:

    :param threshold:


    :return:
        Binary image {0, 1}
    """
    crossings = np.zeros(image.shape)

    for col in range(image.shape[1]):
        for i in range(image.shape[0] - 1):
            if image[i][col] > threshold and image[i + 1][col] < -threshold:
                crossings[i][col] = 1

    for row in range(image.shape[0]):
        for i in range(image.shape[1] - 1):
            if image[row][i] > threshold and image[row][i + 1] < -threshold:
                crossings[row][i] = 1

    return crossings


def add_noise(img):
    """
    Add gaussian zero mean noise to noise to image img

    :param img:
    :return:
    """
    # todo set own noise variance
    noise = np.random.rand(*img.shape)
    noise = noise - np.average(noise)
    return img + noise


def to_uint8(image, type_=np.uint8):
    """
    Maps to [0, 255] with given type

    :param image:
        numpy.ndarray

    :param type_:
        return type of img

    :return:
        image mapped to [0, 255]
    """
    normalized = normalize(image)
    return (normalized * 255).astype(type_)


def normalize(image):
    """
    Maps first to [0, image.max() - image.min()]
    then to [0, 1]

    :param image:
        numpy.ndarray

    :return:
        image mapped to [0, 1]
    """
    image = image.astype(float)
    if image.min() != image.max():
        image -= image.min()
    
    nonzeros = np.nonzero(image)
    image[nonzeros] = image[nonzeros] / image[nonzeros].max()
    return image


def pad_image(img, m, n, pad_with='zeros'):
    """
    Pad an image with some value

    :param img:
        Image to pad, shape (M, N)

    :param m:
        Add m extra rows to beginning and end

    :param n:
        Number of columns to add
    :param pad_with:
        Value to be placed on the new space. {'zeros', 'ones', int/float}

    :return:
        Padded image of shape (2*m + M, 2*n + N)
    """
    if pad_with == 'ones':
        padding = np.ones
    elif isinstance(pad_with, (int, float)):
        padding = lambda x: np.ones(x) * pad_with
    else:
        padding = np.zeros

    padded = padding((img.shape[0] + 2 * m, img.shape[1] + 2 * n))
    padded[m:-m, n:-n] = img
    return padded


def random_dictionary(rows, n_atoms):
    """
    Create a uniform random l2 normalized dictionary

    :param rows:
        Size of signals, ie dictlearn.Patches.size

    :param n_atoms:
        Number of atoms/columns in dictionary

    :return:
        Dictionary of shape (rows, n_atoms)
    """
    dictionary = np.random.rand(rows, n_atoms)
    for c in range(n_atoms):
        col = dictionary[:, c]
        dictionary[:, c] = col / np.linalg.norm(col)

    return dictionary


def dct_dict(n_atoms, size):
    """
    Create a dictionary using the Discrete Cosine Transform (DCT) basis. If n_atoms is
    not a perfect square, the returned dictionary will have ceil(sqrt(n_atoms))**2 atoms

    :param n_atoms:
        Number of atoms in dict

    :param size:
        Size of first patch dim

    :return:
        DCT dictionary, shape (size*size, ceil(sqrt(n_atoms))**2)
    """
    # todo flip arguments to match random_dictionary
    p = int(math.ceil(math.sqrt(n_atoms)))
    dct = np.zeros((size, p))

    for k in range(p):
        basis = np.cos(np.arange(size) * k * math.pi / p)
        if k > 0:
            basis = basis - np.mean(basis)

        dct[:, k] = basis

    kron = np.kron(dct, dct)

    for col in range(kron.shape[1]):
        norm = np.linalg.norm(kron[:, col]) or 1
        kron[:, col] /= norm

    return kron


def ndct_dict(size, n_atoms):
    """
        Create arbitrary sized DCT dictionary

        :param size:
            Number of rows, same as signal size

        :param n_atoms:
            Number of dictionary atoms

        :return:
            Dictionary of size (size, n_atoms)
    """
    dico = np.zeros((size, n_atoms))

    for k in range(n_atoms):
        basis = np.cos(np.arange(size) * k * math.pi / n_atoms)
        dico[:, k] = basis / np.linalg.norm(basis)

    return dico


def numpy_from_vti(path):
    """
    Read and convert a VTI file into numpy array

    :param path: Path to VTI file
    :return: numpy.ndarray of data in VTI file
    """
    warnings.warn('Use dictlearn.vtk_image.VTKImage.read()',
                  VisibleDeprecationWarning)
    from .vtk import VTKImage
    return VTKImage.read(path)


def numpy_to_vti(path, image):
    """
        Write a numpy array to vti file

        :param image: 3D ndarray with image data
        :param path: Where to save the image
        :returns: True if writing successful
    """
    warnings.warn('Use dictlearn.vtk_image.VTKImage.write_vti()',
                  VisibleDeprecationWarning)
    from .vtk import VTKImage
    return VTKImage.write_vti(path, image)


def psnr(original, noisy, max_pixel_value=255):
    """
        Peak Signal to Noise ratio

        :param original:
            Image to compare against, numpy.ndarray with ndim = 2 or 3

        :param noisy:
            numpy.ndarray same shape as 'original'

        :param max_pixel_value:
            Maximum possible value an element in original and noisy can take.
            For a RBG image this is 255.

        :return:
            If original and noisy is greyscale images this is a single number.
            If original have ndim = 3, then this is an array with
            shape = (original.shape[2], )

    """
    if original.ndim == 2:
        mse = ((original - noisy) ** 2).mean()

        if mse == 0:
            return np.inf

        return 20 * np.log10(max_pixel_value) - 10 * np.log10(mse)
    elif original.ndim == 3:
        n_channels = original.shape[2]
        err = np.zeros(n_channels)

        for ch in range(n_channels):
            err[ch] = psnr(original[:, :, ch], noisy[:, :, ch], max_pixel_value)

        return err


def visualize_dictionary(dictionary, rows, cols, show=True, title=None):
    """
        Visualize a dictionary

        :param dictionary:
            Dictionary to plot

        :param rows:
            Number of rows in plot

        :param cols:
            Number of columns in plot. Rows*cols has to be equal
            to the number of atoms in dictionary

        :param show:
            Call pyplot.show() is True

        :param title:
            Title for figure
    """
    import matplotlib.pyplot as plt
    size = int(np.sqrt(dictionary.shape[0])) + 2
    img = np.zeros((rows * size, cols * size))

    for row in range(rows):
        for col in range(cols):
            atom = row * cols + col
            at = normalize(dictionary[:, atom].reshape(size - 2, size - 2))
            padded = np.pad(at, 1, mode='constant', constant_values=1)
            img[row * size:row * size + size, col * size:col * size + size] = padded

    plt.imshow(img, cmap=plt.cm.bone, interpolation='nearest')
    plt.axis('off')

    if title is not None:
        title = title if isinstance(title, str) else str(title)
        plt.title(title)

    if show:
        plt.show()


def ycbcr2rgb(image):
    image = image.astype(float)

    if image.max() > 1:
        image /= 255

    raise RuntimeError()


def rgb2ycbcr(image):
    """
        Convert RGB image to YCbCr

    :param image:
        numpy array with ndim=3 and shape[2] == 3

    :return:
        ycbcr image with same shape as 'image'
    """
    image = image.astype(float)

    if image.max() > 1:
        image /= 255

    ycbcr = np.zeros_like(image)
    ycbcr[:, :, 0] = 16 + 65.481 * image[:, :, 0] + 128.553 * image[:, :, 2] \
                     + 24.996 * image[:, :, 1]

    ycbcr[:, :, 1] = 128 - 37.797 * image[:, :, 0] - 74.203 * image[:, :, 2] \
                     + 112 * image[:, :, 1]

    ycbcr[:, :, 2] = 128 + 112 * image[:, :, 0] - 93.786 * image[:, :, 2] \
                     - 18.214 * image[:, :, 1]

    return ycbcr


def rgb2gray(img):
    if img.ndim != 3 and img.shape[-1] != 3:
        raise ValueError('Image is not RGB')

    return 0.2989 * img[:, :, 0] + 0.5870 * img[:, :, 1] + 0.1140 * img[:, :, 2]


def imread(path, dtype=None):
    """
        Read image as numpy array

    :param path:
        Path to image

    :param dtype:
        Convert image to type 'dtype'

    :return:
        Image at path as numpy array

    """
    img = Image.open(path)

    if dtype is not None:
        return np.asarray(img).astype(dtype)

    return np.asarray(img)


def imsave(fp, image, format=None):
    """
        Save the image 'image' to file 'fp'

    :param fp:
        Path/filename string or a file object

    :param image:
        Image to save, numpy array or PIL Image

    :param format:
        Format to use for saving. If this is None the format is
        determined from filename extension

    :return:
        None
    """
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    image.save(fp, format)
