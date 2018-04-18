from __future__ import print_function, division

import os
import math
import numpy as np
from . import optimize, inpaint, sparse
from .preprocess import Patches
from .utils import dct_dict, random_dictionary, imread


try:  # Py2
    reduce
except NameError:  # Py3
    from functools import reduce


# TODO not tested
class Trainer(object):
    """
        :param signals:
            Training data, shape (n_features, n_samples)

        :param method:
            Training algorithm, 'online' or 'batch'

        :param regularization:
            'l0' or 'l1', 'l0' is faster, but 'l1' can sometimes
            be more accurate
    """
    METHODS = ['online', 'batch', 'ksvd']
    REGULARIZATION = ['l0', 'l1']

    def __init__(self, signals, method='online', regularization='l0'):
        self.signals = signals

        if method not in self.METHODS:
            raise ValueError('Method {} not supported. Choose {}'
                             .format(method, self.METHODS))
        self.method = method

        if regularization not in self.REGULARIZATION:
            raise ValueError('Regularization {} not supported. Choose {}'
                             .format(method, self.REGULARIZATION))

        self.regularization = regularization
        self.dictionary = None
        self.codes = None
        self.iters_default = 1000 if self.method == 'online' else 10

    def train(self, dictionary=None, n_atoms=None, iters=None, n_nonzero=10,
              tolerance=0, n_threads=1, verbose=False, **kwargs):
        """

        Train a dictionary on training signals using 'Online Dictionary 
        Learning (ODL)' or 'K-SVD'. 


        Both methods update the dictionary once very iteration. ODL will
        find the sparse coding on one signal and then update the dictionary
        using a variant of block coordinate-descent with momentum. K-SVD will
        sparse code all signals before doing the dictionary update, thus every
        iteration of K-SVD is a lot slower. Both produces similar results
        given the same running time


        :param dictionary: Optional.
            Dictionary, ndarray with shape (signal_size, n_atoms)

        :param n_atoms:
            Number of dictionary atoms, default 2*signal_size

        :param iters:
            Training iterations, default 10 if 'batch', 1000 for 'online'

        :param n_nonzero:
            Max number of nonzero coefficients in sparse codes. Default 10

        :param tolerance:
            Sparse coding tolerance. Adds coefficients to the sparse 
            approximation until the tolerance is achieved, or all 
            coefficients are active

        :param n_threads:
            Number of threads to use. Default 1

        :param verbose:
            Print progress if True. Default False
        """
        # Get the requested training algorithm. Subclasses can add extra methods
        # by appending the method <name> in METHODS and implementing a func with
        # the same name, ie Inpaint class
        method = getattr(self, 'batch' if self.method == 'ksvd' else self.method)

        if self.dictionary is None and dictionary is None:
            if n_atoms is None:
                n_atoms = 2 * self.signals.shape[0]

            size = int(math.sqrt(self.signals.shape[0]))

            if size * size == self.signals.shape[0]:
                dictionary = dct_dict(n_atoms, size)
            else:
                dictionary = random_dictionary(self.signals.shape[0], n_atoms)
        elif self.dictionary is not None:
            dictionary = self.dictionary.copy()

        if self.signals.shape[0] != dictionary.shape[0]:
            raise ValueError('Need first dim in signals and' 
                             ' dictionary to be equal. {} != {}'
                             .format(self.signals.shape[0], dictionary.shape[0]))
            
        iters = self.iters_default if iters is None else iters
        method(dictionary, iters, n_nonzero, tolerance, n_threads, verbose)
        return self

    def batch(self, dictionary, iters, n_nonzero,
              tolerance, n_threads, verbose):
        if self.regularization == 'l0':
            ret = optimize.ksvd(self.signals, dictionary, iters,
                                n_nonzero, tolerance, 0, verbose, n_threads)
            self.dictionary, self.codes = ret
        else:
            raise RuntimeError('No batch l1 regularization')

    def online(self, dictionary, iters, n_nonzero,
               tolerance, n_threads, verbose):
        if self.regularization == 'l1':
            raise RuntimeError('No online l1 regularization')
        else:
            self.dictionary = optimize.odl(self.signals, dictionary, iters,
                                           n_nonzero, tolerance, verbose)


# TODO not tested
class ImageTrainer(Trainer):
    """
    :param image:
        Train dictionary on this image (data). Can be a path to image, numpy
        array, dl.Patches or dl.Patches3D. If path or numpy array then dl.Patches
        are created. If the image is to large for keeping all image patches in memory
        pass dl.Patches3D instance

    :param patch_size:
        Size of image patches

    :param method:
        Method for training, 'online', or 'batch'

    :param regularization:
        Regularization to use, 'l0' or 'l1'
    """
    def __init__(self, image, patch_size=8, method='online', regularization='l0'):
        if isinstance(image, Patches):
            self.patches = image
        elif isinstance(image, np.ndarray):
            self.patches = Patches(image.astype(float), patch_size)
        elif isinstance(image, str):
            self._open_image(image, patch_size)
        else:
            raise ValueError('Cannot understand image type {}. '
                             'Image can be Patches, numpy.ndarray or path')

        super(ImageTrainer, self).__init__(self.patches.patches, method, regularization)

    def _open_image(self, path, patch_size):
        fn, ext = os.path.splitext(path)

        if ext == '.npy':
            try:
                img = np.load(path).astype(float)
            except IOError:
                raise
        else:
            img = imread(path).astype(float)

        self.patches = Patches(img, patch_size)


# TODO not tested
class Denoise(ImageTrainer):
    """
    Image Denoising with Dictionary Learning

    Train a dictionary on the noisy image, then denoise using sparse
    coding. If method = 'ksvd' a dictionary is learned using K-SVD and the 
    image is denoised using the sparse coefficients from the last training 
    iteration. If method = 'batch' or 'online' an additional sparse
    coding step is used to compute the sparse codes for denoising.

    Both adaptive and *static* denoising is supported. 

    
    Example adaptive denoise:

    >>> denoiser = Denoise('noisy/image.png', 12)
    >>> denoiser.train(iters=5000, n_nonzero=2, n_atoms=256)
    >>> cleaned = denoiser.denoise(sigma=30)

    Example pre-trained (static) dictionary:

    >>> import numpy as np
    >>> dictionary = np.load('dictionary.npy')
    >>> denoiser = Denoise('noisy/image.png', 12)
    >>> denoiser.dictionary = dictionary
    >>> denoiser.train()  # Optional training
    >>> cleaned = denoiser.denoise(sigma=30)


    The size of the image patches and sigma in denoise() will have a large effect
    on the denoised image. If either are too large the image will look blurry,
    if too low the difference between the original noisy image and the 
    denoised image will be small. A patch size in [8, 12] is usually a good 
    choice for most images. If the image has very small details then a smaller 
    patch size might be needed. If the structures in the image are 
    large and smooth, larger patches can produce better results. 

    The value of sigma is highly dependent on the image and its scale. If the
    image is in [0, 1] then 0 <= sigma <= 1. An image in [0, 255] gives sigma
    in [0, 255]. 


    :param image:
        Noisy image. Can be a path to image, numpy array, dl.Patches or 
        dl.Patches3D. If path or numpy array then dl.Patches are created. 
        If the image is to large for keeping all image patches in memory 
        pass dl.Patches3D instance

    :param patch_size:
        Size of image patches

    :param method:
        Method for training, 'online', 'batch' or 'ksvd'

    :param regularization:
        Regularization to use, 'l0' or 'l1'
    """
    def denoise(self, sigma=20, n_threads=1, noise_gain=1.15):
        """
        Denoise the image

        Sigma is the parameter that has the largest effect on the final 
        result. For the best results sigma should be close to the variance
        of the noise. If the difference between the original and the denoised
        image is small sigma is probably too low. If the denoised image is
        very blurry then sigma is too large.


        **Blurry image?** Reduce sigma

        **Noisy image?** Increase sigma 

        :param sigma:
            Noise variance
        
        :param n_threads:
            Number of threads. Default 1

        :param noise_gain:
            Average number of nonzero coefficients in sparse approximation.
            Default 1.15, which has been shown to give good results. 

        :return:
            Reconstructed and denoised image
        """
        if self.dictionary is None:
            raise ValueError('The dictionary needs to be learned before denoising.'
                             ' Call self.train first')

        if self.codes is not None and self.method == 'ksvd':
            # Use sparse coeffs from K-SVD algorithm
            codes = self.codes
        else:
            tol = self.patches.shape[0] * (noise_gain * sigma) ** 2
            codes = sparse.omp_batch(self.patches.patches, self.dictionary,
                                     tol=tol, n_threads=n_threads)

        return self.patches.reconstruct(np.dot(self.dictionary, codes))


# TODO not tested
class Inpaint(ImageTrainer):
    """
        Image inpainting

        Fill in missing areas of an image, or remove unwanted objects. Works
        very well if the missing areas are fairly small, and smaller than
        the image patches. If the missing areas are large use 
        TextureSynthesis


        >>> import dictlearn as dl
        >>> import numpy as np
        >>> image imread('some/img.png')
        >>> # Mask with 60% of pixels marked missing
        >>> mask = np.random.rand(*image.shape) < 0.6
        >>> corrupted = image * mask
        >>> # Plot corrupted
        >>> inp = Inpaint(corrupted, mask)
        >>> reconstructed = inp.train().inpaint()


        :param image:
            Corrupted image

        :param mask: 
            Binary mask for image. Pixels in mask marked 0 is will be inpainted.
            locations marked 1 is kept as is.

        :param patch_size:
            Size of image patches. Default 8

        :param method:
            Inpainting method. 'online' or 'itkrmm'.

    """
    def __init__(self, image, mask, patch_size=8, method='online'):
        # Add itkrmm to known learning algorithms for Trainer
        self.METHODS.append('itkrmm')
        super(Inpaint, self).__init__(image, patch_size, method)

        if isinstance(mask, Patches):
            self.mask = mask
        elif isinstance(mask, str):
            mask = imread(mask).astype(bool)
            self.mask = Patches(mask, patch_size)
        elif isinstance(mask, np.ndarray):
            self.mask = Patches(mask.astype(bool), patch_size)
        else:
            raise ValueError('Unsupported mask type {} '
                             'pass numpy.ndarray, preprocess.Patches or path'
                             .format(type(mask)))

        ratio = np.count_nonzero(self.mask.image)/self.mask.image.size
        # Choose inpainting method based on mask structure
        # If itkrmm not specifically asked for, make a guess for which
        # method is best
        if (ratio < 3 or self.method == 'online') and not method == 'itkrmm':
            self._use_gsr = True
            self.dictionary = '_use_gsr'
        else:
            self._use_gsr = False
            self.method = 'itkrmm'

        self._final = None
        self.iters_default = 10
        self._iters = self.iters_default
        self.low_rank = None

    def train(self, **kwargs):
        if self._use_gsr:
            # Using GSR - All work done in self.inpaint
            self._iters = kwargs.get('iters', self.iters_default)
            return self

        low_rank = kwargs.get('n_low_rank')
        initial = kwargs.get('init_low_rank')
        if low_rank is not None and low_rank > 0:
            self.low_rank = optimize.reconstruct_low_rank(
                self.patches.patches,self.mask.patches, low_rank, 
                initial, self._iters
            )

        return super(ImageTrainer, self).train(**kwargs)

    def itkrmm(self, dictionary, iters, n_nonzero,
               tolerance, n_threads, verbose):

        self.dictionary = optimize.itkrmm(
            self.patches.patches, self.mask.patches,
            dictionary, n_nonzero, iters, self.low_rank, verbose
        )

    def inpaint(self, n_nonzero=20, group_size=60, search_space=20,
                stride=4, callback=None, tol=0, verbose=False):
        """
            :param n_nonzero:
                Number of nonzero coefficients for reconstruction

            :param group_size:
                Size of group for 'online' inpaint. Finds the 'group_size' 
                most similar image patches and trains a dictionary on this 
                group

            :param search_space:
                How far from the current pixel (i, j) to search for similar
                patches. Will search all pixels (i - s, j - s) to 
                (i + s, j + s), s = search_space.

            :param stride:
                Distance between image patches

            :param callback:
                Callback function for online inpaint. Called with two 
                arguments  
                           (1) the current reconstruction and 
                           (2) current iteration

            :param tol:
                For method = 'itkrmm', tolerance for sparse coding 
                reconstruction. Set this the same way as sigma in denoise, 
                to also denoise the image if needed. If the image is noise 
                free, use n_nonzero

            :param verbose:
                Print progress

            :return:
                Inpainted image

        """
        if self.dictionary is None:
            raise ValueError('The dictionary needs to be learned or assigned '
                             'before inpainting. Call self.train or '
                             'or set self.dictionary variable')

        if isinstance(self.dictionary, np.ndarray):
            codes = sparse.omp_mask(
                self.patches.patches, self.mask.patches,
                self.dictionary, n_nonzero, tol=tol, verbose=verbose
            )
            new_patches = np.dot(self.dictionary, codes)
            return self.patches.reconstruct(new_patches)

        final = inpaint.gsr(
            self.patches.image, self.mask.image, self._iters,
            self.patches.size, group_size, search_space,
            stride, callback=callback
        )
        return final


class TextureSynthesis(Inpaint, Denoise):
    """
        Inpaint by texture synthesis

        For each missing pixel we create an image patch centered at 
        this pixel, and search the image for the most similar patch. 
        Then the missing pixel is replaced with the center pixel in the 
        most similar patch. Repeat until all pixels are filled.

    """
    def inpaint(self, max_iters=None, verbose=False):
        """
            :param max_iters:
                If None, run until the entire image is filled. Otherwise
                stop after 'max_iters' iterations
            :param verbose:
                Print progress
        """
        if self.patches.size % 2 == 0:
            patch_size = self.patches.size + 1
        else:
            patch_size = self.patches.size

        return inpaint.inpaint_exemplar(
            self.patches.image, self.mask.image, patch_size, 
            max_iters, verbose
        )


def ksvd_denoise(image, patch_size=8, iters=20, n_atoms=128, sigma=10,
                 verbose=False, ret_dict=False, n_threads=1):
    """
    Adaptive denoising where the dictionary is learned from
    the noisy image using K-SVD

        :param image: Noisy image
        :param patch_size: Size of image patches
        :param iters: Number of K-SVD iterations
        :param n_atoms: Number of dictionary atoms
        :param sigma: Noise variance
        :param verbose: Print progress
        :param ret_dict: Return dictionary
        :param n_threads: Number of threads to use for sparse coding
        :return: Denoised image, [dictionary]
    """
    den = Denoise(image, patch_size, method='ksvd')
    den.patches.remove_mean()
    den.train(iters=iters, n_atoms=n_atoms, 
              verbose=verbose, n_threads=n_threads)

    new_image = den.denoise(sigma=sigma, n_threads=n_threads)

    if ret_dict:
        return new_image, den.dictionary

    return new_image
