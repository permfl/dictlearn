from __future__ import print_function, division

import operator
from functools import partial
import numpy as np
from numpy import linalg

# TODO: Drop using these. Write stuff in C
from sklearn.feature_extraction.image import (
    extract_patches_2d, reconstruct_from_patches_2d
)

try:
    # use the generator xrange if py2
    range = xrange
    reduce
except NameError:
    # Python 3
    from functools import reduce


from . import utils

# TODO
AVAILABLE_MEMORY = 2  # GB
# os.system('wmic computersystem get TotalPhysicalMemory')
# os.system('cat /proc/meminfo')


class Patches(object):
    REMOVE_MEAN = 'remove_mean'
    """
        Generate and reconstruct image patches

        :param image:
            ndarray, 2D or 3D

        :param size:
            Patch size, since all patches are square (cube) this is just \
            the size of the first dimension. Ie 8 for (8, 8) patches
            
        :param stride:
            Stride/distance between patches in image. Can be int or list type. If int
            then the stride is the same in every dimension. If list then each stride[i] 
            denotes the stride on axis i. Patches cannot be reconstructed if the stride 
            in one dimension is larger the than the patch size in the same dimension. Ie.
            stride[i] > size[i] for any i

        :param max_patches:
            Maximum number of patches

        :param random:
            True for taking patches from random locations in image. \
            Overwritten if max_patches=None

        :param order:
            C or F for C or FORTRAN order on underlying data
    """
    def __init__(self, image, size, stride=1, max_patches=None, random=None, order='C'):
        self.image = image
        self.ndim = self.image.ndim
        self._shape = self.image.shape
        if self.ndim not in [2, 3]:
            raise ValueError('Image is of unsupported dimensions, {}. Only'
                             ' 2D or 3D images'.format(self.ndim))

        if isinstance(size, int):
            self._size = size * np.ones(self.ndim, dtype=int)
        elif len(size) == self.ndim:
            self._size = size
        else:
            raise ValueError('Size has to be int or list/array with '
                             'len(size) == image.ndim')

        if isinstance(stride, int):
            self.stride = stride * np.ones(self.ndim, dtype=int)
        elif len(stride) == self.ndim:
            self.stride = stride
        else:
            raise ValueError('Stride has to be int or list/array with '
                             'len(stride) == image.ndim')

        self.patch_size = reduce(operator.mul, self._size)
        self._patches = None
        self._raw_patches_shape = None  # Shape of patches from sklearn
        self.order = order
        self.reconstructed = None  # For storing reconstructed image
        self.weights = None  # Weights for pixel averaging
        self.rgb = self.ndim == 3 and self.image.shape[2] == 3

        if max_patches is not None:
            self.max_patches = max_patches
            if self.n_patches < self.max_patches:
                raise ValueError('Cannot extract more than all patches. '
                                 ' n_patches < max_patches')
        else:
            self.max_patches = max_patches

        if random is not None:
            if isinstance(random, (int, float)):
                self.random = int(random)
            else:
                self.random = np.random.randint(int(10e6))
        else:
            self.random = None

        self._mean = 0
        self.ops = []

    @property
    def patches(self):
        """
        :return:
            Image patches, shape (size[0]*size[1]*..., n_patches)
        """
        if self._patches is None:
            self._construct()

            if len(self.ops) > 0:
                self._do_ops()

        return self._patches

    @property
    def shape(self):
        """
           Shape of patch matrix, (patch_size, n_patches)
        """
        if self._patches is None:
            return self.patch_size, self.n_patches

        return self._patches.shape

    @property
    def size(self):
        """
            Size of patches
        """
        if np.all(self._size == self._size[0]):
            return self._size[0]

        return self._size

    @property
    def n_patches(self):
        """

        :return:
            Number of patches
        """
        if self._patches is None:
            return self._check_size(False)

        return self.patch_size[1]

    def reconstruct(self, new_patches, save=False):
        """
        Reconstruct the image with new_patches. Overlapping
        regions are averaged. The reconstructed patches are not saved by default

        self.patches are the same object before and after this method is called,
        as long as save=False

        :param new_patches:
            `ndarray` (patch_size, n_patches). Patches returned from Patches.patches

        :param save:
            Overwrite current patches with new_patches

        :return:
            Reconstructed image
        """
        if self.random is not None or self.max_patches is not None:
            raise ValueError('Cannot reconstruct when random or '
                             'max_patches is not None')

        if self.order != 'C':
            raise ValueError('Can only reconstruct C ordered patches')

        new_patches += self._mean

        if save:
            self._patches = new_patches

        if self.ndim == 2:
            p = new_patches.T.reshape(self._raw_patches_shape)
            reconstructed_image = reconstruct_from_patches_2d(
                patches=p, image_size=self._shape
            )
        elif self.ndim == 3:
            if self.rgb:
                raise NotImplementedError()
            else:
                return self._reconstruct_3d(new_patches)
        else:
            raise ValueError()

        return reconstructed_image

    def generator(self, batch_size, callback=False):
        """

        Create and reconstruct a batch iteratively.

        If Patches.patches is too large to keep all in memory use this. Only
        'batch_size' patches are generated. This requires approximately 'batch_size'
        times less memory. If batch_size is 100 and Patches.patches need 100 memories
        then this need only one memory.

        >>> import numpy as np
        >>> volume = np.load('some_image.npy')
        >>> size, stride = [10, 10, 10], [1, 1, 1]
        >>> patches = Patches(volume, size, stride)
        >>> for batch in patches.generator(100):
        >>>     # Handle batch
        >>>     assert batch.shape[1] == 100
        >>>     assert batch.shape[0] == 1000, 'Can fail at last batch, see stride'


        One matrix of size (patch_size, batch_size) is created per iteration.
        This generator return (batch, callback) with batch a numpy array of shape
        (patch_size, batch_size) and callback(batch) reconstruct the part of the
        volume which contains the given batch. It is required that the argument to
        callback has shape identical to the batch returned

        This can be used if Patches3D.create() requires too much memory. The amount
        of memory required by this method is\
            batch_size*size[0]*size[1]*size[2]*volume.dtype.itemsize bytes

        >>> import numpy as np
        >>> volume = np.load('some_image_volume.npy')
        >>> size, stride = [10, 10, 10], [1, 1, 1]
        >>> patches = Patches(volume, size, stride)
        >>> for batch, reconstruct in patches.generator(100, callback=True):
        >>>    # Handle batch, here we do nothing
        >>>    reconstruct(batch)
        >>> assert np.array_equal(volume, patches.reconstructed)

        :param batch_size:
            Size of batches. The last batch can be smaller if
            n_patches % batch_size != 0

        :param callback:
            If True a callback function 'callback(batch)' is returned such the the image
            can be partially reconstructed

        :return:
            Generator
        """
        if self.ndim == 2:
            raise NotImplementedError()

        if callback:
            self.check_batch_size_or_raise(batch_size)
            self.reconstructed = np.zeros_like(self.image)
            self.weights = np.zeros_like(self.image)
            return self._next_batch_3d(batch_size, True)
        else:
            self.check_batch_size_or_raise(batch_size)
            return self._next_batch_3d(batch_size)

    def remove_mean(self, add_back=True):
        """
            Remove the mean from every patch, this is automatically
            added back if the image is reconstructed

            :param add_back:
                Automatically add back the mean to patches on reconstruction
        """
        if self._patches is None:
            self.ops.append((Patches.REMOVE_MEAN, (add_back, )))
            return

        self._patches, self._mean = center(self._patches, retmean=True)

        if not add_back:
            self._mean = 0

    def _next_batch_3d(self, batch_size, reconstruct=False):
        """
            Creates 3d patches iteratively, keep only batch_size
            patches in memory
        """
        sx, sy, sz = self._size
        x, y, z = self._shape

        # Keep batches transposed st. write to row not column.
        # Fewer cache misses -> 50% faster
        batch = np.zeros((batch_size, self.patch_size),
                         dtype=self.image.dtype)
        indices = []
        # todo cython
        s = 0
        kk = 0
        for i in range(0, x - sx + 1, self.stride[0]):
            for j in range(0, y - sy + 1, self.stride[1]):
                for k in range(0, z - sz + 1, self.stride[2]):
                    a = self.image[i:i + sx, j:j + sy, k:k + sz].flatten()
                    batch[s] = a
                    s += 1

                    if reconstruct:
                        indices.append((kk, (i, i + sx), (j, j + sy), (k, k + sz)))
                        kk += 1

                    if s == batch_size:
                        if len(self.ops) > 0:
                            self._patches = batch.T
                            self._do_ops(clear=False)
                            batch, self._patches = self._patches.T, None

                        if reconstruct:
                            yield batch.T, partial(self._receive, indices)
                        else:
                            yield batch.T

                        s = 0
                        indices = []

        if len(self.ops) > 0:
            self._patches = batch[:s].T
            self._do_ops(clear=False)
            batch[:s], self._patches = self._patches.T, None

        if reconstruct:
            yield batch[:s].T, partial(self._receive, indices[:s])
            idx = self.weights > 0
            self.reconstructed[idx] /= self.weights[idx]
        else:
            yield batch[:s].T

    def _receive(self, indices, batch):
        """

        :param indices:
            Indices of each patch,  (global_patch_number, (start axis0, end axis0), ...)

        :param batch:
            Reconstruct this batch, and put back in image on location specified
            by indices
        """
        # todo cython
        if self.reconstructed.dtype.kind == 'f' and \
                self.reconstructed.dtype != batch.dtype:
            batch = batch.astype(self.reconstructed.dtype)
        batch += self._mean
        for i, tup in enumerate(indices):
            a, x, y, z = tup
            # i, j, k = self._expand_index(a)
            self.reconstructed[x[0]:x[1], y[0]:y[1], z[0]:z[1]] += \
                batch[:, i].reshape(self.size[0], self.size[1], self.size[2])
            self.weights[x[0]:x[1], y[0]:y[1], z[0]:z[1]] += 1

    def _construct(self):
        """
            Create 2D or 3D patches and keep all in Patches.patches.
            Raise MemoryError is too much data
        """
        self._check_size()

        if self.ndim == 2:
            patches = extract_patches_2d(
                image=self.image, patch_size=self._size,
                max_patches=self.max_patches, random_state=self.random
            )
            self._raw_patches_shape = patches.shape
            self._patches = patches.reshape(patches.shape[0], -1,
                                            order=self.order).T
        elif self.ndim == 3:
            if self.rgb:
                raise NotImplementedError()
            else:
                self._construct_3d()
        else:
            raise ValueError('Only 2D or 3D Images')

    def _construct_3d(self):
        """
            Create flattened 3d patches
        """
        x, y, z = self.image.shape
        size = self._size
        stride = self.stride
        n_patches = self._check_size(False)
        self._patches = np.zeros((n_patches, size[0] * size[1] * size[2]),
                                 dtype=self.image.dtype)
        s = 0
        for i in range(0, x - size[0] + 1, stride[0]):
            for j in range(0, y - size[1] + 1, stride[1]):
                for k in range(0, z - size[2] + 1, stride[2]):
                    a = self.image[i:i + size[0], j:j + size[1],
                        k:k + size[2]].flatten()
                    self._patches[s] = a
                    s += 1

        self._patches = self._patches.T

    def _reconstruct_3d(self, new_patches):
        """
         Reconstruct patches created by Patches._construct_3d

        :return:
            Reconstructed image volume, shape (volume_shape)
        """
        size = np.asarray(self._size)
        stride = np.asarray(self.stride)
        volume = np.zeros(self._shape, dtype=new_patches.dtype)
        weights = np.zeros(self._shape, dtype=int)
        x, y, z = volume.shape

        if np.any(stride > size):
            raise ValueError('Cannot reconstruct volume when stride is larger '
                             'than patch size. Need np.all(stride <= size), '
                             'size={}, stride={}'.format(size, stride))

        s = 0
        patches = new_patches.T
        for i in range(0, x - size[0] + 1, stride[0]):
            for j in range(0, y - size[1] + 1, stride[1]):
                for k in range(0, z - size[2] + 1, stride[2]):
                    a = patches[s].reshape(size[0], size[1], size[2])
                    volume[i:i + size[0], j:j + size[1], k:k + size[2]] += a
                    weights[i:i + size[0], j:j + size[1], k:k + size[2]] += 1
                    s += 1

        idx = weights != 0
        volume[idx] /= weights[idx]
        return volume

    def _check_size(self, throw=True, retdims=False):
        """
            Check required memory and calculate number of patches

        :param throw:
            Raise MemoryError if too large

        :param retdims:
            Return the number of patches in each dimension if True

        :return:
            n_patches, or (n_patches, dims). n_patches is the total number of
            patches and dims a list where dims[i] the number of patches in dimension i
        """
        n_patches = 1
        dims = []
        for i in range(self.ndim):
            dim = self._shape[i] - self._size[i] + 1

            if self.stride[i] > 1:
                dim = dim // self.stride[i] + 1

            n_patches *= dim
            dims.append(dim)

        dtype_size = self.image.dtype.itemsize
        memory_req_gb = n_patches * self.patch_size * dtype_size * 1e-9

        if memory_req_gb > AVAILABLE_MEMORY and throw:
            msg = 'Not enough memory for patches, need {} GB.'
            msg += ' Use Patches.generator(batch_size)'
            raise MemoryError(msg.format(memory_req_gb))

        if retdims:
            return n_patches, dims

        return n_patches

    def check_batch_size_or_raise(self, batch_size):
        """
            Check if there's enough memory to store 'batch_size' patches.
            Raise MemoryError if not
        """
        memory = (batch_size * self.patch_size) * self.image.dtype.itemsize * 1e-9

        if memory > AVAILABLE_MEMORY:
            msg = 'Not enough memory for batches, need {} GB.'
            msg += ' Try smaller batch_size'
            raise MemoryError(msg.format(memory))

    def _expand_index(self, num):
        """
            Should expand 1D index into 3D index (x, y, z)
        """
        # todo fix strides
        if self.ndim == 3:
            num_z = self._shape[2] - self._size[2] + 1
            num_y = self._shape[1] - self._size[1] + 1
            k = self.stride[2] * num % num_z
            j = self.stride[1] * (self.stride[2] * num // num_z) % num_y
            i = self.stride[0] * (num // (num_z * num_y //
                                          (self.stride[2] * self.stride[1])))
            return i, j, k
        else:
            raise NotImplementedError()

    def _do_ops(self, clear=True):
        """
        Do operations on patches. Since the patches are created lazily,
        they may not be created when remove_mean, etc are called. Keep track
        of the operations to do in self.ops, and this can be called after the
        patches are created to execute those ops

        :param clear:
            Clear operation stack

        """
        if clear:
            while len(self.ops) > 0:
                attr, args = self.ops.pop()
                self._handle_op(attr, args)
        else:
            for attr, args in self.ops:
                self._handle_op(attr, args)

    def _handle_op(self, attr, args):
        if hasattr(self, attr):
            getattr(self, attr)(*args)


class Patches3D(Patches):
    """
        Create and reconstruct image patches from 3D volume.

        :param volume:
            3D ndarray

        :param size:
            Size of image patches, (x, y, z)

        :param stride:
            Stride between each patch, (i, j, k). 'volume' cannot be reconstructed
            if i > x, j > y or k > z
    """
    def __init__(self, volume, size, stride):
        super(Patches3D, self).__init__(volume, size, stride)
        import warnings
        warnings.warn('Patches3D deprecated, use Patches',
                      utils.VisibleDeprecationWarning)

    def next_batch(self, batch_size):
        """

        :param batch_size:
            Number of image patches per batch

        :return:
            Generator, next() returns a ndarray of shape (n, batch_size)
        """
        return self.generator(batch_size)

    def create_batch_and_reconstruct(self, batch_size):
        """
            Create and reconstruct a batch iteratively.

            One matrix of 'batch_size' is created per iteration. This generator return
            (batch, callback) with batch a numpy array of shape (n, batch_size) and
            callback(batch) reconstruct the part of the volume which contains the
            given batch.

            This can be used if Patches3D.create() requires too much memory. The amount
            of memory required by this method is\
                batch_size*size[0]*size[1]*size[2]*volume.dtype.itemsize bytes

            >>> import numpy as np
            >>> import dictlearn as dl
            >>> dictionary = np.load('some_dictionary.npy')
            >>> volume = np.load('some_image_volume.npy')
            >>> size, stride = [1, 1, 1], [1, 1, 1]

            >>> patches = Patches3D(volume, size, stride)
            >>> for batch, reconstruct in patches.create_batch_and_reconstruct(100):
            >>>    new_batch = dl.omp_batch(batch, dictionary)
            >>>    reconstruct(new_batch)

            >>> reconstructed_volume = patches.reconstructed

        :param batch_size:
            Number of patches per batch.

        :return:
            Generator, next() returns (batch, reconstruct(new_batch)
        """
        return self.generator(batch_size, callback=True)

    @property
    def volume(self):
        return self.image


def center(data, dim=0, retmean=False, inplace=False):
    """
    Remove the mean at dim from every patch

    :param data: 
        ndarray, data to center
    
    :param dim: 
        Dimension to calculate mean, default 0 (columns)
    
    :param retmean: 
        Return mean if True
    
    :param inplace: 
        Change argument data directly if True, returns mean only
    
    :return: 
        Centered patches and mean if retmean is True. 
        Or just mean if inplace is True
    """
    if inplace:
        mean = data.mean(dim)
        data -= mean
        return mean

    mean = data.mean(dim)
    data = data - mean
    return (data, mean) if retmean else data


def normalize(patches, lim=0.2):
    """
    L2 normalization. If l2 norm of a patch is smaller than lim
    the the patch is divided element wise by lim

    :param patches: ndarray, (size, n_patches)
    :param lim: Threshold for low intensity patches
    :return:
    """
    if patches.ndim == 1:
        return patches / max(linalg.norm(patches), lim)

    for i in range(patches.shape[1]):
        p = patches[:, i]
        patches[:, i] = p / max(linalg.norm(p), lim)

    return patches


