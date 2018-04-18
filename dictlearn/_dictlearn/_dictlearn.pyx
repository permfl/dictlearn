#!python
#cython: wraparound=False, boundscheck=False, cdivision=True
from __future__ import print_function
import numpy as np
cimport numpy as np

np.import_array()


ctypedef long long size
ctypedef unsigned long PySize


def omp_cholesky(signals, dictionary, n_nonzero=0, tol=0, n_threads=1):
    """
        :param signals:
            Signals to encode, shape (signal_size, n_signals)

        :param dictionary:
            shape (signal_size, n_atoms)

        :param n_nonzero:
            Number of nonzero coeffs

        :param tol:
            Accuracy of approx, overrides n_nonzero

        :param n_threads:
            Number of threads to use

        :return:
            Sparse codes, shape (n_atoms, n_signals)
    """
    cdef size_t signal_size = signals.shape[0]
    cdef size_t n_signals
    
    if signals.ndim == 1:
        n_signals = 1
        signals = signals[:, np.newaxis]
    elif signals.ndim == 2:
        n_signals = signals.shape[1]  
    else:
        raise ValueError('Max 2D signals')

    if n_nonzero == 0 and tol == 0:
        raise ValueError('Both n_coeffs and tol can\'t be zero')

    if signal_size != dictionary.shape[0]:
        raise ValueError('Invalid dimensions, need first dim of '
            'dictionary to be of same size as signal size')

    cdef size_t n_nonzero_ = n_nonzero
    cdef double tolerance = tol
    cdef size_t n_atoms = dictionary.shape[1]
    cdef size_t threads = n_threads

    cdef double[:, :] signals_ = np.ascontiguousarray(signals.T)
    cdef double[:, :] alpha = np.zeros((n_signals, n_atoms), dtype=np.float64)
    cdef double[:, :] dict_ = np.ascontiguousarray(dictionary)
    cdef double[:, :] dict_t = np.ascontiguousarray(dictionary.T)

    # Signals and alpha transposed gives fewer cache missed. Since we're
    # reading and writing to these one column at the time it's cheaper
    # that entries in each column are close together in memory

    # Release the GIL such that we can use multiple threads. Not sure if
    # this is necessary since the C code uses no python objects
    with nogil:
        _omp_cholesky(&signals_[0, 0], signal_size, n_signals,
                      &dict_[0, 0], n_atoms, &dict_t[0, 0],
                      &alpha[0, 0], n_nonzero_, threads, tolerance)


    return np.squeeze(np.asarray(alpha).T)


def omp_batch(np.ndarray[double, ndim=2] signals, 
              np.ndarray[double, ndim=2] dictionary,
              n_nonzero=0, tol=0, n_threads=1):
    """

    :param signals:
        Signals to encode, shape (signal_size, n_signals)

    :param dictionary:
        shape (signal_size, n_atoms)

    :param n_nonzero:
        Number of nonzero coeffs

    :param tol:
        Accuracy of approx, overrides n_nonzero

    :param n_threads:
        Number of threads to use

    :return:
        Sparse codes, shape (n_atoms, n_signals)
    """
    if signals.ndim == 1:
        raise ValueError('Cannot do Batch-OMP on a single signal')

    tol = 0 if tol is None else tol
    if n_nonzero == 0 and tol == 0:
        raise ValueError('Need a stopping criteria. ' +
                         'Set n_nonzero_coeffs or tolerance > 0')

    cdef size_t signal_size = signals.shape[0]
    cdef size_t n_signals = signals.shape[1]
    cdef size_t n_atoms = dictionary.shape[1]
    cdef size_t n_th = n_threads 

    if signal_size != dictionary.shape[0]:
        raise ValueError(
            'Invalid dimensions, need first dim of '
            'dictionary to be of same size as signal size'
        )

    cdef double target_error = tol
    cdef size_t max_n_nonzero = n_nonzero

    cdef double[:, :] alpha = np.dot(signals.T, dictionary)
    cdef double[:, :] gram = np.dot(dictionary.T, dictionary)
    cdef double[:] norms = np.linalg.norm(signals, ord=2, axis=0)**2
    cdef double[:, :] out = np.zeros((n_signals, n_atoms), dtype=np.float64)

    cdef int result

    with nogil:
        res = _omp_batch(&alpha[0, 0], n_signals, n_atoms, 
                         &norms[0], &gram[0, 0], target_error,
                         max_n_nonzero, n_th, &out[0, 0])

    if res < 0:
        raise MemoryError('Could not allocate the required memory')

    return np.squeeze(np.asarray(out).T)


def gsr_patch_search(patches, size row, size col, size this, size group_size,
                     size window_size, indices):
    cdef size r_min
    cdef size r_max
    cdef size c_min
    cdef size c_max
    cdef size N
    cdef size M
    cdef size aa
    cdef size bb
    N = indices.shape[0]
    M = indices.shape[1]

    # Finds patch indices
    aa = row - window_size
    if aa > 0:
        r_min = aa
    else:
        r_min = 0

    aa = row + window_size + 1
    bb = N + 1
    if aa > bb:
        r_max = bb
    else:
        r_max = aa

    aa = col - window_size
    if aa > 0:
        c_min = aa
    else:
        c_min = 0

    aa = col + window_size + 1
    bb = M + 1
    if aa > bb:
        c_max = bb
    else:
        c_max = aa

    # Index of patches in search window
    idx = indices[r_min:r_max, c_min:c_max]
    idx = idx.flatten('F')

    B = patches[idx, :]  # Patches within search window
    v = patches[this, :]  # Current patch

    # Squared L2 distance between current and all other patches in
    # search window. Try different similarity measures
    distance = np.sum((B - v) ** 2, axis=1)
    # distance /= patch_size

    # Don't need to sort everything
    # Keep 'group_size' largest in some fitting ds
    ind = np.argsort(distance)

    # Pick index on the 'group_size' closest patches
    best_matching = idx[ind[:group_size]]
    best_matching[0] = this
    return best_matching


def prow_idx(int patch_index, int height):
    if patch_index + 1 % height == 0:
        row = height
    else:
        row = patch_index % height

    return row


def pcol_idx(int patch_index, int width):
    return patch_index // width


def best_patch(np.ndarray[double, ndim=2] image,
               np.ndarray[double, ndim=2] patch,
               np.ndarray[unsigned int, ndim=2] to_fill,
               np.ndarray[unsigned int, ndim=2] source):
    """ Wrapper for bestexemplar in bestexemplar.c """

    cdef size_t height = image.shape[0]
    cdef size_t width = image.shape[1]
    cdef size_t patch_height = patch.shape[0]
    cdef size_t patch_width = patch.shape[1]
    cdef unsigned int[:] best = np.empty(4, np.uint32)

    bestexemplar(&image[0, 0], height, width,
                 &patch[0, 0], patch_height, patch_width,
                 &to_fill[0, 0], &source[0, 0], &best[0])

    return np.asarray(best, np.uint32)


def best_patch_3d(np.ndarray[double, ndim=3] image,
                  np.ndarray[double, ndim=3] patch,
                  np.ndarray[unsigned int, ndim=3] to_fill,
                  np.ndarray[unsigned int, ndim=3] source):
    """ Wrapper bestexemplar_3d in bestexemplar.c """
    cdef size_t height = image.shape[0]
    cdef size_t width = image.shape[1]
    cdef size_t depth = image.shape[2]

    cdef unsigned int patch_height = patch.shape[0]
    cdef unsigned int patch_width = patch.shape[1]
    cdef unsigned int patch_depth = patch.shape[2]
    print(patch_height, patch_width, patch_depth)
    cdef unsigned int[:] best = np.empty(6, np.uint32)

    bestexemplar_3d(&image[0, 0, 0], height, width, depth,
                    &patch[0, 0, 0], patch_height, patch_width, patch_depth,
                    &to_fill[0, 0, 0], &source[0, 0, 0], &best[0])
    
    return np.asarray(best, np.uint32)


# Include wrappers for testing
include "_dictlearn_test_wrappers.pxi"

