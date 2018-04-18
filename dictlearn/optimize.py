# coding: utf-8
from __future__ import print_function, division
import sys
import warnings

import numpy as np
from numpy import linalg as LA
from scipy import linalg

from . import sparse, preprocess

"""
    Optimization algorithms for dictionary learning and sparse coding

"""


try:
    # Alias range as xrange if py2
    range = xrange
except NameError:
    pass  # py3


itkrmm_warning = """\n\
{}:
  Mask {} is all zeros, skipping all new signals with zero mask, without warning.
  Try increasing the patch size, or inpainting method algorithms.TextureSynthesis.
  In the case where there's no overlapping pixel values between two consecutive 
  patches, when masks[:, n-1]*masks[:, n] is all zeros -
  this may end with a linalg.LinalgError: SVD did not converge. Comments in code for
  more details.
"""


def project_c(mat):
    r"""
    Normalize columns using :math:`\ell _2` norm

    :param mat: Matrix to project
    :return: Updated matrix
    """
    for c in range(mat.shape[1]):
        col = mat[:, c]
        norm = LA.norm(col)
        if norm > 0:
            mat[:, c] /= norm

    return mat


def mod(signals, dictionary, n_nonzero, iters, n_threads=1):
    """
    Method of optimal directions

    The first alternate minimization algorithm [3] for dictionary
    learning.

    1. Find sparse codes A given signals X and dictionary D
    2. Update D given the new A by approximately solving for D in
       X = DA. That is D = X*pinv(A), with pinv(A) = A.T*(A*A.T)^-1

    :param signals:
        Training signals

    :param dictionary:
        Initial dictionary

    :param n_nonzero:
        Sparsity target for signal approximation

    :param iters:
        Number of dictionary update iterations

    :param n_threads: Default 1.
        Number of threads to use for sparse coding step

    :return:
        New dictionary

    """
    dictionary = dictionary.copy()

    for _ in range(iters):
        decomp = sparse.omp_batch(signals, dictionary, n_nonzero,
                                  n_threads=n_threads)
        dictionary = project_c(np.dot(signals, LA.pinv(decomp)))

    return dictionary


def ksvd(signals, dictionary, iters, n_nonzero=0, omp_tol=0,
         tol=0, verbose=False, n_threads=1, retcodes=False):
    r"""
        Iterative batch algorithm [1, 2] for fitting a dictionary D to a set of
        signals X. Each iteration consists of two stages:

            1. Fix **D**. Find sparse codes **A** such that **X** is approx equal **DA**
            2. Fix the sparse codes. Find **D_new** such that
               :code:`norm(X - D_new*A) < norm(X - DA)`


        Need one of (or both) :code:`n_nonzero` and :code:`omp_tol` different from zero.
        If :code:`n_nonzero > 0` and :code:`omp_tol == 0`
        then KSVD finds an approximate solution to:

        .. math::

            \underset{\mathbf{D, A}}{min}\quad \left \| \mathbf{X - DA} \right \|_F^2
            \text{ such that }
            \left \| \mathbf{A} \right \|_0 \leq n\_nonzero

        If omp_tol is not None then KSVD finds an approximate solution to:

        .. math::

            \underset{\mathbf{D, A}}{argmin}\quad\left \| \mathbf{A} \right \|_0
            \text{ such that }
            \left \| \mathbf{X - DA} \right \|_F^2 \leq omp\_tol


        >>> import dictlearn as dl
        >>> from numpy import linalg as LA
        >>> image = dl.Patches(dl.imread('some-image'), 8).patches
        >>> dictionary = dl.random_dictionary(8*8, 128)
        >>> sparse_1 = dl.omp_batch(image, dictionary, 10)
        >>> new_dict, _ = dl.ksvd(image, dictionary, 20, 10)
        >>> err_initial = LA.norm(image - dictionary.dot(sparse_1))
        >>> sparse_2 = dl.omp_batch(image, new_dict, 10)
        >>> err_trained = LA.norm(image - new_dict.dot(sparse_2))
        >>> assert err_trained < err_initial


        :param signals:
            Training signals. One signal per column
            numpy.ndarray of shape (signal_size, n_signals)

        :param dictionary:
            Initial dictionary, shape (signal_size, n_atoms)

        :param iters:
            Max number of iterations

        :param n_nonzero: Default 0.
            Max nonzero coefficients in sparse decomposition

        :param omp_tol: Default 0.
            Tolerance of sparse approximation. Overrides n_nonzero

        :param tol: Default 0.
            Stop learning if :code:`norm(signals - dictionary.dot(sparse_codes) < tol`

        :param verbose:
            Print progress

        :param n_threads: Default 1.
            Number of threads to use for sparse coding.

        :param retcodes:
            Return sparse codes from last iteration

        :return:
            dictionary[, sparse decomposition if retcodes = True]
    """
    m, p = dictionary.shape
    iters = int(iters)
    decomp = None

    if isinstance(signals, preprocess.Patches):
        signals = signals.patches

    for t in range(iters):
        if verbose:
            print('K-SVD iteration %d/%d' % (t + 1, iters))

        if verbose:
            print('  OMP: Finding sparse decomposition', end='...')
            sys.stdout.flush()

        decomp = sparse.omp_batch(signals, dictionary, n_nonzero=n_nonzero,
                                  tol=omp_tol, n_threads=n_threads)

        if verbose:
            print('DONE')

        if tol > 0 and LA.norm(signals - dictionary.dot(decomp)) < tol:
            return dictionary, decomp

        if verbose:
            print('  K-SVD: Updating dictionary', end='...')
            sys.stdout.flush()

        for k in range(p):
            row_k = decomp[k]
            # Index of coeffs using atom number k
            w = np.nonzero(row_k)[0]

            if len(w) == 0:
                continue  # ksvd_replace_atom()

            dictionary[:, k] = 0

            g = decomp[k, w]
            decomp_w = decomp[:, w]
            signals_w = signals[:, w]
            dict_d_w = dictionary.dot(decomp_w)
            d = signals_w.dot(g) - dict_d_w.dot(g)
            d /= LA.norm(d)
            g = signals_w.T.dot(d) - dict_d_w.T.dot(d)

            dictionary[:, k] = d
            decomp[k, w] = g

        if verbose:
            print('DONE')

    return dictionary, decomp


def odl(signals, dictionary, iters=1000, n_nonzero=10, tol=0, 
        verbose=False, batch_size=1, n_threads=1, seed=None):
    """
        Online dictionary learning algorithm

        This algorithm sparsely encode one training signal at the time and updates
        the dictionary given this signal. The number if iterations also determines how
        many of the training signals are used. If the number of iterations is less than
        the number of signals, then :code:`iters` signals is drawn at random. If iters is
        equal to the number of signals all signals are used in a random order.

        The dictionary atoms are updated using block-coordinate descent.
        See [4] for details

    :param signals:
        Training signals. One signal per column
        numpy.ndarray of shape (signal_size, n_signals)

    :param dictionary:
        Initial dictionary, shape (signal_size, n_atoms)

    :param iters: Default 1000.
        Number of training iterations to use. This is also equal to the number of
        signals used in training.

    :param n_nonzero: Default 10.
        Max nonzero coefficients in sparse decomposition

    :param tol: Default 0.
        Tolerance of sparse approximation. Overrides n_nonzero

    :param verbose:
        Print progress

    :param batch_size:
        The number of signals to use for each dictionary update

    :param seed:
        Seed the drawing of random signals

    :return:
        Trained and improved dictionary
    """
    n_signals = signals.shape[1]
    if iters == n_signals:
        indices = range(n_signals)
    else:
        np.random.seed(seed)
        indices = np.random.randint(0, n_signals, iters)

    m, p = dictionary.shape
    A = np.zeros((p, p))
    B = np.zeros((m, p))

    if verbose:
        print('ODL:')

    for i, t in enumerate(indices):
        if verbose:
            print('{:10d}/{}'.format(i + 1, iters), end='\r')

        x = signals[:, t]
        alpha = sparse.omp_cholesky(x, dictionary, n_nonzero=n_nonzero, tol=tol)
        alpha = alpha.reshape(-1, alpha.shape[0])
        A += 0.5*np.dot(alpha, alpha.T)
        tmp = np.zeros((p, m))
        tmp[:] = x
        B += tmp.T * alpha

        dictionary = _online_dict_update(dictionary, A, B)

    if verbose:
        print()
    
    return dictionary


def _online_dict_update(dictionary, A, B):
    """

    :param dictionary: assume shape (m, p)
    :param A: assumes shape (p, p)
    :param B: assumes shape (m, p)
    :return: updated dictionary
    """
    for j in range(dictionary.shape[1]):
        u_j = 1/A[j, j]*(B[:, j] -
                         np.dot(dictionary, A[:, j])) + dictionary[:, j]
        d_j = 1/max(LA.norm(u_j), 1)*u_j
        dictionary[:, j] = d_j

    return dictionary


def itkrmm(signals, masks, dictionary, n_nonzero, iters, low_rank=None, verbose=False):
    """
    
    Train a dictionary from corrupted image patches.

    Need signals and masks of same shape. Data point :code:`signals[i, j]` is used if the
    corresponding point in the mask, :code:`masks[i, j] == True`.
    All points :code:`signals[i, j]` with :code:`masks[i, j] == False` are ignored.

    See [5] for details.
    
    :param signals:
        Corrupted image patches, shape (patch_size, n_patches)

    :param masks:
        Binary mask for signals, same shape as signal.

    :param dictionary:
        Initial dictionary (patch_size, n_atoms)

    :param n_nonzero:
        Number of nonzero coeffs to use for training

    :param iters:
        Max number of iterations

    :param low_rank:
        Matrix of low rank components, shape (patch_size, n_low_rank)

    :param verbose:
        Print progress

    :return:
        Dictionary. Shape (patch_size, n_atoms + n_low_rank)
    """
    signals = signals.copy()
    signal_size, n_signals = signals.shape
    n_atoms = dictionary.shape[1]

    # This algorithm doesn't really work for all zero masks
    # Need to skip all these signal, print warning on first seen
    # all zero mask, silently skips all other
    is_warned = False

    if low_rank is None:
        n_low_rank = 0
    else:
        n_low_rank = low_rank.shape[1]

    signals *= masks

    if n_low_rank > 0:
        LL = np.dot(low_rank, low_rank.T)
        dictionary -= np.dot(LL, dictionary)

        for k in range(n_atoms):
            norm = LA.norm(dictionary[:, k])
            dictionary[:, k] /= norm

        # Remove low rank from signals
        for n in range(n_signals):
            lr_dict = low_rank * np.outer(masks[:, n], np.ones(n_low_rank))
            p_lr = np.dot(LA.pinv(lr_dict), signals[:, n])
            signals[:, n] -= np.dot(lr_dict, p_lr)

    dict_prev = dictionary.copy()
    d = signal_size
    K = n_atoms + n_low_rank
    N = n_signals
    S = n_nonzero

    if verbose:
        print('ITKrmm:')

    for t in range(iters):
        if verbose:
            sys.stdout.write('  %03d/%d\r' % (t + 1, iters))
            sys.stdout.flush()

        dictionary = np.zeros((d, K), float)
        mask_weight = np.zeros((d, K), float)

        for n in range(N):  # N >> 1e5
            mask_n = masks[:, n]
            signal_n = signals[:, n]

            if np.all(mask_n == 0):
                if not is_warned:
                    warnings.warn(itkrmm_warning.format('itkrmm', n),
                                  RuntimeWarning)
                    is_warned = True

                continue

            # SKipping patches:
            # When the signals are very corrupted and two consecutive masks
            # have no overlapping seen data points, all entries
            # signals[:, n-1]*signals[:, n] are zero. Whenever this happens
            # Some part of a dictionary atom (same as zero entries in signal n)
            # turns to zero in iteration n, and in iteration n+1
            # when the dictionary from the prev iteration is multiplied with current
            # signal the nonzero entries of the dict and signal don't overlap which
            # leads to zeros in prod__. Because of zeros in dict_prev there's zeros
            # in norm_nonzero and prod = prod__ / norm_nonzero results in nan
            # for those indices where signal[:, n+1] and dict_prev[:, x] are both zero
            # The problem is the zeros in atom x in dict_prev

            not_zero = np.where(mask_n > 0)[0]
            norm_nonzero = LA.norm(dict_prev[not_zero, :], axis=0)

            prod__ = np.dot(dict_prev.T, signal_n)
            prod = prod__ / norm_nonzero

            abs_prod = np.abs(prod)
            sign_prod = np.sign(prod)
            In = np.argsort(abs_prod)[::-1]
            In_S = In[:S]  # S biggest coeffs, hard thresh
            dInm__ = dict_prev[:, In_S] * np.outer(mask_n, np.ones(S))
            dInm = dInm__ / norm_nonzero[In_S]

            if n_low_rank > 0:
                lr_dict_masked = low_rank * np.outer(mask_n, np.ones(n_low_rank))
                dILnm = np.concatenate((lr_dict_masked, dInm), axis=1)
                inv_dILnm = LA.pinv(dILnm)  # May fail if warned
                lhs_ = (np.zeros(n_low_rank), prod[In_S])
                lhs = np.concatenate(lhs_, axis=0)
                res = np.real(signal_n - np.dot(inv_dILnm.T, lhs))
            else:
                inv = LA.pinv(dInm)  # May fail is warned
                res = signal_n - np.dot(inv.T, prod[In_S])

            dictionary[:, In_S] += np.outer(res, sign_prod[In_S].T * np.ones(S)) + \
                                   np.dot(dInm, np.diag(abs_prod[In_S]))

            mask_weight[:, In_S] += np.outer(mask_n, np.ones(S))

        if mask_weight.min() > 0:
            dictionary /= mask_weight
            dictionary *= N
        else:
            dictionary /= mask_weight + 1e-3
            dictionary *= N

        if n_low_rank > 0:
            dictionary -= low_rank.dot(low_rank.T).dot(dictionary)

        norms = LA.norm(dictionary, axis=0)
        zeros = np.where(norms < 1e-3)[0]

        dictionary[:, zeros] = np.random.randn(d, zeros.size)
        norms[zeros] = LA.norm(dictionary[:, zeros], axis=0)
        dictionary /= norms
        dict_prev = dictionary.copy()

    if low_rank is not None:
        return np.concatenate((low_rank, dict_prev), axis=1)

    return dict_prev


def reconstruct_low_rank(signals, masks, n_low_rank, initial=None, iters=10):
    """
    Reconstruct low rank components from image patches, by ITKrMM.

    Low rank components or atoms capture low rank signal features. In the case where
    signals are image patches low rank atoms can capture average intensities and low
    variance features in the image. When these are included in a dictionary most of the
    signals will use at least one of the low rank atoms leaving the normal atoms to
    represent more specific image features

    :param signals:
        Image patches

    :param masks:
        Masks for image patches

    :param n_low_rank:
        Number of low rank components to reconstruct

    :param initial:
        Initial low rank dictionary, shape (signals.shape[0], n_low_rank)

    :param iters:
        Number of iterations for each component

    :return:
        Low rank dictionary, shape (signals.shape[0], n_low_rank)
    """
    signals = signals.copy()
    if signals.shape != masks.shape:
        raise ValueError('Need signals and masks of same shape')

    d, N = signals.shape

    if n_low_rank == 0:
        raise ValueError('Need n_low_rank >= 1, not {}'.format(n_low_rank))

    if initial is None:
        initial = np.random.rand(d, n_low_rank)
        initial /= LA.norm(initial)

    lrc = None

    for i in range(n_low_rank):
        atom = initial[:, i]

        if i > 0:
            atom -= np.dot(lrc, lrc.T).dot(atom)

        atom = _reconstruct_low_rank(signals, masks, lrc, iters, atom)

        if lrc is not None:
            lrc = np.concatenate((lrc, atom[:, np.newaxis]), axis=1)
        else:
            lrc = atom.copy().reshape(atom.shape[0], 1)

    return lrc


def _reconstruct_low_rank(signals, masks, dictionary, iters, init_atom):
    """
        Reconstruct one low rank component using ITKrMM algorithm

    :param signals:
        Signals to recover from

    :param masks:
        Masks for signal

    :param dictionary:
        Matrix of already extracted low rank components

    :param iters:
        Number of iterations for extraction

    :param init_atom:
        Initial atom

    :return:
        Low rank atom, shape (signals.shape[0], )
    """
    d, N = signals.shape
    signals *= masks

    # If one (or more) of the signals is complete missing
    # --> mask[:, i] is zeros only this signal has to be skipped
    is_warned = False

    if dictionary is None:
        dictionary = np.zeros(d)
        L = 0
    else:
        L = dictionary.shape[1]

    if L > 0:
        init_atom -= np.dot(dictionary, dictionary.T).dot(init_atom)
        init_atom /= LA.norm(init_atom)

        # Remove low rank components from patches
        for n in range(N):
            masked_dict = dictionary*np.outer(masks[:, n], np.ones(L))
            inner = np.dot(LA.pinv(masked_dict), signals[:, n])
            signals[:, n] -= masked_dict.dot(inner)

    prev_a = init_atom

    for _ in range(iters):
        ip = np.dot(prev_a.T, signals)
        mask_weight = np.sum(masks, axis=1)

        if L == 0:
            atom = np.dot(signals, np.sign(ip.T))
        else:
            atom = np.zeros(d)
            for n in range(N):
                if np.all(masks[:, n] == 0):
                    if not is_warned:
                        warnings.warn(itkrmm_warning.format('reconstruct_low_rank', n),
                                      RuntimeWarning)
                        is_warned = True

                    continue

                prev_a_masked = prev_a * masks[:, n]
                new = dictionary*np.outer(masks[:, n], np.ones(L))
                dico = np.concatenate((new, prev_a_masked[:, np.newaxis]), axis=1)
                prod = dico.dot(np.dot(LA.pinv(dico), signals[:, n]))
                res = signals[:, n] - prod
                atom += np.sign(ip[n]) * res
                prev_a_masked /= LA.norm(prev_a_masked)**2
                atom += np.abs(ip[n]) * prev_a_masked

        if mask_weight.min() > 0:
            atom /= mask_weight
        else:
            atom /= mask_weight + 1e-2

        if L > 0:
            atom -= np.dot(dictionary, np.dot(dictionary.T, atom))

        atom /= LA.norm(atom)
        prev_a = atom

    return prev_a


def sparse_ksvd(signals, base_dict, dict_repr, atom_sparse,
                target_sparse, iters, n_threads=1):
    r"""
    An extension of the K-SVD algorithm. The idea is that the
    dictionaries has an underlying sparse structure such that
    :math:`D = \Phi\mathbf{A}`, and :math:`x = \Phi\mathbf{Ay}`.
    \Phi is a (separable) base dictionary and A dictionary sparse
    codes.

    With A being sparse there should be a substantial speedup in the
    first sparse coding stage. Leading to a total speedup of
    5x - 30x depending on signals and sparsity targets.

    This implementation is not faster than K-SVD nor does it find
    a good decomposition of the training signals.

    :param signals: Training signals
    :param base_dict: Separable base dictionary representation
    :param dict_repr: Sparse dictionary representation
    :param atom_sparse: Sparsity target for the dictionary
    :param target_sparse: Sparsity target for signal representation
    :return: sparse codes of dictionary, sparse codes of signals
    """
    A = dict_repr.copy()
    for _ in range(iters):
        # Sparse decomposition
        decomp = sparse.omp_batch(signals, np.dot(base_dict, A), target_sparse,
                                  n_threads=n_threads)
        print(type(decomp), decomp.min(), decomp.max(),
              np.count_nonzero(decomp), decomp.size)

        assert decomp.shape[0] == A.shape[1]
        assert decomp.shape[1] == signals.shape[1]

        # Something grows to inf and A -> nan
        for j in range(A.shape[1]):
            I = np.nonzero(decomp[j])[0]

            if len(I) == 0:
                continue

            A[:, j] = 0

            g = decomp[j, I].T
            g /= linalg.norm(g)

            D = np.dot(base_dict, A)
            z = signals[:, I].dot(g) - np.dot(D.dot(decomp[:, I]), g)
            a = sparse.omp_cholesky(z, base_dict, atom_sparse)
            Da = np.dot(base_dict, a)
            a /= linalg.norm(Da)

            A[:, j] = a
            t1 = np.dot(signals[:, I].T, Da)
            t2 = np.dot(base_dict.dot(A), decomp[:, I]).T
            t3 = np.dot(t2, Da)
            final = (t1 - t3).T
            decomp[j, I] = final

    return base_dict, A

