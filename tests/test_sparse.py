import os
import timeit
import numpy as np
from numpy import linalg as LA
from dictlearn import sparse
from dictlearn import utils
from dictlearn import filters
from dictlearn import preprocess

from sklearn.linear_model import orthogonal_mp_gram, orthogonal_mp

import pytest


@pytest.fixture
def sparse_signal():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    img = np.load(os.path.join(dir_path, 'test_img1.npy'))
    img = img.astype(float)
    img = filters.threshold(utils.normalize(img), 0.4)

    return preprocess.Patches(img, 8).patches


@pytest.fixture
def dense_signal():
    def f(img_num):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        img = np.load(os.path.join(dir_path, 'test_img%d.npy' % img_num))
        img = img.astype(float)
        return preprocess.Patches(img, 8).patches
    return f


def test_omp_batch_dense_signal_n_nonzero(dense_signal):
    signals = dense_signal(1)[:, :200]
    n_atoms = 256
    n_nonzero = 10
    n_threads = 1
    dictionary = utils.dct_dict(n_atoms, 8)
    sparse_c = sparse.omp_batch(signals, dictionary,
                                n_nonzero=n_nonzero, n_threads=n_threads)

    # Compare with sklearn
    Xy = np.dot(dictionary.T, signals)
    Gram = np.dot(dictionary.T, dictionary)
    sparse_sk = orthogonal_mp_gram(Gram, Xy, n_nonzero_coefs=n_nonzero)

    assert sparse_c.shape == sparse_sk.shape
    assert np.count_nonzero(sparse_c) <= n_nonzero*signals.shape[1]

    c = np.dot(dictionary, sparse_c)
    sk = np.dot(dictionary, sparse_sk)
    assert np.all(np.linalg.norm(c - sk, axis=0) < 1e-6)


def test_omp_batch_dense_signal_tol(dense_signal):
    signals = dense_signal(1)[:, :20]
    n_atoms = 256
    TOL = 100
    n_threads = 1
    dictionary = utils.dct_dict(n_atoms, 8)
    sparse_c = sparse.omp_batch(signals, dictionary,
                                tol=TOL, n_threads=n_threads)

    c = np.dot(dictionary, sparse_c)
    assert np.all(np.linalg.norm(c - signals, axis=0) < TOL)


def test_omp_batch_sparse_signal_n_nonzero(sparse_signal):
    signals = sparse_signal[:, :200]
    n_atoms = 256
    n_nonzero = 10
    n_threads = 1
    dictionary = utils.dct_dict(n_atoms, 8)
    sparse_c = sparse.omp_batch(signals, dictionary,
                                n_nonzero=n_nonzero, n_threads=n_threads)

    # Compare with sklearn
    Xy = np.dot(dictionary.T, signals)
    Gram = np.dot(dictionary.T, dictionary)
    sparse_sk = orthogonal_mp_gram(Gram, Xy, n_nonzero_coefs=n_nonzero)

    assert sparse_c.shape == sparse_sk.shape
    assert np.count_nonzero(sparse_c) <= n_nonzero * signals.shape[1]

    c = np.dot(dictionary, sparse_c)
    sk = np.dot(dictionary, sparse_sk)
    assert np.all(np.linalg.norm(c - sk, axis=0) < 1e-6)


def test_omp_batch_sparse_signal_tol(sparse_signal):
    signals = sparse_signal[:, :20]
    n_atoms = 256
    TOL = 100
    n_threads = 1
    dictionary = utils.dct_dict(n_atoms, 8)
    sparse_c = sparse.omp_batch(signals, dictionary,
                                tol=TOL, n_threads=n_threads)

    c = np.dot(dictionary, sparse_c)
    assert np.all(np.linalg.norm(c - signals, axis=0) < TOL)


def test_omp_cholseky_dense_signals(dense_signal):
    """
        Not entirely sure how to test this.
        Just comparing with scikit learn for now
    """
    n_atoms = 256
    n_nonzero = 10
    n_threads = 1

    dictionary = utils.dct_dict(n_atoms, 8)
    signals = dense_signal(1)[:, 12]
    codes = sparse.omp_cholesky(signals, dictionary,
                                n_nonzero=n_nonzero, n_threads=n_threads)
    sklearn_codes = orthogonal_mp(dictionary, signals, n_nonzero)

    c_sig = np.dot(dictionary, codes)
    sk_sig = np.dot(dictionary, sklearn_codes)

    assert codes.shape == sklearn_codes.shape
    assert np.count_nonzero(codes) == np.count_nonzero(sklearn_codes)
    assert abs(np.linalg.norm(c_sig - signals) - np.linalg.norm(sk_sig - signals)) < 1e-12

    n_nonzero = 10
    n_threads = 1

    dictionary = utils.dct_dict(n_atoms, 8)
    signals = dense_signal(2)[:, 10]
    codes = sparse.omp_cholesky(signals, dictionary,
                                n_nonzero=n_nonzero, n_threads=n_threads)
    sklearn_codes = orthogonal_mp(dictionary, signals, n_nonzero)

    c_sig = np.dot(dictionary, codes)
    sk_sig = np.dot(dictionary, sklearn_codes)

    assert codes.shape == (n_atoms,)
    assert codes.shape == sklearn_codes.shape
    assert np.count_nonzero(codes) <= np.count_nonzero(sklearn_codes)
    assert abs(np.linalg.norm(c_sig - signals) - np.linalg.norm(sk_sig - signals)) < 1e-12


def test_omp_cholesky_sparse_signal(sparse_signal):
    n_atoms = 256
    n_nonzero = 10
    n_threads = 1

    dictionary = utils.dct_dict(n_atoms, 8)
    signals = sparse_signal[:, 10]
    codes = sparse.omp_cholesky(signals, dictionary,
                                n_nonzero=n_nonzero, n_threads=n_threads)
    sklearn_codes = orthogonal_mp(dictionary, signals, n_nonzero)

    c_sig = np.dot(dictionary, codes)

    assert codes.shape == (n_atoms,)
    assert np.count_nonzero(codes) == np.count_nonzero(sklearn_codes)
    assert np.linalg.norm(c_sig - signals) < 1e-12


def test_omp_cholesky_tol(dense_signal):
    n_atoms = 256
    n_threads = 1
    signals = dense_signal(1)
    # Find sparse approx, codes, of signal such that |signal - D*codes|_2 < TOL
    # In practice (denoising etc) this tolerance is often of order 10^3
    TOL = 0.5

    dictionary = utils.dct_dict(n_atoms, 8)
    # Check if single signal is within given tolerance
    signals_ = signals[:, 100]
    codes = sparse.omp_cholesky(signals_, dictionary, tol=TOL, n_threads=n_threads)
    new_signal = np.dot(dictionary, codes)
    assert np.linalg.norm(new_signal - signals_, axis=0)**2 < TOL

    # Check all decomps within tolerance when multiple signals passed to OMP
    signals = signals[:, 12:250]
    codes = sparse.omp_cholesky(signals, dictionary, tol=TOL, n_threads=n_threads)
    new_signal = np.dot(dictionary, codes)

    assert np.all(np.linalg.norm(new_signal - signals, axis=0)**2 < TOL)


def test_omp_mask():
    signals = np.random.rand(16, 1000)
    mask = np.random.rand(16, 1000) < 0.5

    corrupted = signals*mask
    error1 = np.linalg.norm(signals - corrupted)
    dictionary = utils.dct_dict(128, 4)

    assert error1 > 0

    sparse_approx = sparse.omp_mask(corrupted, mask, dictionary, tol=1e-6)
    better_signals = np.dot(dictionary, sparse_approx)
    error2 = np.linalg.norm(signals - better_signals)

    assert error2 < error1


def test_lars_dense_signal(dense_signal):
    # Checking if lars finds an approximation
    # that is better than just random coeffs
    signal = dense_signal(2)[:, :100]
    n_signals = signal.shape[1]
    dict_ = utils.dct_dict(100, 8)

    approx_rnd = np.random.rand(100, 100)
    err = np.linalg.norm(signal - np.dot(dict_, approx_rnd))

    approx_lars = sparse.lars(signal, dict_, n_nonzero=6)
    assert np.count_nonzero(approx_lars) <= n_signals*6

    err2 = np.linalg.norm(signal - np.dot(dict_, approx_lars))
    assert err2 < err

    t1 = timeit.default_timer()
    lars_alpha10 = sparse.lars(signal, dict_, alpha=10)
    t_faster = timeit.default_timer() - t1

    t1 = timeit.default_timer()
    lars_alpha1 = sparse.lars(signal, dict_, alpha=0.1)
    t_slow = timeit.default_timer() - t1

    err10 = np.linalg.norm(signal - dict_.dot(lars_alpha10))
    err1 = np.linalg.norm(signal - dict_.dot(lars_alpha1))

    assert err1 < err10
    assert t_faster < t_slow


def test_lasso(dense_signal):
    # Checking if lars finds an approximation
    # that is better than just random coeffs
    signal = dense_signal(2)[:, :100]
    dict_ = utils.dct_dict(100, 8)

    approx_rnd = np.random.rand(100, 100)
    err_rnd = np.linalg.norm(signal - np.dot(dict_, approx_rnd))

    lars_alpha10 = sparse.lasso(signal, dict_, alpha=10)
    lars_alpha1 = sparse.lasso(signal, dict_, alpha=0.1)

    err10 = np.linalg.norm(signal - dict_.dot(lars_alpha10))
    err1 = np.linalg.norm(signal - dict_.dot(lars_alpha1))

    assert err10 < err_rnd
    assert err1 < err10


def test_iterative_hard_thresh_nonzero(dense_signal):
    signals = dense_signal(2)[:, 100]
    dict_ = utils.dct_dict(144, 8)

    init_a = np.zeros(144)
    n_nonzero = 10
    init_a[:n_nonzero] = 1
    sparse_1 = sparse.iterative_hard_thresholding(
        signals, dict_, 1, 0.1, init_a, n_nonzero
    )

    new_1 = np.dot(dict_, init_a)
    new_2 = np.dot(dict_, sparse_1)

    err1 = LA.norm(signals - new_1)
    err2 = LA.norm(signals - new_2)

    # Since one iteration should improve the sparse coeffs
    assert err2 < err1
    assert np.count_nonzero(sparse_1) <= n_nonzero

    sparse_1_iter = sparse.iterative_hard_thresholding(
        signals, dict_, 1, 0.1, init_a, n_nonzero
    )

    sparse_10_iters = sparse.iterative_hard_thresholding(
        signals, dict_, 10, 0.1, init_a, n_nonzero
    )

    new1 = np.dot(dict_, sparse_1_iter)
    new2 = np.dot(dict_, sparse_10_iters)

    # Because ten iterations should be better than one
    err2 = LA.norm(signals - new2)
    err1 = LA.norm(signals - new1)

    assert err2 < err1, "Ten iters not better than one" \
                        ", %f not < %f" % (err2, err1)
    assert np.count_nonzero(sparse_1_iter) <= n_nonzero
    assert np.count_nonzero(sparse_10_iters) <= n_nonzero


def test_iterative_hard_thresh_penalty(dense_signal):
    signal = dense_signal(2)[:, 1000]
    dict_ = utils.dct_dict(144, 8)
    init_sparse = np.random.rand(144)

    # Check the this algorithm is better then
    # a random coefficient vector
    sparse_1 = sparse.iterative_hard_thresholding(
        signal, dict_, 1, 0.1, init_sparse, penalty=10
    )

    new_init = np.dot(dict_, init_sparse)
    new_sparse = np.dot(dict_, sparse_1)

    err_init = LA.norm(signal - new_init)
    err_sparse = LA.norm(signal - new_sparse)
    assert err_sparse < err_init

    # Check that more iterations decrease the error
    sparse_1_iter = sparse.iterative_hard_thresholding(
        signal, dict_, 1, 0.1, init_sparse, penalty=10
    )

    sparse_10_iters = sparse.iterative_hard_thresholding(
        signal, dict_, 10, 0.1, init_sparse, penalty=10
    )

    new_1 = np.dot(dict_, sparse_1_iter)
    new_10 = np.dot(dict_, sparse_10_iters)

    err_1 = LA.norm(signal - new_1)
    err_10 = LA.norm(signal - new_10)
    assert err_10 < err_1

    # Smaller penalty gives a more accurate result
    # But less sparse
    sparse_1_pen = sparse.iterative_hard_thresholding(
        signal, dict_, 10, 0.1, init_sparse, penalty=1
    )

    sparse_10_pen = sparse.iterative_hard_thresholding(
        signal, dict_, 10, 0.1, init_sparse, penalty=10
    )

    new_1 = np.dot(dict_, sparse_1_pen)
    new_10 = np.dot(dict_, sparse_10_pen)

    err_1 = LA.norm(signal - new_1)
    err_10 = LA.norm(signal - new_10)
    assert err_1 < err_10
    assert np.count_nonzero(sparse_10_pen) < np.count_nonzero(sparse_1_pen)


def test_iterative_soft_thresh(dense_signal):
    signal = dense_signal(2)[:, 1000]
    dict_ = utils.dct_dict(144, 8)
    init_sparse = np.random.rand(144)

    # Check algorithm better than random
    sparse1 = sparse.iterative_soft_thresholding(
        signal, dict_, init_sparse, 10, iters=1
    )

    new1 = np.dot(dict_, sparse1)
    new0 = np.dot(dict_, init_sparse)

    assert LA.norm(signal - new1) <= LA.norm(signal - new0)

    # check more iterations better
    sparse_1 = sparse.iterative_soft_thresholding(
        signal, dict_, init_sparse, 10, iters=1
    )

    sparse_10 = sparse.iterative_soft_thresholding(
        signal, dict_, init_sparse, 10, iters=10
    )

    new1 = np.dot(dict_, sparse_1)
    new10 = np.dot(dict_, sparse_10)
    assert LA.norm(signal - new10) < LA.norm(signal - new1)

    # Lower reg_param more accurate but higher l1 norm or vector
    sparse_1 = sparse.iterative_soft_thresholding(
        signal, dict_, init_sparse, 1, iters=10
    )

    sparse_10 = sparse.iterative_soft_thresholding(
        signal, dict_, init_sparse, 10, iters=10
    )

    new1 = np.dot(dict_, sparse_1)
    new10 = np.dot(dict_, sparse_10)
    assert LA.norm(signal - new1) < LA.norm(signal - new10)
    assert LA.norm(new10, ord=1) < LA.norm(new1, ord=1)
