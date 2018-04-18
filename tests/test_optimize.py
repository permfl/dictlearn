import pytest
import numpy as np
from numpy import linalg as LA

from dictlearn import optimize
from dictlearn import sparse
from dictlearn import utils


@pytest.fixture
def initial_data():
    def _data(n_signals=100, n_atoms=32):
        signals = np.random.rand(10, n_signals)
        dictionary = utils.random_dictionary(10, n_atoms)
        return signals, dictionary

    return _data


def test_ksvd_num_iters(initial_data):
    """
        Test more iterations == better approximation
    """
    signals, dictionary = initial_data()

    d, a = optimize.ksvd(signals, dictionary, iters=5, n_nonzero=3)
    d1, a1 = optimize.ksvd(signals, dictionary, iters=6, n_nonzero=3)

    assert d.shape == dictionary.shape

    recon1 = np.dot(d, a)
    recon2 = np.dot(d1, a1)

    assert recon1.shape == signals.shape

    err1 = np.linalg.norm(signals - recon1)
    err2 = np.linalg.norm(signals - recon2)

    msg = 'Error with six iterations not less than error with five'
    assert err2 < err1, msg


def test_ksvd_tolerance(initial_data):
    """
        Test approximation is within given tolerance
    """
    signals, dictionary = initial_data()
    tol = 2

    d, a = optimize.ksvd(signals, dictionary, iters=1000, 
                         tol=tol, n_nonzero=5)

    msg = 'Approx not within specified tolerance %.3f not < %.3f'
    error = np.linalg.norm(signals - np.dot(d, a))
    assert error < tol, msg % (error, tol)


def test_project_c():
    mat = np.random.randint(0, 1000, 10000).reshape(100, 100)
    mat = mat.astype(float)
    normalized = optimize.project_c(mat)
    sums = np.linalg.norm(normalized, axis=0)
    assert np.all(np.abs(sums - 1) < 1e-12)


def test_mod(initial_data):
    signals, dico = initial_data()
    n_nonzero = 4
    init_codes = sparse.omp_batch(signals, dico, n_nonzero)
    new_dict = optimize.mod(signals, dico, n_nonzero, 10)
    new_codes = sparse.omp_batch(signals, new_dict, n_nonzero)

    new1 = np.dot(dico, init_codes)
    new2 = np.dot(new_dict, new_codes)

    err1 = LA.norm(signals - new1)
    err2 = LA.norm(signals - new2)
    assert err2 < err1

    new_dict = optimize.mod(signals, dico, n_nonzero, 10)
    new_codes = sparse.omp_batch(signals, new_dict, n_nonzero)
    new3 = np.dot(new_dict, new_codes)

    err3 = LA.norm(signals - new3)
    assert err3 <= err2


def test_odl(initial_data):
    signals, dico = initial_data()
    n_nonzero = 4
    init_codes = sparse.omp_batch(signals, dico, n_nonzero)
    new_dict = optimize.odl(signals, dico, n_nonzero=n_nonzero)
    new_codes = sparse.omp_batch(signals, new_dict, n_nonzero)

    new1 = np.dot(dico, init_codes)
    new2 = np.dot(new_dict, new_codes)

    err1 = LA.norm(signals - new1)
    err2 = LA.norm(signals - new2)
    assert err2 < err1


def test_odl_max_iters():
    patches = np.random.rand(64, 50)
    dico = utils.dct_dict(128, 8)
    n_nonzero = 8
    new_dict = optimize.odl(patches, dico, n_nonzero=n_nonzero, iters=50)
    assert new_dict.shape == dico.shape


def test_reconstruct_low_rank(initial_data):
    signals, dico = initial_data()
    masks = np.random.rand(signals.shape[0], signals.shape[1]) > 0.4
    low_rank = optimize.reconstruct_low_rank(signals, masks, n_low_rank=2, iters=10)
    assert low_rank.shape == (signals.shape[0], 2)

    low_rank = optimize.reconstruct_low_rank(signals, masks, n_low_rank=2, iters=10,
                                             initial=np.random.rand(dico.shape[0], 2))
    assert low_rank.shape == (signals.shape[0], 2)

    low_rank = optimize.reconstruct_low_rank(signals, masks, n_low_rank=2, iters=10)
    assert low_rank.shape == (signals.shape[0], 2)

    with pytest.warns(RuntimeWarning):
        masks[:, 5] = 0
        optimize.reconstruct_low_rank(signals, masks, n_low_rank=2)

    with pytest.raises(ValueError):
        optimize.reconstruct_low_rank(signals, masks, n_low_rank=0)

    with pytest.raises(ValueError):
        masks = np.zeros((10, 10))
        optimize.reconstruct_low_rank(signals, masks, n_low_rank=2)


def test_itkrmm(initial_data):
    signals, dico = initial_data()
    err1 = LA.norm(signals - np.dot(dico, sparse.omp_batch(signals, dico, 2)))
    masks = np.ones_like(signals)
    new_dico = optimize.itkrmm(signals.copy(), masks, dico, n_nonzero=2, iters=20)
    new_signals = sparse.omp_batch(signals, new_dico, 2)
    err2 = LA.norm(signals - np.dot(new_dico, new_signals))
    assert dico.shape == new_dico.shape
    assert err2 < err1

    lr = optimize.reconstruct_low_rank(signals.copy(), masks.copy(), n_low_rank=2)
    newer_dico = optimize.itkrmm(signals.copy(), masks, dico, 2, 20, low_rank=lr)
    newer_signals = sparse.omp_batch(signals.copy(), newer_dico, 2)
    err3 = LA.norm(signals - np.dot(newer_dico, newer_signals))
    assert newer_dico.shape == (dico.shape[0], dico.shape[1] + 2*lr.shape[1])
    assert err3 < err1

    with pytest.warns(RuntimeWarning):
        masks[:, 1] = 0
        optimize.itkrmm(signals.copy(), masks, dico, 2, 20)