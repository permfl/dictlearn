import os
import pytest
import numpy as np
from numpy import linalg
from scipy.linalg import solve_triangular
from sklearn.linear_model import orthogonal_mp, orthogonal_mp_gram

from dictlearn import utils, algorithms, filters, preprocess
from dictlearn._dictlearn import _dictlearn


@pytest.fixture
def lower_system():
    l1 = np.eye(4, dtype=float)
    b1 = np.arange(4, dtype=float)

    l2 = np.array(
        [
            [1, 0, 0, 0],
            [-3, 1.2, 0, 0],
            [0, 3.2, 5, 0],
            [1, 2, 3, 4]
        ], dtype=float
    )

    b2 = np.array([-2, 3.4, 3, 8], dtype=float)

    return l1, b1, l2, b2


@pytest.fixture
def mat_vec():
    def f(m, n):
        mat = np.random.rand(m, n)
        vec = np.random.rand(n)
        return mat, vec
    return f


@pytest.fixture
def mat_mat():
    def f(r1, c1, r2, c2):
        m1 = np.random.rand(r1, c1)
        m2 = np.random.rand(r2, c2)
        return m1, m2
    return f


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


def test_forward_solver(lower_system):
    l1, b1, l2, b2 = lower_system

    res_ = _dictlearn.wraps_forward(l1, l1.shape[0], b1.copy(), l1.shape[0])
    res_ex = solve_triangular(l1, b1.copy(), lower=True)
    assert res_.shape == res_ex.shape
    assert np.allclose(res_, res_ex)

    res_ = _dictlearn.wraps_forward(l2, l2.shape[0], b2.copy(), l2.shape[0])
    res_ex = solve_triangular(l2, b2.copy(), lower=True)
    assert res_.shape == res_ex.shape
    assert np.allclose(res_, res_ex)

    res_ = _dictlearn.wraps_forward(l2, 2, b2.copy(), l2.shape[0])[:2]
    res_ex = solve_triangular(l2[:2, :2], b2[:2].copy(), lower=True)
    assert res_.shape == res_ex.shape
    assert np.allclose(res_, res_ex)


def test_backward_solver(lower_system):
    l1, b1, l2, b2 = lower_system
    l1 = l1.T
    l2 = l2.T

    res_ = _dictlearn.wraps_backward(l1, l1.shape[0], b1.copy(), l1.shape[0])
    res_ex = solve_triangular(l1, b1.copy())
    assert res_.shape == res_ex.shape
    assert np.allclose(res_, res_ex)

    res_ = _dictlearn.wraps_backward(l2, l2.shape[0], b2.copy(), l2.shape[0])
    res_ex = solve_triangular(l2, b2.copy())
    assert res_.shape == res_ex.shape
    assert np.allclose(res_, res_ex)

    res_ = _dictlearn.wraps_backward(l2, 2, b2.copy(), l2.shape[0])[:2]
    res_ex = solve_triangular(l2[:2, :2], b2[:2].copy())
    assert res_.shape == res_ex.shape
    assert np.allclose(res_, res_ex)


def test_lu_solver(lower_system):
    l1, b1, l2, b2 = lower_system
    CH1 = np.dot(l1, l1.T)
    CH2 = np.dot(l2, l2.T)

    res_ = _dictlearn.wraps_lu(l1, l1.shape[0], b1.copy(), l1.shape[0])
    res_ex = linalg.solve(CH1, b1.copy())
    assert res_.shape == res_ex.shape
    assert np.allclose(res_, res_ex)

    res_ = _dictlearn.wraps_lu(l2, l2.shape[0], b2.copy(), l2.shape[0])
    res_ex = linalg.solve(CH2, b2.copy())
    assert res_.shape == res_ex.shape
    assert np.allclose(res_, res_ex)

    res_ = _dictlearn.wraps_lu(l2, 2, b2.copy(), l2.shape[0])[:2]
    res_ex = linalg.solve(CH2[:2, :2], b2[:2].copy())
    assert res_.shape == res_ex.shape
    assert np.allclose(res_, res_ex)


def test_argmax_mat_vec():
    A = np.random.randint(-5, 5, 100).reshape(10, 10) * np.random.rand()
    b = np.random.rand(10)

    expected = np.argmax(np.abs(np.dot(A, b)))
    got = _dictlearn.wraps_argmax_mat_vec(A, b)
    assert got == expected


def test_set_entries():
    n, m = 10, 100
    index_set = np.random.randint(0, m, n)
    dest = np.zeros(m)
    src = np.arange(n)

    got = _dictlearn.wraps_set_entries(dest.copy(), src, index_set, n)

    expected = dest.copy()
    expected[index_set] = src

    assert np.array_equal(got, expected)


def test_fill_entries():
    n, m = 10, 100
    index_set = np.random.randint(0, m, n)
    dest = np.zeros(n)
    src = np.arange(m)

    got = _dictlearn.wraps_fill_entries(dest.copy(), src, index_set, n)
    expected = src[index_set]

    assert np.array_equal(got, expected)


def test_copy_of():
    a = np.arange(100, dtype=float)
    res = _dictlearn.wraps_copy_of(a)
    assert np.array_equal(a, res)


def test_transpose():
    A = np.random.rand(10, 10)

    expected = A.T
    res = _dictlearn.wraps_transpose(A)
    assert np.array_equal(res, expected)


def test_dot():
    a = np.arange(10, dtype=float)
    b = np.random.rand(10)

    expected = np.dot(a, b)
    got = _dictlearn.wraps_dot(a, b, 10)
    assert np.allclose(got, expected)

    expected = np.dot(a[:5], b[:5])
    got = _dictlearn.wraps_dot(a, b, 5)
    assert np.allclose(got, expected)

    b *= 0
    expected = np.dot(a, b)
    got = _dictlearn.wraps_dot(a, b, 10)
    assert np.allclose(got, expected)

    a = np.arange(1, dtype=float)
    b = np.random.rand(1)

    expected = np.dot(a, b)
    got = _dictlearn.wraps_dot(a, b, 1)
    assert np.allclose(got, expected)


def test_mat_vec(mat_vec):
    m, v = mat_vec(10, 10)
    got = _dictlearn.wraps_mat_vec(m, v)
    expected = np.dot(m, v)
    assert got.shape == expected.shape
    assert np.allclose(got, expected)

    m, v = mat_vec(100, 100)
    got = _dictlearn.wraps_mat_vec(m, v)
    expected = np.dot(m, v)
    assert got.shape == expected.shape
    assert np.allclose(got, expected)

    m, v = mat_vec(1, 100)
    got = _dictlearn.wraps_mat_vec(m, v)
    expected = np.dot(m, v)
    assert got.shape == expected.shape
    assert np.allclose(got, expected)


def test_faster_lu_solver(lower_system):
    l1, b1, l2, b2 = lower_system
    CH1 = np.dot(l1, l1.T)
    CH2 = np.dot(l2, l2.T)

    res_ = _dictlearn.wraps_faster_lu(l1, l1.shape[0], b1.copy(), l1.shape[0])
    res_ex = linalg.solve(CH1, b1.copy())
    assert res_.shape == res_ex.shape
    assert np.allclose(res_, res_ex)

    res_ = _dictlearn.wraps_faster_lu(l2, l2.shape[0], b2.copy(), l2.shape[0])
    res_ex = linalg.solve(CH2, b2.copy())
    assert res_.shape == res_ex.shape
    assert np.allclose(res_, res_ex)

    res_ = _dictlearn.wraps_faster_lu(l2, 2, b2.copy(), l2.shape[0])[:2]
    res_ex = linalg.solve(CH2[:2, :2], b2[:2].copy())
    assert res_.shape == res_ex.shape
    assert np.allclose(res_, res_ex)


def test_contains():
    arr = np.array([1, 2, 3, 4, 5, 6], dtype=np.uint64)
    assert _dictlearn.wraps_contains(1, arr)
    assert _dictlearn.wraps_contains(5, arr)
    assert not _dictlearn.wraps_contains(8, arr)