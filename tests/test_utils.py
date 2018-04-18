import os
import math
import pytest
import numpy as np
from dictlearn import utils

try:
    import vtk
except ImportError:
    vtk = None

import helpers


def test_dct_dict():
    size = 8
    n_atoms = 256
    atoms = int(math.ceil(math.sqrt(n_atoms)))**2
    dct = utils.dct_dict(n_atoms, size)

    assert dct.shape == (size*size, atoms)

    for col in range(dct.shape[1]):
        assert abs(np.linalg.norm(dct[:, col]) - 1) < 1e-12


def test_random_dictionary():
    dict_ = utils.random_dictionary(64, 256)

    assert dict_.shape == (64, 256)

    for col in range(dict_.shape[1]):
        assert abs(np.linalg.norm(dict_[:, col]) - 1) < 1e-12


def test_pad_image():
    a = np.ones((2, 2))

    padded = utils.pad_image(a, 2, 2, 'zeros')
    should_eq = np.zeros((6, 6))
    should_eq[2:-2, 2:-2] += 1
    assert np.allclose(padded, should_eq)

    padded = utils.pad_image(a, 2, 2, 'ones')
    assert np.allclose(padded, np.ones((6, 6)))

    padded = utils.pad_image(a, 2, 2, pad_with=12.432)
    should_eq = np.ones((6, 6))*12.432
    should_eq[2:-2, 2:-2] = 1
    assert np.allclose(padded, should_eq)


def test_normalize():
    a = np.arange(11)
    aa = utils.normalize(a.copy())

    assert abs(aa.max() - 1) < 1e-12
    assert abs(aa.min()) < 1e-12
    assert np.allclose(aa, a/10.0)

    ab = np.arange(-5, 6)
    aa = utils.normalize(ab)
    assert abs(aa.max() - 1) < 1e-12
    assert abs(aa.min()) < 1e-12
    assert np.allclose(aa, a / 10.0)


def test_to_uint8():
    a = np.arange(11)
    aa = utils.to_uint8(a.copy())

    assert abs(aa.max() - 255) < 1e-12
    assert abs(aa.min()) < 1e-12
    assert type(aa[0]) == np.uint8

    ab = np.arange(-5, 6)
    aa = utils.to_uint8(ab)
    assert abs(aa.max() - 255) < 1e-12
    assert abs(aa.min()) < 1e-12


def test_psnr_2d():
    original = np.ones((100, 100))

    assert utils.psnr(original, original, 1) == np.inf

    little_corruption = original.copy()
    original[:10, :10] = np.random.rand(10, 10)
    psnr_1 = utils.psnr(original, little_corruption, 1)

    bigger_corruption = little_corruption.copy()
    bigger_corruption[:43, :57] = np.random.rand(43, 57)
    psnr_2 = utils.psnr(original, bigger_corruption, 1)

    assert psnr_1 > psnr_2
    assert psnr_1 < np.inf


def test_psnr_3d():
    original = np.ones((10, 10, 10))

    assert np.all(utils.psnr(original, original, 1) == np.inf)

    little_corruption = original.copy()
    original[:2, :2, :2] = np.random.rand(2, 2, 2)
    psnr_1 = utils.psnr(original, little_corruption, 1)

    bigger_corruption = little_corruption.copy()
    bigger_corruption[:4, :7] = np.random.rand(4, 7, 10)
    psnr_2 = utils.psnr(original, bigger_corruption, 1)

    assert np.all(psnr_1 > psnr_2)
