import pytest
import numpy as np
from dictlearn import preprocess
import helpers
TOL = 1e-12


# noinspection PyTypeChecker
def test_patches_1d():
    a = np.ones(100)*np.random.rand()
    expected_mean = a[0]
    new_a, mean = preprocess.center(a, retmean=True)

    assert isinstance(mean, float)
    assert abs(expected_mean - mean) < TOL
    assert np.allclose(a, new_a + mean, atol=TOL)


def test_center_2d_ndarray():
    a = np.arange(4)
    b = np.zeros((4, 4))
    b[:] = a
    expected_mean = np.array([0, 1, 2, 3])
    new_b, means = preprocess.center(b, retmean=True)

    assert np.allclose(expected_mean, means, atol=TOL)
    assert np.allclose(new_b + means, b, atol=TOL)
    assert b.shape == new_b.shape
    assert expected_mean.shape == means.shape


def test_center_2d_ndarray_inplace():
    a = np.arange(4)
    b = np.zeros((4, 4))
    b[:] = a
    b_test = b.copy()
    expected_mean = np.array([0, 1, 2, 3])
    mean = preprocess.center(b_test, inplace=True)

    assert np.allclose(b_test + expected_mean, b, atol=TOL)
    assert np.array_equal(expected_mean, mean)


def test_normalize_1d():
    a = np.arange(11)
    b = preprocess.normalize(a)

    assert a.shape == b.shape
    assert abs(np.linalg.norm(b) - 1) < TOL


def test_normalize_2d():
    a = np.arange(11)
    b = np.zeros((11, 11))
    b[:] = a
    b = b.T  # All cols == 0, 1, 2,.., 10
    bn = preprocess.normalize(b)

    assert b.shape == bn.shape

    for col in range(bn.shape[1]):
        assert abs(np.linalg.norm(bn[:, col]) - 1) < TOL


def test_patches():
    img = helpers.get_image(1)
    patch_size = 8
    patch_maker = preprocess.Patches(img, patch_size)
    patches = patch_maker.patches
    reconstructed = patch_maker.reconstruct(patches)

    assert patches.shape[0] == patch_size**2
    assert np.allclose(img, reconstructed)

    patch_size = 16
    patch_maker = preprocess.Patches(img, patch_size)
    assert patch_maker.size == patch_size
    patches = patch_maker.patches
    reconstructed = patch_maker.reconstruct(patches)

    assert patches.shape[0] == patch_size ** 2
    assert np.allclose(img, reconstructed)

    img = np.random.rand(256, 256, 256)
    patch_size = 16
    with pytest.raises(MemoryError):
        preprocess.Patches(img, patch_size).patches


def test_patches_reconstruct():
    img = np.random.rand(128, 128)

    with pytest.raises(ValueError):
        patches = preprocess.Patches(img.copy(), 8, order='F')
        patches.reconstruct(patches.patches)

    patches = preprocess.Patches(img.copy(), 8)
    before = patches.patches.copy()
    recon = patches.reconstruct(patches.patches)
    assert recon.shape == img.shape
    assert np.allclose(img, recon)
    assert np.array_equal(before, patches.patches)

    new_patches = patches.patches + 10
    patches.reconstruct(new_patches, save=True)
    assert np.array_equal(new_patches, patches.patches)
    assert not np.allclose(before, patches.patches)


def test_patches_mean():
    img = np.random.rand(100, 100)

    patches = preprocess.Patches(img.copy(), 10)
    patches.remove_mean(add_back=True)
    new = patches.reconstruct(patches.patches)
    assert np.allclose(img, new)

    patches = preprocess.Patches(img.copy(), 10)
    patches.remove_mean(add_back=False)
    new = patches.reconstruct(patches.patches)
    assert not np.allclose(img - img.mean(), new)


def test_patches2d_bad_input():
    with pytest.raises(ValueError):
        preprocess.Patches(np.zeros((1, 1, 1, 1)), 10)

    with pytest.raises(ValueError):
        preprocess.Patches(np.arange(2), 2)

    with pytest.raises(ValueError):
        preprocess.Patches(np.zeros((10, 10)), 10, max_patches=10)


def test_patches_3d():
    vol = np.ones((10, 10, 10))
    patches = preprocess.Patches(vol, 8)
    assert patches.shape == (8*8*8, (10 - 8 + 1)**3)
    assert patches.size == 8
    new = patches.reconstruct(patches.patches)
    assert np.array_equal(vol, new)


# noinspection PyStatementEffect
def test_patches3d_static():
    vol = np.random.rand(20, 20, 20)
    patches = preprocess.Patches(vol, (10, 10, 10), (1, 1, 1))
    assert patches.shape == (10*10*10, 11**3), '3D patches incorrect shape'
    vol_a = patches.reconstruct(patches.patches)
    assert np.allclose(vol, vol_a), 'Recon 3D patches incorr values'

    with pytest.raises(MemoryError):
        vol = np.random.rand(256, 256, 256)
        preprocess.Patches(vol, (12, 12, 12), (1, 1, 1)).patches


def test_patches3d_stride():
    vol = np.random.rand(20, 20, 20)
    size = (10, 10, 10)
    patches = preprocess.Patches(vol, size, (1, 1, 11))

    with pytest.raises(ValueError):
        patches.reconstruct(patches.patches)

    patches = preprocess.Patches(vol, size, (1, 1, 10))
    vol_a = patches.reconstruct(patches.patches)
    patch_size = 10*10*10
    n_patches = 11*11*2
    assert patches.shape == (patch_size, n_patches)
    assert np.allclose(vol, vol_a)

    patches = preprocess.Patches(vol, size, (2, 2, 2))
    recon = patches.reconstruct(patches.patches)
    assert patches.shape == (10*10*10, 6*6*6)
    assert np.allclose(vol, recon)

    with pytest.raises(ValueError):
        too_many_dims = np.zeros((2, 2, 2, 2))
        preprocess.Patches(too_many_dims, [1]*3, [1]*3)

    with pytest.raises(ValueError):
        too_few_dims = np.zeros((2, 2))
        preprocess.Patches(too_few_dims, [1]*3, [1]*3)


def test_patches3d_batch():
    vol = np.random.rand(20, 20, 20)
    for i in range(9):
        vol[:, i, :] = i
    size = (9, 9, 9)
    stride = (1, 1, 1)

    corr = preprocess.Patches(vol.copy(), size, stride)
    corr = corr.patches
    iter_n_patches = 0

    patches = preprocess.Patches(vol, size, stride)
    iter_patches = None
    for batch in patches.generator(11):
        if iter_patches is None:
            iter_patches = batch.copy()
        else:
            iter_patches = np.hstack((iter_patches, batch.copy()))

        iter_n_patches += batch.shape[1]

    assert iter_n_patches == corr.shape[1]
    assert corr.shape == iter_patches.shape
    assert np.array_equal(corr, iter_patches)
    assert patches.n_patches == (20 - 9 + 1)**3

    stride = (1, 1, 3)
    patches = preprocess.Patches(vol, size, stride)
    assert patches.n_patches == (20 - 9 + 1)**2 * ((20 - 9 + 1) // stride[2] + 1)

    stride = (1, 3, 1)
    patches = preprocess.Patches(vol, size, stride)
    assert patches.n_patches == (20 - 9 + 1) ** 2 * ((20 - 9 + 1) // stride[1] + 1)

    stride = (3, 1, 1)
    patches = preprocess.Patches(vol, size, stride)
    assert patches.n_patches == (20 - 9 + 1) ** 2 * ((20 - 9 + 1) // stride[0] + 1)


def test_patches3d_check_size():

    vol = np.zeros((25, 25, 25), dtype=np.float64)
    patches = preprocess.Patches(vol, (10, 10, 10), (1, 1, 1))
    size, dims = patches._check_size(False, True)
    size = [size] + dims
    assert size[0] == (25 - 10 + 1)**3
    assert size[1] == 25 - 10 + 1
    assert size[2] == 25 - 10 + 1
    assert size[3] == 25 - 10 + 1

    with pytest.raises(MemoryError):
        p = preprocess.Patches(np.zeros((256, 256, 256), dtype=np.float64),
                               (10, 10, 10), (2, 2, 2))
        _ = p.patches

    p = preprocess.Patches(vol, (10, 10, 10), (2, 2, 2))
    size, dims = p._check_size(False, True)
    x, y, z = dims
    exp = (25 - 10 + 1) // 2 + 1
    assert x == exp
    assert y == exp
    assert z == exp
    assert size == exp**3 and size == x*y*z

    with pytest.raises(ValueError):
        preprocess.Patches(np.zeros((1, 1, 1, 1)), (1, 1, 1), (1, 1, 1))


def test_patches3d_reconstruct_batch():
    vol = np.random.rand(10, 10, 10)
    size = (7, 7, 7)
    stride = (1, 1, 1)
    patches = preprocess.Patches(vol, size, stride)

    for batch, reconstruct in patches.generator(10, callback=True):
        reconstruct(batch)

    assert np.array_equal(patches.image, vol)

    vol = np.random.rand(20, 20, 20)
    size = (7, 7, 7)
    stride = (3, 1, 2)
    patches = preprocess.Patches(vol, size, stride)

    for batch, reconstruct in patches.generator(10, callback=True):
        reconstruct(batch)

    assert np.array_equal(patches.image, vol)

    patches = preprocess.Patches(np.zeros((256, 256, 256)), [10]*3, [1]*3)

    with pytest.raises(MemoryError):
        patches.generator(1e11)

    with pytest.raises(MemoryError):
        patches.generator(1e10)

    vol = np.random.rand(20, 20, 20)
    size = (7, 7, 7)
    stride = (1, 1, 7)
    patches = preprocess.Patches(vol, size, stride)

    for batch, recon in patches.generator(10, callback=True):
        recon(batch)

    assert np.allclose(vol, patches.image)


def test_expand_index():
    vol = np.random.rand(15, 15, 15)
    stride = (1, 1, 1)
    size = (10, 10, 10)
    patches = preprocess.Patches(vol, size, stride)
    count = 0

    n_patches, dims = patches._check_size(False, True)
    xx, yy, zz = dims

    for i in range(0, xx, stride[0]):
        for j in range(0, yy, stride[1]):
            for k in range(0, zz, stride[2]):
                ii, jj, kk = patches._expand_index(count)
                msg = 'stride={}, {} -> {} != {}'.format(stride, count,
                                                         [ii, jj, kk], [i, j, k])
                assert i == ii, msg
                assert j == jj, msg
                assert k == kk, msg
                count += 1
