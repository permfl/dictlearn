from __future__ import print_function
import os
import sys
import multiprocessing

import numpy as np

import dictlearn as dl
from dictlearn import utils, filters


DICT = np.load('frame_dict_trained.npy')
IMAGE_VOLUME = np.load('images/image_volume.npy')
IMAGE_VOLUME = utils.normalize(IMAGE_VOLUME)
IMAGE_VOLUME = filters.threshold(IMAGE_VOLUME, 0.4)


def slice_dir():
    home = os.path.expanduser('~')
    slices_dir = os.path.join(home, 'slices')
    try:
        os.mkdir(slices_dir)
    except OSError:
        pass

    return slices_dir


def train(data):
    rank, start, stop = data
    image = IMAGE_VOLUME[start:stop]
    fn = 'dct_slice_%d.npy'
    slice_path = slice_dir()

    for i in range(image.shape[0]):
        img = image[i]
        patch_size = (16, 16)
        signals = dl.Patches(img, patch_size[0]).patches
        D = DICT.copy()
        D, _ = dl.ksvd(signals, dictionary=D, iters=5, n_nonzero=8)
        path = os.path.join(slice_path, fn % i)
        np.save(path, D)


def reconstruct_adept(data):
    rank, start, stop = data
    image = IMAGE_VOLUME[start:stop]
    fn = 'dct_slice_%d.npy'
    slice_path = slice_dir()

    for i in range(image.shape[0]):
        img = IMAGE_VOLUME[i]
        img = filters.threshold(utils.normalize(img), 0.4)
        patch_size = (16, 16)
        sigma = 10
        path = os.path.join(slice_path, fn % i)
        D = np.load(path)
        den = dl.Denoise(img, patch_size[0])
        den.dictionary = D
        denoised = den.denoise(sigma)
        image[i] = denoised.copy()

    slices_dir = slice_dir()
    fn = 'slice_%d_%d.npy' % (start, stop)
    np.save(os.path.join(slices_dir, fn), image)


def reconstruct(rng):
    rank, start, stop = rng
    new_img = IMAGE_VOLUME[start:stop]

    for i in range(new_img.shape[0]):
        image = new_img[i]
        image = utils.normalize(image)
        image = filters.threshold(image, 0.4)

        patch_size = (16, 16)
        sigma = 10
        den = dl.Denoise(image, patch_size[0])
        den.dictionary = DICT
        denoised = den.denoise(sigma)
        new_img[i] = denoised.copy()

    slices_dir = slice_dir()
    fn = 'slice_%d_%d.npy' % (start, stop)
    np.save(os.path.join(slices_dir, fn), new_img)


if __name__ == '__main__':
    pool = multiprocessing.Pool()
    slices = IMAGE_VOLUME.shape[0]
    proc = multiprocessing.cpu_count()

    slices_pr_proc = slices//proc
    rest = slices % proc
    ranges = []
    start = 0
    stop = 0
    for p in range(proc):
        start = p*slices_pr_proc
        stop = (p+1)*slices_pr_proc

        if p == proc - 1:
            ranges.append((p, start, stop + rest))
        else:
            ranges.append((p, start, stop))

    if sys.argv[1] == 'train':
        pool.map(train, ranges)
    elif sys.argv[1] == 'denoise':
        pool.map(reconstruct_adept, ranges)
    else:
        pool.map(reconstruct, ranges)



