from __future__ import print_function
import argparse
import math
import os
import timeit
import numpy as np

import matplotlib.pyplot as plt
from scipy import misc

from dictlearn import utils, filters, kernels
from dictlearn.algorithms import ksvd_denoise


def save(args):
    if args.format == 'npy':
        np.save(args.output + '.npy', denoised)
    else:
        plt.savefig(args.output)


descr = """
Denoise an image with K-SVD.

Patch size has the biggest impact on the noise removal
Bigger patches --> smoother image

8x8 works well for natural images
"""

parser = argparse.ArgumentParser(description=descr)
parser.add_argument('image', help='Path to image')
parser.add_argument(
    '-p', '--patch-size', default=8, type=int,
    help='Size of image patches. Default 8'
)
parser.add_argument(
    '-c', '--num-atoms', default=256, type=int,
    help='Number of atoms (cols) in dictionary'
)
parser.add_argument('--iters', default=5, type=int,
                    help='Number of K-SVD iterations')
parser.add_argument('-s', '--sigma', default=10, type=float,
                    help='Standard deviation of noise')
parser.add_argument('-o', '--output',
                    help='Path for saving denoised image')
parser.add_argument('-m', '--method', default='adaptive',
                    choices=['adaptive', 'dct'],
                    help='Denoising method.'
                         '\nAdaptive learns dict from noisy image'
                         '\nDCT uses discrete cosine transform basis as dict')
parser.add_argument('-f', '--format', default='png', help='Format to save image')
parser.add_argument('-t', '--threshold', type=float, help='Threshold original' + 
                    ' image at this minimum intensity')
parser.add_argument('--vis-dict', action='store_true',
                    help='Visualize the dictionary')

parser.add_argument('--smooth', type=int,
                    help='Apply gaussion smoothing with kernel size SIZE (odd)')

parser.add_argument('-j', '--n-threads', type=int, default=1,
                    help="Number of threads to use, default 1")

args = parser.parse_args()

image = misc.imread(args.image)
image = utils.normalize(image)


if args.threshold is not None:
    image = filters.threshold(image, args.threshold)

image *= 255

patch_size = args.patch_size
n_atoms = args.num_atoms
iters = args.iters
sigma = args.sigma


t1 = timeit.default_timer()

denoised = ksvd_denoise(image, patch_size, iters, n_atoms,
                        sigma, verbose=True, retDict=args.vis_dict,
                        n_threads=args.n_threads)

t2 = timeit.default_timer()

print('Time: ', t2 - t1)

if args.vis_dict:
    denoised, dictionary = denoised
    size = int(math.floor(math.sqrt(args.num_components)))
    size_component = int(math.floor(math.sqrt(dictionary.shape[0])))
    print('Generating dictionary plot')
    for i in range(args.num_components):
        plt.subplot(size, size, i + 1)
        plt.imshow(dictionary[:, i].reshape(size_component, size_component))
        plt.axis('off')

    if args.output:
        fn, ext = os.path.splitext(args.output)
        fn += '.dict'
        args.output = '{}{}'.format(fn, ext)
        save(args)
    else:
        plt.show()


if args.smooth is not None:
    denoised = utils.convolve2d(denoised, kernels.gaussian(args.smooth))[1:-1, 1:-1]


plt.subplot(121)
plt.imshow(image, cmap=plt.cm.bone)
plt.title('Original')
plt.axis('off')
plt.subplot(122)
plt.imshow(denoised, cmap=plt.cm.bone)
plt.title('Denoised')
plt.axis('off')

if args.output:
    save(args)
else:

    plt.show()
