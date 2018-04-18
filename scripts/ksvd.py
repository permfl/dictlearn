# coding: utf-8
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from dictlearn.preprocess import Patches, center, normalize
from dictlearn.utils import random_dictionary, to_uint8
from dictlearn.optimize import ksvd


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('image')
parser.add_argument('iters', type=int, help='K-SVD iterations')
parser.add_argument('mu', type=int, help='Sparsity target')
parser.add_argument('--patch_size', default=8, type=int, help='Size of patch is this^2')
parser.add_argument('--atoms', default=128, type=int)

args = parser.parse_args()
iters = args.iters
mu = args.mu
patch_size = args.patch_size

if args.image.endswith('.npy'):
    img = np.load(args.image)
else:
    from scipy import misc
    print('aa')
    img = misc.imread(args.image)


img = center(normalize(to_uint8(img, float)))
patches = Patches(img, patch_size)
p = patches.patches

if args.atoms is not None:
    n_atoms = args.atoms
else:
    n_atoms = p.shape[0]*2


D, Y = ksvd(p, random_dictionary(p.shape[0], n_atoms), iters, mu, verbose=True)

image = patches.reconstruct(D.dot(Y))
print('Error (Frobenius): ', np.linalg.norm(img - image))

plt.subplot(121)
plt.imshow(img, cmap=plt.cm.bone)
plt.title('Original')
plt.subplot(122)
plt.imshow(image, plt.cm.bone)
plt.title('Sparse reconstruction')
plt.show()





