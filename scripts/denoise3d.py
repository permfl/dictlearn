from __future__ import print_function
import os
import argparse
import numpy as np
import dictlearn as dl
import timeit


def train_3d(patches, batch_size, dictionary, iters, 
             n_nonzero, verbose, n_threads):
    
    count = 1

    for batch in patches.next_batch(batch_size):
        t1 = timeit.default_timer()
        dictionary = dl.ksvd(batch, dictionary, iters, n_nonzero, 
                             verbose=verbose, n_threads=n_threads)[0]
        t2 = timeit.default_timer()

        if verbose:
            print('Batch %d with %d signals took %.2f seconds'
                  % (count, batch.shape[1], t2 - t1))

        count += 1

    return dictionary, count


def train_slice():
    pass


def denoise_3d(patches, batch_size, patch_size, sigma, dictionary, 
               tol, n_threads, verbose):
    """
    """
    count = 0

    for batch, reconstruct in patches.create_batch_and_reconstruct(batch_size):
        t1 = timeit.default_timer()
        tol = patch_size * (1.15 * sigma) ** 2
        codes = dl.omp_batch(batch, dictionary, tol=tol, n_threads=n_threads)
        reconstruct(np.dot(dictionary, codes))
        t2 = timeit.default_timer()

        if verbose:
            print('Batch %d with %d signals took %.2f seconds'
                  % (count, batch.shape[1], t2 - t1))

        count += 1

    return count


def denoise_slice():
    pass


def slized(size):
    return any(i == 0 or i == 1 for i in size)

parser = argparse.ArgumentParser()
parser.add_argument('image', type=str,
                    help='Image volume to use for training')

parser.add_argument('--patch-size', nargs=3, type=int, default=[8, 8, 8],
                    help='Size of patches, in format "x y z"')

parser.add_argument('--strides', nargs=3, type=int, default=[1, 1, 1],
                    help='Stride between patches')

parser.add_argument('--n-atoms', type=int, help='Number of dictionary atoms')
parser.add_argument('--batch-size', type=int, default=10000, help='Batch size')
parser.add_argument('--iters', type=int, default=500, help='Iterations per batch')
parser.add_argument('--n-nonzero', type=int, default=20,
                    help='Number of nonzero for sparse coding')
parser.add_argument('--base-dir', type=str, required=True, help='Save output here')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--n-threads', type=int, default=1, help='Number of threads')
parser.add_argument('--init-dict', type=str, help='Initial dictionary path')
parser.add_argument('--sigma', type=float, help='Variance of noise')


args = parser.parse_args()
filename = args.image
size = args.patch_size
stride = args.strides
n_atoms = args.n_atoms
iters = args.iters
base_dir = args.base_dir
n_threads = args.n_threads
verbose = args.verbose
n_nonzero = args.n_nonzero
batch_size = args.batch_size
sigma = args.sigma
patch_size = size[0]*size[1]*size[2]

fn = '__dico_size_{}_{}_{}_stride_{}_{}_{}_atoms_{}_iters_{}_v2.npy'.format(
    size[0], size[1], size[2], stride[0], stride[1], stride[2], n_atoms, iters
)


if verbose:
    print('Training and denoising image at: ', filename)
    print('Patches size', size, sep='\t')
    print('Patches stride', stride, sep='\t')
    print('Number atoms', n_atoms, sep='\t')
    print('Iterations', iters, sep='\t')
    print('Number threads', n_threads, sep='\t')
    print('Number nonzero', n_nonzero, sep='\t')
    print('Batch size', batch_size, sep='\t')
    print('Sigma', sigma, sep='\t')
    print('Saving stuff at', base_dir, sep='\t')
    print('\n\n')

_, ext = os.path.splitext(filename)
if ext == '.npy':
    volume = np.load(filename)
elif ext == '.vti':
    volume = dl.utils.numpy_from_vti(filename)
else:
    raise SystemExit('Unknown file format %s' % filename)


try:
    if args.init_dict is None:
        dictionary = np.load(os.path.join(base_dir, fn))
    else:
        dictionary = np.load(os.path.join(base_dir, args.init_dict))
except IOError:
    dictionary = dl.random_dictionary(patch_size, n_atoms)


patches = dl.Patches(volume, size, stride)

if verbose:
    print('Training dictionary')

_t1 = timeit.default_timer()
if slized(size):
    dictionary, count = train_slice()
else:
    dictionary, count = train_3d(
        patches, batch_size, dictionary, iters, n_nonzero, verbose, n_threads
)
_t2 = timeit.default_timer()

np.save(os.path.join(base_dir, fn), dictionary)

if verbose:
    print('Training done after {} batches and {:.2f} seconds'
          .format(count, _t2 - _t1))

if verbose:
    print('\n\nStarting denoise')

count = 1
_t1 = timeit.default_timer()
if slized(size):
    denoise_slice()
else:
    count = denoise_3d(
        patches, batch_size, patch_size, sigma, dictionary, tol, n_threads, verbose
    )

_t2 = timeit.default_timer()

if verbose:
    print('Denoise done after %d batches and %.2f seconds' % (count, _t2 - _t1))


fn = 'volume_size_{}_{}_{}_stride_{}_{}_{}_atoms_{}_iters_{}_sigma_{}.npy'\
     .format(size[0], size[1], size[2], stride[0], stride[1], stride[2], 
             n_atoms, iters, sigma)

if verbose:
    print('Saving denoised image at', fn)

np.save(fn, patches.reconstructed)

