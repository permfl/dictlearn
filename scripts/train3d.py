from __future__ import print_function
import os
import argparse
import numpy as np
import dictlearn as dl
import timeit


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
parser.add_argument('--output-dir', type=str, required=True, help='Save output here')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--n-threads', type=int, default=1, help='Number of threads')


args = parser.parse_args()


filename = args.image
name, ext = os.path.splitext(filename)

if ext == '.npy':
    volume = np.load(filename)
elif ext == '.vti':
    volume = dl.utils.numpy_from_vti(filename)
else:
    raise SystemExit('Unknown file format %s' % filename)

size = args.patch_size

if args.n_atoms is None:
    n_atoms = 2*size[0]*size[1]*size[2]
else:
    n_atoms = args.n_atoms

if args.verbose:
    print('Volume', filename)
    print('Patch size', size)
    print('N atoms', n_atoms)

dictionary = dl.random_dictionary(size[0]*size[1]*size[2], n_atoms)
patches = dl.preprocess.Patches(volume, size, args.strides)
iters = args.iters
n_nonzero = args.n_nonzero

if args.verbose:
    print('patches.volume.shape', patches.image.shape)
    print('patches.size', patches.size)
    print('patches.stride', patches.stride)
    print('iters pr batch', iters)
    print('batch size', args.batch_size)
    print('n_nonzero', n_nonzero)

count = 0
t1 = timeit.default_timer()
for batch in patches.generator(args.batch_size):
    count += 1
    dictionary = dl.ksvd(batch, dictionary, iters, n_nonzero, n_threads=args.n_threads)[0]
t2 = timeit.default_timer()

print('Time', t2 - t1)
print('Avg pr batch', (t2 - t1) / count)
print('Num batches', count)


np.save(os.path.join(args.output_dir, '__dictionary.npy'), dictionary)

