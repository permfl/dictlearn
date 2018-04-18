#!/usr/bin/env python

from __future__ import print_function

import os
import argparse
import timeit
import numpy as np

from scipy import misc

description = """\
Script for inpainting grayscale images

Group-based sparse representation (GSR) and itkrmm works best for inpainting
smaller structures, where the missing pixels in one area are not bigger than
the image patches. ITKrMM first trains an explicit dictionary from the corrupted
image before reconstruction with OMP. Because of this OMP step, itkrmm is slower than
GSR which uses an implicit dictionary. The results from both methods are similar.

Exemplar based inpainting works very well when the area to inpaint is larger. 
This method if also the fastest, as the algorithm uses texture synthesis to fill
the missing are. That is replace missing image patches with the most similar patch from
a different location in the image.

Exemplar based arguments:
    --patch-size: Size of one side in image patches. As image patches are square the 
                  patch size will be (patch-size, patch-size)
    --max-iter:   Maximum number of iteration. If this is to low the algorithm
                  will stop before all the missing pixels are filled in.
    --verbose:    Print progress
    

GSR arguments:
    --patch-size: Size of one side in image patches. As image patches are square the 
                  patch size will be (patch-size, patch-size)
    --iters: Number of iterations. More iteration -> better results, but slower.
    --group-size: How many patches to group together.
    --search-space: Size of the are to search for similar patches
    --stride: Distance between each image patch.
    --verbose:    Print progress
     
     
ITKrMM arguments:
    --patch-size: Size of one side in image patches. As image patches are square the 
                  patch size will be (patch-size, patch-size)
    --iters: Number of iterations. More iteration -> better results, but slower.
    --nonzero-train: Number of nonzero coefficients to use during training
    --low-rank: Number of low rank dictionary atoms
    --nonzero-omp: Number of nonzero coefficients to use during 
                   reconstruction with OMP
    --tolerance: Tolerance for OMP. This adds nonzero coefficients until the 
                 reconstruction error is less then tolerance and overwrites 
                 --nonzero-omp
    --verbose:    Print progress
    

The default arguments for all methods will give fairly good
results for all images and masks. If ran with the default arguments and
the inpainted image turns out very bad, the inpainting algorithm is probably wrong.

"""

__doc__ = description


def gsr(image, mask, args):
    inpainter = dl.Inpaint(image, mask, args.patch_size).train(iters=args.iters)

    def callback(recon, it):
        print(' Iteration {}/{}'.format(it + 1, args.iters), end='\r')

    if args.verbose:
        print('GSR:')
        print(' Iteration {}/{}'.format(0, args.iters), end='\r')
        callback = callback
    else:
        callback = None

    return inpainter.inpaint(group_size=args.group_size,
                             search_space=args.search_space,
                             sliding_step=args.stride,
                             callback=callback)


def exemplar(image, mask, args):
    return dl.inpaint.inpaint_exemplar(image, mask, args.patch_size,
                                       args.max_iters, args.verbose)


def itkrmm(image, mask, args):
    inpainter = dl.Inpaint(image, mask, args.patch_size, method='itkrmm')
    inpainter.train(iters=args.iters,
                    n_nonzero=args.nonzero_train,
                    n_low_rank=args.low_rank,
                    verbose=args.verbose)

    return inpainter.inpaint(n_nonzero=args.nonzero_omp,
                             tol=args.tolerance,
                             verbose=args.verbose)


def write_image(image, path, output=None):
    if output is None:
        name, ext = os.path.splitext(path)
        output = '{}_inpainted{}'.format(name, ext)

    _, ext = os.path.splitext(output)

    if ext == '.npy':
        np.save(output, image)
    else:
        misc.imsave(output, image)

    print('Mask saved to: {}'.format(output))


parser = argparse.ArgumentParser(description=description,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)

parser.add_argument(
    'image', type=str,
    help='Path to image to inpaint'
)

parser.add_argument(
    'mask', type=str,
    help='Path to mask for image'
)

parser.add_argument(
    'method', type=str, choices=['gsr', 'exemplar', 'itkrmm'],
    help='Choice of inpainting method'
)

parser.add_argument(
    '-p', '--patch-size', type=int, default=9,
    help='Size of image patches'
)

parser.add_argument(
    '-i', '--iters', type=int, default=10,
    help='Number of iterations'
)

parser.add_argument(
    '--max-iters', type=int,
    help='Maximum number of iterations for exemplar inpaint'
)

parser.add_argument(
    '-g', '--group-size', type=int, default=60,
    help='Number of patches per group in GSR'
)

parser.add_argument(
    '-w', '--search-space', type=int, default=20,
    help='Size of search space/window for GSR'
)

parser.add_argument(
    '-s', '--stride', type=int, default=4,
    help='Stride (distance) between each image patch for GSR'
)

parser.add_argument(
    '--nonzero-train', type=int, default=10,
    help='ITKrMM: Number of nonzero coefficients to use for training'
)

parser.add_argument(
    '--nonzero-omp', type=int, default=10,
    help='ITKrMM: Number of nonzero coefficients for image reconstruction with OMP'
)

parser.add_argument(
    '--tolerance', type=float, default=1e-6,
    help='ITKrMM: Tolerance for image reconstruction with OMP. Overwrites --nonzero-omp'
)

parser.add_argument(
    '--low-rank', type=int, default=0,
    help='ITKrMM: Number of low rank dictionary atoms'
)

parser.add_argument(
    '-v', '--verbose', action='store_true',
    help='Print progress'
)

parser.add_argument(
    '-o', '--output', type=str,
    help='Path for storing inpainted image'
)


args = parser.parse_args()


if not os.path.isfile(args.image):
    raise SystemExit('Cannot find image {}'.format(args.image))

if not os.path.isfile(args.mask):
    raise SystemExit('Cannot find mask {}'.format(args.mask))


import dictlearn as dl

img = dl.imread(args.image).astype(float)
mask = dl.imread(args.mask).astype(bool)

img = img*mask

if args.method == 'gsr':
    method = gsr
elif args.method == 'itkrmm':
    method = itkrmm
else:
    method = exemplar


t1 = timeit.default_timer()
inpainted = method(img, mask, args)
t2 = timeit.default_timer()

write_image(inpainted, args.image, args.output)
print('Time:', t2 - t1, 'seconds')
