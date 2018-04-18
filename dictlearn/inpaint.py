from __future__ import print_function
import math
import numpy as np
from scipy.interpolate import griddata

from ._dictlearn import _dictlearn as _dl
from . import operators


def interpolate(incomplete_image, mask):
    y, x = np.where(mask.T == False)
    M, N = incomplete_image.shape
    x1, y1 = np.meshgrid(np.arange(M), np.arange(N))
    return griddata((x, y), incomplete_image[x, y].flatten(), (x1, y1), 
                    fill_value=128).T


def gsr_core(image, patch_size, group_size, search_space,
             sliding_step, mu, reg_param, factor):
    """

    :param image: Corrupted image
    :param patch_size: Size each side in patch
    :param group_size: Number of patches in a group
    :param search_space: Size of windows to search for similar patches
    :param mu: Stabilisation factor
    :param sliding_step: Distance between first pixel in each patch
    :param reg_param: Regularization parameter, weight on l_1 norm
    :param factor:
    :return:
    """
    height = image.shape[0]
    width = image.shape[1]

    tau = reg_param*factor/mu
    thresh = math.sqrt(2 * tau)

    N = height - patch_size + 1
    M = width - patch_size + 1
    L = N * M

    # Indices for first pixel in all patches

    row = np.arange(0, N, sliding_step, dtype=np.uint32)
    col = np.arange(0, M, sliding_step, np.uint32)

    # We might be missing some image patches at the end of each dimension
    # ignore the sliding step at the end s.t also last rows and columns
    # are included in some patches
    rows = np.concatenate((row, np.arange(row[len(row)-1] + 1, N)))
    cols = np.concatenate((col, np.arange(col[len(col)-1] + 1, M)))

    patches = np.zeros((patch_size * patch_size, L), dtype=np.float64)

    count = 0
    for i in range(patch_size):
        for j in range(patch_size):
            img = image[i:height - patch_size + i + 1,
                        j:width - patch_size + j + 1]

            patch = img.flatten('F')
            patches[count, :] = patch
            count += 1

    # patches   -> one patch pr column
    # patches_t ->           pr row
    patches_t = patches.T

    # Index to all possible patches in the image
    I = np.arange(L, dtype=np.uint32)
    I = I.reshape(N, M, order='F')

    reconstruction = np.zeros((height, width))
    # How many times each pixel is seen
    weights = np.zeros((height, width))
    for row in rows:
        for col in cols:
            this = col * N + row

            # Find group_size closest matching patches
            # ~15%
            group_indices = _dl.gsr_patch_search(patches_t, row, col, this,
                                                 group_size, search_space, I)

            # Construct the group
            group = patches[:, group_indices]

            # ~50%
            U, Sigma_, Vt = np.linalg.svd(group)
            Sigma = np.zeros((U.shape[0], Vt.shape[1]))
            Sigma[np.diag_indices(Sigma.shape[1])] = Sigma_

            # Iterative hard thresholding-like sparse coding
            # keep only coeffs bigger than thresh
            sparse_codes = Sigma * (np.abs(Sigma) > thresh)

            # The dictionary:
            # dictionary = U
            # rowss = dictionary.shape[1] // 8
            # colss = rowss + dictionary.shape[1] % 8
            # utils.visualize_dictionary(dictionary, rowss, colss)
            group_recon = U.dot(sparse_codes).dot(Vt)

            # For each patch in this group put it back into image
            for k in range(len(group_indices)):
                gi = group_indices[k]
                r_ = _dl.prow_idx(gi, N)
                c_ = gi // N

                # Both of the two following assignments
                # ~10%
                reconstruction[r_:r_ + patch_size, c_:c_ + patch_size] += \
                    group_recon[:, k].reshape(patch_size, patch_size)
                weights[r_:r_ + patch_size, c_:c_ + patch_size] += 1

    return reconstruction / (weights + np.spacing(1))


def gsr(broken_img, mask, iters, patch_size=8,
        group_size=60, search_space=20, sliding_step=4,
        mu=2.5e-3, reg_param=0.082, factor=240, callback=None):
    """
    Group-based sparse image inpainting

    :param broken_img: 
        Image to inpaint
    
    :param mask: 
        Mask in [0, 1] entries 0 denoted pixels to be inpainted. 
        Same shape as broken_img
    
    :param iters: 
        Number of iterations
    
    :param patch_size: 
        Size of patches, (patch_size, patch_size)
    
    :param group_size: 
        Number of patches to group together
    
    :param search_space: 
        Size of search window
    
    :param sliding_step: 
        Distance between patches. =1 takes all possible
        patches, 2 every other patch etc.
    
    :param mu: 
        Stabilisation Factor
    
    :param reg_param: 
        Regularization parameter
    
    :param factor: 
        Magic value 3 from paper
    
    :param callback: 
        called after each iteration with image estimate and  iteration number
    
    :return: Reconstructed image
    """
    HTy = broken_img*mask
    x = interpolate(broken_img, np.logical_not(mask))
    c = 0
    muinv = 1 / mu
    invHHT = 1 / (mu + mask)
    
    for k in range(iters):
        #SBI
        est = gsr_core(x - c, patch_size, group_size,
                       search_space, sliding_step, mu,
                       reg_param, factor)

        r = HTy + mu * (est + c)
        x = muinv * (r - mask*(invHHT * (mask*r)))
        c += (est - x)

        if callback is not None:
            callback(x, k)

    return x


def best_patch(image, patch, to_fill, source):
    """
        Finds the patch in image that best matches patch
    :param image: Image to search
    :param patch: Find closest matching patch to this
    :param to_fill: Which pixels in patch needs to be filled
    :param source: What parts of the image are already filled
    :return: index set, [first row, last row, first column, last column]
    """
    last_row = image.shape[0] - patch.shape[0]
    last_col = image.shape[1] - patch.shape[1]
    best_error = 1e6
    best = np.zeros(4, np.uint32)

    for row in range(last_row + 1):
        for col in range(last_col + 1):
            if np.any(np.logical_not(source[row:row + patch.shape[0],
                                     col:col + patch.shape[1]])):
                continue

            current_patch = image[row:row + patch.shape[0],
                                  col:col + patch.shape[1]]

            # Compare already filled part of patch with current patch
            error = np.sum((current_patch[np.logical_not(to_fill)] -
                            patch[np.logical_not(to_fill)]) ** 2)

            current_error = error
            if current_error < best_error:
                best_error = current_error
                best = np.array((row, row + patch.shape[0],
                                 col, col + patch.shape[1]))

    return best


def inpaint_exemplar(image, mask, patch_size=9, max_iters=None, verbose=False):
    """
        Inpainting by texture synthesis. This method works better
        for filling in (or removing) bigger structures in an image compared
        to inpainting by dictionary learning

    :param image: 
        Image to inpaint

    :param mask: 
        Mask of area to inpaint, same size as image

    :param patch_size: 
        Size of image patches, default: 9

    :param max_iters: 
        Maximum number of iterations to use, default: None

    :param verbose: 
        Print progress if true
    
    :return: 
        Inpainted image
    """

    if patch_size % 2 == 0:
        raise ValueError('Need patch size odd')

    if image.ndim != 2:
        raise ValueError('Only greyscale images')

    image = image.astype(float)
    mask = mask.astype(bool)
    X, Y = image.shape

    if verbose:
        count = mask.size - np.count_nonzero(mask)
        print('Exemplar based inpainting')
        print(' {} pixels to be filled'.format(count))

    IND = np.arange(X * Y).reshape(X, Y)

    confidence = np.zeros(image.shape)
    priorities = np.zeros(image.shape)
    data_terms = np.tile(-0.1, image.shape)

    confidence[mask] = 1
    patch_offset = patch_size // 2

    # Isophotes.
    # Using plain gradients are very good for capturing
    # horizontal and vertical edges, not so good for curves
    # todo: Curvelet based isophotes?
    iso_x, iso_y = np.gradient(image)

    iso_x /= -255
    iso_y /= 255

    iteration = 1

    # Loop until all pixels are filled
    while np.any(np.logical_not(mask)):
        # Fill front: All missing pixels mask[i, j] == False
        # where at least one neighbour is already filled
        xx, yy = np.where(operators.convolve2d(mask.astype(float),
                                               operators.laplacian, 'same') > 0)
        fill_front = list(zip(xx, yy))

        # Compute confidence values
        for pixel in fill_front:
            # Get indices for patch centered at pixel
            # or what ever patch that includes pixel
            # if at the boundary
            x_start = max(pixel[0] - patch_offset, 0)
            x_end = min(pixel[0] + patch_offset, image.shape[0])
            y_start = max(pixel[1] - patch_offset, 0)
            y_end = min(pixel[1] + patch_offset, image.shape[1])

            conf = 0
            for i in range(x_start, x_end):
                for j in range(y_start, y_end):
                    if mask[i, j]:
                        conf += confidence[i, j]

            confidence[pixel] = conf / patch_size ** 2

        # Compute data terms
        rows = [i[0] for i in fill_front]
        cols = [i[1] for i in fill_front]
        indices = (rows, cols)
        Nx, Ny = np.gradient(mask.astype(float))
        Nx = Nx[indices].reshape(len(indices[0]), 1)
        Ny = Ny[indices].reshape(len(indices[1]), 1)
        N = np.concatenate((Nx, Ny), axis=1)

        data_terms[indices] = abs(iso_x[indices] * N[:, 0] +
                                  iso_y[indices] * N[:, 1]) + 0.001

        priorities[indices] = confidence[indices] * data_terms[indices]

        # Highest priority pixel, ie the one we are the most
        # confident about what the values should be
        h = np.argmax(priorities[indices])
        max_pri = np.unravel_index(IND[indices][h], image.shape)

        # Get indices of patch centered at this pixel h
        x_start = max(max_pri[0] - patch_offset, 0)
        x_end = min(max_pri[0] + patch_offset, image.shape[0])
        y_start = max(max_pri[1] - patch_offset, 0)
        y_end = min(max_pri[1] + patch_offset, image.shape[1])

        patch = image[x_start:x_end + 1, y_start:y_end + 1]
        to_fill = np.logical_not(mask[x_start:x_end + 1, y_start:y_end + 1])
        to_fill = to_fill.astype(np.uint32)

        best_exemplar = _dl.best_patch(image, patch.copy(), to_fill,
                                       mask.astype(np.uint32))
        x1, x2, y1, y2 = best_exemplar

        for i in range(patch_size):
            p_row = x_start + i
            q_row = x1 + i

            if p_row >= image.shape[0] or q_row >= image.shape[0]:
                continue

            for j in range(patch_size):
                p_col = y_start + j
                q_col = y1 + j

                if p_col >= image.shape[1] or q_col >= image.shape[1]:
                    continue

                # Now move data from patch q onto patch p
                # but only for pixels which is not filled yet
                if not mask[p_row, p_col]:
                    image[p_row, p_col] = image[q_row, q_col]
                    mask[p_row, p_col] = True
                    iso_x[p_row, p_col] = iso_x[q_row, q_col]
                    iso_y[p_row, p_col] = iso_y[q_row, q_col]
                    confidence[p_row, p_col] = confidence[max_pri]

        if verbose:
            progress = 1 - (mask.size - np.count_nonzero(mask))/float(count)
            print(' Iteration {}, {:.2f}% done'
                  .format(iteration, progress*100), end='\r')

        iteration += 1
        if max_iters is not None and iteration >= max_iters:
            break

    if verbose:
        print()

    return image
