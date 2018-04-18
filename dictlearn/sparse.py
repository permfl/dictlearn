from __future__ import print_function

import math
import warnings
import numpy as np

from numpy import linalg as LA
from scipy import linalg
from sklearn import linear_model

from . import filters
from . import preprocess
from ._dictlearn import _dictlearn


def omp_batch(signals, dictionary, n_nonzero=10, tol=0, n_threads=1):
    """
        Batch Orthogonal Matching Pursuit. A more effective version than omp_cholesky
        if the number of signals is high. Saves time and calculations by\
        doing some pre-computations. See [2] for details

        >>> import dictlearn as dl
        >>> image = dl.imread('some/image.png')
        >>> dictionary = dl.load_dictionary('some-dictionary')
        >>> image_patches = dl.Patches(image, 8)
        >>> sparse_codes = dl.omp_batch(image_patches, dictionary, n_nonzero=4)
        >>> sparse_approx = dictionary.dot(sparse_codes)

        :param signals:
            Signals to encode. numpy.ndarray shape (signal_size, n_signals) or
            dictlearn.preprocess.Patches

        :param dictionary:
            ndarray, shape (signal_size, n_atoms)

        :param n_nonzero: Default 10.
            Max number of nonzero coeffs for sparse codes

        :param tol: Default 0.
            Add nonzero coeffs until :code:`norm(signal - dict*sparse_code) < tol`

        :param n_threads:  Default 1.
            Number of threads to use.

        :return:
            Sparse codes, shape (n_atoms, n_signals)
    """
    if isinstance(signals, preprocess.Patches):
        signals = signals.patches

    if signals.dtype != np.float64:
        signals = signals.astype(np.float64)

    return _dictlearn.omp_batch(signals, dictionary, n_nonzero, tol, n_threads)


def omp_cholesky(signals, dictionary, n_nonzero=10, tol=0, n_threads=1):
    """
        Cholesky Orthogonal Matching pursuits. Use omp_batch if many signals need
        to be sparse coded. See [2] for details

        >>> import dictlearn as dl
        >>> image = dl.imread('some/image.png')
        >>> dictionary = dl.load_dictionary('some-dictionary')
        >>> image_patches = dl.Patches(image, 8)
        >>> sparse_codes = dl.omp_cholesky(image_patches, dictionary, n_nonzero=4)
        >>> sparse_approx = dictionary.dot(sparse_codes)

        :param signals:
            Signals to sparse code, shape (signal_size,) or (signal_size, n_signals)

        :param dictionary:
            ndarray, shape (signal_size, n_atoms)

        :param n_nonzero: Default 10.
            Max number of nonzero coeffs for sparse codes

        :param tol: Default 0.
            Add nonzero coeffs until norm(signal - dict*sparse_code) < tol

        :param n_threads:  Default 1.
            Number of threads to use.

        :return:
            Sparse codes, shape (n_atoms, n_signals)
    """
    if isinstance(signals, preprocess.Patches):
        signals = signals.patches

    if signals.dtype != np.float64:
        signals = signals.astype(np.float64)

    return _dictlearn.omp_cholesky(signals, dictionary, n_nonzero,
                                   tol, n_threads)


def omp_mask(signals, masks, dictionary, n_nonzero=None,
             tol=1e-6, n_threads=1, verbose=False):
    """
        Orthogonal Matching Pursuit for masked data. Tries to reconstruct the full
         set of signals, by ignoring data points signals[i, j] where mask[i, j] == 0.


         >>> import dictlearn as dl
         >>> broken_image = dl.imread('some-broken-image')
         >>> mask = dl.imread('mask-for-some-image')
         >>> # Create patches from broken_image and mask + get dictionary
         >>> sparse_codes = dl.omp_mask(broken_image, mask, dictionary)
         >>> reconstructed_image_patches = dictionary.dot(sparse_codes)


    :param signals:
        Corrupted signals, shape (size, n_signals)

    :param masks:
        Masks, shape (size, n_signals). masks[:, i] is the mask for signals[:, i]

    :param dictionary:
        Trained dictionary, shape (size, n_atoms)

    :param n_nonzero: Default None.
        Max number of nonzero coeffs to use

    :param tol:  Default 1e-6.
        Stop if signal approximation is within the accuracy. Overwrites n_nonzero

    :param n_threads: Not used

    :param verbose: Default False.
        Print progress

    :return:
        Sparse approximation to signals, shape (n_atoms, n_signals)
    """

    size, n_signals = signals.shape
    size_mask, n_masks = masks.shape
    n_atoms = dictionary.shape[1]

    if size != size_mask or n_signals != n_masks:
        raise ValueError('The size signals and masks has to be equal '
                         'signals.shape != masks.shape, {} != {}'
                         .format(signals.shape, masks.shape))

    if size != dictionary.shape[0]:
        raise ValueError('Need signal size equal to atom size.'
                         ' signals.shape[0] != dictionary.shape[0] -> {} != {}'
                         .format(size, dictionary.shape[0]))

    sparse = np.zeros((n_atoms, n_signals))
    n_nonzero = n_atoms if n_nonzero is None else n_nonzero

    if verbose:
        print('OMP Mask:')

    for k in range(n_signals):
        if verbose:
            print(' {}/{}'.format(k + 1, n_signals), end='\r')

        y = signals[:, k]
        mask = masks[:, k]
        dic = dictionary.copy()
        y = mask * y
        r = y.copy()

        dic = dic * np.outer(mask, np.ones((1, n_atoms)))
        scale = np.sqrt(np.sum(dic * dic, axis=0))
        nonzero = np.nonzero(scale)
        scale[nonzero] = 1.0 / scale[nonzero]

        n_active = 0
        active = np.zeros(n_atoms, dtype=np.uint16)

        while True:  # np.sum(r * r) > tol:
            h = np.abs(np.dot(dic.T, r))
            # h *= scale
            active[n_active] = np.argmax(h)
            # alpha = lstsq(dic[:, active[:n_active+1], y)[0]
            alpha = np.dot(linalg.pinv(dic[:, active[:n_active+1]]), y)
            r = y - np.dot(dic[:, active[:n_active+1]], alpha)
            n_active += 1

            if n_active >= n_nonzero:
                break

            if tol > 0 and np.sum(r * r) < tol:
                break

        sparse[active[:n_active], k] = alpha

    return sparse


def lars(signals, dictionary, n_nonzero=0, alpha=0, lars_params=None, **kwargs):
    """
        "Homotopy" algorithm for solving the Lasso

            argmin 0.5*||X - DA||_2^2 + r*||A||_1

        for all r.

        This algorithm is supposedly the most accurate for l1
        regularization.

        This is terribly slow, and not very accurate. ~20x slower
        than OMP. Find this strange as OMP solves a NP-Hard problem
        and this a convex

        :param signals:
            Signals to encode. Shape (signal_size, n_signals) or (signal_size,)

        :param dictionary:
            Dictionary, shape (signal_size, n_atoms)

        :param n_nonzero:
            Number of nonzero coefficients to use

        :param alpha:
            Regularization parameter. Overwrites n_nonzero

        :param lars_params:
            See sklearn.linear_models.LassoLars docs

        :param kwargs:
            Not used. Just to make calling API for all regularization algorithms the same

        :return:
            Sparse codes, shape (n_atoms, n_signals) or (n_atoms,)

    """
    params = {
        'precompute': True,
        'fit_path': False,
        'normalize': True
    }

    if n_nonzero > 0 and alpha == 0:
        params['n_nonzero_coefs'] = int(n_nonzero)
        model = linear_model.Lars()
    elif alpha > 0:
        params['alpha'] = alpha
        model = linear_model.LassoLars()
    else:
        raise ValueError('Need to specify either regularization '
                         'parameter alpha or number of nonzero '
                         'coefficients n_nonzero')

    if isinstance(lars_params, dict):
        params.update(lars_params)

    model.set_params(**params)
    model.fit(dictionary.copy(), signals)
    return model.coef_.T.copy()


def lasso(signals, dictionary, alpha, lasso_params=None, **kwargs):
    """

    :param signals:
        Signals to encode, shape (signal_size,) or (signal_size, n_signals)

    :param dictionary:
        Dictionary, shape (signal_size, n_atoms)

    :param alpha:
        Regularization parameter. <~0.9 yields more accurate results than OMP, but slower

    :param lasso_params:
        Other parameters. See sklearn.linear_model.Lasso

    :param kwargs:
        Not used, just for making calling compatible with other sparse coding methods

    :return:
        Sparse codes, shape (n_atoms,) or (n_atoms, n_signals)
    """
    params = {
        'alpha': alpha,
        'precompute': True,
        'fit_intercept': False
    }

    if isinstance(lasso_params, dict):
        params.update(lasso_params)

    model = linear_model.Lasso()
    model.set_params(**params)
    model.fit(dictionary.copy(), signals)
    return model.coef_.T.copy()


def matching_pursuit(signal, dictionary, sparsity=None, tol=None):
    """
    Coordinate descent algorithm: Matching pursuit
    Mallat and Zhang 1993

    Finds a sparse decomposition of the signal x given
    a dictionary D.

    Not guaranteed to converge

    :param signal:
        signal to approximate

    :param dictionary:
        Pre-learned dictionary

    :param sparsity:
        stop if ||approx||_0 < sparsity

    :param tol:
        stop if ||signal - dict*decomp|| < tol

    :return:
        sparse decomposition of signal
    """
    assert len(signal.shape) == 1
    assert len(dictionary.shape) == 2
    warnings.warn('Use _dictlearn.omp_[cholesky, batch]', DeprecationWarning)
    _ = signal.shape[0]
    p = dictionary.shape[1]

    decomp = np.zeros(p)

    if sparsity is None and tol is None:
        raise ValueError('Need a stopping criterion, sparsity or tol')

    while True:
        approx = signal - dictionary.dot(decomp)
        j = np.argmax(np.dot(dictionary.transpose(), approx))
        decomp[j] = decomp[j] + dictionary[:, j].dot(approx)

        if tol is not None:
            error = LA.norm(signal - dictionary.dot(decomp)) ** 2
            if error < tol:
                break

        if sparsity is not None:
            if np.count_nonzero(decomp) == sparsity:
                break

    return decomp


def iterative_hard_thresholding(signal, dictionary, iters, step_size,
                                initial, n_nonzero=None, penalty=None):
    """

    If penalty make sure sqrt(2*penalty) is not bigger than
    all elements in initial_a, if that's the case the solution
    is just the zero vector

    Requires tuning of the hyper parameters step_size, initial_a
    and penalty. optimize.omp_cholesky may be a better choice.

    If reg_param is supplied this solves:

        min 0.5*||signal - D*alpha||_2^2 + reg_param||alpha||_0

    If n_nonzero is supplied then the following is solved for alpha

        min || signal - D*alpha||_2^2 such that ||alpha||_0 <= n_nonzero

    :param signal:
        Signal in R^m

    :param dictionary:
        Dictionary (D) in R^(m,p)

    :param iters:
        Number of iterations

    :param step_size:
        Step size for gradient descent step

    :param initial:
        Initial sparse coefficients. Need ||initial_a||_0 <= nonzero if n_nonzero not None

    :param n_nonzero:
        Sparsity target

    :param penalty:
        Penalty parameter

    :return:
        Sparse decomposition of signal
    """
    alpha = initial.copy()

    if n_nonzero is None and penalty is None:
        raise ValueError('Need to supply target sparsity n_nonzero, '
                         'or penalty parameter, penalty')

    if n_nonzero is not None and LA.norm(initial, ord=0) > n_nonzero:
        raise ValueError('Initial_a has to satisfy sparsity target n_nonzero')

    m, p = dictionary.shape

    assert signal.shape[0] == m
    assert alpha.shape[0] == p

    for i in range(iters):
        # One step gradient descent
        alpha += step_size * dictionary.T.dot(signal - dictionary.dot(alpha))
        abs_alpha = np.abs(alpha)
        # Choose threshold
        if n_nonzero is None:
            thresh = math.sqrt(2 * penalty)
        else:
            # Keep only n_nonzero biggest coeffs
            thresh = np.unique(abs_alpha)[-n_nonzero]

        alpha[np.where(abs_alpha < thresh)] = 0

    return alpha


def iterative_soft_thresholding(signal, dictionary, initial, reg_param=None,
                                n_nonzero=None, step_size=0.1, iters=10):
    """
    l1 reg using iterative soft thresholding

    if regularization parameter is given solve:

        min_a 1/2 * || x - D*alpha||_2^2 + reg_param*||alpha||_1

    if number of nonzero is given solve:

        min_a ||signal - D*alpha||_2^2 such that ||alpha||_1 <= n_nonzero

    This method requires tuning of the initial value for alpha, step_size
    and iters/res_param.

    :param signal:
        Signal, shape (signal_size, )

    :param dictionary:
        Dictionary, shape (signal_size, n_atoms)

    :param initial:
        Initial sparse codes, shape (n_atoms, )

    :param reg_param:
        Regularization parameter

    :param n_nonzero:
        Max number of nonzero coeffs

    :param step_size:
        Gradient descent step size

    :param iters:
        Number of iterations

    :return:
        Sparse codes, shape (n_atoms, )
    """
    signal = signal.T
    dictT = dictionary.T
    alpha = initial.copy()
    if reg_param is None and n_nonzero is None:
        raise ValueError('Regularization parameter or number '
                         'of nonzero coeffs is needed')

    for _ in range(iters):
        # One step gradient descent
        err = np.dot(dictT, signal) - np.dot(dictT, dictionary.dot(alpha))
        alpha += step_size*err

        if reg_param is not None:
            alpha = filters.threshold(alpha, reg_param, type='soft')
        else:
            raise NotImplementedError()
            alpha = l1_ball(alpha, n_nonzero)

    return alpha


def l1_ball(vector, target):
    """
        Create sparse approximation of 'vector' by
        projecting onto to l1-ball keeping at most
        'target' coefficients active

    :param vector:
        Vector to project to sparse

    :param target:
        Max number of nonzero coefficients

    :return:
        Sparse vector, same shape as 'vector'
    """
    if LA.norm(vector, ord=1) <= target:
        return vector

    sign = np.sign(vector)
    u = project_simplex(np.abs(vector), target)
    return sign*u


def project_simplex(vector, target):
    assert vector.ndim == 1
    sign = np.sign(vector)
    vector = np.abs(vector)
    U = np.arange(vector.size)
    s = 0
    p = 0.0

    ii = 0

    # TODO fix inf loop big when L doesn't empty

    while U.size > 0:
        ii += 1
        k = np.where(U == np.random.choice(U, 1))[0][0]
        G = []
        L = []

        for idx in U:
            val = vector[k]
            if vector[idx] < val:
                L.append(idx)
            else:
                G.append(idx)

        grad_p = len(G)
        grad_s = np.sum(vector[G])

        if s + grad_s - (p + grad_p)*vector[k] < target:
            s += grad_s
            p += grad_p
            U = np.asarray(L)
        else:
            U = np.array([i for i in G if i != k])

    thresh = (s - target)/p
    return sign*np.maximum(vector - thresh, 0)


def lasso_coordinate_descent(signals, dictionary, iters, reg_param, initial=None):
    """
    :param signals:
    :param dictionary:
    :param iters:
    :param reg_param:
    :param initial:
    :return:
    """
    K = dictionary.shape[1]

    if initial is None:
        sparse = np.zeros((K, signals.shape[1]), dtype=signals.dtype)

    if iters < K:
        iters = K

    for j in range(signals.shape[1]):
        signal = signals[:, j]
        residual = signal.astype(float)

        for i in range(iters):
            n = i % (K - 1)
            alpha_prev = sparse[j, n]
            atom = dictionary[:, n]
            alpha_new = sparse[j, n] + np.dot(atom, residual)

            if alpha_new <= -reg_param:
                alpha_new += reg_param
            elif alpha_new >= reg_param:
                alpha_new -= reg_param
            else:
                alpha_new = 0

            sparse[j, n] = alpha_new

            change = alpha_prev - alpha_new
            residual += np.dot(atom, change)

    return sparse.T


def subset_selection(signals, dictionary, n_nonzero):
    """
    :param signals:
    :param dictionary:
    :param n_nonzero:
    :return:
    """
    sparse = np.zeros((dictionary.shape[1], signals.shape[1]))

    for i in range(signals.shape[1]):
        signal = signals[:, i]
        resp = np.abs(np.dot(dictionary.T, signal))
        active = np.argsort(resp)[-n_nonzero:]
        alpha = np.dot(LA.pinv(dictionary[:, active]), signal)
        sparse[active, i] = alpha

    return sparse


def feature_sign_search(signals, dictionary, reg_param, max_iters=1000):
    """
    :param signals:
    :param dictionary:
    :param reg_param:
    :param max_iters:
    :return:
    """
    # Todo optimize -> cython
    DtY = np.dot(dictionary.T, signals).T
    Gram = np.dot(dictionary.T, dictionary)
    signals = signals.T
    sparse = np.zeros((Gram.shape[0], DtY.shape[0]), np.double)

    for i in range(DtY.shape[0]):
        sparse[:, i] = fss(Gram, DtY[i], signals[i], dictionary, max_iters, reg_param)

    return sparse


def fss(gram, dty, signal, dictionary, max_iters, gamma):
    """
    :param gram:
    :param dty:
    :param signal:
    :param dictionary:
    :param max_iters:
    :param gamma:
    :return:
    """
    K = gram.shape[0]
    solution = np.zeros(K, dtype=np.double)
    theta = np.zeros(K)
    active = np.zeros(K)
    n_active = 0
    active_set = np.zeros(K, dtype=np.int32)
    activate = True

    for i in range(max_iters):
        if activate:
            not_active = np.where(theta == 0)[0]
            grad = np.dot(gram, solution) - dty
            theta = np.sign(solution)
            h = np.argmax(np.abs(grad[not_active]))

            if abs(grad[h]) >= gamma:
                active[not_active[h]] = 1
                active_set[n_active] = not_active[h]
                theta[not_active[h]] = -np.sign(grad[not_active[h]])
                n_active += 1

        # Check rank todo
        # x_new
        idx = active_set[:n_active]
        x_new = np.dot(np.linalg.pinv(gram[np.ix_(idx, idx)]),
                                 dty[idx] - gamma / 2 * theta[idx])
        signs_x_new = np.sign(x_new)
        diff = np.where(np.abs(signs_x_new - theta[idx]))[0]

        if diff.size > 0:
            x_new = line_search(x_new, solution[idx], dictionary[np.ix_(idx, idx)], gamma, diff, signal)

        solution[idx] = x_new
        zero = np.abs(solution) < 1e-12
        solution[zero] = 0
        nonzero = np.nonzero(solution)[0]
        active_set = np.zeros(len(nonzero), dtype=int)
        n_active = len(nonzero) + 1
        active_set[:n_active] = nonzero

        grad = np.dot(gram, solution) - dty
        # Cond 4a
        activate = np.allclose(grad[nonzero] + gamma * np.sign(solution[nonzero]), 0)

        # Cond 4b
        something = np.all(np.abs(grad[zero]) <= gamma)
        if something:
            return solution

    return solution


def line_search(new, old, dico, gamma, diff, signal):
    solution = new
    cost = lambda z: abs(np.sum(signal - np.dot(dico, z))) + gamma * np.sum(np.abs(z))
    best = cost(solution)

    for i in diff:
        x, y = old[i], new[i]
        step = x / (x - y)
        cand = old - step*(old - new)
        fss_cost = cost(cand)

        if fss_cost < best:
            best = fss_cost
            solution = cand

    return solution
