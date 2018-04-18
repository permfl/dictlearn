#!python
#cython: wraparound=False, boundscheck=False, cdivision=True
from __future__ import print_function
import numpy as np
cimport numpy as np

np.import_array()

from scipy.linalg.cython_lapack cimport dsyev
from libc cimport math as cmath

ctypedef long long size
ctypedef unsigned long PySize


def vesselness_single_scale(double alpha, double beta, double c, 
                            Py_ssize_t x, Py_ssize_t y, Py_ssize_t z,
                            np.ndarray[double, ndim=3] ddx,
                            np.ndarray[double, ndim=3] ddy,
                            np.ndarray[double, ndim=3] ddz,
                            np.ndarray[double, ndim=3] dxdy,
                            np.ndarray[double, ndim=3] dxdz,
                            np.ndarray[double, ndim=3] dydz):
    """
    """
    cdef double[:, :] hessian = np.zeros((3, 3), dtype=np.double)
    cdef double[:, :, :] response = np.zeros((x, y, z))
    cdef double[:] eigvals = np.zeros(3, dtype=np.double)
    cdef int i, j, k, info
    cdef int largest, medium, smallest, n
    cdef double vesselness = 0

    for i in range(x):
        for j in range(y):
            for k in range(z):
                hessian[0, 0] = ddx[i, j, k]
                hessian[0, 1] = dxdy[i, j, k]
                hessian[0, 2] = dxdz[i, j, k]
                hessian[1, 0] = dxdy[i, j, k]
                hessian[1, 1] = ddy[i, j, k]
                hessian[1, 2] = dydz[i, j, k]
                hessian[2, 0] = dxdz[i, j, k]
                hessian[2, 1] = dydz[i, j, k]
                hessian[2, 2] = ddz[i, j, k]

                info = hessian_eigenvalues(hessian, eigvals)

                if info != 0:
                    lapack_error(info, 'dsyev')

                largest = 2
                medium = 1
                smallest = 0

                for n in range(3):
                    if abs(eigvals[n]) > abs(eigvals[largest]):
                        largest = n

                for n in range(3):
                    if abs(eigvals[n]) < abs(eigvals[smallest]):
                        smallest = n

                medium = 3 - largest - smallest

                if eigvals[medium] < 0 and eigvals[largest] < 0:
                    vesselness = frangi_response(
                        eigvals[smallest], eigvals[medium], eigvals[largest], 
                        alpha, beta, c
                    )

                    response[i, j, k] = vesselness
    return np.asarray(response)


cdef inline double frangi_response(double e1, double e2, double e3, 
                                   double alpha, double beta, double c):
    cdef double fe1 = cmath.fabs(e1)
    cdef double fe2 = cmath.fabs(e2)
    cdef double fe3 = cmath.fabs(e3)
    cdef double ratio_b = fe1 / cmath.sqrt(fe2 * fe3)
    cdef double ratio_a = fe2 / fe3
    cdef double norm = cmath.sqrt(e1*e1 + e2*e2 + e3*e3)
    cdef double left, middle, rigth

    left = 1 - cmath.exp(-ratio_a*ratio_a / (2*alpha*alpha))
    middle = cmath.exp(-ratio_b*ratio_b / (2*beta*beta))
    right = 1 - cmath.exp(-norm*norm / (2*c*c))
    return left*middle*right


def tubular_candidate_points(Py_ssize_t rows, Py_ssize_t  cols, Py_ssize_t  depth,
                             np.ndarray[double, ndim=3] ddx,
                             np.ndarray[double, ndim=3] ddy,
                             np.ndarray[double, ndim=3] ddz,
                             np.ndarray[double, ndim=3] dxdy,
                             np.ndarray[double, ndim=3] dxdz,
                             np.ndarray[double, ndim=3] dydz):
    """

    Find points that looks like a tube by checking the concavity of the
    hessian matrix.

    :param rows:
        Height of derivative array

    :param cols:
        Width of derivative array

    :param depth:
        Depth of derivative array

    :param ddx:
        2nd derivative in x direction

    :param ddy:
        2nd derivative in y direction

    :param ddz:
        2nd derivative in z direction

    :param dxdy:
    :param dxdz:
    :param dydz:

    :return:
        Array of coordinates, (n_points, 3). The columns are x, y, and z coords.
    """
    cdef double[:, :] hessian = np.zeros((3, 3), dtype=np.double)
    cdef unsigned int[:, :] selection = np.zeros((rows*cols*depth, 3), dtype=np.uint32)
    cdef double[:] eigvals = np.zeros(3, dtype=np.double)
    cdef int i, j, k, info
    cdef int largest, medium, smallest, n
    cdef Py_ssize_t  count = 0

    for i in range(rows):
        for j in range(cols):
            for k in range(depth):
                hessian[0, 0] = ddx[i, j, k]
                hessian[0, 1] = dxdy[i, j, k]
                hessian[0, 2] = dxdz[i, j, k]
                hessian[1, 0] = dxdy[i, j, k]
                hessian[1, 1] = ddy[i, j, k]
                hessian[1, 2] = dydz[i, j, k]
                hessian[2, 0] = dxdz[i, j, k]
                hessian[2, 1] = dydz[i, j, k]
                hessian[2, 2] = ddz[i, j, k]

                info = hessian_eigenvalues(hessian, eigvals)

                if info != 0:
                    lapack_error(info, 'dsyev')

                largest = 2
                medium = 1
                smallest = 0

                for n in range(3):
                    if abs(eigvals[n]) > abs(eigvals[largest]):
                        largest = n

                for n in range(3):
                    if abs(eigvals[n]) < abs(eigvals[smallest]):
                        smallest = n

                medium = 3 - largest - smallest

                if eigvals[medium] < 0 and eigvals[largest] < 0:
                    selection[count][0] = i
                    selection[count][1] = j
                    selection[count][2] = k
                    count += 1

    return np.asarray(selection[:count])


def single_scale_hessian_response(Py_ssize_t rows, Py_ssize_t cols, Py_ssize_t depth,
                                  double scale, np.ndarray[double, ndim=4] gradient,
                                  np.ndarray[unsigned int, ndim=2] preselection,
                                  np.ndarray[double, ndim=3] ddx,
                                  np.ndarray[double, ndim=3] ddy,
                                  np.ndarray[double, ndim=3] ddz,
                                  np.ndarray[double, ndim=3] dxdy,
                                  np.ndarray[double, ndim=3] dxdz,
                                  np.ndarray[double, ndim=3] dydz):
    """

    :param rows:
    :param cols:
    :param depth:
    :param scale:
    :param gradient:
    :param preselection:
    :param ddx:
    :param ddy:
    :param ddz:
    :param dxdy:
    :param dxdz:
    :param dydz:
    :return:
    """
    cdef double[:, :] hessian = np.zeros((3, 3), dtype=np.double)
    cdef double[:, :, :] response = np.zeros((rows, cols, depth), dtype=np.double)
    cdef double[:] eigvals = np.zeros(3, dtype=np.double)
    cdef double[:, :] eigvecs = np.zeros((3, 3), dtype=np.double)
    cdef int i, info, x, y, z
    cdef int largest, medium, smallest, n
    cdef double resp, theta
    cdef Py_ssize_t n_points = preselection.shape[0]

    for i in range(n_points):
        x = preselection[i][0]
        y = preselection[i][1]
        z = preselection[i][2]

        hessian[0, 0] = ddx[x, y, z]
        hessian[0, 1] = dxdy[x, y, z]
        hessian[0, 2] = dxdz[x, y, z]
        hessian[1, 0] = dxdy[x, y, z]
        hessian[1, 1] = ddy[x, y, z]
        hessian[1, 2] = dydz[x, y, z]
        hessian[2, 0] = dxdz[x, y, z]
        hessian[2, 1] = dydz[x, y, z]
        hessian[2, 2] = ddz[x, y, z]

        info = eigenhessian(hessian, eigvals)

        if info != 0:
            lapack_error(info, 'dsyev')

        largest = 2
        medium = 1
        smallest = 0

        for n in range(3):
            if abs(eigvals[n]) > abs(eigvals[largest]):
                largest = n

        for n in range(3):
            if abs(eigvals[n]) < abs(eigvals[smallest]):
                smallest = n

        medium = 3 - largest - smallest
        theta = 1.73205080757  # sqrt 3

        resp = circular_response(x, y, z, gradient, &hessian[0, 0], eigvals[largest],
                                 eigvals[medium], eigvals[smallest], scale, theta)

        response[x, y, z] = resp

    return response


cdef double circular_response(Py_ssize_t x, Py_ssize_t y, Py_ssize_t z,
                              np.ndarray[double, ndim=4] gradient,
                              double *eigenvectors, double eig1, double eig2,
                              double eig3, double scale, double theta):
    """
    
    :param x: 
    :param y: 
    :param z: 
    :param gradient: 
    :param eigenvectors: 
    :param eig1: 
    :param eig2: 
    :param eig3: 
    :param scale: 
    :param theta: 
    :return: 
    """
    cdef int N = <int> (2 * np.pi * cmath.sqrt(scale) + 1)
    cdef size H, W, D, i, point, n_points
    cdef count = 0
    cdef double alpha, cos, sin
    cdef double[:, :] v_alphas = np.zeros((N, 3), dtype=np.double)
    cdef double theta_scale = theta * cmath.sqrt(scale)

    for point in range(N):
        alpha = 2 * np.pi * point / N
        cos = cmath.cos(alpha)
        sin = cmath.sin(alpha)
        v_alphas[point][0] = cos*eigenvectors[0] + sin*eigenvectors[3]
        v_alphas[point][1] = cos*eigenvectors[1] + sin*eigenvectors[4]
        v_alphas[point][2] = cos*eigenvectors[2] + sin*eigenvectors[5]

    H = gradient.shape[0]
    W = gradient.shape[1]
    D = gradient.shape[2]

    cdef double resp = 0
    cdef cx, cy, cz, dx, dy, dz, vax, vay, vaz
    dx = x / H
    dy = y / W
    dz = z / D

    for point in range(N):
        vax = v_alphas[point][0]
        vay = v_alphas[point][1]
        vaz = v_alphas[point][2]

        cx = x + theta_scale * vax
        cy = y + theta_scale * vay
        cz = z + theta_scale * vaz

        fx = trilinear_interpolation(cx, cy, cz, dx, dy, dz, H, W, D, 0, gradient)
        fy = trilinear_interpolation(cx, cy, cz, dx, dy, dz, H, W, D, 1, gradient)
        fz = trilinear_interpolation(cx, cy, cz, dx, dy, dz, H, W, D, 2, gradient)

        resp = resp + fx *vax + fy * vay + fz * vaz

    return resp / N


def lapack_error(int code, method):
    """
        Check lapack error and raise
    """
    if method == 'dsyev':
        if code > 0:
            raise RuntimeError('Eigenvalues did not converge')

    msg = 'Argument %d to %s had an illegal value' % (abs(code), method)
    raise ValueError(msg)


cdef inline int hessian_eigenvalues(double[:, :] hessian, double[:] eigvals):
    """
        Calculate eigenvalues of hessian    
    """
    # Eigenvalues only
    cdef char JOBZ = 'N'

    # Lower triangular
    # The matrix 'hessian' is really upper triangular and C order
    # but lapack reads in fortran order thus upper triangular c matrix
    # is eqv with lower triangular fortran matrix
    cdef char PLO = 'L'
    # Order of hessian
    cdef int N = 3
    # Distance between each row
    cdef int LDA = 3
    # don't know
    cdef double[:] WORK = np.zeros(20, dtype=np.double)
    # Length of work
    cdef int LWORK = 20
    # info about run
    cdef int INFO

    with nogil:
        dsyev(&JOBZ, &PLO, &N, &hessian[0, 0], &LDA, &eigvals[0],
              &WORK[0], &LWORK, &INFO)

    return INFO


cdef inline int eigenhessian(double[:, :] hessian, double[:] eigvals):
    """
        Calculate eigenvectors and eigenvalues of hessian. Eigenvalues are
        stored such the hessian[i] corresponds to the eigenvalue eigvals[i]    
    """
    cdef char JOBZ = 'V'
    cdef char PLO = 'U'
    cdef int N = 3
    cdef int LDA = 3
    cdef double[:] WORK = np.zeros(30, dtype=np.double)
    cdef int LWORK = 20
    cdef int INFO

    with nogil:
        dsyev(&JOBZ, &PLO, &N, &hessian[0, 0], &LDA, &eigvals[0],
              &WORK[0], &LWORK, &INFO)

    return INFO


cdef inline double linear_interpolate(double p1, double p2, double dx):
    return p1*(1 - dx) + p2*dx


cdef inline double bilinear_interpolate(double p1, double p2, double p3, double p4,
                                        double dx, double dy):
    cdef double x = linear_interpolate(p1, p2, dx)
    cdef double y = linear_interpolate(p3, p4, dx)
    return linear_interpolate(x, y, dy)


cdef inline double trilinear_interpolation(double x, double y, double z,
                                           double dx, double dy, double dz,
                                           int H, int W, int D, int axis,
                                           double[:, :, :, :] array):
    xp = <Py_ssize_t> min(H - 1, max(0, cmath.floor(x)))
    xl = <Py_ssize_t> max(0, min(H - 1, cmath.ceil(x)))

    yp = <Py_ssize_t> min(W - 1, max(0, cmath.floor(y)))
    yl = <Py_ssize_t> max(0, min(W - 1, cmath.ceil(y)))

    zp = <Py_ssize_t> min(D - 1, max(0, cmath.floor(z)))
    zl = <Py_ssize_t> max(0, min(D - 1, cmath.ceil(z)))

    p1 = array[xp, yl, zp, axis]
    p2 = array[xl, yl, zp, axis]
    p3 = array[xp, yp, zp, axis]
    p4 = array[xl, yp, zp, axis]
    p5 = array[xp, yl, zl, axis]
    p6 = array[xl, yl, zl, axis]
    p7 = array[xp, yp, zl, axis]
    p8 = array[xl, yp, zl, axis]

    cdef double xi = bilinear_interpolate(p1, p2, p3, p4, dx, dy)
    cdef double yi = bilinear_interpolate(p5, p6, p7, p8, dx, dy)
    return linear_interpolate(xi, yi, dz)
