def wraps_faster_lu(lower, active_s, b, r_size):
    cdef size_t active_size = active_s
    cdef size_t real_size = r_size
    cdef double[:, :] L = np.ascontiguousarray(lower)
    cdef double[:] b_ = np.ascontiguousarray(b)

    faster_cholesky(&L[0, 0], active_size, &b_[0], real_size)

    return np.asarray(b)


def wraps_forward(lower, active_size, b, width):
    cdef double[:, :] lower_ = np.ascontiguousarray(lower, dtype=np.float64)
    cdef double[:] b_ = np.ascontiguousarray(b, np.float64)
    cdef int active_size_ = active_size
    cdef int width_ = width

    forward_solve(&lower_[0, 0], active_size_, &b_[0], width_)

    return np.asarray(b)


def wraps_backward(lower, active_size, b, width):
    cdef double[:, :] lower_ = np.ascontiguousarray(lower, dtype=np.float64)
    cdef double[:] b_ = np.ascontiguousarray(b, np.float64)
    cdef int active_size_ = active_size
    cdef int width_ = width

    backward_solve(&lower_[0, 0], active_size_, &b_[0], width_)

    return np.asarray(b)


def wraps_lu(lower, active_size, b, real_size):
    cdef double[:, :] lower_ = np.ascontiguousarray(lower, dtype=np.float64)
    cdef double[:] b_ = np.ascontiguousarray(b, np.float64)
    cdef int active_size_ = active_size
    cdef int width_ = real_size

    cholesky_solve(&lower_[0, 0], active_size_, &b_[0], width_)

    return np.asarray(b)


def wraps_argmax_mat_vec(matrix, vector):
    cdef double[:, :] mat = np.ascontiguousarray(matrix, dtype=np.float64)
    cdef double[:] vec = np.ascontiguousarray(vector, dtype=np.float64)
    cdef size_t rows = mat.shape[0]
    cdef size_t cols = mat.shape[1]

    return argmax_mat_vec(&mat[0, 0], rows, cols, &vec[0])


def wraps_set_entries(dest, src, indices, size):
    # void set_entries(double *dest, double *src, size_t *indices, size_t n)
    cdef double[:] dest_ = np.ascontiguousarray(dest, dtype=np.float64)
    cdef double[:] src_ = np.ascontiguousarray(src, dtype=np.float64)
    cdef size_t[:] idx = np.ascontiguousarray(indices, dtype=np.uint64)
    cdef size_t n = size

    set_entries(&dest_[0], &src_[0], &idx[0], n)
    return np.asarray(dest_)


def wraps_fill_entries(dest, src, indices, size):
    # void fill_entries(double *dest, double *src, size_t *indices, size_t n)
    cdef double[:] dest_ = np.ascontiguousarray(dest, dtype=np.float64)
    cdef double[:] src_ = np.ascontiguousarray(src, dtype=np.float64)
    cdef size_t[:] idx = np.ascontiguousarray(indices, dtype=np.uint64)
    cdef size_t n = size

    fill_entries(&dest_[0], &src_[0], &idx[0], n)
    return np.asarray(dest_)


def wraps_copy_of(arr):
    cdef size_t n = len(arr)
    cdef double[:] a = np.ascontiguousarray(arr)

    res = copy_of(&a[0], n)
    cdef double [:] r = <double[:n]> res
    return np.asarray(r)


def wraps_transpose(matrix):
    # double *transpose(double *mat, size_t rows, size_t cols)
    cdef size_t rows = matrix.shape[0]
    cdef size_t cols = matrix.shape[1]
    cdef double[:, :] mat = np.ascontiguousarray(matrix)
    res = transpose(&mat[0, 0], rows, cols)
    cdef double [:, :] r = <double[:cols, :rows]> res
    return np.asarray(r)


def wraps_dot(arr1, arr2, size):
    # double dot(double *arr1, double *arr2, size_t n)
    cdef size_t n = size
    cdef double[:] a1 = np.ascontiguousarray(arr1, dtype=np.float64)
    cdef double[:] a2 = np.ascontiguousarray(arr2, dtype=np.float64)
    return dot(&a1[0], &a2[0], n)


def wraps_mat_vec(matrix, vector):
    # double *mat_vec_mult(double *mat, double *vec, size_t rows, size_t cols)
    cdef size_t rows = matrix.shape[0]
    cdef size_t cols = matrix.shape[1]
    cdef double[:, :] mat = np.ascontiguousarray(matrix)
    cdef double[:] res = np.zeros(rows, dtype=np.float64)
    cdef double[:] vec = np.ascontiguousarray(vector)
    mat_vec_mult(&mat[0, 0], &vec[0], rows, cols, &res[0])
    #cdef double [:] r = <double[:rows]> res
    return np.asarray(res)


def wraps_mat_mat(matrix1, matrix2):
    # double *mat_mat_mult(double *mat1, size_t r1, size_t c1,
    #                      double *mat2, size_t r2, size_t c2)
    cdef size_t r1 = matrix1.shape[0]
    cdef size_t c1 = matrix1.shape[1]
    cdef double[:, :] mat1 = np.ascontiguousarray(matrix1)
    cdef size_t r2 = matrix2.shape[0]
    cdef size_t c2 = matrix2.shape[1]
    cdef double[:, :] mat2 = np.ascontiguousarray(matrix2)
    res = mat_mat_mult(&mat1[0, 0], r1, c1, &mat2[0, 0], r2, c2)
    cdef double [:, :] r = <double[:r1, :c2]> res
    return np.asarray(r)


def wraps_contains(value, array):
    cdef size_t val = value
    cdef size_t n = len(array)
    cdef size_t[:] arr = np.ascontiguousarray(array, dtype=np.uint64)
    cdef int res
    res = contains(val, &arr[0], n)
    return bool(res)