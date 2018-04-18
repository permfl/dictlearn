cdef extern from "orthogonal_mp.h":
    ctypedef void (*BLAS_DGEMV)(char *, int *, int *, double *, double *, int *,
              double *, int *, double *, double *, int *);

    ctypedef void(*DPOTRS)(char *, int *, int *, double *, int *,
                      double*, int *, int *)

    int _omp_cholesky(double *signals, size_t size, size_t n_signals,
                      double *dictionary, size_t n_atoms,
                      double *dictionary_t, double *alpha,
                      size_t n_nonzero_coeffs, size_t n_threads,
                      double tolerance) nogil

    void forward_solve(double *lower, size_t active_size,
                       double *b, size_t width)

    void backward_solve(double *upper, size_t active_size,
                        double *b, size_t width)

    void cholesky_solve(double *lower, size_t active_size, double *b, size_t real_size)
    size_t argmax_mat_vec(double *mat, size_t r, size_t c, double *vec)
    void set_entries(double *dest, double *src, size_t *indices, size_t n)
    void fill_entries(double *dest, double *src, size_t *indices, size_t n)
    double *copy_of(double *arr, size_t n)
    double *transpose(double *mat, size_t rows, size_t cols)
    double dot(double *arr1, double *arr2, size_t n)
    void mat_vec_mult(double *mat, double *vec, size_t rows, size_t cols,
                         double *out)
    double *mat_mat_mult(double *mat1, size_t r1, size_t c1,
                         double *mat2, size_t r2, size_t c2)

    int _omp_batch(double *alphas, size_t n_signals, size_t n_atoms,
                   double *norms, double *gram, double target_error,
                   size_t max_n_nonzero, size_t n_threads, double *out) nogil

    void faster_cholesky(double *lower, size_t active_size, double *b, size_t real_size)
    int contains(size_t value, size_t *array, size_t n)


cdef extern  from "bestexemplar.h":
    void bestexemplar(double* image, size_t height, size_t width,
                      const double* patch, unsigned int patch_height,
                      unsigned int patch_width, unsigned int* to_fill,
                      unsigned int* source, unsigned int* best)

    void bestexemplar_3d(double* image, size_t height, size_t width, size_t depth,
                         const double* patch, unsigned int patch_height, 
                         unsigned int patch_width, unsigned int patch_depth,
                         unsigned int* to_fill, unsigned int* source, 
                         unsigned int* best)


