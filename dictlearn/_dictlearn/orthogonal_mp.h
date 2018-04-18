#ifndef __ORTHOGONAL_MP
#define __ORTHOGONAL_MP
#ifdef __cplusplus
extern "C" {
#endif

/*
    Solves LL'x = b.
*/
typedef void(*DPOTRS)(char *, int *, int *, double *, int *, 
                      double*, int *, int *);

/*
    Double precision general matrix matrix product
*/
typedef void (*DGEMM)(char *, char *, int *, int *, int *, double *, double *, 
                      int *, double *, int *, double *, double *, int *);


/*
    Double precision general matrix vector product
*/
typedef void (*BLAS_DGEMV)(char *, int *, int *, double *, double *, int *, 
              double *, int *, double *, double *, int *);


#include <stddef.h>
#include "common.h"


/*
    Main Cholesky-OMP function. This is just a parallel for loop calling 
    omp_cholesky_core with appropriate arguments

    Args
    ----
        signals:      Matrix with one signal per row. 
                      Shape: (signal_size*n_signals)
        size:         Size of signal
        n_signals:    Number of signals
        dictionary:   (size x n_atoms)
        n_atoms:      number of atoms / columns in dictionary
        dictionary_t: transposed dictionary
        alpha:        Transpose sparse codes, overwritten with solution.
                      Shape (n_signals * n_atoms)
        n_nonzero_coeffs:  Max number of nonzero entries in sparse codes
        n_threads:     Number of threads to use

    Returns
    -------
        Number >= 0 if everything ok.
*/
int _omp_cholesky(double *signals, size_t size, size_t n_signals,
             double *dictionary, size_t n_atoms,
             double *dictionary_t, double *alpha, 
             size_t n_nonzero_coeffs, size_t n_threads, 
             double tolerance);


/*
    Similar to omp_cholesky and omp_batch
    Solves the formulation M*(signals - D*alpha)
*/
int _omp_cholesky_mask(double *signals, size_t mask,
             size_t size, size_t n_signals, double *dictionary, 
             size_t n_atoms, double *alpha, size_t n_nonzero_coeffs, 
             size_t n_threads, double tolerance);


/*
    OMP-Cholesky
    ************

    This is the implementation of the OMP-Cholesky variant presented in
        [1] Rubinstein, Ron, Michael Zibulevsky, and Michael Elad. "Efficient 
        implementation of the K-SVD algorithm using batch orthogonal 
        matching pursuit." Cs Technion 40.8 (2008): 1-15.


    Args
    ----
        signal: Signal to encode
        dict: Dictionary, (signal_size * n_atoms)
        dict_t: Pre-transposed dictionary 
        signal_size: Size of signal
        n_atoms: Number of dictionary atoms
        n_nonzero_coeffs: Max number of nonzero coeffs in sparse approx.
    
    Returns
    -------
        Sparse approximation of the signal, of length n_atoms
*/
double *omp_cholesky_core(double *signal, double *dict, double *dict_t,
                  size_t signal_size, size_t n_atoms,
                  size_t n_nonzero_coeffs, double tolerance);

/*
    Internal Cholesky OMP func

    Updates residual according to line 12 in algorithm 2 in [1], eqv with
    residual = x - numpy.dot(dict[:, indices], gamma[indices])


    Args
    ----
        dict: Dictionary
        n_atoms: Number of columns in dict
        signal_size: Rows in dict and entries in signal
        residual: Previous residual, length signal_size
        gamma: The sparse codes gamma[I], in [2]
        indices: Index set of active atoms
        n_active: Number of active atoms, length of indices
*/
void update_residual(double *dict, size_t n_atoms, size_t signal_size, 
                     double *signal, double *residual, double *gamma, 
                     size_t *indices, size_t n_active);


/*
    OMP Batch
    *********

    Accelerated OMP using pre-computations, see [1]

    Args
    ----
        alphas: np.dot(signals.T, dictionary), see _omp_cholesky for signals and
                dictionary.
        n_signals: Number of signals
        n_atoms: Number of dictionary atoms
        norms: An array of n_signals elements where each element is the l2 norm
               of the corresponding signal
        gram: Gram matrix of dictionary
        target_error: Error tolerance, use max_n_nonzero if == 0
        max_n_nonzero: Max number of nonzero coefficients
        n_threads: Number of OpenMP threads
        out: Sparse codes saved in this array. out[:, i] codes for signal i
*/
int _omp_batch(double *alphas, size_t n_signals, size_t n_atoms, 
              double *norms, double *gram, double target_error,
              size_t max_n_nonzero, size_t n_threads, double *out);


/*
    Encodes on signal

    Args
    ----
        alpha_init: np.dot(signal, dictionary)
        gram: Gram matrix of dictionary, shape (n_atoms, n_atoms)
        n_atoms: Number of dictionary atoms
        error: Initial error, l2 norm of signal
        target_error: Error tolerance
        max_n_nonzero: Max size of support
        gamma: Output, sparse codes

*/
int omp_batch_core(double *alpha_init, double *gram, size_t n_atoms, 
                   double error, double target_error, 
                   size_t max_n_nonzero, double *gamma);


/*
    Prints size elements of arr
*/
void print_arr(double *arr, size_t size, const char *name);


/*
    Prints row*cols elements of mat
*/
void print_mat(double *mat, size_t rows, size_t cols, const char *name);


#ifdef __cplusplus
}  /* extern C */
#endif
#endif /* __ORTHOGONAL_MP */
