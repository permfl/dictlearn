#include <stdlib.h>
#include <stddef.h>
#include <math.h>
#include <float.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>

#include "orthogonal_mp.h"

#define check(x) \
    do { \
        if(x == NULL) \
            return -1; \
    } while(0)


/*
    Prints size elements of arr
*/
void print_arr(double *arr, size_t size, const char *name);


/*
    Prints row*cols elements of mat
*/
void print_mat(double *mat, size_t rows, size_t cols, const char *name);


/*
    alphas: product (D^T*X)^T, shape (n_signals, n_atoms)
    norms: array of length n_signals with x^T*x for all signals x
    gram: D^T*D, shape (n_atoms, n_atoms)

*/
int _omp_batch(double *alphas, size_t n_signals, size_t n_atoms, 
              double *norms, double *gram, double target_error,
              size_t max_n_nonzero, size_t n_threads, double *out)
{
    int i, err, status;
    status = 1;

    omp_set_num_threads((int)n_threads);

    #pragma omp parallel for private(i) 
    for(i = 0; i < n_signals; i++) {
        if(status >= 0) {
            err = omp_batch_core(&alphas[i*n_atoms], gram, n_atoms,
                                 norms[i], target_error, max_n_nonzero,
                                 &out[i*n_atoms]);
        }

        if(err < 0) {
            status = -1;
        }
    } 

    return status;
}


/*
    alpha: D^T*x
    error: x^T*x
    gram: D^T*D

    Very similar to omp_cholesky
*/
int omp_batch_core(double *alpha_init, double *gram, size_t n_atoms, 
                   double error, double target_error, 
                   size_t max_n_nonzero, double *gamma)
{
    int status;
    size_t i; /*loop counter*/
    size_t n_active, k_max_idx, n_nonzero_coeffs, *indices, L_size;
    double *alpha, *L, *beta, delta, delta_prev, c;
    double tmp, *b, *a; /* variable and arrays for holding various stuff*/

    if(target_error > 0) {
        n_nonzero_coeffs = n_atoms;
    } else {
        target_error = 0;
        n_nonzero_coeffs = max_n_nonzero;
    }

    alpha = copy_of(alpha_init, n_atoms);

    if(alpha == NULL)
        return -1;
    
    indices = (size_t*)calloc(n_nonzero_coeffs, sizeof(size_t));

    if(indices == NULL) {
        free(alpha);
        return -1;
    }

    L = (double*)calloc(n_nonzero_coeffs*n_nonzero_coeffs, sizeof(double));

    if(L == NULL) {
        free(alpha);
        free(indices);
        return -1;
    }

    b = (double*)calloc(n_nonzero_coeffs, sizeof(double));

    if(b == NULL) {
        free(alpha);
        free(indices);
        free(L);
        return -1;
    }

    a = (double*)calloc(n_nonzero_coeffs, sizeof(double));
    if(a == NULL) {
        free(alpha);
        free(indices);
        free(L);
        free(b);
        return -1;
    }

    L[0] = 1;
    L_size = 1;
    status = 0;
    n_active = 1;
    delta = 0;
    delta_prev = 0;

    while(1) {
        k_max_idx = argmax_vec(alpha, n_atoms);

        /*
            Enforce orthogonality
            Stop adding coeffs in k_max_idx is already used
        */
        if(contains(k_max_idx, indices, n_active - 1))
            break;

        if(n_active > 1) {

            for(i=0; i < n_active - 1; i++) {
                b[i] = gram[k_max_idx*n_atoms + indices[i]];
            }
            
            forward_solve(L, L_size, b, n_nonzero_coeffs);
            c = 1 - dot(b, b, L_size);

            /* 
                1.0 + DBL_EPSILON != 1.0
                Skip if c == 0 s.t we don't get a 0
                on diagonal in L
            */
            if(c < DBL_EPSILON) {
                break;
            }

            /* Set "last" row of L equal to w*/
            for(i = 0; i <= L_size; i++) {
                L[L_size*n_nonzero_coeffs + i] = b[i];
            }
            /* Set "last" diagonal element*/
            L[(n_active - 1)*n_nonzero_coeffs + (n_active - 1)] = sqrt(c);
            L_size++;
        }

        indices[n_active - 1] = k_max_idx;
        fill_entries(b, alpha_init, indices, n_active);
        faster_cholesky(L, L_size, b, n_nonzero_coeffs);


        
        /*
            Update the solution, gamma
            gamma[indices] = b
        */
        set_entries(gamma, b, indices, n_active);

        if(n_active >= n_nonzero_coeffs) {
            break; /*wanted sparsity reached */
        }

        beta = mat_vec_mult_indices(gram, n_atoms, n_atoms, gamma, 
                                    indices, n_active);

        if(beta == NULL) {
            status = -1;
            break;
        }

        for(i = 0; i < n_atoms; i++) {
            alpha[i] = alpha_init[i] - beta[i];
        }


        if(target_error > 0) {
            /* a = beta[indices] */
            fill_entries(a, beta, indices, n_active);
            /*error = error + delta;*/
            delta = dot(b, a, n_active);
            error = error - delta + delta_prev;

            if(fabs(error) <= target_error) {
                free(beta);
                break; /*wanted error reached*/
            }

            tmp = delta_prev;
            delta_prev = delta;
            delta = tmp;
        }
         
        free(beta);
        n_active++;
    }

    /* clean up*/
    free(alpha);
    free(indices);
    free(L);
    free(a);
    free(b);

    return status;
}


double *matmat(double *mat1, double *mat2, size_t m, size_t k, size_t n)
{
    double *new, prod;
    size_t i, j, p;

    new = (double*)malloc(m*n*sizeof(double));

    for(i = 0; i < m; i++) {
        for(j = 0; j < n; j++) {
            prod = 0;
            for(p = 0; p < k; p++) {
                prod = prod + mat1[i*k + p]*mat2[p*k + j];
            }

            new[i*n + j] = prod;
        }
    }

    return new;
}

