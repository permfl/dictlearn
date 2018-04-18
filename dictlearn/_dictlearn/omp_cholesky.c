#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <stdio.h>
#include <string.h>
#include <float.h>
#include "omp.h"

#include "orthogonal_mp.h"


int _omp_cholesky(double *signals, size_t size, size_t n_signals,
             double *dictionary, size_t n_atoms,
             double *dictionary_t, double *alpha,
             size_t n_nonzero_coeffs, size_t n_threads,
             double tolerance)
{

    size_t k;
    int i;
    double *res;
    
    /**/
    omp_set_num_threads((int)n_threads);

    #pragma omp parallel for private(i, k) 
    for(i = 0; i < n_signals; i++) {
        res = omp_cholesky_core(&signals[i*size], dictionary, dictionary_t,
                        size, n_atoms, n_nonzero_coeffs, tolerance);

        for(k = 0; k < n_atoms; k++)
            alpha[i*n_atoms + k] = res[k];

    } 
    return 1;
}


double *omp_cholesky_core(double *signal, double *dict, double *dict_t,
                  size_t signal_size, size_t n_atoms,
                  size_t n_nonzero_coeffs, double tolerance)
{
    size_t i, k; /* loop counters */
    size_t n_active, __size, L_size, k_max_idx;
    size_t *indices;
    double *residual, *gamma, *alpha, *L;
    double *active_dict, *atom, c, *b;

    if(tolerance > 0) {
        n_nonzero_coeffs = signal_size;
    } else {
        tolerance = 0;
    }

    residual = copy_of(signal, signal_size);
    n_active = 1;

    /* Sparse codes*/
    gamma = (double*)calloc(n_atoms, sizeof(double));
    alpha = (double*)calloc(n_atoms, sizeof(double));
    mat_vec_mult(dict_t, signal, n_atoms, signal_size, alpha);
    
    /* Index set. Think these might need to be sorted*/
    indices = (size_t*)malloc(n_nonzero_coeffs*sizeof(size_t));
    
    /* Rewrite mat_vec_mult s.t this is used istead of w */
    b = (double*)calloc(n_nonzero_coeffs, sizeof(double));
    __size = n_nonzero_coeffs*n_nonzero_coeffs;
    
    /*
        L rectangular lower triangular matrix of shape
        n_nonzero_coeffs x n_nonzero_coeffs
    */
    L = (double*)calloc(__size, sizeof(double));
    L[0] = 1.0;
    L_size = 1;


    while(1) {           
        k_max_idx = argmax_mat_vec(dict_t, n_atoms, signal_size, residual);
        /*
            Enforce orthogonality
            Stop adding coeffs in k_max_idx is already used
        */
        if(contains(k_max_idx, indices, n_active - 1))
            break;


        if(n_active > 1) {
            /*
            Matrix for columns <indices> from dictionary
              Shape: (n_active - 1) x signal_size
            */
            active_dict = (double*)malloc((n_active - 1)*signal_size*sizeof(double));
            
            for(i = 0; i < n_active - 1; i++) {
                for(k = 0; k < signal_size; k++) {
                    active_dict[i*signal_size + k] = \
                    dict_t[indices[i]*signal_size + k];
                }
            }

            /* Atom k_max_idx of dictionary*/
            atom = &dict_t[k_max_idx*signal_size];
            mat_vec_mult(active_dict, atom, n_active - 1, signal_size, b);
            forward_solve(L, L_size, b, n_nonzero_coeffs);

            c = 1 - dot(b, b, n_active - 1);

            /* 
                1.0 + DBL_EPSILON != 1.0
                Skip if c == 0 s.t we don't get a 0
                on diagonal in L
            */
            if(c < DBL_EPSILON) {
                free(active_dict);
                break;
            }

            /* Set "last" row of L equal to w*/
            for(i = 0; i <= L_size; i++) {
                L[L_size*n_nonzero_coeffs + i] = b[i];
            }
            /* Set "last" diagonal element*/
            L[(n_active - 1)*n_nonzero_coeffs + (n_active - 1)] = sqrt(c);
            L_size++;

            free(active_dict);
        }

        indices[n_active - 1] = k_max_idx;

        /* b = alpha[indices] */
        fill_entries(b, alpha, indices, n_active);
        
        /*Solve LL^Tc = [Entries <indices> of alpha]*/
        faster_cholesky(L, L_size, b, n_nonzero_coeffs);
        update_residual(dict, n_atoms, signal_size, signal, residual,
                        b, indices, n_active);

        /* gamma[indices] = b */
        set_entries(gamma, b, indices, n_active);

        if(tolerance > 0 && dot(residual, residual, signal_size) < tolerance)
            break;

        if(n_active >= n_nonzero_coeffs)
            break;

        n_active++;

    }

    free(residual);
    free(alpha);
    free(indices);
    free(L);
    free(b);

    return gamma;
}


void update_residual(double *dict, size_t n_atoms, size_t signal_size,
                     double *signal, double *residual, double *gamma,
                     size_t *indices, size_t n_active)
{
    size_t i, j, idx;

    for(i = 0; i < signal_size; i++) {
        residual[i] = signal[i];
        for(j = 0; j < n_active; j++) {
            idx = indices[j];
            residual[i] = residual[i] - dict[i*n_atoms + idx]*gamma[j];
        }
    }
}


double std(double* arr, size_t n)
{
    int i;
    double var, exp;
    exp  = 0;

    for(i = 0; i < n; i++) {
        exp = exp + arr[i];
    }

    exp = exp/n;
    var = 0;

    for(i = 0; i < n; i++) {
        var = var + (arr[i] - exp)*(arr[i] - exp);
    }

    return sqrt(var/n);
}


/***
    For easier debuggin
*/

void print_arr(double *arr, size_t size, const char *name)
{
    size_t i;
    printf("\n### %s\n", name);
    printf("[");
    for(i = 0; i < size; i++) {
        printf("%.2f, ", arr[i]);
    }
    printf("]\n");
}


void print_mat(double *mat, size_t rows, size_t cols, const char *name)
{
    size_t i, j;
    printf("\n### %s\n", name);
    for(j = 0; j < rows; j++) {
        for(i = 0; i < cols; i++) {
            printf("%.2f, ", mat[j*cols + i]);
        }
        printf("\n");
    }
    printf("\n\n");
}



