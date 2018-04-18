#include <stddef.h>
#include <stdlib.h>
#include <math.h>

#include "common.h"


void forward_solve(double *lower, size_t active_size, 
                   double *b, size_t real_size)
{
    size_t k;

    b[0] = b[0]/lower[0];
    for(k = 1; k < active_size; k++) {
        b[k] = b[k] - dot(&lower[k*real_size], b, k);
        b[k] = b[k]/lower[k*real_size + k];
    }
}

void backward_solve(double *upper, size_t active_size, 
                    double *b, size_t real_size)
{
    size_t k;
    active_size--;

    b[active_size] = b[active_size]/upper[active_size*real_size + active_size];

    for(k = active_size - 1; k != -1; k--) {
        b[k] = b[k] - dot(&upper[k*real_size + k + 1], 
                          &b[k + 1], active_size - k);
        b[k] = b[k]/upper[k*real_size + k];
    }
}

/*

*/
void cholesky_solve(double *lower, size_t active_size, double *b, size_t real_size)
{
    double *upper;
    forward_solve(lower, active_size, &b[0], real_size);
    upper = transpose(lower, real_size, real_size);
    backward_solve(upper, active_size, &b[0], real_size);
    free(upper);
}


double dot(double *arr1, double *arr2, size_t n)
{
    int i;
    double prod;
    prod = 0;
    
    for(i = 0; i < n; i++) {
        prod = prod + arr1[i]*arr2[i];
    }

    return prod;
}


double *transpose(double *mat, size_t rows, size_t cols)
{
    int r, c;
    double *new;
    new = (double*)malloc(rows*cols*sizeof(double));

    for(r = 0; r < rows; r++) {
        for(c = 0; c < cols; c++) {
            new[c*rows + r] = mat[r*cols + c];
        }
    }

    return new;
}


double *mat_mat_mult(double *mat1, size_t r1, size_t c1,
                     double *mat2, size_t r2, size_t c2)
{

    int r, c;
    double *mat2_trans, *mat3;
    mat3 = (double*)malloc(r1*c2*sizeof(double));
    mat2_trans = transpose(mat2, r2, c2);

    for(r = 0; r < r1; r++) {
        for(c = 0; c < c2; c++) {
            mat3[r*c2 + c] = dot(&mat1[r*c1], &mat2[c*r2], c1);
        }
    }

    free(mat2_trans);
    return mat3;
}


void mat_vec_mult(double *mat, double *vec, size_t rows, size_t cols, 
                     double *out)
{
    int row;

    for(row = 0; row < rows; row++) {
        out[row] = dot(&mat[row*cols], vec, cols);
    }
}


size_t argmax_mat_vec(double *mat, size_t r, size_t c, double *vec)
{

    size_t k_idx_max, i, j;
    double prod, k_val_max;

    k_idx_max = 0;
    k_val_max = 0;


    for(i = 0; i < r; i++) {
        prod = 0;
        for(j = 0; j < c; j++) {
            prod = prod + mat[i*c + j]*vec[j];
        }

        prod = prod < 0?-1.0*prod: prod;

        if(prod > k_val_max) {
            k_val_max = prod;
            k_idx_max = i;
        }
    }

    return k_idx_max;
}


size_t argmax_vec(double *vec, size_t n)
{
    double max, tmp;
    size_t i, max_i;
    max_i = 0;
    max = 0;

    for(i = 0; i < n; i++) {
        tmp = vec[i];
        tmp = tmp < 0? -1.0*tmp: tmp;
        if(tmp > max) {
            max_i = i;
            max = tmp;
        }
    }

    return max_i;
}


void set_entries(double *dest, double *src, size_t *indices, size_t n)
{
    size_t i;

    for(i = 0; i < n; i++) {
        dest[indices[i]] = src[i];
    }
}


void fill_entries(double *dest, double *src, size_t *indices, size_t n)
{
    size_t i;
    for(i = 0; i < n; i++) {
        dest[i] = src[indices[i]];
    }
}


double *copy_of(double *arr, size_t n)
{
    int i;
    double *new;
    new = (double*)malloc(n*sizeof(double));

    if(new == NULL)
        return NULL;

    for(i = 0; i < n; i++)
        new[i] = arr[i];

    return new;
}


double *mat_vec_mult_indices(double* mat, size_t rows, size_t cols,
                             double *vec, size_t *indices, size_t n)
{
    size_t i, j, c;
    double *res;
    res = (double*)malloc(rows*sizeof(double));

    if(res == NULL)
        return NULL;

    for(i = 0; i < rows; i++) {
        res[i] = 0;
        for(j = 0; j < n; j++) {
            c = indices[j];
            res[i] = res[i] + mat[i*cols + c]*vec[c];
        }
    }

    return res;
}


void faster_cholesky(double *lower, size_t active_size, double *b, size_t real_size)
{
    size_t row, col;
    double prod;

    forward_solve(lower, active_size, &b[0], real_size);

    /*  
        "Backward" substitution"

        Solving this way, row index on inner loop is slower
        than normal backsub, but the time saved on not transposing
        is a lot more than the time lost here because of more
        cache misses
    */
    b[active_size - 1] = b[active_size - 1]/lower[(active_size - 1)*real_size + (active_size - 1)];

    for(col = active_size - 2; col != -1; col--) {
        prod = b[col];
        for(row = col + 1; row < active_size; row++) {
            prod = prod - lower[row*real_size + col]*b[row];
        }
        b[col] = prod/lower[col*real_size + col];
    }
}


int contains(size_t value, size_t *array, size_t n)
{
    size_t i;

    for(i = 0; i < n; i++) {
        if (array[i] == value)
            return 1;
    }

    return 0;
}