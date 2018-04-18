#include <stddef.h>


/*
    Forward substitution solver
    ***************************

    Solver lower triangular system Ax = b, where A is rectangular lower
    triangular of size (active_size * active_size). real_size is the size of
    the allocated memory. As this is used for solving the Cholesky system LL^T
    in _omp_core the L matrix is always of size real_size*real_size, but we
    want to solve the leading principal minor of size active_size*active_size

    Example:
        If active_size = 2 and real_size = 4, and
        L = [
            [1, 0, 0, 0],
            [1, 2, 0, 0],
            [1, 2, 3, 0],
            [1, 2, 3, 4]
        ]
        and
        b = [1, 2, 3, 4]

        Then we'll solve the equation Ax = y where
    
        A = [[1, 0],
            [1, 2]]
        
        and b = [1, 2]


    
    Args
    ----
        lower: Lower triangular active_size*active_size matrix
        active_size: Size of leading principal minor to solve
        b, IN/OUT: RHS in equation of length active_size. 
                   Overwritten with solution
        real_size: Size of underlying array


*/
void forward_solve(double *lower, size_t active_size, 
                   double *b, size_t real_size);


/*
    Backward substitution solver
    ****************************
    
    Solver upper triangular system Ax = b, where A is rectangular upper
    triangular of size (active_size * active_size). Same as forward_solve, 
    but with upper triag matrix. See forward_solve for more details


    Args
    ----
        upper: Upper triangular active_size*active_size matrix
        active_size: Size of leading principal minor to solve
        b, IN/OUT: RHS in equation of length active_size. 
                   Overwritten with solution
        real_size: Size of underlying array

*/
void backward_solve(double *upper, size_t active_size, 
                    double *b, size_t real_size);


/*
    Cholesky-Solver?
    ****************

    Solves the linear system LL^Tx = b, where L is rectangular, 
    lower triangular with ones on the diagonal. Solver first Lz = b with
    forward_solver, then L^Tx = z with backward_solver


    Args
    ----
        lower: Rectangular, lower triangular matrix
        active_size: Size of lower matrix
        b IN/OUT: RHS in equation, is overwritten with the solution
        real_size: Size underlying array
*/
void cholesky_solve(double *lower, size_t active_size, double *b, size_t real_size);


/*
    Argmax <row of mat, vec>

    Finds k such that the absolute value of the inner product
    of row k in mat and vec is bigger than inner product with 
    all other rows


    _omp_core calls this with the transposed dictionary which reduces
    cache misses. Transposed dictionary has the entries in the atoms
    close to each other in memory


    Args
    ----
        mat: Matrix of size (r*c)
        r: Number of rows in mat
        c: Number of columns in mat and entries in vec
        vec: Vector of size c


    Returns
    -------
        Index of row in mat with biggest inner prod with vec

*/
size_t argmax_mat_vec(double *mat, size_t r, size_t c, double *vec);


/*
    Argmax vector
    *************

    Find index of entry with bigger absolute value in a vector


    Args
    ----
        vec: vector
        n: length of vector


    Returns
    -------
        Index to biggest entry
*/
size_t argmax_vec(double *vec, size_t n);


/*
    Transpose matrix
    ****************

    Args
    ----
        mat: Matrix to transpose
        rows: Rows of mat
        cols: Columns of mat

    Returns
    -------
        Transposed copy of mat, shape: (cols, rows)

*/
double *transpose(double *mat, size_t rows, size_t cols);


/*
    Dot-product
    ***********

    Simplest dot product implementation

    Args
    ----
        arr1: Array of length n
        arr2: Array of length n
        n: Length of arrays


    Returns
    -------
        Dot product or the two arrays
*/
double dot(double *arr1, double *arr2, size_t n);


/*
    Matrix Vector Product
    *********************

    Simplest mat-vec product. Loop through rows in matrix and take inner
    prod with vector.


    Args
    ----
        mat: Matrix, shape (rows, cols)
        vec: Vector, shape (cols, )
        rows: Rows in matrix, and solution
        cols: Columns in matrix, and vector

    Returns
    -------
        Vector, shape (rows, )
*/
void mat_vec_mult(double *mat, double *vec, size_t rows, size_t cols, 
                   double *out);


/*
    Matrix Matrix Product
    *********************
    
    Multiplies matrices A*B
    O(n^3) implementation, O(r1*c1*c2) in this case


    Args
    ----
        mat1: Matrix, shape (r1, c1)
        r1: Rows mat1
        c1: Cols mat1
        mat2: Matrix, shape (c1 = r2, c2)
        r2: Rows mat2, r2 == c1
        c2: Columns mat2

    Returns
    -------
        matrix product, shape (r1, c2)
*/
double *mat_mat_mult(double *mat1, size_t r1, size_t c1,
                     double *mat2, size_t r2, size_t c2);



/*
    Take elements in src array and copies to dest at places specified by 
    indices. I.e dest[indices[i]] = src[i] for all i

    Equivalent to numpy/python:
        dest[indices] = smaller


    Args
    ----
        dest OUT: Copy elements to this. Atleast as big sa biggest entry in
                  indices
        src: Copy elements from this array, size atleast n
        indices: Indices for where to put entries in src in dest
        n: Number of entries in indices
*/
void set_entries(double *dest, double *src, size_t *indices, size_t n);


/*
    Takes elements in src array specified by index set indices and copies
    to dest array. I.e dest[i] = src[indices[i]]
    
    Equivalent to numpy/python:
        dest = src[indices]


    Args
    ----
        dest: Copy to this array. Size atleast n
        src: Copy elements from here. Size atleast as big as biggest
             entry in indices
        indices: Index set
        n: size of index seg
*/
void fill_entries(double *dest, double *src, size_t *indices, size_t n);


/*
    Returns a copy of array arr

    Args
    ----
        arr: Array to copy
        n: Size of array

    Returns
    -------
        A copy of arr
*/
double *copy_of(double *arr, size_t n);


/*
    
    Multiply columns "indices" in a matrix with elements 
    "indices" in a vector
    Equivalent to np.dot(mat[:, indices], vec[indices])

    Args
    ----
        mat: Matrix (rows, cols)
        rows: Num rows in mat and length of output
        cols: Columnas in mat and length of vector
        indices: index set
        n: Length of index set

    Returns
    -------
        product of matric and vector, length rows
*/
double *mat_vec_mult_indices(double* mat, size_t rows, size_t cols,
                             double *vec, size_t *indices, size_t n);


/*
    Variation of cholesky_solve. Solves LL'x = b for a lower triangular matrix L
    without transposing. See lu_solve for details
*/
void faster_cholesky(double *lower, size_t active_size, double *b, size_t real_size);


/*
    Check if a value is contained in an array

    Args
    ----
        value: Value to look for
        array:
        n: Length of array

    Returns
    -------
        1 if array contains value, 0 otherwise
*/
int contains(size_t value, size_t *array, size_t n);