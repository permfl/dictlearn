int _omp_cholesky_mask(double *signals, size_t mask,
                       size_t size, size_t n_signals, double *dictionary,
                       size_t n_atoms, double *alpha, size_t n_nonzero_coeffs,
                       size_t n_threads, double tolerance)
{
    /*
        Mask is of same size as signals. First column in mask
        correspond to the first signal. Each column vector in mask is
        really a size,size diagonal matrix


        1. Multiply signals[:, i] and mask[:, i] elementwise to get
           the masked signal.

        2. Multiply m = mask[:, i] with the dictionary. Ie multiply
           m[i] elementwise with row i of the dictionary to get the
           masked dictionary.

           TODO
    */
    return 0;
}