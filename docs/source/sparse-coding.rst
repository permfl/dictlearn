Sparse Coding
=============

The goal of these algorithms is to find a sparse coefficient matrix (or vector) for some
signals given a dictionary of signal features.



The two norms are defined as:

    :math:`\left \| \mathbf{x} \right \|_0 = \#\{\mathbf{x}_i \neq 0: \forall \mathbf{x}_i \in \mathbf{x}\}`

    :math:`\left \| \mathbf{x} \right \|_1 = \sum_i |x_i|`


Thus the two problem formulations are:

.. math::

    \hat{a} = \underset{a}{argmin} \quad \frac{1}{2} \left \| x - Da \right \| _F^2 + \lambda \left \| a \right \|_0


.. math::

    \hat{a} = \underset{a}{argmin} \quad \frac{1}{2} \left \| x - Da \right \| _F^2 + \lambda \left \| a \right \|_1


With signal :math:`x`, dictionary :math:`D`, and :math:`\hat{a}` the sparse coefficients


:math:`\ell_0`-regularization
*****************************

.. autofunction:: dictlearn.sparse.omp_batch
.. autofunction:: dictlearn.sparse.omp_cholesky
.. autofunction:: dictlearn.sparse.omp_mask
.. autofunction:: dictlearn.sparse.iterative_hard_thresholding


:math:`\ell_1`-regularization
*****************************

.. autofunction:: dictlearn.sparse.lars
.. autofunction:: dictlearn.sparse.lasso
.. autofunction:: dictlearn.sparse.iterative_soft_thresholding
.. autofunction:: dictlearn.sparse.l1_ball


References
**********
[1] Rubinstein, Ron, Michael Zibulevsky, and Michael Elad. \
"Efficient implementation of the K-SVD algorithm using batch \
orthogonal matching pursuit." Cs Technion 40.8 (2008): 1-15.