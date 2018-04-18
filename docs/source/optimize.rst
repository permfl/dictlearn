.. _optimize:

========
Optimize
========


Optimization methods for learning dictionaries.


Standard algorithms
*******************

Algorithms for training on complete data (ie. when you don't need to mask your data).
These are the algorithms needed for most use cases.


.. autofunction:: dictlearn.optimize.ksvd
.. autofunction:: dictlearn.optimize.odl
.. autofunction:: dictlearn.optimize.mod


Masked data
***********

Use these algorithms when you need to explicitly mark which data points to use
and which to discard/ignore. All masks should have the same shape as the training
data, with values [0, 1]. A data point is ignored if 0.

.. autofunction:: dictlearn.optimize.itkrmm
.. autofunction:: dictlearn.optimize.reconstruct_low_rank



References
**********

    [1] *M. Aharon, M. Elad, and A. Bruckstein, “The K-SVD: An algorithm for \
    designing of overcomplete dictionaries for sparse representation,” \
    IEEE Trans. on Signal Processing, vol. 54, no. 11, pp. 4311–4322, 2006.*

    [2] *Rubinstein, Ron, Michael Zibulevsky, and Michael Elad. \
    "Efficient implementation of the K-SVD algorithm using \
    batch orthogonal matching pursuit." Cs Technion 40.8 (2008): 1-15.*

    [3] *Engan, Kjersti, Sven Ole Aase, and J. Hakon Husoy. \
    "Method of optimal directions for frame design." Acoustics, Speech, \
    and Signal Processing, 1999. Proceedings., \
    1999 IEEE International Conference on. Vol. 5. IEEE, 1999.*

    [4] *Mairal, Julien, et al. "Online dictionary learning for sparse \
    coding." Proceedings of the 26th annual international conference \
    on machine learning. ACM, 2009.*

    [5] *Naumova, Valeriya, and Karin Schnass. "Dictionary learning from \
    incomplete data." arXiv preprint arXiv:1701.03655 (2017).*