###############
Getting Started
###############


Overview
========
`dictlearn` is a module for signal and image processing. This tool include easy-to-use algorithms for denoising,
inpainting, feature enhancement and detection, and image segmentation. Additionally, this tool has some methods
designed specifically for medical image processing, among these are vessel segmentation and denoising of
large 3D images.

* Multiple algorithms for dictionary learning and sparse coding
* Accelerated with Cython and C
* Built on numpy, scipy and scikit-learn


This module is a part of a masters thesis in applied mathematics, which can be read `here`__.

.. _dictionary learning: https://en.wikipedia.org/wiki/Sparse_dictionary_learning




Installation
============
Clone the repository::

    $ git clone git@github.com:permfl/dictlearn.git


Linux/OSX
---------
Install dependencies with::

    $ pip install -r requirements.txt

Make sure `scipy` and `numpy` are linked with `BLAS/lapack`. See the installation guides for
numpy_ and scipy_ for more details.

.. _numpy: https://docs.scipy.org/doc/numpy-1.10.1/user/install.html
.. _scipy: https://www.scipy.org/install.html

Then install the library with::

    $ python setup.py install


Windows
-------
Using Anaconda_ is strongly recommended. The `PyWavelet` package in `requirement.txt` are not listed in
anaconda package repository.
Comment out this dependency with `#`, then install dependencies with `conda install`::

    $ conda install --file requirements.txt
    $ pip install PyWavelets


Then install the library with::

    $ python setup.py install


Cython not compiling on Windows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Make sure you have the Microsoft C++ compiler. Download the compiler for python 2.7:
 https://www.microsoft.com/en-us/download/details.aspx?id=44266
For python 3 you need :code:`Build Tools for Visual Studio 2017` :
 https://www.visualstudio.com/downloads/#build-tools-for-visual-studio-2017

See here_ if downloading the above compiler doesn't fix the problem.

Next steps
==========
.. Make links

See the examples, or the dictionary learning tutorial


.. _here: https://github.com/cython/cython/wiki/InstallingOnWindows
.. _master: https://www.duo.uio.no/bitstream/handle/10852/63309/master-florvaag.pdf
__ master_
.. _Anaconda: https://www.continuum.io/




