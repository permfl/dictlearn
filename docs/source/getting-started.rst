###############
Getting Started
###############


Overview
========
This is a module for various image processing operations using `dictionary learning`_
and sparse coding.


Features
--------

* Feature 1
* Feature 2
* Feature 3
* Feature 4
* Feature 5

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
Using Anaconda_ is strongly recommended. The `PyWavelet` package in `requirement.txt` are not listed in anaconda package
repository. Comment out this dependency with `#`, then install dependencies with `conda install`::

    $ conda install --file requirements.txt
    $ pip install PyWavelets


Then install the library with::

    $ python setup.py install


Cython not compiling on Windows
-------------------------------
Make sure you have the Microsoft C++ compiler. Download the compiler for python 2.7:
 https://www.microsoft.com/en-us/download/details.aspx?id=44266
For python 3 you need :code:`Build Tools for Visual Studio 2017` :
 https://www.visualstudio.com/downloads/#build-tools-for-visual-studio-2017

See here_ if downloading the above compiler doesn't fix the problem.


.. _here: https://github.com/cython/cython/wiki/InstallingOnWindows
.. _Anaconda: https://www.continuum.io/




