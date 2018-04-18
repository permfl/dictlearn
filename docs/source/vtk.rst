================
Working with VTK
================


The Visualization Toolkit (VTK) is an open-source, freely available software system for
3D computer graphics, image processing, and visualization. It consists of a C++ class
library and several interpreted interface layers including Tcl/Tk, Java, and Python.
VTK supports a wide variety of visualization algorithms including scalar, vector, tensor,
texture, and volumetric methods, as well as advanced modeling techniques such as
implicit modeling, polygon reduction, mesh smoothing, cutting, contouring, and Delaunay
triangulation. VTK has an extensive information visualization framework and a suite of
3D interaction widgets. The toolkit supports parallel processing and integrates with
various databases on GUI toolkits such as Qt and Tk. VTK is cross-platform and runs on
Linux, Windows, Mac, and Unix platforms. VTK is part of Kitware’s collection of
commercially supported open-source platforms for software development.


Installing
----------


VTK version 8.0.0 and later is available on PyPi for python versions 2.7, 3.4, 3.5 and 3.6

.. code-block:: none

    $ pip install vtk


On Windows only python 3.5 and 3.6 are supported. VTK can also be installed with
anaconda. Available versions are at https://anaconda.org/conda-forge/vtk/files.


If you want to build VTK yourself, download it from https://www.vtk.org.
When building toggle the flag `VTK_WRAP_PYTHON` to generate the wrapping files.
Detailed instructions can bee seen
`here <http://ghoshbishakh.github.io/blog/blogpost/2016/07/13/building-vtk-with-python3-wrappers.html>`_


Wrappers
""""""""

Reading and writing VTK image files to and from numpy array requires a lot of
boilerplate code. The classes below, :code:`VTKImage` and :code:`VTKInformation` wraps
reading and writing :code:`vti` files, ie images of type :code:`vtkImageData`. To read an
image, write: :code:`image = VTKImage.read('path.vti')`. If :code:`image` will be modified
you have to save its attributes: :code:`info = image.information()`.

:code:`VTKImage` is a subclass of :code:`numpy.ndarray` and can be used as any normal numpy array.

Finally the image can be written to disk

.. code-block:: python

    VTKImage.write_vti('new_path.vti', image, info)


.. autoclass:: dictlearn.vtk.VTKImage
    :members:

.. autoclass:: dictlearn.vtk.VTKInformation
    :members:

.. autofunction:: dictlearn.vtk.vtp_to_vti
