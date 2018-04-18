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

.. py:class:: VTKImage

    Numpy ndarray wrapper for VTK images. This class holds meta data 
    about the image such that reading and writing to file will keep 
    the correct attributes
        
    >>> import numpy as np
    >>> import dictlearn as dl
    >>> # volume is a numpy array
    >>> volume = VTKImage.read('path/to/volume.vti')  
    >>> assert isinstance(volume, np.ndarray)
    >>> prod = np.dot(volume[:10, :10], np.random.rand(10, 5))
    >>> assert prod.shape == (10, 5)
    >>> patches = dl.Patches(volume)
    >>> patch_generator = patches.create_batch_and_reconstruct(10000):
    >>> for batch, reconstruct in patch_generator:
    >>>     # Handle batch
    >>>     reconstruct(batch)
        
    Write 'patches.reconstructed' to disk with the
    same attributes as 'path/to/volume.vti'
        
    >>> volume.write('path/to/volume_new.vti', patches.volume)

    .. py:method:: information(self)

        Get image meta data. See VTKInformation

    .. py:staticmethod:: from_array(array, info=None)

        Create a VTKImage from a numpy array

    .. py:staticmethod:: from_image_data(image_data, name=None)

        Crate VTKImage from vtkImageData

        :param image_data:
            vtk.vtkImageData instance

        :param name:
            Name of point array to extract. Defaults to array at 
            index 0

        :return:
            VTKImage

    .. py:staticmethod:: read(path, name=None)

        Read a vti image. 

        :param path:
            Path to file

        :param name:
            Name or index of array. 
            If 'name' is None then array at index 0 is returned

        :return:
            VTKImage instance

    .. py:method:: write(self, path, array=None)

        Write data (array or self) to 'vti' file. This file is written with
        self.extent, self.origin, self.spacing and self.dtype. 
        If the instance is created with VTKImage.read() these attributes 
        are copied from the read file, otherwise the default values are used:
        
            * extent = [0, self.dimensions[0] - 1,
                        0, self.dimensions[1] - 1,
                        0, self.dimensions[2] - 1]

            * origin = [0, 0, 0]
            * spacing = [1, 1, 1]
            * dtype = np.float64


        :param path:
            Filename, where to save

        :param array:
            Optional, if array is None 'self' is written to file. 
            If array is not None then array is written to file

        :return:
            True if writing successful

    .. py:staticmethod:: write_vti(path, array, info=None, extent=None, origin=None, spacing=None, use_array_type=True, name='ImageScalars')

        Write 'array' to 'path' as vti file

        :param path:
            Where to write

        :param array:
            Data to write, ndarray with array.ndim == 3

        :param info:
            Optional instance of VTKInformation, overwrites extent, origin 
            and spacing.

        :param extent:
            Data extent, array like, len(extent) == 6. Default [0, array.shape[0] - 1,\
            0, array.shape[1] - 1, 0, array.shape[2] - 1]

        :param origin:
            Data origin, default [0, 0, 0]

        :param spacing:
            Spacing between voxels, default [1, 1, 1]

        :param use_array_type:
            Only used if info is not None. If this is False the image is 
            saved with the data type given by info, otherwise
            array.dtype is used

        :param name:
            Name of scalar array

        :return:
            True if write successful


    .. py:method:: print(self)

        Print image information

    .. py:method:: copy(self, order='C')

        Return a copy of the image

        :param order: {'C', 'F', 'A', 'K'}, optional
            Controls the memory layout of the copy. 'C' means C-order,
            'F' means F-order, 'A' means 'F' if `a` is Fortran contiguous,
            'C' otherwise. 'K' means match the layout of `a` as closely
            as possible.

.. py:class:: VTKInformation(path=None, reader=None)
    
    Holds image metadata

        * datatype, VTK datatype, int
        * bounds, bounds of the geometry, size 6
        * center, center of the geometry, size 3
        * dimensions, size of the geometry, size 6
        * extent, six integers - give the index of the first and last
                  point in each direction
        * origin, 
        * spacing, 

    :param path:
        Path to vtk image

    :param reader:
        Instance of a vtk image reader

.. function:: vtp_to_vti(surface, information, invalue=1, outvalue=0, flip=None)

    Convert a closed surface to ImageData using
    vtkPolyDataToImageStencil. All points on or inside
    the takes 'invalue' while all point outside the
    surface takes 'outvalue'

    :param surface:
        Path to surface file
    :param information:
        Information about the volume to create. Either an
        instance of VTKInformation or path to a vti file. If
        this is a path to an image, its attributes are copied
        to the converted image
    :param invalue:
        Value of points inside or of the surface
    :param outvalue:
        Value of points outside the surface.
    :param flip:
        Flip around an axis, options: 'x', 'y', 'z' or None to keep
        as is
    :return:
        An instance of VTKImage
