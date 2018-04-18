from __future__ import print_function, absolute_import

import os
import vtk
import platform
import numpy as np
from vtk.util import numpy_support

if platform.uname().system.lower() == 'windows':
    # VTK warnings crashes python on windows
    vtk.vtkObject.SetGlobalWarningDisplay(False)


def _indices(extent):
    """
        Indices to data in full volume
    """
    if len(extent) == 4:
        return np.meshgrid(np.arange(extent[2], extent[3] + 1),
                           np.arange(extent[0], extent[1] + 1))

    return np.meshgrid(np.arange(extent[2], extent[3] + 1),
                       np.arange(extent[0], extent[1] + 1),
                       np.arange(extent[4], extent[5] + 1))


def dtype(i):
    """
        Map VTK type i to numpy.dtype
    """
    if i == vtk.VTK_CHAR:
        return np.int8
    elif i == vtk.VTK_UNSIGNED_CHAR:
        return np.uint8
    elif i == vtk.VTK_SHORT:
        return np.int16
    elif i == vtk.VTK_UNSIGNED_SHORT:
        return np.uint16
    elif i == vtk.VTK_INT:
        return np.int32
    elif i == vtk.VTK_UNSIGNED_INT:
        return np.uint32
    elif i == vtk.VTK_LONG:
        return np.int64
    elif i == vtk.VTK_UNSIGNED_LONG:
        return np.uint64
    elif i == vtk.VTK_FLOAT:
        return np.float32
    elif i == vtk.VTK_DOUBLE:
        return np.float64
    elif i == vtk.VTK_LONG_LONG:
        return np.int128 if hasattr(np, 'int128') else np.int64
    elif hasattr(np, 'uint128') and i == vtk.VTK_UNSIGNED_LONG_LONG:
        return np.uint128
    else:
        raise ValueError('Cannot understand type {}'.format(i))


def vtk_type(i):
    """
        Map numpy.dtype i to VTK type
    """
    if i == np.int8:
        return vtk.VTK_CHAR
    elif i == np.uint8:
        return vtk.VTK_UNSIGNED_CHAR
    elif i == np.int16:
        return vtk.VTK_SHORT
    elif i == np.uint16:
        return vtk.VTK_UNSIGNED_SHORT
    elif i == np.int32:
        return vtk.VTK_INT
    elif i == np.uint32:
        return vtk.VTK_UNSIGNED_INT
    elif i == np.int64:
        return vtk.VTK_LONG
    elif i == np.uint64:
        return vtk.VTK_UNSIGNED_LONG
    elif i == np.float32:
        return vtk.VTK_FLOAT
    elif i == np.float64:
        return vtk.VTK_DOUBLE
    elif hasattr(np, 'int128') and i == np.int128:
        return vtk.VTK_LONG_LONG
    elif hasattr(np, 'uint128') and i == np.uint128:
        return vtk.VTK_UNSIGNED_LONG_LONG
    else:
        raise ValueError('Cannot understand dtype {}'.format(i))


def create(array):
    """
        Return numpy array 'arr' as VTKImage array
    """
    return array.view(VTKImage)


class VTKInformation(dict):
    """
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
    """
    def __init__(self, path=None, reader=None):
        self.path = path

        if self.path is None and reader is None:
            return

        _del_reader = False

        if reader is None:
            if not isinstance(path, str):
                raise ValueError('path has to be of type string, not {}'
                                 .format(type(path)))

            if not os.path.exists(self.path):
                raise IOError("No such file or directory: '{}'"
                              .format(self.path))

            reader = vtk.vtkXMLImageDataReader()
            reader.SetFileName(self.path)
            _del_reader = True
        elif isinstance(reader, vtk.vtkXMLImageDataReader):
            self.path = reader.GetFileName()

            if self.path is None or not os.path.exists(self.path):
                raise ValueError('vtkXMLImageDataReader improperly configured.'
                                 ' Image path is empty or does not exist')
        else:
            if not hasattr(reader, 'Update') and not hasattr(reader, 'GetOutput'):
                if hasattr(reader, 'GetClassName'):
                    name = reader.GetClassName()
                else:
                    name = type(reader)

                raise ValueError('Unsupported reader, {}'.format(name))

        reader.Update()
        output = reader.GetOutput()

        self.datatype = output.GetPointData().GetArray(0).GetDataType()
        self.bounds = output.GetBounds()
        self.center = output.GetCenter()
        self.dimensions = output.GetDimensions()
        self.extent = output.GetExtent()
        self.origin = output.GetOrigin()
        self.spacing = output.GetSpacing()

        if _del_reader:
            del reader
            del output

        super(VTKInformation, self).__init__(self, extent=self.extent,
                                             origin=self.origin,
                                             spacing=self.spacing)

    @property
    def dtype(self):
        return dtype(self.datatype)


    def __repr__(self):
        return str(dict(
            datatype=self.datatype,
            dtype=dtype(self.datatype),
            bounds=self.bounds,
            center=self.center,
            dimensions=self.dimensions,
            extent=self.extent,
            origin=self.origin,
            spacing=self.spacing,
        ))


class VTKImage(np.ndarray):
    """
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
    """
    ORIGIN = np.array([0, 0, 0], np.float32)
    SPACING = np.array([1, 1, 1], np.float32)

    # todo pickle support for faster to/from python

    @property
    def dimensions(self):
        return (self.extent[1] - self.extent[0] + 1,
                self.extent[3] - self.extent[2] + 1,
                self.extent[5] - self.extent[4] + 1)

    def information(self):
        """
            Get image meta data. See VTKInformation
        """
        if not hasattr(self, '_info'):
            self._info = VTKInformation()
            self._info.datatype = self.datatype
            self._info.bounds = self.bounds
            self._info.center = self.center
            self._info.dimensions = self.dimensions
            self._info.extent = self.extent
            self._info.origin = self.origin
            self._info.spacing = self.spacing

        return self._info

    @property
    def info(self):
        return self.information()

    @staticmethod
    def from_array(array, info=None):
        """
            Create a VTKImage from a numpy array
        """

        if array.ndim != 3:
            raise ValueError('Need array.ndim == 3, not {}'.format(array.ndim))

        if not isinstance(array, np.ndarray):
            array = np.asarray(array)

        obj = array.view(VTKImage)

        if info is None:
            obj.origin = (0, 0, 0)
            obj.spacing = (1, 1, 1)
            x, y, z = array.shape
            obj.extent = (0, x - 1, 0, y - 1, 0, z - 1)
            obj.bounds = obj.extent
            obj.center = np.array(obj.extent)[1::2] / 2.0
            obj.datatype = vtk_type(array.dtype)
        else:
            obj.origin = info.origin
            obj.spacing = info.spacing
            obj.extent = info.extent
            obj.bounds = info.bounds
            obj.center = info.center
            obj.datatype = info.datatype

        return obj

    @staticmethod
    def from_image_data(image_data, name=None):
        """
            Crate VTKImage from vtkImageData

        :param image_data:
            vtk.vtkImageData instance

        :param name:
            Name of point array to extract. Defaults to array at 
            index 0

        :return:
            VTKImage
        """

        if not isinstance(image_data, vtk.vtkImageData):
            raise ValueError("'image_data' is not vtk.vtkImageData, but {}"
                             .format(str(image_data)))

        bounds = image_data.GetBounds()
        center = image_data.GetCenter()
        dimensions = image_data.GetDimensions()
        extent = image_data.GetExtent()
        origin = image_data.GetOrigin()
        spacing = image_data.GetSpacing()

        array = image_data.GetPointData().GetArray(0 if name is None else name)
        datatype = array.GetDataType()

        array = numpy_support.vtk_to_numpy(array)
        array = array.reshape(*dimensions, order='F')
        obj = array.view(VTKImage)
        obj.bounds = bounds
        obj.center = center
        obj.extent = extent
        obj.origin = origin
        obj.spacing = spacing
        obj.datatype = datatype

        return obj

    @staticmethod
    def from_dicom(filename):
        raise NotImplementedError()

    @staticmethod
    def read(path, name=None):
        """
        Read a vti image. 

        :param path:
            Path to file

        :param name:
            Name or index of array. 
            If 'name' is None then array at index 0 is returned

        :return:
            VTKImage instance
        """
        if not os.path.exists(path):
            raise IOError("No such file or directory: '{}'"
                          .format(path))

        reader = vtk.vtkXMLImageDataReader()
        reader.SetFileName(path)
        reader.Update()

        output = reader.GetOutput()
        obj = VTKImage.from_image_data(output, name)
        obj.info.path = path
        del output
        del reader
        return obj

    def write(self, path, array=None):
        """
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
        """
        if array is not None:
            if array.shape != self.dimensions:
                raise ValueError(
                    'Incompatible dimensions read image {} and array {}'
                    .format(self.dimensions, array.shape)
                )

        image = self if array is None else array

        return VTKImage.write_vti(path, image, self.extent,
                                  self.origin, self.spacing)

    @staticmethod
    def write_vti(path, array, info=None, extent=None, origin=None, 
                  spacing=None, use_array_type=True, name='ImageScalars'):
        """
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
        """
        dimensions = array.shape
        datatype = vtk_type(array.dtype)

        if info is not None:
            extent = info.extent
            origin = info.origin
            spacing = info.spacing

            if not use_array_type:
                datatype = info.datatype
        else:
            if extent is None:
                extent = np.zeros(2 * array.ndim, np.int32)
                extent[1::2] = np.asarray(dimensions, np.int32) - 1

            if origin is None:
                origin = VTKImage.ORIGIN

            if spacing is None:
                spacing = VTKImage.SPACING

        importer = vtk.vtkImageImport()
        importer.SetDataScalarType(datatype)
        importer.SetNumberOfScalarComponents(1)
        importer.SetDataExtent(extent)
        importer.SetWholeExtent(extent)
        importer.SetDataOrigin(origin)
        importer.SetDataSpacing(spacing)
        importer.SetScalarArrayName(name)

        array = array.astype(dtype(datatype))
        size = len(array.flat)*array.dtype.itemsize
        array = array.flatten('F')
        vtk_array = numpy_support.numpy_to_vtk(array, 0, datatype)
        # Set pointer to image scalars
        importer.CopyImportVoidPointer(vtk_array.GetVoidPointer(0), size)
        importer.Update()
        data = importer.GetOutput()

        writer = vtk.vtkXMLImageDataWriter()
        writer.SetFileName(path)
        writer.SetInputData(data)
        return bool(writer.Write())

    def print(self):
        """
            Print image information
        """
        string = 'VTKImage:\n  dimensions {}\n'.format(self.dimensions)
        string += '  bounds {}\n'.format(self.bounds)
        string += '  center {}\n'.format(self.center)
        string += '  extent {}\n'.format(self.extent)
        string += '  origin {}\n'.format(self.origin)
        string += '  spacing {}\n'.format(self.spacing)
        print(string)

    def copy(self, order='C'):
        """
        Return a copy of the image

        :param order: {'C', 'F', 'A', 'K'}, optional
            Controls the memory layout of the copy. 'C' means C-order,
            'F' means F-order, 'A' means 'F' if `a` is Fortran contiguous,
            'C' otherwise. 'K' means match the layout of `a` as closely
            as possible.
        """
        res = np.ndarray.copy(self, order)
        try:
            res.bounds = self.bounds
            res.center = self.center
            res.extent = self.extent
            res.origin = self.origin
            res.spacing = self.spacing
            res.datatype = self.datatype
        except AttributeError:
            # Trying to copy a slice of this array
            # Don't need to keep track of the stuff above.
            # Ex: self[12].copy()
            # Implement __getitem__ to fix
            pass

        return res


def vtp_to_vti(surface, information, invalue=1, outvalue=0, flip=None):
    """
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
    """

    if not isinstance(surface, str) or not os.path.isfile(surface):
        raise IOError("No such file or directory: '{}'".format(surface))

    if not isinstance(information, VTKInformation):
        if os.path.isfile(information):
            information = VTKInformation(information)
        else:
            raise IOError("No such file or directory: '{}'".format(surface))

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(surface)
    reader.Update()
    polydata = reader.GetOutput()
    del reader

    # Foreground, takes value invalue on all point
    # inside or on the surface
    image_data = vtk.vtkImageData()
    image_data.SetSpacing(information.spacing)
    image_data.SetOrigin(information.origin)
    image_data.SetDimensions(information.dimensions)
    image_data.SetExtent(information.extent)
    image_data.AllocateScalars(information.datatype, 1)

    num_points = image_data.GetNumberOfPoints()

    for i in range(num_points):
        image_data.GetPointData().GetScalars().SetTuple1(i, invalue)

    # Surface stencil
    stencil = vtk.vtkPolyDataToImageStencil()
    stencil.SetInputData(polydata)
    stencil.SetOutputOrigin(information.origin)
    stencil.SetOutputSpacing(information.spacing)
    stencil.SetOutputWholeExtent(image_data.GetExtent())
    stencil.Update()

    # Image stencil. All points not in stencil
    # gets value outvalue
    img_stencil = vtk.vtkImageStencil()
    img_stencil.SetInputData(image_data)
    img_stencil.SetStencilConnection(stencil.GetOutputPort())
    img_stencil.ReverseStencilOff()
    img_stencil.SetBackgroundValue(outvalue)
    img_stencil.Update()

    image = img_stencil.GetOutput()

    if flip is not None:
        axes = {'x': 0, 'y': 1, 'z': 2}
        flipper = vtk.vtkImageFlip()
        flipper.SetFilteredAxis(axes[flip])
        flipper.SetInputData(image)
        flipper.Update()
        image = flipper.GetOutput()

    return VTKImage.from_image_data(image)


def as_image_data(volume, info=None):
    """
        Convert a numpy array 'volume' to vtk.vtkImageData
    """
    if info is None and not isinstance(volume, VTKImage):
        raise ValueError('Need info instance of VTKInformation or '
                         'volume instance of VTKImage')

    if info is None:
        info = volume.information()

    # TODO: Fix seg fault when extent is not (0, x, 0, y, 0, z)
    # it's an issue with python's gc. The allocated memory
    # for importer.GetOutput() is freed when vtk_array goes
    # out of scope
    importer = vtk.vtkImageImport()
    importer.SetDataScalarType(info.datatype)
    importer.SetNumberOfScalarComponents(1)
    importer.SetDataExtent(info.extent)
    importer.SetWholeExtent(info.extent)
    importer.SetDataOrigin(info.origin)
    importer.SetDataSpacing(info.spacing)

    size = len(volume.flat)*volume.dtype.itemsize
    array = volume.flatten('F')
    vtk_array = numpy_support.numpy_to_vtk(array, 1, info.datatype)
    # Set pointer to image scalars
    importer.CopyImportVoidPointer(vtk_array.GetVoidPointer(0), size)
    importer.Update()
    return vtk_array, importer.GetOutput()


def marching_cubes(image_data, contours, connectivity=False, path=None):
    """
    """
    if isinstance(image_data, VTKImage):
        info = image_data.information()
        importer = vtk.vtkImageImport()
        importer.SetDataScalarType(info.datatype)
        importer.SetNumberOfScalarComponents(1)
        importer.SetDataExtent(info.extent)
        importer.SetWholeExtent(info.extent)
        importer.SetDataOrigin(info.origin)
        importer.SetDataSpacing(info.spacing)

        size = len(image_data.flat) * image_data.dtype.itemsize
        array = image_data.flatten('F')
        vtk_array = numpy_support.numpy_to_vtk(array, 1, info.datatype)
        importer.CopyImportVoidPointer(vtk_array.GetVoidPointer(0), size)
        importer.Update()
        image_data = importer.GetOutput()

    if isinstance(contours, (int, float)):
        contours = (contours, )

    contour = vtk.vtkMarchingContourFilter()
    contour.SetInputData(image_data)
    
    for i, value in enumerate(contours):
        contour.SetValue(0, value)

    contour.Update()
    poly_data = contour.GetOutput()

    if connectivity:
        connectivity = vtk.vtkConnectivityFilter()
        connectivity.SetInputData(poly_data)
        connectivity.SetExtractionModeToLargestRegion()
        connectivity.Update()

        mapper = vtk.vtkGeometryFilter()
        mapper.SetInputData(connectivity.GetOutput())
        mapper.Update()
        poly_data = mapper.GetOutput()

    if isinstance(path, str):
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(path)
        writer.SetInputData(poly_data)
        writer.Write()

    return poly_data
