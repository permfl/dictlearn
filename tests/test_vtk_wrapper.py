import pytest

try:
    from dictlearn.vtk import VTKInformation, VTKImage
    import vtk
    skip = vtk.VTK_MAJOR_VERSION < 7
except ImportError:
    skip = True

import os
import helpers
import numpy as np


@pytest.mark.skipif(skip, reason='No vtk installed')
def test_information():
    filename = 'test_file.vti'
    path = os.path.join(helpers.get_dir(), filename)

    info1 = VTKInformation(path)
    info2 = VTKInformation(path=path)

    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(path)
    reader.Update()

    info3 = VTKInformation(reader=reader)

    assert info1 == info2
    assert info1 == info3

    with pytest.raises(IOError):
        VTKInformation('bad_path')

    with pytest.raises(ValueError):
        reader = vtk.vtkXMLImageDataReader()
        VTKInformation(reader=reader)  # No path

    with pytest.raises(ValueError):
        reader = vtk.vtkXMLImageDataReader()
        reader.SetFileName('bad_path')
        VTKInformation(reader=reader)

    assert info1.dtype == np.int32
    assert info1.datatype == vtk.VTK_INT


@pytest.mark.skipif(skip, reason='Needs VTK')
def test_numpy_to_vti():
    volume = np.arange(10*10*10, dtype=np.int32).reshape(10, 10, 10)
    here = helpers.get_dir()
    fn = os.path.join(here, 'tmp_file.vti')
    VTKImage.write_vti(fn, volume)

    f1 = open(fn, 'r')
    f2 = open(os.path.join(here, 'test_file.vti'), 'r')

    assert f1.read() == f2.read()
    f1.close()
    f2.close()

    os.remove(fn)


@pytest.mark.skipif(skip, reason='Needs VTK')
def test_numpy_from_vti():
    fn = os.path.join(helpers.get_dir(), 'test_file.vti')
    should_be = np.arange(1000, dtype=np.int32).reshape(10, 10, 10)
    content = VTKImage.read(fn)

    assert should_be.dtype == content.dtype
    assert np.array_equal(should_be.flatten(), content.flatten())


@pytest.mark.skipif(skip, reason='Needs VTK')
def test_extent():
    fn = os.path.join(helpers.get_dir(), 'tmp_file.vti')

    arr = np.arange(1000, dtype=np.float64).reshape(10, 10, 10)
    extent = (0, 9, 10, 19, 0, 9)
    origin = (0, 3, 0)
    VTKImage.write_vti(fn, arr, extent=extent, origin=origin)

    arr2 = VTKImage.read(fn)

    assert np.array_equal(arr, arr2)
    assert arr2.extent == extent
    assert arr2.origin == origin

    os.remove(fn)