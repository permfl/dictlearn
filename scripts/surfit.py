from __future__ import print_function
import argparse
import os
import sys
import glob
import math
import time
import operator
import textwrap
from functools import partial

try:
    import cPickle as pickle
except ImportError:
    # py3
    import pickle
    from functools import reduce

import vtk
import itk
import yaml
import numpy as np
from scipy import signal
from skimage import morphology as _m

import dictlearn as dl
from dictlearn import vtk as dl_vtk


"""
Surfit
======


Create a surface image from CT volume

This script consists of three main steps:

    1. Denoise
    2. Feature Enhancement
    3. Surface Generation


1. Denoise
**********
TODO

2. Feature Enahncement
**********************
TODO


3. Surface Generation
*********************
TODO





"""

START_BEGINNING = 0
START_DENOISE = 1
START_ENHANCE = 2

vtk.vtkObject.SetGlobalWarningDisplay(0)

class ParsePatchSize(argparse.Action):
    def __init__(self, option_strings, dest, nargs='+', **kwargs):
        super(ParsePatchSize, self).__init__(option_strings, dest, nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        if len(values) not in [1, 3]:
            msg = 'Patch size cannot be "{}".'.format(' '.join(values))
            msg += ' Allowed values are "X" for 2D patches and "X Y Z" for 3D'
            raise SystemExit(msg)

        if len(values) == 1:
            setattr(namespace, self.dest, int(values[0]))
        else:
            values = [int(val) for val in values]
            setattr(namespace, self.dest, values)


class Config:
    def __init__(self, **kwargs):
        self.alpha = kwargs.get('alpha', 0.2)
        self.axis = kwargs.get('axis', 'x')
        self.batch_size = kwargs.get('batch_size', 5000)
        self.dictionary = kwargs.get('dictionary')
        self.iters = kwargs.get('iters', 10)
        self.method = kwargs.get('method', 'online')
        self.n_nonzero = kwargs.get('n_nonzero', 0)
        self.n_threads = kwargs.get('n_threads', 1)
        self.patch_size = kwargs.get('patch_size', 8)
        self.n_atoms = kwargs.get('n_atoms', self.default_atoms())
        self.sigma = kwargs.get('sigma', -1)
        self.stride = kwargs.get('stride', 1)
        self.threshold = kwargs.get('threshold')
        self.sharpen = kwargs.get('sharpen', True)
        self.verbose = kwargs.get('verbose', True)
        self.vessel_size = kwargs.get('vessel_size', 1)
        self.filter_size = kwargs.get('filter_size', 3)
        self.scale_min = kwargs.get('min_scale', 1)
        self.scale_max = kwargs.get('max_scale', 5)
        self.n_scales = kwargs.get('n_scales', 5)
        self.level = kwargs.get('level', 0)
        
    def save(self, path):
        path = path + '_config.pkl'
        with open(path, 'wb') as f:
            pickle.dump(self.__dict__, f)

    @staticmethod
    def restore(path):
        path = path + '_config.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as f:
                return Config(**pickle.load(f))

        return Config()

    def set(self, **kwargs):
        for key, value in kwargs.items():
            if value is None:
                continue 

            if hasattr(self, key):
                setattr(self, key, value)

    def default_atoms(self):
        if isinstance(self.patch_size, int):
            return 2*self.patch_size*self.patch_size
        else:
            return 2*reduce(operator.mul, self.patch_size)


class SurfitBase(object):
    def __init__(self, config):
        self.config = config
        
        with open(self.config, 'r') as f:
            self.data = yaml.load(f)

        self.configured = False
        self._method = ''
        self.input = None
        self.output = None
        self.inputs = []
        self.outputs = []
        self.params = Config()

    def denoise(self, **kwargs):
        print('denoise')
        volume, info = read_volume(self.input)
        patch_size = self._param('patch_size', kwargs)
        stride = self._param('stride', kwargs)
        axis = self._param('axis', kwargs)
        batch_size = self._param('batch_size', kwargs)
        dictionary = self._param('dictionary', kwargs)
        n_atoms = self._param('n_atoms', kwargs)
        iters = self._param('iters', kwargs)
        n_nonzero = self._param('n_nonzero', kwargs)
        sharpen = self._param('sharpen', kwargs)
        n_threads = self._param('n_threads', kwargs)
        method = self._param('method', kwargs)
        sigma = self._param('sigma', kwargs)
        verbose = self._param('verbose', kwargs)

        if verbose:
            self.print_params('Denoise', **{
                'Volume shape': volume.shape,
                'Patch Size': patch_size,
                'Stride': stride,
                'Axis': axis,
                'Dictionary': dictionary,
                'Batch Size': batch_size,
                'Number of atoms': n_atoms,
                'Iterations': iters,
                'Number of nonzero coeffs': n_nonzero,
                'Number of threads': n_threads,
                'Training method': method,
                'Sigma (noise)': sigma
             })

        volume = dl.normalize(volume)

        if sharpen:
            volume = laplacian_sharpening(volume)

        if isinstance(patch_size, int) and isinstance(stride, int):
            volume, dictionary = denoise_2d(
                volume, patch_size, axis, dictionary, n_atoms,
                iters, n_nonzero, n_threads, sigma, verbose
            )
        else:
            volume, dictionary = denoise_3d(
                volume, patch_size, stride, batch_size, dictionary, n_atoms, 
                iters, n_nonzero, n_threads, method, sigma, verbose
            )

        fn = self._get_filename('dictionary', '.npy')
        self.write(dictionary, fn=fn)
        return self.write(volume, info)

    def inpaint(self, **kwargs):
        volume, info = read_volume(self.input)
        mask_path = self._param('mask', kwargs)
        if mask_path is not None:
            mask = read_volume(mask_path).astype(bool)
        else:
            raise ValueError('Missing parameter mask for inpainting')

        patch_size = self._param('patch_size', kwargs)
        stride = self._param('stride', kwargs)
        axis = self._param('axis', kwargs)
        batch_size = self._param('batch_size', kwargs)
        dictionary = self._param('dictionary', kwargs)
        n_atoms = self._param('n_atoms', kwargs)
        iters = self._param('iters', kwargs)
        n_nonzero = self._param('n_nonzero', kwargs)
        n_threads = self._param('n_threads', kwargs)
        method = self._param('method', 'odl')
        sigma = self._param('sigma', kwargs)
        verbose = self._param('verbose', kwargs)

        if isinstance(batch_size, int) and isinstance(stride, int):
            print('  inpaint 2d')
            volume, dictionary = inpaint_2d(
                volume, mask, patch_size, axis, dictionary, n_atoms,
                iters, n_nonzero, n_threads, sigma, verbose
            )
        else:
            print('  inpaint 3d')
            volume, dictionary = inpaint_3d(
                volume, mask, patch_size, stride, batch_size, dictionary, n_atoms, 
                iters, n_nonzero, n_threads, method, sigma, verbose
            )

        return self.write(volume, info)

    def kmeans_enhance(self, **kwargs):
        """
            Implementation of <- hessian_enhance>
        """
        volume, info = read_volume(self.input)
        vessel_size = self._param('vessel_size', kwargs)
        alpha = self._param('alpha', kwargs)
        axis = self._param('axis', kwargs)
        volume = kmeans_enhance(volume, vessel_size, alpha, axis)
        return self.write(volume, info)

    def hessian_enhance(self, **kwargs):
        """
            Implementation of <- hessian_enhance>
        """
        volume, info = read_volume(self.input)
        filter_size = self._param('filter_size', kwargs)
        scale_min = self._param('scale_min', kwargs)
        scale_max = self._param('scale_max', kwargs)
        n_scales = self._param('scale_max', kwargs)

        volume, mean, cands = hessian_enhance(volume, filter_size, scale_min,
                                              scale_max, n_scales)
        fn = self._get_filename('hessian_mean')
        dl_vtk.VTKImage.write_vti(fn, mean, info)

        fn = self._get_filename('hessian_candidates')
        dl_vtk.VTKImage.write_vti(fn, cands, info)
        return self.write(volume, info)

    def create_seed(self, **kwargs):
        """
            Implementation of <- create_seed>
        """
        volume, info = read_volume(self.input)
        size = self._param('filter_size', kwargs)
        sigma = self._param('vessel_size', kwargs)
        level = self._param('level', kwargs, 0.1)
        verbose = self._param('verbose', kwargs)
        shrink = self._param('shrink', kwargs, 1)

        if verbose:
            self.print_params('Create Seed', **{
                'Filter Size': size,
                'Vessel size': sigma,
                'Level': level,
                'Shrink': shrink

             })
        seed, preselection = seed_region_growing(volume, info, size, sigma, 
                                                 level, shrink)
        # Shrink seed

        fn = self._get_filename('preselection')
        self.write(preselection, info, fn)

        return self.write(seed, info)

    def active_contours(self, **kwargs):
        """
            Implementation of <- active_contour>
        """
        seed_image, info = read_volume(self.input)
        feature_image = kwargs.get('feature_image')
        verbose = self._param('verbose', kwargs)

        if feature_image is None:
            source = self.first
            sigma = kwargs.get('sigma', 1)
            normalize = kwargs.get('normalize', False)
            remap = kwargs.get('remap', True)

            if verbose:
                self.print_params('Creating default feature image', **{
                    'Scales': sigma,
                    'Normalize across scales': normalize,
                    'Remap': remap
                })

            feature_image = potential(read_volume(source)[0], sigma, 
                                      normalize, remap, True)
        elif isinstance(feature_image, dict):
            source = self._param('input', feature_image, self.first)
            sigma = self._param('sigma', feature_image, 1)
            normalize = self._param('normalize', feature_image, -1)
            
            if normalize == -1:
                normalize = hasattr(sigma, '__iter__')

            remap = self._param('remap', feature_image, True)
            
            if verbose:
                self.print_params('Creating feature image', **{
                    'Input': source,
                    'Scales': sigma,
                    'Normalize across scales': normalize,
                    'Remap': remap
                })

            feature_image = potential(read_volume(source)[0], sigma,
                                      normalize, remap, True)

        else:
            if verbose:
                self.print_params('Reading feature image', **{
                    'Path': feature_image
                })

            feature_image, _ = read_volume(feature_image)

        fn = self._get_filename('feature')
        self.write(feature_image, info, fn)

        isosurface_value = kwargs.get('isosurface_value', 0.5)
        advection_scaling = kwargs.get('advection_scaling', 1.0)
        curvature_scaling = kwargs.get('curvature_scaling', 1.0)
        propagation_scaling = kwargs.get('propagation_scaling', -0.5)
        max_iters = kwargs.get('max_iters', 1000)

        if self._param('verbose', kwargs):
            self.print_params('Active Contours', **{
                'Isosurface Value': isosurface_value,
                'Advection Scaling': advection_scaling,
                'Curvature Scaling': curvature_scaling,
                'Propagation Scaling': propagation_scaling,
                'Max iters': max_iters
            })

        levelsets = geodesic_active_contours(
            seed_image, feature_image, isosurface_value, advection_scaling, 
            curvature_scaling, propagation_scaling, max_iters
        )

        return self.write(levelsets, info)

    def surface(self, **kwargs):
        """
            Implementation of <- surface>

            :param input: 
                Path to input file, default is output from previous method

            :param output:
                Path for output file, default <input + _surface.vtp>

            :param level:
                Intensity value for surface. Number or threshold function
        """
        volume, info = read_volume(self.input)
        level = self._param('level', kwargs)
        self.output = self._get_filename('surface', '.vtp')

        if self._param('verbose', kwargs):
            print('Surface')
            print('  Surface level:', level)
            print('  Writing to:', self.output)

        polydata = surface(volume, info, level)

        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(self.output)
        writer.SetInputData(polydata)
        return bool(writer.Write())

    def configure(self, data):
        """
            Set global configuration
        """
        self.first = self._param('input', data, default='')        
        self.output = self.first
        self.write_dir = data.get('write_dir')
        self.params.set(n_threads=data.get('n_threads'))
        self.params.set(verbose=data.get('verbose'))

        self.write_dir, self.prefix = output_directory(self.first, 
                                                       self.write_dir)
        self.configured = True

    def start(self):
        """
            Look for methods in the config file and execute
        """
        for entry in self.data:
            if isinstance(entry, str):
                method = entry
                entry = {}
            else:
                method = list(entry.keys())[0]
            
            if method == 'config':
                self.configure(entry[method])
                continue

            conf = self._param(method, entry, {})

            if conf.get('skip', False):
                continue

            inp = conf.get('input')
            out = conf.get('output')
            self._set_input_output(inp, out, method)
            
            if not hasattr(self, method):
                raise ValueError('Missing method: ', method)

            self._method = method
            op = getattr(self, method)
            kwargs = conf.get('args', {})
            op(**kwargs)

        return 1

    def _set_input_output(self, inp, out, name):
        """
            Get input and output paths for the current method


            :param inp:
                Path input file

            :param out:
                Path output file

            :param name:
                Name of next method to be executed
        """
        if inp is None and self.output is None:
            raise SystemExit('Need input file specified for ' + name)

        if inp is not None:
            self.input = inp

            if self.first == '':
                self.first = self.input
                _, self.prefix = output_directory(self.first)
        else:
            self.input = self.output

        if out is not None:
            self.output = out 
        else:
            self.output = self._get_filename(name)

    def _get_filename(self, step, ext='.vti'):
        """
            :param step:
                Name of the next method to be executed
        """
        name = self.prefix + '_{}'.format(step) + ext
        return os.path.join(self.write_dir, name)

    def _param(self, name, ns=None, default=None):
        """
            Get param 'name'. First check if its given as a keyword
            argument, if not check self.params
        """
        if ns is None:
            ns = {}

        param = ns.get(name)

        if param is None:
            if default is not None:
                param = default
            else:
                if not hasattr(self.params, name):
                    raise ValueError('Method {} takes no argument {}'
                                     .format(self._method, name))
                else:
                    param = getattr(self.params, name)

        return param

    def print_params(self, name, **kwargs):
        """
            Print parameters for method 'name'

            Ex:

            <name>
              Output: <output>
              Key: value
              Key: value
              ...
        """
        print(name)

        for key in kwargs:
            print(' ', key, ':', kwargs[key])

        if 'output' not in kwargs:
            print('  Output:', self.output)

    def write(self, data, info=None, fn=None):
        """
            :param data:
                Data to write, VTKImage or ndarray

            :param info:
                VTKInformation instance

            :param fn:
                Filename, if None the file is written with fn=self.output

        """
        if fn is None:
            if info is None:
                raise ValueError()

            return dl_vtk.VTKImage.write_vti(self.output, data, info)

        if fn.endswith('.vti'):
            return dl_vtk.VTKImage.write_vti(fn, data, info)
        elif fn.endswith('.npy'):
            return np.save(fn, data)
        else:
            raise ValueError()


def index(i, axis, read=True):
    """
        Return slice obj for element i for a given axis
        TODO use other index class in detection.py
    """
    if read:
        if axis == 0:
            return np.index_exp[i, :, :]
        elif axis == 1:
            return np.index_exp[:, i, :]
        elif axis == 2:
            return np.index_exp[:, :, i]
        else:
            raise ValueError('Bad axis, {}'.format(axis))
    else:
        if axis == 0:
            return np.index_exp[i, :, :, 0]
        elif axis == 1:
            return np.index_exp[:, i, :, 1]
        elif axis == 2:
            return np.index_exp[:, :, i, 2]
        else:
            raise ValueError('Bad axis, {}'.format(axis))


def is_dicom(path):
    """
        Check if file at path is a dicom file
        # TODO write this -> VTKImage.from_dicom
    """
    return 'dicom' in path.lower()


def _save_volume(volume, suffix, info, path, name):
    path = os.path.join(path, '{}_{}.vti'.format(name, suffix))
    return dl_vtk.VTKImage.write_vti(path, volume, info=info)


def output_directory(input_image, output_dir=None):
    if input_image is None:
        raise ValueError()

    path, name = os.path.split(input_image)
    prefix, ext = os.path.splitext(name)

    if output_dir is not None:
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        return output_dir, prefix

    dir_name = os.path.join(path, prefix + '_dir')
    
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
    else:
        dir_name = dir_name + '_{}'.format(int(time.time()*1000))
        os.mkdir(dir_name)

    return dir_name, prefix


def get_dictionary(dico):
    if isinstance(dico, str) and not dico.endswith('.npy'):
        with open(dico, 'rb') as f:
            return pickle.load(f)
    elif isinstance(dico, str) and dico.endswith('.npy'):
        return np.load(dico)
    elif isinstance(dico, np.ndarray):
        return dico
    else:
        msg = 'Unsupported dictionary type. Dictionary can be \n'
        msg += 'numpy dump (.npy), pickeled object or numpy array'
        raise SystemExit(msg)


def start_run(input_image, output_dir, start_from):
    """
        Figure out at which step the script should start and
        read required input data. 
    """   
    name, ext = os.path.splitext(os.path.basename(input_image))
    base_path, prefix = output_directory(input_image, output_dir)

    required = {
        'denoise': os.path.join(base_path, '{}_denoised.vti'.format(name)),
        'enhance': os.path.join(base_path, '{}_features.vti'.format(name))
    }

    value = {
        'denoise': START_DENOISE,
        'enhance': START_ENHANCE
    }

    if start_from == START_BEGINNING:
        volume = read_volume(input_image)
        return volume, base_path, prefix, START_BEGINNING
    elif start_from not in required:
        raise SystemExit('Unknown argument --start-from {}'
                         .format(start_from))
    
    if os.path.isfile(required[start_from]):
        volume = read_volume(required[start_from])
        return volume, base_path, prefix, value[start_from]
    else:
        msg = 'Missing input file for starting at {}'.format(start_from)
        print(msg + ', starting from beginning with {}'.format(input_image))
        volume = read_volume(input_image)
        return volume, base_path, prefix, START_BEGINNING
    

# Read input file
#   DICOM
#   VTI
#   NPY
def read_volume(filename):
    path, name = os.path.split(filename)
    fn, ext = os.path.splitext(name)

    if ext.lower() == '.vti':
        volume = dl_vtk.VTKImage.read(filename)
    elif ext.lower() == '.npy':
        array = np.load(filename)
        volume = dl_vtk.VTKImage.from_array(array)
    elif is_dicom(filename):
        volume = dl_vtk.VTKImage.from_dicom(filename)
    else:
        raise ValueError('Unsupported image format, {}'.format(filename))

    info = volume.information()
    info.path = filename
    return volume, info


def threshold(volume, kind):
    """
        Threshold volume.

        kind can be the name of a threshold function in filters.py 
        or a number

        :param volume:
            Volume to theshold. ndarray or VTKImage

        :param kind:
            Name of threshold function or value

        :return:
            Thresholded image
    """
    try:
        thresh = float(kind)
    except ValueError:
        # Thresh is a name, get correct method
        if kind is None:
            return volume

        attr = 'threshold_' + kind.lower()
        if not hasattr(dl.filters, attr):
            raise SystemExit('Unsupported threshold {}'.format(kind))

        func = getattr(dl.filters, attr)
        thresh = func(volume)
    except TypeError:
        # Thresh is none
        return volume

    volume = volume.copy()
    volume[volume < thresh] = 0
    return volume


def laplacian_sharpening(volume):
    """
        Enhancement of gradient/edges

        The Laplacian is calculated from 'volume' and then
        added back

        result = volume + laplacian(volume)

        :param volume:
            VTKImage

        :return:
            VTKImage
    """
    if hasattr(itk, 'LaplacianSharpening'):
        img = itk.GetImageFromArray(volume)
        sharp = itk.LaplacianSharpening(img)
        sharp = itk.GetArrayFromImage(sharp)
    else:
        img = itk.GetImageFromArray(volume.astype(np.float32))
        filt = itk.LaplacianSharpeningImageFilter.IF3IF3.New()
        filt.SetInput(img)
        filt.Update()
        sharp = filt.GetOutput()
        sharp = itk.GetArrayFromImage(sharp).T

    return volume + sharp


def denoise_2d(volume, patch_size, axis, dictionary, n_atoms,
               iters, n_nonzero, n_threads, sigma, verbose):
    """
        Denoise volume using 2D image patches

        :param volume:
            Volume to denoise, ndarray or  VTKImage

        :param patch_size:
            Int, size of image patches. 

        :param axis:
            Which axis to denoise. 'xyz' for all axis, 'x' for only
            x-axis. If multiple axis the result is averaged

        :param dictionary:
            Path to trained dictionary or None

        :param n_atoms:
            If 'dictionary' is None, create one with n_atoms

        :param iters:
            Number of training iterations

        :param n_nonzero:
            Number of nonzero coefficients to use for training

        :param n_threads:
            Number of threads

        :param sigma:
            Noise variance, if None it's calculated using SURE SHRINK

        :param verbose:
            Print progress if true

        :return:
            Reconstructed volume, dictionary
    """
    if dictionary is not None:
        dictionary = get_dictionary(dictionary)
        if isinstance(n_atoms, int):
            if 0 < n_atoms < dictionary.shape[1]:
                dictionary = dictionary[:, :n_atoms]
        else:
            n_atoms = dictionary.shape[1]

    elif n_atoms is not None:
        if n_atoms == 0:
            n_atoms = 2*patch_size*patch_size

        dictionary = dl.dct_dict(n_atoms, patch_size)
        n_atoms = dictionary.shape[1]
    else:
        raise SystemExit('Denoise2D: Need dictionary or n_atoms not None')

    if sigma == -1:
        sigma = dl.filters.estimate_sigma(volume)

    denoised = np.zeros_like(volume)
    axes = {'x': 0, 'y': 1, 'z': 2}

    if n_nonzero == 0:
        n_nonzero = int(math.ceil(0.1*patch_size**2))

    for ax in axis:
        if verbose:
            print('Denoising axis:', ax)

        for i in range(volume.shape[axes[ax]]):
            if verbose:
                sys.stdout.write(' Slice %d/%d\r' % (i+1, volume.shape[axes[ax]]))
                sys.stdout.flush()

            image = volume[index(i, axes[ax])]

            denoiser = dl.Denoise(image, patch_size, 'batch')
            denoiser.dictionary = dictionary
            denoiser.train(iters=iters, n_nonzero=n_nonzero,
                           n_threads=n_threads)

            recon = denoiser.denoise(sigma=sigma, n_threads=n_threads)
            dictionary = denoiser.dictionary
            denoised[index(i, axes[ax])] += recon

        print()

    return denoised / len(axis), dictionary


def denoise_3d(volume, patch_size, stride, batch_size, dictionary, n_atoms, 
               iters, n_nonzero, n_threads, method, sigma, verbose):
    """
        Denoise volume using 3D image patches

        :param volume:
            Volume to denoise, ndarray or  VTKImage

        :param patch_size:
            List/tuple, size of image patches. [size_x, size_y, size_z]

        :param stride:
            List/tuple, stride, [stride_x, stride_y, stride_z]

        :param batch_size:
            Size of batches to use for training. The dictionary is trained
            looking at 'batch_size' patches at the time. This need to be 
            small enough such that all 'batch_size' patches can be kept
            in memory

        :param dictionary:
            Path to trained dictionary or None

        :param n_atoms:
            If 'dictionary' is None, create one with n_atoms

        :param iters:
            Number of training iterations

        :param n_nonzero:
            Number of nonzero coefficients to use for training

        :param n_threads:
            Number of threads

        :param method:
            Training algorithm, 'batch' or 'odl'. ODL recommended

        :param sigma:
            Noise variance, if None it's calculated using SURE SHRINK

        :param verbose:
            Print progress if true

        :return:
            Reconstructed volume, dictionary
    """
    patches = dl.Patches(volume, patch_size, stride)

    try:
        patch_generator = patches.generator(batch_size)
    except MemoryError as ex:
        raise SystemExit(ex)

    if dictionary is None:
        if n_atoms == 0:
            raise SystemExit('Need number of atoms for 3D denoising')

        dictionary = dl.random_dictionary(patches.patch_size, n_atoms)

    if n_nonzero == 0:
        n_nonzero = int(math.ceil(0.1*patches.patch_size**2))

    trainer = dl.Trainer(next(patch_generator), method)
    trainer.dictionary = dictionary

    trainer.train(n_atoms=n_atoms, iters=iters, n_nonzero=n_nonzero,
                  n_threads=n_threads, verbose=verbose)
    i = batch_size
    if verbose:
        print('Training, batch:')

    n_batches = int(patches.n_patches) + 1
    for batch in patch_generator:
        if verbose:
            sys.stdout.write('  %d/%d\r' % (i, n_batches))
            sys.stdout.flush()
            i += batch_size

        trainer.signals = batch
        trainer.train(n_atoms=n_atoms, iters=iters, n_nonzero=n_nonzero,
                      n_threads=n_threads, verbose=False)
        
    print()

    dictionary = trainer.dictionary

    if sigma == -1:
        sigma = dl.filters.estimate_sigma(volume)

    tolerance = patches.patch_size * (1.15 * sigma) ** 2

    i = batch_size

    if verbose:
        print('Reconstruction, batch:')

    for batch, recon in patches.generator(batch_size, callback=True):
        if verbose:
            sys.stdout.write('  %d/%d\r' % (i, n_batches))
            sys.stdout.flush()
            i += batch_size

        codes = dl.omp_batch(batch, dictionary, tol=tolerance, 
                             n_threads=n_threads)
        recon(np.dot(dictionary, codes))

    print()

    return patches.reconstructed, dictionary

# Inpaint
#   ITKrMM
#   Synthesis
def inpaint_2d(volume, mask, dictionary, patch_size, n_atoms, axis, n_nonzero=0,
               iters=10, verbose=False, n_threads=1, n_low_rank=0, which='itkrmm'):
    """
        Inpaint volume using 2D image patches

    """
    if dictionary is not None:
        dictionary = get_dictionary(dictionary)

        if isinstance(n_atoms, int):
            if 0 < n_atoms < dictionary.shape[1]:
                atoms = np.argsort(np.var(dictionary, axis=0))[:-n_atoms] 
                dictionary = dictionary[:, atoms]
        else:
            n_atoms = dictionary.shape[1]

    elif n_atoms is not None:
        if n_atoms == 0:
            n_atoms = 2*patch_size*patch_size

        dictionary = dl.dct_dict(n_atoms, patch_size)
        n_atoms = dictionary.shape[1]
    else:
        raise SystemExit('Inpaint2D: Need dictionary or n_atoms not None')

    inpainted = np.zeros_like(volume)
    axes = {'x': 0, 'y': 1, 'z': 2}

    if n_nonzero == 0:
        n_nonzero = int(math.ceil(0.1*patch_size**2))

    index = dl.detection.Index(inpainted.shape)

    for ax in axis:
        if verbose:
            print('Inpainting axis:', ax)

        for i in range(volume.shape[axes[ax]]):
            if verbose:
                sys.stdout.write(' Slice %d/%d\r' % (i+1, volume.shape[axes[ax]]))
                sys.stdout.flush()

            image = volume[index(i, axes[ax])]
            mask_ = mask[index(i, axis[ax])]

            if which == 'itkrmm':
                inpainter = dl.Inpaint(image, mask_, patch_size, 'itkrmm')
                inpainter.dictionary = dictionary
                inpainter.train(
                    iters=iters, n_nonzero=n_nonzero, n_low_rank=n_low_rank,
                    n_threads=n_threads, verbose=verbose
                )

                recon = inpainter.inpaint(n_nonzero=n_nonzero, verbose=verbose)
                dictionary = inpainter.dictionary
            else:
                synth = dl.TextureSynthesis(image, mask_, patch_size)
                recon = synth.inpaint(None if iters == 10 else iters, verbose)

            inpainted[index(i, axes[ax])] += recon

        print()

    return inpainted / len(axis), dictionary


def inpaint_3d(volume, mask, patch_size, stride, batch_size, dictionary, n_atoms,
                iters, n_nonzero, n_threads, which, sigma, verbose, n_low_rank):

    patches = dl.Patches(volume, patch_size, stride)
    masks = dl.Patches(mask, patch_size, stride)

    try:
        patches.patches
        patch_generator = None
    except MemoryError:
        try:
            patch_generator = patches.generator(batch_size)
        except MemoryError as ex:
            raise SystemExit(ex)

    if dictionary is None:
        if n_atoms == 0:
            raise SystemExit('Need number of atoms for 3D inpainting')

        dictionary = dl.random_dictionary(patches.patch_size, n_atoms)

    if n_nonzero == 0:
        n_nonzero = int(math.ceil(0.1*patches.patch_size**2))

    if patch_generator is not None:
        i = batch_size
        if verbose:
            print('Training, batch:')

        n_batches = int(patches.n_patches) + 1
        masks = masks.generator(batch_size)

        for batch in patch_generator:
            if verbose:
                sys.stdout.write('  %d/%d\r' % (i, n_batches))
                sys.stdout.flush()
                i += batch_size

            dictionary = dl.itkrmm(batch, next(masks), dictionary, n_nonzero, 
                                   iters, low_rank=None, verbose=verbose)
          
        print()

        i = batch_size

        if verbose:
            print('Reconstruction, batch:')

        masks = dl.Patches(mask, patch_size, stride).generator(batch_size)

        for batch, recon in patches.generator(batch_size, callback=True):
            if verbose:
                sys.stdout.write('  %d/%d\r' % (i, n_batches))
                sys.stdout.flush()
                i += batch_size

            codes = dl.omp_mask(batch, next(masks), dictionary, n_nonzero,
                                verbose=verbose)
            recon(np.dot(dictionary, codes))

        print()

        return patches.reconstructed, dictionary
    else:
        if which == 'itkrmm':
            inpainter = dl.Inpaint(patches, masks, patch_size, which)
            inpainter.dictionary = dictionary
            inpainter.train(
                n_low_rank=n_low_rank, iters=iters, n_nonzer=n_nonzero,
                verbose=verbose, n_threads=n_threads
            )
            dictionary = inpainter.dictionary
            recon = inpainter.inpaint(n_nonzero=n_nonzero, verbose=verbose)
            return recon, dictionary
        else:
            raise NotImplementedError()


def kmeans_enhance(volume, size, alpha, axis='xyz', verbose=True):
    """
        Enhance vessel-like features using KMeans

        The volume is transformed into image patches (2D or 3D) then these
        patches are clustered into two clusters. If the image is not too
        noisy the cluster with the least number of members will contain 
        the patches being part of a vessel. 

        All patches in the non-vessel cluster are set to zero while the others
        are kept, and a new set of patches are created by:

            enhanced = alpha*old_patches + (1 - alpha)*new_patches


        If 2D patches are used, then for each slice on the axis specified 
        by 'axis' are updated according to the rule above. The final volume 
        if created by taking the maximum over the axis 'axis'

        :param volume:
            VTKImage

        :param size:
            Size of the image patches. The largest side of the patches 
            should not be larger than twice the diameter of the smallest
            vessels

        :param alpha:
            Weighing of the new patches, see formula above

        :param axis:
            Compute response of slices in these direction. 'x' for x-axis
            only, 'y' for y-axis etc. 'xz' for computing x- and z- axis.

        :param verbose:
            Print progress

        :return:
            VTKImage, same shape as volume
        
    """
    axes = {'x': 0, 'y': 1, 'z': 2}
    
    if isinstance(size, (list, tuple)):
        patches = dl.Patches(volume, size)
        labels = dl.detection.smallest_cluster(patches.patches.T, 2, False)
        cleaned = patches.patches * labels
        enhanced = alpha*patches.patches + (1 - alpha)*cleaned
        return patches.reconstruct(enhanced)
    else:
        x, y, z = volume.shape
        feature_image = np.zeros((x, y, z, 3))

        for ax in axis:
            if verbose:
                print('Clustering axis:', ax)

            for i in range(feature_image.shape[axes[ax]]):
                patches = dl.Patches(volume[index(i, axes[ax])].copy(), size)
                labels = dl.detection.smallest_cluster(patches.patches.T, 2, False)
                cleaned = patches.patches * labels
                enhanced = alpha*patches.patches + (1 - alpha)*cleaned
                feature_image[index(i, axes[ax], False)] += patches.reconstruct(enhanced)

        return np.max(feature_image, axis=3)


def hessian_enhance(volume, size, scale_min, scale_max, n_scales, 
                    save_func=None):
    """
        Enhance tubular features using hessian eigenvalues.

        One hessian matrix is created for each entry in volume for each 
        scale by convolving the volume with a gaussian gradient filter
        with kernel size 'size' and sigma in np.linspace(scale_min, scale_max,
        n_scales)

        :param volume:
            VTKImage

        :param size:
            Size of gaussian filter kernel

        :param scale_min:
            Used for sigma in the gaussian kernel. This value corresponds to
            the smallest vessel diameter

        :param scale_max:
            Used for sigma in the gaussian kernel. This value corresponds to
            the largest vessel diameter

        :param n_scales:
            Size of the interval [scale_min, scale_max]

        :return:
            tuple of 3 VTKImage instances. First: the maximum tubular 
            response over all scales for each entry, 2nd is the mean response,
            3rd all entries that looks like part of a tube
    """
    if volume.min() >= 0:
        volume = 2 * dl.normalize(volume) - 1

    candidates = dl.detection.tubular_candidates(volume, size, scale_min)

    cands = np.zeros_like(volume)
    for x, y, z in candidates:
        cands[x, y, z] = 1

    if save_func is not None:
        save_func(cands, 'hessian_candidates')

    tdf = dl.detection.tube(volume, candidates, size, scale_min, 
                            scale_max, n_scales)
    mean_response = tdf.mean(axis=0)
    
    if save_func is not None:
        save_func(mean_response, 'mean_hessian_response')
    
    kernel = dl.kernels.gaussian3d(3, 1)
    kernel /= kernel.sum()
    enhanced = signal.convolve(mean_response, kernel, mode='same')
    
    if save_func is not None:
        save_func(enhanced, 'response_gauss')
        return enhanced

    return enhanced, mean_response, cands


#   NN
def nn_enhance(volume, alpha):
    raise NotImplementedError()
    network = dl.nn.network('params')
    scores = network.predict(volume)
    features = volume*scores
    return alpha*volume + (1 - alpha)*features


def seed_region_growing(volume, info, size=3, sigma=1, level=0.3, shrink=0):
    """
        Create binary seed image using Hessian eigenvalues

        :param volume:
            Input volume, ndarray or VTKImage

        :param info:
            Volume attributes, VTKInformation

        :param size:
            Size of filter kernel. The hessian matrix is calculated by 
            convolving 'volume' with a gaussian kernel of size 'size'

        :param sigma:
            For gaussian kernel

        :param level:


        :return:
            VTKImage
    """
    volume = 2*dl.normalize(volume) - 1
    tdf = dl.detection.tubular_candidates(volume, size, sigma)
    preselection = np.zeros(volume.shape)

    for x, y, z in tdf:
        preselection[x, y, z] = 1
    
    kernel = dl.kernels.gaussian3d(size, sigma)
    preselection = dl.normalize(signal.convolve(preselection, kernel, mode='same'))
    preselection[preselection < level] = 0
    preselection[preselection > 0 ] = 1

    if isinstance(shrink, int) and shrink > 0:
        selem = _m.ball(shrink)
        preselection = _m.erosion(preselection, selem)

    tmpfile = str(time.time()) + '_vtk.vti'
    dl_vtk.VTKImage.write_vti(tmpfile, preselection, info)
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(tmpfile)
    reader.Update()
    os.remove(tmpfile)

    connectivity = vtk.vtkImageConnectivityFilter()
    connectivity.SetInputConnection(reader.GetOutputPort())
    connectivity.SetExtractionModeToLargestRegion()
    connectivity.Update()

    seed = dl_vtk.VTKImage.from_image_data(connectivity.GetOutput())

    return seed, dl_vtk.VTKImage.from_array(preselection, info)


def potential(image, sigma=1.0, normalize=True, 
              remap=True, flip_gradient=False):
    """
        Create edge potential map using gradient of gaussian. If the radius
        of the smallest and largest vessels in 'image' is very different, 
        then sigma should be given as a list. sigma=[1, 2, ..., 5] this way
        no vessels are "missed". 

        By default edges are close to one and areas inside the vessels are 
        close to zero. Set flip_gradient to True to have edges close to zero
        and areas inside the vessels close to one

        When multiple sigmas are given the final potential volume is the 
        elementwise maximum over the sigmas.

        :param image:
            Source image, numpy array or VTKImage.

        :param normalize:
            Normalize the gradient given sigma. The gaussian gradient is 
            decreasing when sigma increase. Setting this to true ensures the
            response from different sigmas have the same scale

        :param remap:
            ???

        :param flip_gradient:
            Set edges close to zero if True, otherwise edges are close to one

        :return:
            Edge potentials volume of same shape as 'image'

    """
    if not isinstance(image, (dl_vtk.VTKImage, np.ndarray)):
        msg = 'Image has to VTKImage or ndarray, not {}'.format(type(image))
        raise ValueError(msg)

    info = getattr(image, 'information', lambda: None)()
    itkImage = itk.GetImageFromArray(image.astype(np.float32))

    if isinstance(sigma, (int, float)):
        sigma = [sigma]

    height, width, depth = image.shape
    outputs = np.zeros((len(sigma), height, width, depth))

    for i in range(len(sigma)):
        filt = itk.GradientMagnitudeRecursiveGaussianImageFilter.IF3IF3.New()
        filt.SetInput(itkImage)
        filt.SetSigma(sigma[i])
        filt.SetNormalizeAcrossScale(normalize)
        filt.Update()
        output = filt.GetOutput()

        if remap:
            smallest, largest = itk.range(output)
            alpha = (largest - smallest) / 6.0
            beta = 0.5 * (largest + smallest)

            sigmoid = itk.SigmoidImageFilter.IF3IF3.New()
            sigmoid.SetInput(output)
            sigmoid.SetAlpha(alpha)
            sigmoid.SetBeta(beta)
            sigmoid.SetOutputMaximum(largest)
            sigmoid.SetOutputMinimum(smallest)
            sigmoid.Update()
            output = sigmoid.GetOutput()

        array = itk.GetArrayFromImage(output).T

        if flip_gradient:
            array = 1.0 / (1 + array)

        outputs[i] = array

    edges = np.min(outputs, axis=0)

    return dl_vtk.VTKImage.from_array(edges, info)


def geodesic_active_contours(seed_image, feature_image, isosurface_value=0.5, 
                             advection_scaling=-1, curvature_scaling=1, 
                             propagation_scaling=-0.5, max_iters=1000):
    """
        Grow seed_image according to edge potential in feature_image.

        Seed image is a binary image containing all structures that is
        needed in the final geometry. GeodesicActiveContour grows 
        seed_image until the best match given the features are found.

        For the best results feature_image should be (near) zero at the 
        object boundaries and (near) one inside. 

        :param seed_image:
            Binary seed for region growing

        :param feature_image:
            Edge potential, same shape as seed_image

        :param isosurface_value:
            Thing

        :param advection_scaling:
            Weight of the advection term in the region growing minimization.
            Larger value is more growing/shrinking. Positive for shrinking,
            negative for growing

        :param curvature_scaling:
            Larger gives more smoothing.

        :param propagation_scaling:
            Pass

        :param max_iters:
            Stop growing after this many iterations

        :return:
            Final geometry as VTKImage. Positive values are part of the 
            geometry, negative elements are outside the structure.
    
    """
    if isinstance(seed_image, (dl_vtk.VTKImage, np.ndarray)):
        seed = seed_image.T.astype(np.float32)
        seed = np.asfortranarray(seed)
        initialImage = itk.GetImageFromArray(seed)
    elif 'itkImage' not in seed_image.__class__.__name__:
        raise ValueError('Need seed_image of type '
                             'VTKImage, ndarray or itkImage')
    else:
        initialImage = seed_image

    if isinstance(feature_image, (dl_vtk.VTKImage, np.ndarray)):
        feat = feature_image.T.astype(np.float32)
        feat = np.asfortranarray(feat)
        featureImage = itk.GetImageFromArray(feat)
    elif 'itkImage' not in feature_image.__class__.__name__:
        raise ValueError('Need feature_image of type '
                             'VTKImage, ndarray or itkImage')
    else:
        featureImage = feature_image

    if hasattr(seed_image, 'information'):
        try:
            info = seed_image.information()
        except AttributeError:
            info = None
    else:
        info = None

    filt = itk.GeodesicActiveContourLevelSetImageFilter.IF3IF3F.New()
    filt.SetReverseExpansionDirection(False)
    filt.SetFeatureImage(featureImage)
    filt.SetInitialImage(initialImage)
    filt.SetIsoSurfaceValue(isosurface_value)
    filt.SetAdvectionScaling(advection_scaling)
    filt.SetCurvatureScaling(curvature_scaling)
    filt.SetPropagationScaling(propagation_scaling)
    filt.SetNumberOfIterations(max_iters)
    filt.Update()

    arr = itk.GetArrayFromImage(filt.GetOutput())

    if info is not None:
        return dl_vtk.VTKImage.from_array(arr, info)

    return arr


# Extract arteries
def surface(volume, info, level=None, connectivity=True):
    """
        Creates a surface from an image volume using Marching Cubes. 
        The surface is created at the intensity value level

        :param volume:
            Input volume, shape (x, y, z), of type dictlearn.VTKImage or 
            vtkImageData
        
        :param info:
            Volume information/attributes, instance of 
            dictlearn.VTKInformation
        
        :param level:
            Intensity to create surface. If this is none, the surface is 
            created at the level given from threshold_minimum. See 
            filters.py

        :param connectivity:
            If True, keep only the largest connected component

        :return:
            Surface as vtkPolyData
    """
    thresh = 0 

    if isinstance(volume, dl_vtk.VTKImage):
        if level is None:
            level = dl.filters.threshold_minimum(volume)

        _, image_data = dl_vtk.as_image_data(volume, info)
    else:
        image_data = vtk.vtkImageData()
        image_data.DeepCopy(volume)
        
        if level is None:
            volume = dl_vtk.VTKImage.from_image_data(volume)
            level = dl.filters.threshold_minimum(volume)

    contour = vtk.vtkMarchingContourFilter()
    contour.SetInputData(image_data)
    contour.SetValue(0, level)
    contour.Update()

    connectivity = vtk.vtkConnectivityFilter()
    connectivity.SetInputData(contour.GetOutput())
    connectivity.SetExtractionModeToLargestRegion()
    connectivity.Update()

    mapper = vtk.vtkGeometryFilter()
    mapper.SetInputData(connectivity.GetOutput())
    mapper.Update()

    return mapper.GetOutput()


# Run automatedPreProcessing?
# mpi4py version for 3D patches?


def main():
    description = 'A Script'  # TODO!
    parser = argparse.ArgumentParser(description=description)
    # TODO This is a mess, there must better way
    parser.add_argument(
        'input_path', type=str, 
        help='Path input image. Supported formats VTI, numpy dump. For'
             ' best results this should be the smallest region enclosing'
             ' the structures of interest. Can be created with:'
             ' http://www.vmtk.org/vmtkscripts/vmtkimagevoiselector.html'
    )

    parser.add_argument(
        'output_dir', type=str,
        help='Output directory for saving intermediate files.'
    )

    parser.add_argument(
        '-t', '--threshold', type=str,
        help='Threshold image before processing. Can be a value or'
             ' one of: "entropy", "median", "mean", "maxlink" or "minimum. '
             'See dictlearn/filters.py for details'
    )

    parser.add_argument(
        '--n-threads', type=int, 
        help='Number of threads. Default 1'
    )

    parser.add_argument(
        '-s', '--patch-size', nargs='+', action=ParsePatchSize,
        help='Size of images patches. If 2D patches this is a single'
             ' number. For 3D patches three numbers, ex: --patch-size 8 9 10.'
             ' Default 8.'
    )

    parser.add_argument(
        '--axis', type=str,
        help='If denoise on 2D slices use slices on this axis. \'x\' for '
             'x-axis, \'y\' for y-axis and \'z\' for z-axis. \'xz\' will '
             'denoise slices on the \'x\' and \'z\' axes separately and '
             'average the final volumes. Default \'x\'.'
    )

    parser.add_argument(
        '--n-atoms', type=int,
        help='Number of atoms in the dictionary. If this is 0 the number '
             ' of atoms will be twice the number of entries in a patch'
    )

    parser.add_argument(
        '--dictionary', type=str,
        help='Path to dictionary'
    )

    parser.add_argument(
        '--iters', type=int, 
        help='Number of training iterations for the dictionary'
    )

    parser.add_argument(
        '--n-nonzero', type=int,
        help='Number of nonzero coefficients for sparse coding. If this is 0 '
             ' it\'s set to approximately 10%% of the number of elements in '
             ' a patch'
    )

    parser.add_argument(
        '--batch-size', type=int,
        help='Batch size for 3D denoise. Trains and denoise "batch-size" image'
             ' patches at the time.'
    )

    parser.add_argument(
        '--stride', action=ParsePatchSize,
        help='Distance between 3D patches ie "2 2 2" takes every 2nd patch '
             ' in each direction'
    )

    parser.add_argument(
        '--method', type=str, choices=['ksvd', 'odl'],
        help='Use the K-SVD or Online dictionary Learning algorithm '
             'for training'
    )

    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='Print progress'
    )

    parser.add_argument(
        '--sigma', type=float,
        help='Standard deviation of the noise. If not given here it\'s '
             'estimated using "Ideal spatial adaptation by wavelet shrinkage" '
             'by Donoho and Johnstone'
    )

    # TODO better name
    parser.add_argument(
        '--vessel-size', type=int,
        help='Approximate vessel diameter, if the smallest vessel to capture is'
             ' n pixels wide then this should not be larger then 2*n'
    )

    parser.add_argument(
        '--alpha', type=float,
        help='For vessel enhancement step. The original image and feature image is'
             ' combined as alpha*original + (1 - alpha)*features. alpha = 0 give '
             'the feature image only, alpha=1 only original'
    )

    parser.add_argument(
        '--mask', type=str, help='Path to a mask for inpainting'
    )

    parser.add_argument(
        '--start-from', type=str, default=START_BEGINNING,
        choices=['denoise', 'enhance'],
        help='Figure it out yourself'
    )

    # Denoise 2D or 3D
    # If 2D which axes
    # Patch size
    # Strides

    # Num training iters
    # Num atoms
    # Num non zeros

    # Sigma

    # Num threads

    args = parser.parse_args()
    
    # if args.start_from is None:
    #    base_path, prefix = output_directory(args.input_image, args.output_dir)
    #    start_from = START_BEGINNING
    # else:
    #    base_path, prefix, start_from = resume_run(args.input_image, 
    #                                               args.output_dir,
    #                                               args.start_from)

    data = start_run(args.input_path, args.output_dir, args.start_from)
    volume, base_path, prefix, start_from = data
    volume, info = volume

    if args.verbose:
        print('Read volume at {}:'.format(info.path))
        volume.print()
        print('Writing data to directory "{}" and files'.format(base_path), end=' ')
        print('prefixed by "{}"'.format(prefix))

    save_volume = partial(_save_volume, path=base_path, name=prefix)
    config = Config.restore(os.path.join(base_path, prefix))
    config.set(args.__dict__)
    config.save(os.path.join(base_path, prefix))

    if start_from <= START_BEGINNING:
        volume = threshold(volume, config.threshold)

        if isinstance(config.patch_size, int):
            volume, dictionary = denoise_2d(
                volume, config.patch_size, config.axis, config.dictionary, 
                config.n_atoms, config.iters, config.n_nonzero, 
                config.n_threads, config.sigma, config.verbose
            )
        else:
            volume, dictionary = denoise_3d(
                volume, config.patch_size, config.stride, 
                config.batch_size, config.dictionary, config.n_atoms, 
                config.iters, config.n_nonzero, config.n_threads, 
                config.method, config.sigma, config.verbose
            )

        save_volume(volume, 'denoised', info)
        np.save(os.path.join(base_path, prefix + '_dictionary'), dictionary)
    
    # Inpaint
    
    # Enhance features
    which_enhance = 'not kmeans'
    if start_from <= START_DENOISE:
        if which_enhance == 'kmeans':
            volume = kmeans_enhance(volume, config.vessel_size,
                                    config.alpha, config.axis)
        else:
            volume = hessian_enhance(volume, config.filter_size, config.scale_min,
                                     config.scale_max, config.n_scales)
        save_volume(volume, 'features', info)

    if start_from <= START_ENHANCE:
        polydata = surface(volume, info)
        writer = vtk.vtkXMLPolyDataWriter()
        surf_name = os.path.join(base_path, '{}_surface.vtp'.format(prefix))
        writer.SetFileName(surf_name)
        writer.SetInputData(polydata)
        writer.Write()

    return 1


if __name__ == '__main__':
    if not sys.argv[1].lower().endswith('.yml'):
        raise SystemExit('Missing config file')

    exe = SurfitBase(sys.argv[1])
    sys.exit(exe.start())
