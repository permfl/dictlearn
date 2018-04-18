"""

Dictionary learning for image processing

This library implements various dictionary learning and sparse coding
algorithms for image and signal processing. 

Features
--------
    * Denoising
    * Inpainting


Overview of this module


`algorithms.py`:
    
    High level interfaces to dictionary learning
    algorithms, and image processing methods





`filters`:
    
    Simple general filtering operations. 

    * Local arithmetic mean
    * Thresholding


`kernels.py`:
    
    Filter kernels


`linalg.py`:
    
    Various factorizations and solver from linear algebra.
    Will remove this file.


`operators.py`:

    Has only the sobel operator. Will remove this file


`optimize.py`:

    This file contains dictionary learning algorithms and
    various sparse coding algorithms


`preprocess.py`:
    
    Image patch generation and other preprocessing methods
    needed to get the best results from the learning algorithms


`utils.py`:

    This has a lot of different stuff. Some can be moved to preprocess.
    Needs some cleaning


`vmtk_utils.py`: 
    common.py from Aslak. Not sure how much of this is needed. 
    Will look into it later


`wavelet.py`:
    
    Wavelet stuff. DWT and IDWT, with some thresholding filters


"""
from __future__ import print_function
__version__ = '0.0.1'


try:
    ___SETUP___
    print('Installing dictlearn...')
    # This file is imported from setup.py, and the code in
    # dictlearn._dictlearn is not yet built thus we cannot
    # import the stuff below
except NameError:
    #### TODO ####
    # Speed up imports: This is way to slow!
    from .algorithms import (
        Trainer, ImageTrainer, Denoise, Inpaint, ksvd_denoise
    )

    __all__ = ['Trainer', 'ImageTrainer', 'Denoise', 'Inpaint', 'ksvd_denoise']

    from .operators import laplacian, convolve2d
    __all__.extend(['laplacian', 'convolve2d'])

    from .optimize import ksvd, odl, itkrmm, reconstruct_low_rank
    from .sparse import omp_batch, omp_cholesky, omp_mask

    __all__.extend(['omp_batch', 'omp_cholesky', 'omp_mask', 'ksvd', 'odl', 'itkrmm',
                    'reconstruct_low_rank'])

    from .detection import Index, tube

    __all__.extend(['Index', 'tube'])

    from .preprocess import Patches
    __all__.extend(['Patches'])

    from . import preprocess
    from . import detection
    from . import filters
    from . import kernels
    from . import utils
    from . import optimize
    from . import operators
    from . import inpaint
    from . import sparse

    __all__.extend(['optimize', 'detection', 'filters', 'preprocess' 
                    'kernels', 'utils', 'operators', 'inpaint', 'sparse'])

    from .utils import (
        psnr, normalize, random_dictionary, dct_dict, 
        visualize_dictionary, rgb2gray, imread, imsave
    )

    __all__.extend([
        'psnr', 'normalize', 'random_dictionary', 'dct_dict',
        'visualize_dictionary', 'rgb2gray', 'imread', 'imsave'
    ])

    import os

    try:
        import cPickle as pickle
    except ImportError:
        import pickle

    _dict_path_dir = os.path.dirname(os.path.realpath(__file__))
    _dict_path = os.path.join(_dict_path_dir, 'dictionaries')

    # TODO
    # Is this really necessary?
    class DictionaryInfo(object):
        """
            Extra info for saving a dictionary
        """
        def __init__(self, patch_size=0, n_atoms=0, algorithm='',
                     n_nonzero=0, regularization='', iters=0, notes=""""""):
            self.patch_size = patch_size
            self.n_atoms = n_atoms
            self.n_nonzero = n_nonzero
            self.algorithm = algorithm
            self.regularization = regularization
            self.iters = iters
            self.notes = notes

        def __str__(self):
            s = '  Patch Size: {}\n  n_atoms: {}\n  n_nonzero: {}\n'\
                .format(self.patch_size, self.n_atoms, self.n_nonzero)
            s += '  iters: {}\n  algorithm: {}\n  regularization: {}'\
                .format(self.iters, self.algorithm, self.regularization)
            s += '\n  {}'.format(self.notes)
            return s

    def load_dictionary(name):
        """
        Load a previously saved dictionary
        :param name: Name of the dictionary to load. Same as name passed\
                     to save_dictionary()
        :return: Dictionary
        """
        for fn in os.listdir(_dict_path):
            short_name = fn.split('__dl__')[0]
            if short_name == name:
                with open(os.path.join(_dict_path, fn), 'r') as dump:
                    return pickle.load(dump).dictionary

        raise ValueError('Cannot find dictionary named {}'.format(name))

    def save_dictionary(name, dictionary, dict_info=None):
        """
        Save a dictionary
        :param name: Name of dictionary
        :param dictionary: Dictionary to save
        :param dict_info: Optional additional info
        """
        name = os.path.splitext(name)[0]
        patch_size, n_atoms = dictionary.shape

        if dict_info is None:
            dict_info = DictionaryInfo(patch_size, n_atoms)
        else:
            if not isinstance(dict_info, DictionaryInfo):
                raise ValueError('dict_info has to be an instance of'
                                 ' DictionaryInfo, not type {}'.format(type(dict_info)))
            if patch_size != dict_info.patch_size or n_atoms != dict_info.n_atoms:
                raise ValueError('Size of dictionary doesn\'t match data in dict_info, '
                                 'need dictionary.shape == (patch_size, n_atoms')

        dict_info.dictionary = dictionary
        fn = name + '__dl__{}__{}__.pkl'.format(patch_size, n_atoms)

        with open(os.path.join(_dict_path, fn), 'w') as dump:
            pickle.dump(dict_info, dump)


    def list_dictionaries(write=True):
        """
        List all saved dictionaries
        :param write: If True the list of dictionaries is printed, false to have it 
                      returned
        :return: List of pre-trained dictionaries, maybe
        """
        dicts = os.listdir(_dict_path)

        if not write:
            return dicts

        for fn in dicts:
            if fn.endswith('.md'):
                continue

            name, data = fn.split('__dl__')

            with open(os.path.join(_dict_path, fn)) as f:
                d = pickle.load(f)
                print(name)
                print(d, end='\n\n')


    def delete_dictionaries():
        # Delete all saved dictionaries
        for fn in os.listdir(_dict_path):
            if fn.startswith('100k_natural_image_patches_Elad'):
                continue

            os.remove(os.path.join(_dict_path, fn))

    __all__.extend(['DictionaryInfo', 'load_dictionary', 'save_dictionary',
                    'list_dictionaries', 'delete_dictionaries'])
