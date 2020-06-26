import re
import os
import sys
import numpy
import platform
from setuptools import setup
from distutils.extension import Extension
from distutils.command.build_ext import build_ext
from Cython.Build import cythonize

if sys.version_info[0] >= 3:
    import builtins
else:
    import __builtin__ as builtins


# Set global variable st we in can import the package before
# the extensions are built. Taken from numpy:
# https://github.com/numpy/numpy/blob/master/setup.py
builtins.___SETUP___ = True
import dictlearn


dir_path = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(dir_path, 'dictlearn/__init__.py'), 'r') as fd:
    version = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
                        fd.read(), re.MULTILINE).group(1)


def sources(which=''):
    sources_base = os.path.join(dir_path, 'dictlearn/_dictlearn')

    if which == 'hessian':
        cython_sources = ['hessian.pyx']
        source_files = []
    else:
        cython_sources = ['_dictlearn.pyx']
        source_files = ['omp_cholesky.c', 'omp_batch.c',
                        'common.c', 'bestexemplar.c']

    return [os.path.join(sources_base, source)
            for source in cython_sources + source_files]


COMPILE_ARGS = {
    'unix': ['-fopenmp', '-O3'],
    'msvc': ['/openmp', '/O2']
}


class build_ext_sub(build_ext):
    def compile_args(self, compiler):
        if platform.system() == 'Darwin':
            args = ['-Xpreprocessor', '-fopenmp', '-O3']
        else:
            args = COMPILE_ARGS.get(compiler, [])

        return args


    def build_extensions(self):
        compiler = self.compiler.compiler_type

        for ext in self.extensions:
            ext.extra_compile_args = self.compile_args(compiler)

        build_ext.build_extensions(self)


def extensions():
    exts = [
        Extension('dictlearn._dictlearn._dictlearn', sources()),
        Extension('dictlearn._dictlearn.hessian', sources('hessian'))
    ]

    for ext in exts:
        ext.include_dirs.append(numpy.get_include())

    return cythonize(exts)


setup(
    name='dictlearn',
    version=version,
    description=dictlearn.__doc__,
    ext_modules=extensions(),
    packages=['dictlearn', 'dictlearn._dictlearn'],
    cmdclass={'build_ext': build_ext_sub},
    install_requires=open('requirements.txt').read().split(),
    license='MIT',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: Unix',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
        'Topic :: Scientific/Engineering'
    ]
)

