[![Build Status](https://travis-ci.org/permfl/dictlearn.svg?branch=master)](https://travis-ci.org/permfl/dictlearn)
[![Documentation Status](https://readthedocs.org/projects/dictlearn/badge/?version=latest)](http://dictlearn.readthedocs.io/en/latest/?badge=latest)

## Supported Python Versions

Supported python versions are: 2.7, 3.5 and 3.6. It should also work with python 3.7, however that is not tested 

## Install Linux

```
$ pip install -r requirements.txt
$ python setup.py install
```

## Install Mac
This package depends in `libomp` which is not installed by default. If you see the following error, `libomp` has to be installed.

```
$ python setup.py install
Installing dictlearn...
running develop
running egg_info
writing dictlearn.egg-info/PKG-INFO
writing dependency_links to dictlearn.egg-info/dependency_links.txt
writing requirements to dictlearn.egg-info/requires.txt
writing top-level names to dictlearn.egg-info/top_level.txt
reading manifest file 'dictlearn.egg-info/SOURCES.txt'
writing manifest file 'dictlearn.egg-info/SOURCES.txt'
running build_ext
building 'dictlearn._dictlearn._dictlearn' extension
.
.
.
clang: error: unsupported option '-fopenmp'
error: command 'gcc' failed with exit status 1
```

Install `libomp` with homebrew:
```
$ brew install libomp
```

and run `python setup.py install` again.


## Install Windows
Using anaconda:
`$ conda install --file requirements.txt`

Building the cython extensions are probably easier using anaconda.

If cython build crashes, install Visual Studio Build Tools. For python 3 you need: 

[http://landinghub.visualstudio.com/visual-cpp-build-tools](http://landinghub.visualstudio.com/visual-cpp-build-tools) 

and for python 2 

[https://www.microsoft.com/en-us/download/details.aspx?id=44266](https://www.microsoft.com/en-us/download/details.aspx?id=44266) 


### VTK and ITK
If you need to read/write VTK files you have to install **VTK**. 
Everything that requires VTK or ITK are located in `dictlearn/vtk.py`  and `scripts/`. The rest of the code can run
without having VTK or ITK installed.


### Denoise (Gray scale images only)
Simple denoising using 20 training iterations with 8x8 image patches.
```python
import matplotlib.pyplot as plt
import dictlearn as dl

denoise = dl.Denoise('noisy_image.png')
denoised_image = denoise.train().denoise()
plt.imshow(denoised_image)
plt.show()
```

### Inpainting 
```python
import matplotlib.pyplot as plt
import dictlearn as dl

inpainter = dl.Inpaint('image.png', 'mask.png')
inpainted_image = inpainter.train().inpaint()

plt.subplot(121)
plt.imshow(inpainter.patches.image)
plt.title('Original')

plt.subplot(122)
plt.imshow(inpainted_image)
plt.title('Inpainted')

plt.show()
```

## Tests
Run tests with

`$ pytest tests`

from root directory
