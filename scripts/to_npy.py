import os
import sys
import numpy as np
import dictlearn as dl 

path = sys.argv[1]

for name in os.listdir(path):
    name, ext = os.path.splitext(name)
    name = os.path.join(path, name)

    if ext != '.vti':
        continue

    np_path = name + '.npy'
    if os.path.isfile(np_path):
        continue

    numpy = dl.utils.numpy_from_vti(name + ext)
    np.save(np_path, numpy)
    print('Numpy volume saved to:', np_path)
