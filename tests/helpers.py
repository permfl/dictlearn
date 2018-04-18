import os
import numpy


def get_dir():
    return os.path.dirname(os.path.realpath(__file__))


def get_image(img_num):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    return numpy.load(os.path.join(dir_path, 'test_img%d.npy' % img_num))