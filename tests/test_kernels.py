import numpy as np
from dictlearn import kernels
import pytest


def test_gaussian():
    kern = kernels.gaussian(5, sigma=1)
    assert kern.shape == (5, 5)


def test_neighbourhood():
    nhood = kernels.get_neighbourhood(3)

    hood = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 0), (0, 1),
        (1, -1), (1, 0), (1, 1),
    ]

    assert nhood == hood

    nhood = kernels.get_neighbourhood((3, 3))
    assert nhood == hood

    with pytest.raises(ValueError):
        kernels.get_neighbourhood(2)

    with pytest.raises(ValueError):
        kernels.get_neighbourhood((2, 2))

    with pytest.raises(ValueError):
        kernels.get_neighbourhood((3, 3, 3))
