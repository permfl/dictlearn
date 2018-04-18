import numpy as np
from dictlearn import filters


def test_arithmetic_mean():
    pass


def test_threshold_hard():
    a = np.ones(10)

    b1 = filters.threshold(a.copy(), min_val=1.1)
    assert np.allclose(b1, a*0)

    b2 = filters.threshold(a.copy(), min_val=0.5, max_val=0.9)
    assert np.allclose(b2, a*0)

    b3 = filters.threshold(a.copy(), min_val=0.9)
    assert np.allclose(b3, a)

    a[:5] = 0.7

    b4 = filters.threshold(a.copy(), min_val=0.9)
    a[:5] = 0
    assert np.allclose(b4, a)
