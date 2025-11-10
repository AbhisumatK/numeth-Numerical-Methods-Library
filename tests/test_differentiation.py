import math

import pytest

import numeth

try:
    import numpy as np
except ImportError:
    pytest.skip("numpy not available", allow_module_level=True)


def test_forward_difference():
    assert np.isclose(numeth.forward_difference(lambda x: x**2, 1, 0.001), 2, atol=0.01)
    with pytest.raises(ValueError):
        numeth.forward_difference(lambda x: x, 1, -0.1)


def test_backward_difference():
    assert np.isclose(
        numeth.backward_difference(lambda x: x**2, 1, 0.001), 2, atol=0.01
    )
    with pytest.raises(ValueError):
        numeth.backward_difference(lambda x: x, 1, 0)


def test_central_difference():
    assert np.isclose(numeth.central_difference(lambda x: x**2, 1, 0.001), 2, atol=1e-3)
    with pytest.raises(ValueError):
        numeth.central_difference(lambda x: x, 1, -1)


def test_central_second_difference():
    assert np.isclose(
        numeth.central_second_difference(lambda x: x**2, 1, 0.001), 2, atol=1e-3
    )
    with pytest.raises(ValueError):
        numeth.central_second_difference(lambda x: x, 1, 0)


def test_richardson_extrapolation():
    assert np.isclose(
        numeth.richardson_extrapolation(lambda x: x**2, 1, 0.1), 2, atol=1e-3
    )
    with pytest.raises(ValueError):
        numeth.richardson_extrapolation(lambda x: x, 1, 0)
