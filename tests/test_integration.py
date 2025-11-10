import math

import pytest

import numeth

try:
    import numpy as np
except ImportError:
    pytest.skip("numpy not available", allow_module_level=True)


def test_trapezoidal():
    assert np.isclose(numeth.trapezoidal(lambda x: x, 0, 1), 0.5)
    with pytest.raises(ValueError):
        numeth.trapezoidal(lambda x: x, 1, 0)


def test_composite_trapezoidal():
    assert np.isclose(
        numeth.composite_trapezoidal(lambda x: x**2, 0, 1, 100), 1 / 3, atol=1e-3
    )
    with pytest.raises(ValueError):
        numeth.composite_trapezoidal(lambda x: x, 1, 0, 10)
    with pytest.raises(ValueError):
        numeth.composite_trapezoidal(lambda x: x, 0, 1, 0)


def test_simpsons_13():
    assert np.isclose(numeth.simpsons_13(lambda x: x**2, 0, 1), 1 / 3)
    with pytest.raises(ValueError):
        numeth.simpsons_13(lambda x: x, 1, 0)


def test_composite_simpsons_13():
    assert np.isclose(
        numeth.composite_simpsons_13(lambda x: x**2, 0, 1, 100), 1 / 3, atol=1e-5
    )
    with pytest.raises(ValueError):
        numeth.composite_simpsons_13(lambda x: x, 0, 1, 1)  # odd
    with pytest.raises(ValueError):
        numeth.composite_simpsons_13(lambda x: x, 1, 0, 2)


def test_simpsons_38():
    assert np.isclose(numeth.simpsons_38(lambda x: x**2, 0, 3), 9.0, atol=0.1)
    with pytest.raises(ValueError):
        numeth.simpsons_38(lambda x: x, 1, 0)


def test_gaussian_quadrature_2():
    assert np.isclose(numeth.gaussian_quadrature_2(lambda x: x**2, 0, 1), 1 / 3)
    with pytest.raises(ValueError):
        numeth.gaussian_quadrature_2(lambda x: x, 1, 0)


def test_gaussian_quadrature_3():
    assert np.isclose(numeth.gaussian_quadrature_3(lambda x: x**2, 0, 1), 1 / 3)
    with pytest.raises(ValueError):
        numeth.gaussian_quadrature_3(lambda x: x, 1, 0)
