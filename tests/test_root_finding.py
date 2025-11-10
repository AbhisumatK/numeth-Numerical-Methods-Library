import math

import pytest

import numeth

try:
    import numpy as np
except ImportError:
    pytest.skip("numpy not available", allow_module_level=True)


def test_bisection():
    root, _, converged = numeth.bisection(lambda x: x**2 - 2, 0, 2)
    assert np.isclose(root, math.sqrt(2))
    assert converged
    with pytest.raises(ValueError):
        numeth.bisection(lambda x: x**2 - 2, 3, 4)


def test_newton_raphson():
    root, _, converged = numeth.newton_raphson(lambda x: x**2 - 2, lambda x: 2 * x, 1.0)
    assert np.isclose(root, math.sqrt(2))
    assert converged
    # With numerical derivative
    root, _, converged = numeth.newton_raphson(lambda x: x**2 - 2, None, 1.0)
    assert np.isclose(root, math.sqrt(2))
    assert converged
    with pytest.raises(ValueError):
        numeth.newton_raphson(lambda x: x, lambda x: 0, 1)


def test_secant():
    root, _, converged = numeth.secant(lambda x: x**2 - 2, 0, 3)
    assert np.isclose(root, math.sqrt(2))
    assert converged
    with pytest.raises(ValueError):
        numeth.secant(lambda x: 1, 0, 1)  # denom zero


def test_false_position():
    root, _, converged = numeth.false_position(lambda x: x - 1, 0, 2)
    assert root == 1.0
    assert converged
    with pytest.raises(ValueError):
        numeth.false_position(lambda x: x**2 - 2, 3, 4)
