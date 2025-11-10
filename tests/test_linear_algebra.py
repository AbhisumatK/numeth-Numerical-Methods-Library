import math

import pytest

import numeth

try:
    import numpy as np
except ImportError:
    pytest.skip("numpy not available", allow_module_level=True)


def test_gauss_elimination():
    x = numeth.gauss_elimination([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]], [8, -11, -3])
    assert np.allclose(x, [2, 3, -1])
    with pytest.raises(ValueError):
        numeth.gauss_elimination([[1, 2], [3, 4]], [5])


def test_lu_decomposition():
    A = [[2, 1, 1], [4, -6, 0], [-2, 7, 2]]
    L, U = numeth.lu_decomposition(A)
    assert np.allclose(np.dot(L, U), A)
    with pytest.raises(ValueError):
        numeth.lu_decomposition([[1, 2], [2, 4]])  # singular


def test_jacobi():
    A = [[10, -1, 2], [1, 11, -1], [3, -2, 10]]
    b = [6, 25, -11]
    x, _, converged = numeth.jacobi(A, b)
    assert converged
    assert np.allclose(np.dot(A, x), b, atol=1e-5)
    with pytest.raises(ValueError):
        numeth.jacobi([[0, 1], [1, 1]], [1, 1])


def test_gauss_seidel():
    A = [[10, -1, 2], [1, 11, -1], [3, -2, 10]]
    b = [6, 25, -11]
    x, _, converged = numeth.gauss_seidel(A, b)
    assert converged
    assert np.allclose(np.dot(A, x), b, atol=1e-5)
    with pytest.raises(ValueError):
        numeth.gauss_seidel([[1, 1], [1, 0]], [1, 1])
