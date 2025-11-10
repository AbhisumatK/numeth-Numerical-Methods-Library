import pytest

import numeth


def test_linear_interpolation():
    assert numeth.linear_interpolation([(0, 0), (2, 4)], 1) == 2
    with pytest.raises(ValueError):
        numeth.linear_interpolation([(0, 0)], 1)
    with pytest.raises(ValueError):
        numeth.linear_interpolation([(0, 0), (0, 1)], 0)
    with pytest.raises(ValueError):
        numeth.linear_interpolation([(0, 0), (1, 1)], 2)


def test_lagrange_interpolation():
    assert numeth.lagrange_interpolation([(0, 0), (1, 1), (2, 4)], 0.5) == 0.25
    with pytest.raises(ValueError):
        numeth.lagrange_interpolation([], 1)
    with pytest.raises(ValueError):
        numeth.lagrange_interpolation([(0, 0), (0, 1)], 0)


def test_newton_divided_difference():
    assert numeth.newton_divided_difference([(0, 0), (1, 1), (2, 4)], 0.5) == 0.25
    with pytest.raises(ValueError):
        numeth.newton_divided_difference([], 1)
    with pytest.raises(ValueError):
        numeth.newton_divided_difference([(0, 0), (0, 1)], 0)
