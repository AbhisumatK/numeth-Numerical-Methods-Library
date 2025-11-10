import math

import pytest

import numeth

try:
    import numpy as np
except ImportError:
    pytest.skip("numpy not available", allow_module_level=True)


def test_golden_section_search():
    x_min, f_min = numeth.golden_section_search(lambda x: x**2, -5, 5)
    assert np.isclose(x_min, 0, atol=1e-5)
    assert np.isclose(f_min, 0, atol=1e-5)
    with pytest.raises(ValueError):
        numeth.golden_section_search(lambda x: x, 1, 0)


def test_newton_optimization():
    x_min, f_min, _, converged = numeth.newton_optimization(
        lambda x: x**2, lambda x: 2 * x, lambda x: 2, 10
    )
    assert np.isclose(x_min, 0)
    assert np.isclose(f_min, 0)
    assert converged
    with pytest.raises(ValueError):
        numeth.newton_optimization(lambda x: x, lambda x: 1, lambda x: 0, 0)
