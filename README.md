<p align="center">
    <img src="https://raw.githubusercontent.com/AbhisumatK/numeth-Numerical-Methods-Library/main/numeth.jpg" alt="numeth Logo" width="300">
</p>
      
[![PyPI version](https://badge.fury.io/py/numeth.svg)](https://badge.fury.io/py/numeth)
[![PyPI downloads](https://img.shields.io/pypi/dm/numeth.svg)](https://pypistats.org/packages/numeth)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/AbhisumatK/numeth-Numerical-Methods-Library/blob/main/LICENSE)

# numeth

A fully functional Python package implementing core numerical methods for engineering and applied mathematics. Designed for usability and educational clarity.

## Installation

Install via pip:

```
pip install numeth
```

## Quick Start

Here's a simple example using the Newton-Raphson method to find the square root of 2:

```python
from numeth import newton_raphson

def f(x):
    return x**2 - 2

def df(x):
    return 2 * x

root, iterations, converged = newton_raphson(f, df, x0=1.0, tol=1e-6, max_iter=100)
print(f"Root: {root}, Iterations: {iterations}, Converged: {converged}")
# Output: Root: 1.414213562373095, Iterations: 4, Converged: True
```

## Supported Methods

### Integration
- Trapezoidal Rule (single and composite)
- Simpson’s 1/3 Rule (single and composite)
- Simpson’s 3/8 Rule
- Gaussian Quadrature (2-point and 3-point)

### Differentiation
- Forward difference (first derivative)
- Backward difference (first derivative)
- Central difference (first derivative)
- Central difference (second derivative)
- Richardson extrapolation (first derivative)

### Root Finding
- Bisection Method
- Newton-Raphson Method
- Secant Method
- False Position Method

### Interpolation
- Linear Interpolation
- Lagrange Interpolation
- Newton’s Divided Difference Interpolation

### Linear Algebra
- Gauss Elimination with partial pivoting
- LU Decomposition (Doolittle’s method)
- Jacobi Iterative Method
- Gauss-Seidel Iterative Method

### Optimization
- Golden Section Search (minimization)
- Newton’s Method for Optimization (1D)

## How to Contribute or Report Issues

Contributions are welcome! Please submit pull requests or open issues on the [GitHub repository](https://github.com/example/numeth).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
