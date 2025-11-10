import math
from typing import Callable, Optional

try:
    import numpy as np
except ImportError:
    np = None


def _has_numpy() -> bool:
    return np is not None


def trapezoidal(f: Callable[[float], float], a: float, b: float) -> float:
    """
    Single trapezoidal rule for numerical integration.

    Args:
        f: Function to integrate.
        a: Lower bound.
        b: Upper bound.

    Returns:
        Approximation of the integral.

    Raises:
        ValueError: If a >= b.

    Example:
        >>> trapezoidal(lambda x: x**2, 0, 1)
        0.5
    """
    if a >= b:
        raise ValueError("Lower bound a must be less than upper bound b")
    h = b - a
    return (h / 2) * (f(a) + f(b))


def composite_trapezoidal(
    f: Callable[[float], float], a: float, b: float, n: int
) -> float:
    """
    Composite trapezoidal rule for numerical integration.

    Args:
        f: Function to integrate.
        a: Lower bound.
        b: Upper bound.
        n: Number of intervals.

    Returns:
        Approximation of the integral.

    Raises:
        ValueError: If a >= b or n < 1.

    Example:
        >>> composite_trapezoidal(lambda x: x**2, 0, 1, 2)
        0.375
    """
    if a >= b:
        raise ValueError("Lower bound a must be less than upper bound b")
    if n < 1:
        raise ValueError("Number of intervals n must be at least 1")
    h = (b - a) / n
    integral = f(a) + f(b)
    for i in range(1, n):
        integral += 2 * f(a + i * h)
    return (h / 2) * integral


def simpsons_13(f: Callable[[float], float], a: float, b: float) -> float:
    """
    Single Simpson's 1/3 rule for numerical integration.

    Args:
        f: Function to integrate.
        a: Lower bound.
        b: Upper bound.

    Returns:
        Approximation of the integral.

    Raises:
        ValueError: If a >= b.

    Example:
        >>> simpsons_13(lambda x: x**2, 0, 1)
        0.3333333333333333
    """
    if a >= b:
        raise ValueError("Lower bound a must be less than upper bound b")
    h = (b - a) / 2
    return (h / 3) * (f(a) + 4 * f(a + h) + f(b))


def composite_simpsons_13(
    f: Callable[[float], float], a: float, b: float, n: int
) -> float:
    """
    Composite Simpson's 1/3 rule for numerical integration. Requires even n.

    Args:
        f: Function to integrate.
        a: Lower bound.
        b: Upper bound.
        n: Number of intervals (must be even).

    Returns:
        Approximation of the integral.

    Raises:
        ValueError: If a >= b, n < 2, or n is odd.

    Example:
        >>> composite_simpsons_13(lambda x: x**2, 0, 1, 2)
        0.3333333333333333
    """
    if a >= b:
        raise ValueError("Lower bound a must be less than upper bound b")
    if n < 2 or n % 2 != 0:
        raise ValueError("Number of intervals n must be even and at least 2")
    h = (b - a) / n
    integral = f(a) + f(b)
    for i in range(1, n, 2):
        integral += 4 * f(a + i * h)
    for i in range(2, n, 2):
        integral += 2 * f(a + i * h)
    return (h / 3) * integral


def simpsons_38(f: Callable[[float], float], a: float, b: float) -> float:
    """
    Simpson's 3/8 rule for numerical integration.

    Args:
        f: Function to integrate.
        a: Lower bound.
        b: Upper bound.

    Returns:
        Approximation of the integral.

    Raises:
        ValueError: If a >= b.

    Example:
        >>> simpsons_38(lambda x: x**2, 0, 1)
        0.3333333333333333
    """
    if a >= b:
        raise ValueError("Lower bound a must be less than upper bound b")
    h = (b - a) / 3
    return (3 * h / 8) * (f(a) + 3 * f(a + h) + 3 * f(a + 2 * h) + f(b))


def gaussian_quadrature_2(f: Callable[[float], float], a: float, b: float) -> float:
    """
    2-point Gaussian quadrature for numerical integration.

    Args:
        f: Function to integrate.
        a: Lower bound.
        b: Upper bound.

    Returns:
        Approximation of the integral.

    Raises:
        ValueError: If a >= b.

    Example:
        >>> gaussian_quadrature_2(lambda x: x**2, 0, 1)
        0.33333333333333337
    """
    if a >= b:
        raise ValueError("Lower bound a must be less than upper bound b")
    # Transform to [-1,1]
    t = lambda x: ((b - a) * x + (b + a)) / 2
    w = (b - a) / 2
    x1, x2 = -math.sqrt(1 / 3), math.sqrt(1 / 3)
    return w * (f(t(x1)) + f(t(x2)))


def gaussian_quadrature_3(f: Callable[[float], float], a: float, b: float) -> float:
    """
    3-point Gaussian quadrature for numerical integration.

    Args:
        f: Function to integrate.
        a: Lower bound.
        b: Upper bound.

    Returns:
        Approximation of the integral.

    Raises:
        ValueError: If a >= b.

    Example:
        >>> gaussian_quadrature_3(lambda x: x**2, 0, 1)
        0.3333333333333333
    """
    if a >= b:
        raise ValueError("Lower bound a must be less than upper bound b")
    # Transform to [-1,1]
    t = lambda x: ((b - a) * x + (b + a)) / 2
    w = (b - a) / 2
    x1, x2, x3 = -math.sqrt(3 / 5), 0, math.sqrt(3 / 5)
    w1, w2, w3 = 5 / 9, 8 / 9, 5 / 9
    return w * (w1 * f(t(x1)) + w2 * f(t(x2)) + w3 * f(t(x3)))
