from typing import Callable
from .results import NumericalResult



def forward_difference(f: Callable[[float], float], x: float, h: float) -> NumericalResult:
    """
    Forward difference approximation of the first derivative.

    Args:
        f: Function to differentiate.
        x: Point at which to evaluate.
        h: Step size.

    Returns:
        Approximation of f'(x).

    Raises:
        ValueError: If h <= 0.

    Example:
        >>> forward_difference(lambda x: x**2, 1, 0.1)
        2.100000000000002
    """
    if h <= 0:
        raise ValueError("Step size h must be positive")
    result = (f(x + h) - f(x)) / h
    return NumericalResult(result, method_info={'type': 'differentiation', 'method': 'forward_difference', 'f': f, 'x': x, 'h': h})


def backward_difference(f: Callable[[float], float], x: float, h: float) -> NumericalResult:
    """
    Backward difference approximation of the first derivative.

    Args:
        f: Function to differentiate.
        x: Point at which to evaluate.
        h: Step size.

    Returns:
        Approximation of f'(x).

    Raises:
        ValueError: If h <= 0.

    Example:
        >>> backward_difference(lambda x: x**2, 1, 0.1)
        1.9000000000000013
    """
    if h <= 0:
        raise ValueError("Step size h must be positive")
    result = (f(x) - f(x - h)) / h
    return NumericalResult(result, method_info={'type': 'differentiation', 'method': 'backward_difference', 'f': f, 'x': x, 'h': h})


def central_difference(f: Callable[[float], float], x: float, h: float) -> NumericalResult:
    """
    Central difference approximation of the first derivative.

    Args:
        f: Function to differentiate.
        x: Point at which to evaluate.
        h: Step size.

    Returns:
        Approximation of f'(x).

    Raises:
        ValueError: If h <= 0.

    Example:
        >>> central_difference(lambda x: x**2, 1, 0.1)
        2.0000000000000018
    """
    if h <= 0:
        raise ValueError("Step size h must be positive")
    result = (f(x + h) - f(x - h)) / (2 * h)
    return NumericalResult(result, method_info={'type': 'differentiation', 'method': 'central_difference', 'f': f, 'x': x, 'h': h})


def central_second_difference(f: Callable[[float], float], x: float, h: float) -> NumericalResult:
    """
    Central difference approximation of the second derivative.

    Args:
        f: Function to differentiate.
        x: Point at which to evaluate.
        h: Step size.

    Returns:
        Approximation of f''(x).

    Raises:
        ValueError: If h <= 0.

    Example:
        >>> central_second_difference(lambda x: x**2, 1, 0.1)
        1.9999999999999991
    """
    if h <= 0:
        raise ValueError("Step size h must be positive")
    result = (f(x + h) - 2 * f(x) + f(x - h)) / (h**2)
    return NumericalResult(result, method_info={'type': 'differentiation', 'method': 'central_second_difference', 'f': f, 'x': x, 'h': h})


def richardson_extrapolation(f: Callable[[float], float], x: float, h: float) -> NumericalResult:
    """
    Richardson extrapolation for first derivative using central differences.

    Args:
        f: Function to differentiate.
        x: Point at which to evaluate.
        h: Initial step size.

    Returns:
        Improved approximation of f'(x).

    Raises:
        ValueError: If h <= 0.

    Example:
        >>> richardson_extrapolation(lambda x: x**2, 1, 0.1)
        2.0000000000000004
    """
    if h <= 0:
        raise ValueError("Step size h must be positive")
    d1 = central_difference(f, x, h)
    d2 = central_difference(f, x, h / 2)
    result = (4 * d2 - d1) / 3
    return NumericalResult(result, method_info={'type': 'differentiation', 'method': 'richardson_extrapolation', 'f': f, 'x': x, 'h': h})
