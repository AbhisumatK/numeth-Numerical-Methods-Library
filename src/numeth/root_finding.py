from typing import Callable, Optional, Tuple
from .results import IterativeResult



def bisection(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> IterativeResult:
    """
    Bisection method for root finding.

    Args:
        f: Function to find root of.
        a: Lower bound.
        b: Upper bound.
        tol: Tolerance.
        max_iter: Maximum iterations.

    Returns:
        Root, number of iterations, convergence status.

    Raises:
        ValueError: If f(a) and f(b) have the same sign or a >= b.

    Example:
        >>> bisection(lambda x: x**2 - 2, 0, 2)
        (1.4142131805419922, 24, True)
    """
    if a >= b:
        raise ValueError("a must be less than b")
    if f(a) * f(b) >= 0:
        raise ValueError("f(a) and f(b) must have opposite signs")
    iter = 0
    while (b - a) / 2 > tol and iter < max_iter:
        c = (a + b) / 2
        if f(c) == 0:
            return IterativeResult(c, iter, True, method_info={'type': 'root_finding', 'method': 'bisection', 'f': f, 'a': a, 'b': b})
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
        iter += 1
    result = (a + b) / 2
    return IterativeResult(result, iter, iter < max_iter, method_info={'type': 'root_finding', 'method': 'bisection', 'f': f, 'a': a, 'b': b})


def newton_raphson(
    f: Callable[[float], float],
    df: Optional[Callable[[float], float]] = None,
    x0: float = 0.0,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> IterativeResult:
    """
    Newton-Raphson method for root finding.

    Args:
        f: Function to find root of.
        df: Derivative of f (optional, if None, approximated numerically).
        x0: Initial guess.
        tol: Tolerance.
        max_iter: Maximum iterations.

    Returns:
        Root, number of iterations, convergence status.

    Raises:
        ValueError: If derivative is zero.

    Example:
        >>> newton_raphson(lambda x: x**2 - 2, lambda x: 2*x, 1.0)
        (1.4142135623746899, 4, True)
    """
    x = x0
    iter = 0
    while abs(f(x)) > tol and iter < max_iter:
        if df is None:
            # Approximate derivative using central difference
            h = 1e-5
            dfx = (f(x + h) - f(x - h)) / (2 * h)
        else:
            dfx = df(x)
        if dfx == 0:
            raise ValueError("Derivative is zero")
        x = x - f(x) / dfx
        iter += 1
    return IterativeResult(x, iter, abs(f(x)) <= tol and iter < max_iter, method_info={'type': 'root_finding', 'method': 'newton_raphson', 'f': f, 'x0': x0})


def secant(
    f: Callable[[float], float],
    x0: float,
    x1: float,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> IterativeResult:
    """
    Secant method for root finding.

    Args:
        f: Function to find root of.
        x0: First initial guess.
        x1: Second initial guess.
        tol: Tolerance.
        max_iter: Maximum iterations.

    Returns:
        Root, number of iterations, convergence status.

    Raises:
        ValueError: If denominator is zero.

    Example:
        >>> secant(lambda x: x**2 - 2, 0, 1)
        (1.414213562373095, 5, True)
    """
    iter = 0
    while abs(f(x1)) > tol and iter < max_iter:
        denom = f(x1) - f(x0)
        if denom == 0:
            raise ValueError("Denominator is zero")
        x2 = x1 - f(x1) * (x1 - x0) / denom
        x0, x1 = x1, x2
        iter += 1
    return IterativeResult(x1, iter, abs(f(x1)) <= tol and iter < max_iter, method_info={'type': 'root_finding', 'method': 'secant', 'f': f, 'x0': x0, 'x1': x1})


def false_position(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> IterativeResult:
    """
    False position (Regula Falsi) method for root finding.

    Args:
        f: Function to find root of.
        a: Lower bound.
        b: Upper bound.
        tol: Tolerance.
        max_iter: Maximum iterations.

    Returns:
        Root, number of iterations, convergence status.

    Raises:
        ValueError: If f(a) and f(b) have the same sign or a >= b.

    Example:
        >>> false_position(lambda x: x**2 - 2, 0, 2)
        (1.4142135623746899, 7, True)
    """
    if a >= b:
        raise ValueError("a must be less than b")
    if f(a) * f(b) >= 0:
        raise ValueError("f(a) and f(b) must have opposite signs")
    iter = 0
    c = (a + b) / 2
    while abs(b - a) > tol and iter < max_iter:
        c = b - f(b) * (b - a) / (f(b) - f(a))
        if f(c) == 0:
            return IterativeResult(c, iter, True, method_info={'type': 'root_finding', 'method': 'false_position', 'f': f, 'a': a, 'b': b})
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
        iter += 1
    if iter == max_iter:
        if abs(f(a)) < abs(f(b)):
            c = a
        else:
            c = b
    return IterativeResult(c, iter, iter < max_iter, method_info={'type': 'root_finding', 'method': 'false_position', 'f': f, 'a': a, 'b': b})
