from typing import Callable, Tuple


def golden_section_search(
    f: Callable[[float], float], a: float, b: float, tol: float = 1e-6
) -> Tuple[float, float]:
    """
    Golden section search for minimization in 1D.

    Args:
        f: Function to minimize.
        a: Lower bound.
        b: Upper bound.
        tol: Tolerance.

    Returns:
        Minimizer, minimum value.

    Raises:
        ValueError: If a >= b.

    Example:
        >>> golden_section_search(lambda x: (x - 2)**2, 0, 5)
        (2.000003999516325, 1.598392519461786e-11)
    """
    if a >= b:
        raise ValueError("a must be less than b")
    gr = (1 + 5**0.5) / 2  # Golden ratio
    c = b - (b - a) / gr
    d = a + (b - a) / gr
    while abs(c - d) > tol:
        if f(c) < f(d):
            b = d
        else:
            a = c
        c = b - (b - a) / gr
        d = a + (b - a) / gr
    x_min = (b + a) / 2
    return x_min, f(x_min)


def newton_optimization(
    f: Callable[[float], float],
    df: Callable[[float], float],
    d2f: Callable[[float], float],
    x0: float,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> Tuple[float, float, int, bool]:
    """
    Newton's method for optimization in 1D (minimization).

    Args:
        f: Function to minimize.
        df: First derivative.
        d2f: Second derivative.
        x0: Initial guess.
        tol: Tolerance.
        max_iter: Maximum iterations.

    Returns:
        Minimizer, minimum value, iterations, converged.

    Raises:
        ValueError: If second derivative is zero.

    Example:
        >>> newton_optimization(lambda x: x**2, lambda x: 2*x, lambda x: 2, 10)
        (0.0, 0.0, 1, True)
    """
    x = x0
    iter = 0
    while abs(df(x)) > tol and iter < max_iter:
        d2 = d2f(x)
        if d2 == 0:
            raise ValueError("Second derivative is zero")
        x = x - df(x) / d2
        iter += 1
    return x, f(x), iter, abs(df(x)) <= tol and iter < max_iter
