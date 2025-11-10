from typing import List, Tuple


def linear_interpolation(points: List[Tuple[float, float]], x: float) -> float:
    """
    Linear interpolation at point x given list of (x_i, y_i).

    Args:
        points: List of (x, y) pairs.
        x: Point to evaluate.

    Returns:
        Interpolated value.

    Raises:
        ValueError: If less than 2 points or x out of range or duplicate x.

    Example:
        >>> linear_interpolation([(0, 0), (1, 1)], 0.5)
        0.5
    """
    if len(points) < 2:
        raise ValueError("At least two points required")
    points = sorted(points)  # Sort by x
    xs = [p[0] for p in points]
    if len(set(xs)) != len(xs):
        raise ValueError("Duplicate x values")
    if x < xs[0] or x > xs[-1]:
        raise ValueError("x out of range")
    for i in range(len(points) - 1):
        if xs[i] <= x <= xs[i + 1]:
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            return y1 + (y2 - y1) * (x - x1) / (x2 - x1)
    raise ValueError("Interpolation failed")  # Should not reach here


def lagrange_interpolation(points: List[Tuple[float, float]], x: float) -> float:
    """
    Lagrange interpolation at point x given list of (x_i, y_i).

    Args:
        points: List of (x, y) pairs.
        x: Point to evaluate.

    Returns:
        Interpolated value.

    Raises:
        ValueError: If less than 1 point or duplicate x.

    Example:
        >>> lagrange_interpolation([(0, 0), (1, 1), (2, 4)], 1.5)
        2.25
    """
    n = len(points)
    if n < 1:
        raise ValueError("At least one point required")
    xs = [p[0] for p in points]
    if len(set(xs)) != n:
        raise ValueError("Duplicate x values")
    result = 0.0
    for i in range(n):
        xi, yi = points[i]
        term = yi
        for j in range(n):
            if i != j:
                xj = points[j][0]
                term *= (x - xj) / (xi - xj)
        result += term
    return result


def newton_divided_difference(points: List[Tuple[float, float]], x: float) -> float:
    """
    Newton's divided difference interpolation at point x given list of (x_i, y_i).

    Args:
        points: List of (x, y) pairs.
        x: Point to evaluate.

    Returns:
        Interpolated value.

    Raises:
        ValueError: If less than 1 point or duplicate x.

    Example:
        >>> newton_divided_difference([(0, 0), (1, 1), (2, 4)], 1.5)
        2.25
    """
    n = len(points)
    if n < 1:
        raise ValueError("At least one point required")
    points = sorted(points)  # Sort by x for consistency
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    if len(set(xs)) != n:
        raise ValueError("Duplicate x values")

    # Build divided difference table
    dd = [[0.0] * n for _ in range(n)]
    for i in range(n):
        dd[i][0] = ys[i]
    for j in range(1, n):
        for i in range(n - j):
            dd[i][j] = (dd[i + 1][j - 1] - dd[i][j - 1]) / (xs[i + j] - xs[i])

    # Evaluate
    result = dd[0][0]
    prod = 1.0
    for i in range(1, n):
        prod *= x - xs[i - 1]
        result += dd[0][i] * prod
    return result
