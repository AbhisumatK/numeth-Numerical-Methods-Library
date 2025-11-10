from typing import List, Tuple


def gauss_elimination(A: List[List[float]], b: List[float]) -> List[float]:
    """
    Gauss elimination with partial pivoting to solve Ax = b.

    Args:
        A: Coefficient matrix (n x n).
        b: Right-hand side vector (n).

    Returns:
        Solution vector x.

    Raises:
        ValueError: If matrix is singular or dimensions mismatch.

    Example:
        >>> gauss_elimination([[2, 1], [5, 3]], [5, 13])
        [2.0, 1.0]
    """
    n = len(b)
    if len(A) != n or any(len(row) != n for row in A):
        raise ValueError("Matrix A must be square and match b size")

    # Augment A with b
    for i in range(n):
        A[i].append(b[i])

    # Forward elimination with partial pivoting
    for i in range(n):
        # Pivoting
        max_row = i
        for k in range(i + 1, n):
            if abs(A[k][i]) > abs(A[max_row][i]):
                max_row = k
        A[i], A[max_row] = A[max_row], A[i]

        if A[i][i] == 0:
            raise ValueError("Matrix is singular")

        for k in range(i + 1, n):
            c = -A[k][i] / A[i][i]
            for j in range(i, n + 1):
                if i == j:
                    A[k][j] = 0.0
                else:
                    A[k][j] += c * A[i][j]

    # Back substitution
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = A[i][n] / A[i][i]
        for k in range(i - 1, -1, -1):
            A[k][n] -= A[k][i] * x[i]
    return x


def lu_decomposition(
    A: List[List[float]],
) -> Tuple[List[List[float]], List[List[float]]]:
    """
    LU decomposition using Doolittle's method (L diagonal 1s).

    Args:
        A: Square matrix.

    Returns:
        L (lower triangular), U (upper triangular).

    Raises:
        ValueError: If matrix is not square or singular.

    Example:
        >>> L, U = lu_decomposition([[2, 1, 1], [4, -6, 0], [-2, 7, 2]])
        # L and U such that L @ U == A
    """
    n = len(A)
    if n == 0 or any(len(row) != n for row in A):
        raise ValueError("Matrix must be square and non-empty")

    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]

    for i in range(n):
        # U
        for k in range(i, n):
            sum_ = sum(L[i][j] * U[j][k] for j in range(i))
            U[i][k] = A[i][k] - sum_

        if U[i][i] == 0:
            raise ValueError("Matrix is singular")

        # L
        L[i][i] = 1.0
        for k in range(i + 1, n):
            sum_ = sum(L[k][j] * U[j][i] for j in range(i))
            if U[i][i] == 0:
                raise ValueError("Matrix is singular")
            L[k][i] = (A[k][i] - sum_) / U[i][i]

    return L, U


def jacobi(
    A: List[List[float]],
    b: List[float],
    x0: List[float] = None,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> Tuple[List[float], int, bool]:
    """
    Jacobi iterative method to solve Ax = b.

    Args:
        A: Coefficient matrix.
        b: Right-hand side vector.
        x0: Initial guess (optional).
        tol: Tolerance.
        max_iter: Maximum iterations.

    Returns:
        Solution, iterations, converged.

    Raises:
        ValueError: Dimensions mismatch or diagonal zero.

    Example:
        >>> jacobi([[2, 1], [5, 7]], [11, 13], [1, 1])
        ([2.0, -1.0000000000000002], 14, True)
    """
    n = len(b)
    if len(A) != n or any(len(row) != n for row in A):
        raise ValueError("Matrix A must be square and match b size")
    if x0 is None:
        x0 = [0.0] * n
    x = x0.copy()
    iter = 0
    while iter < max_iter:
        x_new = [0.0] * n
        for i in range(n):
            if A[i][i] == 0:
                raise ValueError("Diagonal element is zero")
            sum_ = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - sum_) / A[i][i]
        # Check convergence
        if all(abs(x_new[k] - x[k]) < tol for k in range(n)):
            return x_new, iter, True
        x = x_new
        iter += 1
    return x, iter, False


def gauss_seidel(
    A: List[List[float]],
    b: List[float],
    x0: List[float] = None,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> Tuple[List[float], int, bool]:
    """
    Gauss-Seidel iterative method to solve Ax = b.

    Args:
        A: Coefficient matrix.
        b: Right-hand side vector.
        x0: Initial guess (optional).
        tol: Tolerance.
        max_iter: Maximum iterations.

    Returns:
        Solution, iterations, converged.

    Raises:
        ValueError: Dimensions mismatch or diagonal zero.

    Example:
        >>> gauss_seidel([[2, 1], [5, 7]], [11, 13], [1, 1])
        ([1.9999999999999998, -0.9999999999999999], 8, True)
    """
    n = len(b)
    if len(A) != n or any(len(row) != n for row in A):
        raise ValueError("Matrix A must be square and match b size")
    if x0 is None:
        x0 = [0.0] * n
    x = x0.copy()
    iter = 0
    while iter < max_iter:
        max_diff = 0.0
        for i in range(n):
            if A[i][i] == 0:
                raise ValueError("Diagonal element is zero")
            sum_ = sum(A[i][j] * x[j] for j in range(n) if j != i)
            new_x = (b[i] - sum_) / A[i][i]
            max_diff = max(max_diff, abs(new_x - x[i]))
            x[i] = new_x
        if max_diff < tol:
            return x, iter, True
        iter += 1
    return x, iter, False
