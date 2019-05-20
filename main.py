import numpy as np
import numpy.linalg as linal
import typing


# Ax = b

def seidel_method(*,
                  A: np.ndarray,
                  b: np.array,
                  w: float = 1,
                  eps: typing.Optional[float] = None,
                  x0: np.array) -> typing.AsyncGenerator[np.array]:
    """
    Implementation of seidel method.

    :return: sequence of approximation.
    """
    n, m = A.shape
    assert m == n
    B = np.zeros(shape=(n, m))
    c = np.zeros(shape=n)
    for i in range(0, n):
        for j in range(0, n):
            if i != j:
                B[i][j] = - A[i][j] / A[i][i]
        c[i] = b[i] / A[i][i]

    if eps is not None:
        B2 = np.triu(B, 1)
        eps = (1 - linal.det(B)) / linal.det(B2) * eps

    cur_x = x0
    yield x0
    while True:
        next_x = np.zeros(shape=n)
        for i in range(0, n):
            for j in range(0, n):
                if i != j:
                    next_x[i] += B[i][j] * (next_x[j] if j < i else cur_x[j])
        yield cur_x
        if eps is not None:
            if linal.norm(next_x - cur_x) < eps:
                break
        cur_x = next_x
