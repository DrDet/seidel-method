import numpy as np
import numpy.linalg as linal
import typing
from itertools import islice

import sample_generator as gen


# Ax = b

def seidel_method(*,
                  A: np.ndarray,
                  b: np.array,
                  w: float = 1,
                  eps: typing.Optional[float] = None,
                  x0: np.array) -> typing.Iterator[np.array]:
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
        eps = (1 - linal.norm(B, ord=1)) / linal.norm(B2, ord=1) * eps

    cur_x = x0
    yield x0
    while True:
        next_x = np.zeros(shape=n)
        for i in range(0, n):
            for j in range(0, n):
                if i != j:
                    next_x[i] += B[i][j] * (next_x[j] if j < i else cur_x[j])
            next_x[i] += c[i]
        yield cur_x
        if eps is not None:
            if linal.norm(next_x - cur_x) < eps:
                break
        cur_x = next_x


if __name__ == '__main__':
    n = 1
    A = gen.gen_random_matrix(n)
    x0 = np.array(np.random.random(n))
    b = np.array(np.random.random(n)) * 20
    print(f'A:\n{A.round(2)},\n'
          f'b: {b.round(2)},\n'
          f'x0: {x0.round(2)}')
    i = 0
    for x in islice(seidel_method(A=A, b=b, x0=x0), 10):
        i = i + 1
        print(f'{i}:{x}')
    print(f'Ax*: {A.dot(x)}')
