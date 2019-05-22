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
        if linal.norm(B, ord=1) > 1:
            print('WARNING: ||B|| >= 1 => estimation of eps is invalid')
        else:
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
            next_x[i] += (w - 1) * (next_x[i] - cur_x[i])  # relaxation offset
        yield cur_x
        if eps is not None:
            if linal.norm(next_x - cur_x) < eps:
                break
        cur_x = next_x


def relaxation_param_choosing(A, b, x0):
    eps = 1e-10
    for w in np.arange(0, 2, 0.1):
        seq = seidel_method(A=A, b=b, x0=x0, w=w, eps=eps)
        iters_cnt = 0
        for x in seq:
            iters_cnt += 1
        print(f'for w = {w.round(2)} iterations cnt = {iters_cnt}')


if __name__ == '__main__':
    n = 10
    A0 = gen.gen_diagonally_dominant_matrix(n)
    A1 = gen.gen_hilbert_matrix(n)
    A2 = gen.gen_random_matrix(n)
    x0 = np.array(np.random.random(n))
    b = np.array(np.random.random(n))
    print('>>> Initialization')
    print(f'A0 - diagonally dominant matrix :\n{A0.round(4)},')
    print(f'A1 - hilbert matrix :\n{A1.round(4)},')
    print(f'A2 - random matrix :\n{A2.round(4)},')
    print(f'b:   {b.round(4)},')
    print(f'x0:  {x0.round(4)}')

    print('\n>>> Test Seidel method on different matrix.')
    print("\nDiagonally dominant matrix")
    EPS = 1e-6
    I = 1000
    print(f'eps = {EPS}, max_iteration = {I}')
    i = 0
    for x in islice(seidel_method(A=A0, b=b, x0=x0, eps=EPS), I):
        i = i + 1
    print('> result')
    if i >= I:
        print('Not coverage.')
    print(f'iteration = {i}')
    print(f'||A(x*) - b|| = {linal.norm(A0.dot(x) - b)}')

    print("\nHilbert matrix")
    EPS = 1e-6
    I = 1000
    print(f'eps = {EPS}, max_iteration = {I}')
    i = 0
    for x in islice(seidel_method(A=A1, b=b, x0=x0, eps=EPS), I):
        i = i + 1
    print('> result')
    if i >= I:
        print('Not coverage.')
    print(f'iteration = {i}')
    print(f'||A(x*) - b|| = {linal.norm(A1.dot(x) - b)}')

    print("\nRandom matrix")
    EPS = 1e-6
    I = 100
    print(f'eps = {EPS}, max_iteration = {I}')
    i = 0
    for x in islice(seidel_method(A=A2, b=b, x0=x0, eps=EPS), I):
        i = i + 1
    print('> result')
    if i >= I:
        print('Not coverage.')
    print(f'iteration = {i}')
    print(f'||A(x*) - b|| = {linal.norm(A2.dot(x) - b)}')
    print(f'\n>>> Relaxation parameter choosing for diagonally dominant matrix:')
    relaxation_param_choosing(A0, b, x0)
