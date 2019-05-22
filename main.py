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
            pass  # print('WARNING: ||B|| >= 1 => estimation of eps is invalid')
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


def relaxation_param_choosing(*, A, b, x0, name: typing.Optional[str] = None, I=1000, EPS=1e-6):
    if name is None:
        print('>>> Relaxation parameter choosing')
    else:
        print(f'>>> Relaxation parameter choosing for {name}')

    for w in np.arange(0.9, 1.1, 0.01):
        iters_cnt = 0
        for x in islice(seidel_method(A=A, b=b, x0=x0, w=w, eps=EPS), I):
            iters_cnt += 1
        if iters_cnt == I:
            print(f'for w = {w.round(2)} not coverage: ||A(x*) - b|| = {linal.norm(A.dot(x) - b)}, iteration = {iters_cnt}, x = {x}')
        else:
            print(f'for w = {w.round(2)} iteration = {iters_cnt}')
    print()


def measure(*, A, b, x0, EPS=1e-6, I=1000, name: str = 'measurement'):
    print(f'>>> {name}')
    print(f'eps = {EPS}, max_iteration = {I}')
    i = 0
    for x in islice(seidel_method(A=A, b=b, x0=x0, eps=EPS), I):
        i = i + 1
    print('> result')
    if i >= I:
        print(f'Not coverage: ||A(x*) - b|| = {linal.norm(A.dot(x) - b)}')
    print(f'iteration = {i}')
    print(f'x* = {x}')
    print()


def main_demo():
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
    measure(A=A0, b=b, x0=x0, name='Diagonally dominant matrix')
    measure(A=A1, b=b, x0=x0, name='Hilbert matrix')
    measure(A=A2, b=b, x0=x0, name='Random matrix')

    relaxation_param_choosing(A=A0, b=b, x0=x0, name='diagonally dominant matrix')


def main_measurement():
    good = np.array([
        [101.126133, 14.177853, 5.570273, 32.704660, 28.181857],
        [46.534970, 531.885739, 144.802686, 139.902591, 149.429588],
        [55.014745, 47.646827, 256.697287, 39.422978, 34.689687],
        [2.581816, 13.412359, 7.144080, 113.486129, 14.905638],
        [110.216649, 34.889634, 1.970613, 7.504255, 989.672536],
    ])

    random = np.array([
        [38.410138, 53.634632, 57.547258, 60.574633, 61.109409],
        [17.457198, 66.641469, 45.628101, 35.860134, 6.646870],
        [61.160772, 78.548540, 80.458022, 52.468398, 30.893063],
        [87.721305, 72.940947, 95.634175, 92.646077, 54.396008],
        [15.091494, 46.745994, 24.297494, 86.361705, 21.750511]
    ])

    bad = np.array([
        [0.333333, 0.250000, 0.200000, 0.166667, 0.142857],
        [0.250000, 0.200000, 0.166667, 0.142857, 0.125000],
        [0.200000, 0.166667, 0.142857, 0.125000, 0.111111],
        [0.166667, 0.142857, 0.125000, 0.111111, 0.100000],
        [0.142857, 0.125000, 0.111111, 0.100000, 0.090909],
    ])
    b = np.array([78.185980, 84.521714, 99.682760, 99.969787, 61.538438])
    x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    print(f'good:\n {good}')
    print(f'bad:\n {bad}')
    print(f'random:\n {random}')
    print(f'b = {b}')
    print(f'x0 = {x0}')

    measure(A=good, b=b, x0=x0, name='good')
    measure(A=random, b=b, x0=x0, I=2000, name='random')
    measure(A=bad, b=b, x0=x0, name='bad')
    relaxation_param_choosing(A=good, b=b, x0=x0, name='good')
    relaxation_param_choosing(A=random, b=b, x0=x0, name='random')
    relaxation_param_choosing(A=bad, b=b, x0=x0, name='bad')


if __name__ == '__main__':
    main_measurement()
