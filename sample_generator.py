# Matrix generator module

import numpy as np
import random as rnd


def gen_random_matrix(n):
    return np.array(np.random.random([n, n]))


def gen_diagonally_dominant_matrix(n):
    matrix = gen_random_matrix(n)
    sum = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                sum += abs(matrix[i][j])
    for i in range(n):
        matrix[i][i] = rnd.uniform(sum, 10 * sum)
    return matrix


def gen_hilbert_matrix(n):
    res = []
    for i in range(n):
        row = []
        for j in range(n):
            e = 1.0 / (i + j + 1)
            row.append(e)
        res.append(row)
    return np.array(res)

# print(gen_random_matrix(3))
# print(gen_diagonally_dominant_matrix(3))
# print(gen_hilbert_matrix(3))
