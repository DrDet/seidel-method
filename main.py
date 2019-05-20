import numpy as np


# Ax = b

def seidel_method(*,
                  A: np.ndarray,
                  b: np.array,
                  w: float = 1,
                  x0: np.array) -> np.ndarray:
    """
    Implementation of seidel method.

    :return: sequence of approximation.
    """
    yield x0
