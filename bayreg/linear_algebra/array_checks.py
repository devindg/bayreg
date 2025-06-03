import numpy as np


def is_square(x: np.ndarray) -> bool:
    return x.shape[0] == x.shape[1]


def is_symmetric(x: np.ndarray) -> bool:
    return np.allclose(x, x.T)


def is_positive_definite(x: np.ndarray) -> bool:
    # noinspection PyBroadException
    # This function assumes x is symmetric.
    try:
        np.linalg.cholesky(x)
        return True
    except Exception:
        return False


def is_positive_semidefinite(x: np.ndarray, tol=1e-9) -> bool:
    return np.all(np.linalg.eigvalsh(x) >= -tol)
