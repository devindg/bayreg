import numpy as np


def diag_2d(x: np.ndarray, as_col: bool = True) -> np.ndarray:
    if as_col:
        return np.diag(x).reshape(-1, 1)
    else:
        return np.diag(x).reshape(1, -1)


def replace_nan(x, val):
    z = x.copy()
    shape = z.shape
    z = z.ravel()
    z[np.isnan(z)] = val
    z = z.reshape(shape)
    return z


# Define matrix inversion routine based on dimension.
# Ideally, the function would look like _mat_inv(dim) -> f(z, s=dim)
# so that the function type would be returned based on instantiation of dim.
# Numba doesn't support returning functions it seems. Thus, instead of
# returning the function type once based on instantiation, using mat_inv(z)
# as defined below will evaluate True/False everytime in a loop and return
# the correct function type.

def mat_inv(z):
    dim = z.shape[0]
    if dim == 1:
        return 1. / z
    else:
        return np.linalg.lstsq(z, np.eye(dim))[0]


def svd(x):
    n, k = x.shape

    # Get SVD of design matrix
    if n >= k:
        U, s, Vt = np.linalg.svd(x, full_matrices=False)
        S = np.diag(s)
    else:
        U, s, Vt = np.linalg.svd(x, full_matrices=True)
        S = np.zeros((n, k))
        S[:n, :n] = np.diag(s)

    return U, S, Vt
