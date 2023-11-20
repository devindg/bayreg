import numpy as np
from scipy.stats import t as t_dist
from numpy.linalg import svd
from typing import NamedTuple
from bayreg.linear_algebra.array_operations import mat_inv


class Outliers(NamedTuple):
    int_stud_resid: np.ndarray
    ext_stud_resid: np.ndarray
    ext_stud_resid_pval: np.ndarray
    resid: np.ndarray
    num_eff_params: float


def get_projection_matrix_diagonal(predictors, prior_coeff_prec=None):
    n, k = predictors.shape
    x = predictors

    U, s, Vt = svd(x, full_matrices=False)
    S = np.diag(s)
    proj_mat_diag = np.empty((n, 1))
    if prior_coeff_prec is None:
        W = None
        for i in range(n):
            proj_mat_diag[i] = U[i, :] @ U.T[:, i]
    else:
        US = U @ S
        W = mat_inv(S**2 + Vt @ prior_coeff_prec @ Vt.T)
        for i in range(n):
            proj_mat_diag[i] = US[i, :] @ W @ US.T[:, i]

    return proj_mat_diag, U, S, Vt, W


def studentized_residuals(response, predictors, prior_coeff_prec=None, prior_coeff_mean=None):
    y = response.copy()
    x = predictors.copy()

    if y.ndim not in (1, 2):
        raise ValueError("The response array must have dimension 1 or 2.")
    elif y.ndim == 1:
        y = y.reshape(-1, 1)
    else:
        if all(i > 1 for i in y.shape):
            raise ValueError("The response array must have shape (1, n) or (n, 1), "
                             "where n is the number of observations. Both the row and column "
                             "count exceed 1.")
        else:
            y = y.reshape(-1, 1)

    if x.ndim not in (1, 2):
        raise ValueError('The predictors array must have dimension 1 or 2. Dimension is 0.')
    elif x.ndim == 1:
        x = x.reshape(-1, 1)
    else:
        if 1 in x.shape:
            x = x.reshape(-1, 1)

    n, m, k = y.shape[0], y.shape[1], x.shape[1]
    if n != x.shape[0]:
        raise ValueError("The response and predictors arrays do not have the same number "
                         "of observations.")

    if prior_coeff_prec is not None and prior_coeff_mean is None:
        raise ValueError("prior_coeff_mean must be provided if prior_coeff_prec is provided.")

    if prior_coeff_prec is None and prior_coeff_mean is not None:
        raise ValueError("prior_coeff_prec must be provided if prior_coeff_mean is provided.")

    if k >= n:
        if prior_coeff_mean is None:
            raise ValueError("The number of predictors exceeds the number of observations. "
                             "Ordinary least squares is not feasible in this case. Provide "
                             "mean and precision priors for the predictor coefficients "
                             "if you want to proceed with an overdetermined system, i.e., "
                             "a rank-deficient design matrix. A strong precision prior may "
                             "be necessary to avoid non-positive degrees of freedom for the "
                             "Student-t distribution associated with externalized residuals.")

    h, U, S, Vt, W = get_projection_matrix_diagonal(predictors=x, prior_coeff_prec=prior_coeff_prec)
    p = sum(h)

    if prior_coeff_mean is None:
        b = Vt.T @ np.diag(np.diag(S) ** (-1)) @ U.T @ y
    else:
        b = (Vt.T @ W @ Vt) @ (x.T @ y + prior_coeff_prec @ prior_coeff_mean)

    response_pred = x @ b
    resid = y - response_pred

    if m in (0, 1):
        mse = np.sum(resid ** 2) / (n - p)
        int_stud_resid = resid / np.sqrt(mse * (1 - h))
    else:
        mse = np.sum(resid ** 2, axis=0) / (n - p) * np.ones((n, m))
        int_stud_resid = resid / np.sqrt(mse * (1 - h)[:, np.newaxis])

    ext_stud_resid = int_stud_resid * np.sqrt((n - p - 1) / (n - p - int_stud_resid ** 2))
    ext_stud_resid_pval = 1 - t_dist.cdf(abs(ext_stud_resid), df=n - p - 1)

    return Outliers(int_stud_resid=int_stud_resid,
                    ext_stud_resid=ext_stud_resid,
                    ext_stud_resid_pval=ext_stud_resid_pval,
                    resid=resid,
                    num_eff_params=p)
