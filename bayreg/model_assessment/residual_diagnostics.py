import numpy as np
from scipy.stats import t as t_dist
from numpy.linalg import svd
from typing import NamedTuple
from linear_algebra.array_operations import mat_inv


class Outliers(NamedTuple):
    int_stud_resid: np.ndarray
    ext_stud_resid: np.ndarray
    ext_stud_resid_pval: np.ndarray
    resid: np.ndarray
    num_eff_params: float


def get_projection_matrix_diagonal(predictors, coeff_prec_prior=None):
    n, k = predictors.shape
    x = predictors

    U, s, Vt = svd(x, full_matrices=False)
    S = np.diag(s)
    proj_mat_diag = np.empty((n, 1))
    if coeff_prec_prior is None:
        W = None
        for i in range(n):
            proj_mat_diag[i] = U[i, :] @ U.T[:, i]
    else:
        US = U @ S
        W = mat_inv(S**2 + Vt @ coeff_prec_prior @ Vt.T)
        for i in range(n):
            proj_mat_diag[i] = US[i, :] @ W @ US.T[:, i]

    return proj_mat_diag, U, S, Vt, W


def studentized_residuals(response, predictors, coeff_prec_prior=None, coeff_mean_prior=None):
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

    if coeff_prec_prior is not None and coeff_mean_prior is None:
        raise ValueError("coeff_mean_prior must be provided if coeff_prec_prior is provided.")

    if coeff_prec_prior is None and coeff_mean_prior is not None:
        raise ValueError("coeff_prec_prior must be provided if coeff_mean_prior is provided.")

    if k >= n:
        if coeff_mean_prior is None:
            raise ValueError("The number of predictors exceeds the number of observations. "
                             "Ordinary least squares is not feasible in this case. Provide "
                             "mean and precision priors for the predictor coefficients "
                             "if you want to proceed with an overdetermined system, i.e., "
                             "a rank-deficient design matrix. A strong precision prior may "
                             "be necessary to avoid non-positive degrees of freedom for the "
                             "Student-t distribution associated with externalized residuals.")

    h, U, S, Vt, W = get_projection_matrix_diagonal(predictors=x, coeff_prec_prior=coeff_prec_prior)
    p = sum(h)

    if coeff_mean_prior is None:
        b = Vt.T @ np.diag(np.diag(S) ** (-1)) @ U.T @ y
    else:
        b = (Vt.T @ W @ Vt) @ (x.T @ y + coeff_prec_prior @ coeff_mean_prior)

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
