import warnings
from typing import NamedTuple, Union
import numpy as np
import pandas as pd
from scipy.stats import invgamma, t
from bayreg.linear_algebra.array_checks import is_positive_semidefinite, is_symmetric
from bayreg.linear_algebra.array_operations import mat_inv, svd
from bayreg.model_assessment.performance import (
    oos_error,
    mean_squared_prediction_error,
    r_squared,
    r_squared_classic,
    watanabe_akaike,
)
from sklearn.preprocessing import StandardScaler


def valid_design_matrix(
        regress_design_matrix: np.ndarray,
        const_tol: float = 1e-6
):
    x = np.array(regress_design_matrix)
    num_obs, num_pred = x.shape
    cols = [j for j in range(num_pred)]

    # Initialize valid columns
    valid_cols = cols

    # Identify intercept, if any
    all_ones = np.all(x == 1, axis=0)
    if np.any(all_ones):
        intercept_index = np.where(all_ones)[0]

        if len(intercept_index) > 1:
            redundant_intercept = intercept_index[1:]
            print(f"Columns with indexes {redundant_intercept} are redundant "
                  f"intercepts and will be ignored"
                  )
        else:
            redundant_intercept = []

        intercept_index = intercept_index[0]  # First index with an intercept
        valid_cols = [j for j in valid_cols if j not in redundant_intercept]
    else:
        intercept_index = None

    # Identify duplicate columns, if any
    _, non_redundant_cols = np.unique(x, axis=1, return_index=True)
    if len(cols) != len(non_redundant_cols):
        redundant_cols = [j for j in cols if j not in non_redundant_cols]
        print(f"Columns with indexes {redundant_cols} are redundant and will be ignored.")
        valid_cols = [j for j in valid_cols if j not in redundant_cols]

    # Identify constant columns, if any
    const_cols = np.where(np.std(x, axis=0) <= const_tol)[0]
    if len(const_cols) > 0:
        if intercept_index is None:
            print(f"Columns with indexes {const_cols} are non-intercept constants and will be ignored.")
            valid_cols = [j for j in valid_cols if j not in const_cols]
        else:
            non_intercept_const_cols = [j for j in const_cols if j != intercept_index]

            if len(non_intercept_const_cols) > 0:
                print(f"Columns with indexes {non_intercept_const_cols} are "
                      f"non-intercept constants and will be ignored."
                      )

            valid_cols = [j for j in valid_cols if j not in non_intercept_const_cols]

    x = x[:, valid_cols]

    return x, valid_cols, intercept_index


def default_zellner_g(x: np.ndarray) -> float:
    n, k = x.shape

    return max([n, k ** 2])


def zellner_covariance(
        x: np.ndarray,
        zellner_g: float,
        max_mat_cond_index: float
) -> np.ndarray:
    num_coeff = x.shape[1]
    if num_coeff > 1:
        ss = StandardScaler()
        x_z = ss.fit_transform(x)
        variable_cols = ~np.all(x_z == 0, axis=0)

        if np.sum(~variable_cols) > 1:
            raise AssertionError(
                "The design matrix cannot have more than one constant."
            )

        xtx = (x_z.T @ x_z)[np.ix_(variable_cols, variable_cols)]
        k_z = x_z.shape[1]
        eig_vals = np.linalg.eigvalsh(xtx)
        eig_cond_index = np.sqrt(np.max(eig_vals) / eig_vals)
        eig_cond_index = np.nan_to_num(eig_cond_index, nan=np.inf)

        if np.any(eig_cond_index > max_mat_cond_index):
            w = 0
        else:
            det_sign, log_det = np.linalg.slogdet(xtx)
            avg_determ = (det_sign * np.exp(log_det)) ** (1 / k_z)
            if not np.isfinite(avg_determ):
                avg_determ = 0.
            avg_trace = np.trace(xtx) / k_z
            w = avg_determ / avg_trace

        xtx = x[:, variable_cols].T @ x[:, variable_cols]
        prior_coeff_cov = mat_inv(
            1 / zellner_g * (w * xtx + (1 - w) * np.diag(np.diag(xtx)))
        )

        # If a constant is present, insert an approximately flat
        # prior for the intercept.
        if np.sum(~variable_cols) == 1:
            max_diag = np.max(np.diag(prior_coeff_cov))
            prior_coeff_cov = np.insert(
                prior_coeff_cov,
                ~variable_cols,
                0,
                axis=0
            )
            prior_coeff_cov = np.insert(
                prior_coeff_cov,
                ~variable_cols,
                0,
                axis=1
            )
            intercept_var = np.min([1e4, max_diag * 1e4])
            prior_coeff_cov[~variable_cols, ~variable_cols] = intercept_var

    else:
        prior_coeff_cov = mat_inv(
            1 / zellner_g * x.T @ x
        )

    return prior_coeff_cov


class Posterior(NamedTuple):
    num_post_samp: int
    post_coeff_mean: np.ndarray
    post_coeff_cov: np.ndarray
    ninvg_post_coeff_cov: np.ndarray
    post_coeff: np.ndarray
    post_err_var_shape: float
    post_err_var_scale: float
    post_err_var: np.ndarray


class Prior(NamedTuple):
    prior_coeff_mean: np.ndarray
    prior_coeff_cov: np.ndarray
    prior_err_var_shape: Union[int, float]
    prior_err_var_scale: Union[int, float]
    zellner_g: Union[int, float]


class ConjugateBayesianLinearRegression:
    """
    Conjugate Bayesian linear regression procedure. The model is:

        Estimating equation:
        y = X.beta + error
        error | X, beta ~ N(0, sigma^2*I)

        Likelihood:
        y | X, beta, sigma^2 ~ N(X.beta, sigma^2*I_n)

        Prior:
        beta ~ N(prior_beta_mean, prior_beta_cov)
        sigma^2 ~ Inverse-Gamma(prior_err_var_shape, prior_err_var_scale)

        Posterior:
        beta, sigma^2 | X, y ~ N-IG(post_beta_mean, ninvg_post_beta_cov, post_err_var_shape, post_err_var_scale)
        beta | sigma^2, X, y ~ N(post_beta_mean, sigma^2 * ninvg_post_beta_cov)
        sigma^2 | X, y ~ N-IG(post_err_var_shape, post_err_var_scale)
        beta | X, y ~ Multivariate-t(df = 2 * post_err_var_shape,
                                     mean = post_coeff_mean,
                                     cov = post_err_var_scale / post_err_var_shape * ninvg_post_beta_cov)

    where

        - y is an n x 1 response vector;
        - X is an n x k design matrix, with k being the number of predictors;
        - error is an unobserved n x 1 vector of perturbations;
        - beta is a k x 1 vector of coefficients;
        - sigma^2 is a scalar representing the homoskedastic variance of error | X, beta;
        - I_n is the n x n identity matrix;
        - prior_beta_mean is a k x 1 vector representing a prior about the mean of beta;
        - prior_beta_cov is a k x k matrix representing a prior about the covariance of beta;
        - prior_err_var_shape is a scalar representing a prior about the shape of sigma^2's distribution;
        - prior_err_var_scale is a scalar representing a prior about the scale of sigma^2's distribution;

        - N(a,b) represents a normally distributed random variable with mean a and variance b;
        - IG(a,b) represents an inverse-gamma distributed random variable with shape a and scale b;
        - N-IG(a,b,c,d) represents a normal inverse-gamma distributed multivariate random variable
            with parameters a, b, c, and d

    """

    def __init__(
            self,
            response: Union[np.ndarray, list, tuple, pd.Series, pd.DataFrame],
            predictors: Union[np.ndarray, list, tuple, pd.Series, pd.DataFrame],
            seed: int = 42,
    ):
        """

        :param response:
        :param predictors:
        :param seed:
        """

        self.response = response
        self.predictors = predictors
        self.response_type = type(response)
        self.predictors_type = type(predictors)
        self.valid_predictors = None
        self.has_intercept = None
        self.fit_intercept = None
        self.intercept_index = None
        self.response_index = None
        self.response_name = None
        self.num_obs = None
        self.num_coeff = None
        self.num_predictors = None
        self.predictors_names = None
        self.prior = None
        self.posterior = None
        self.post_pred_dist = None
        self.standardize_data = None
        self.data_transformer = None
        self._back_transform_means = None
        self._back_transform_sds = None
        self._intercept_name = '__INTERCEPT__'

        # Define new index for intercept if intercept is present
        self._intercept_index = 0

        if not isinstance(seed, int):
            raise TypeError("seed must be an integer.")

        if not 0 < seed < 2 ** 32 - 1:
            raise ValueError("seed must be an integer between 0 and 2**32 - 1.")

        self.seed = seed

    def _posterior_exists_check(self):
        if self.posterior is None:
            raise AttributeError(
                "No posterior distribution was found. The fit() method must be called."
            )

        return

    def _post_pred_dist_exists_check(self):
        if self.post_pred_dist is None:
            raise AttributeError(
                "No posterior predictive distribution was found. "
                "The fit() and posterior_predictive_distribution() methods "
                "must be called."
            )

        return

    def _process_response(
            self,
            response
    ):
        y = response

        if not isinstance(y, (np.ndarray, list, tuple, pd.Series, pd.DataFrame)):
            raise TypeError(
                "The response array must be a Numpy array, list, tuple, Pandas Series, \n"
                "or Pandas DataFrame."
            )
        else:
            if isinstance(y, (list, tuple)):
                y = np.asarray(y, dtype=np.float64)
            else:
                y = y.copy()

            if isinstance(y, (pd.Series, pd.DataFrame)):
                if isinstance(y, pd.DataFrame):
                    self.response_name = y.columns.tolist()
                elif isinstance(y, pd.Series):
                    self.response_name = [y.name]

                self.response_index = y.index
                y = y.to_numpy()

        if y.ndim not in (1, 2):
            raise ValueError("The response array must have dimension 1 or 2.")
        elif y.ndim == 1:
            y = y.reshape(-1, 1)
        else:
            if all(i > 1 for i in y.shape):
                raise ValueError(
                    "The response array must have shape (1, n) or (n, 1), "
                    "where n is the number of observations. Both the row and column "
                    "count exceed 1."
                )
            else:
                y = y.reshape(-1, 1)

        if np.any(np.isnan(y)):
            raise ValueError("The response array cannot have null values.")

        if np.any(np.isinf(y)):
            raise ValueError("The response array cannot have Inf and/or -Inf values.")

        self.num_obs = y.shape[0]

        if self.response_index is None:
            self.response_index = np.arange(y.shape[0])

        return y

    def _process_predictors(
            self,
            predictors
    ):
        x = predictors

        if not isinstance(
                x, (np.ndarray, list, tuple, pd.Series, pd.DataFrame)
        ):
            raise TypeError(
                "The predictors array must be a Numpy array, list, tuple, Pandas Series, "
                "or Pandas DataFrame."
            )
        else:
            if isinstance(x, (list, tuple)):
                x = np.asarray(x, dtype=np.float64)
            else:
                x = x.copy()

            # Check if response and predictors are the same data type.
            if isinstance(x, (np.ndarray, list, tuple)) and not isinstance(
                    x, (np.ndarray, list, tuple)
            ):
                raise TypeError(
                    "The response array provided is a NumPy array, list, or tuple, but the predictors "
                    "array is not. Object types must match."
                )

            if isinstance(x, (pd.Series, pd.DataFrame)) and not isinstance(
                    x, (pd.Series, pd.DataFrame)
            ):
                raise TypeError(
                    "The response array provided is a Pandas Series/DataFrame, but the predictors "
                    "array is not. Object types must match."
                )

            # Get predictor names if a Pandas object
            if isinstance(x, (pd.Series, pd.DataFrame)):
                if not np.all(x.index == x.index):
                    raise ValueError("The response and predictors indexes must match.")

                if isinstance(x, pd.DataFrame):
                    self.predictors_names = x.columns.tolist()
                elif isinstance(x, pd.Series):
                    self.predictors_names = [x.name]

                x = x.to_numpy()
            else:
                self.predictors_names = [f"x{i + 1}" for i in range(x.shape[1])]

        # Check dimensions
        if x.ndim not in (1, 2):
            raise ValueError("The predictors array must have dimension 1 or 2.")
        elif x.ndim == 1:
            x = x.reshape(-1, 1)

        # Check for nan and inf values
        if np.any(np.isnan(x)):
            raise ValueError("The predictors array cannot have null values.")

        if np.any(np.isinf(x)):
            raise ValueError("The predictors array cannot have Inf and/or -Inf values.")

        # Check if the number of observations conforms to response
        if x.shape[0] != self.num_obs:
            raise ValueError(
                "The number of observations in the predictors array must match "
                "the number of observations in the response array."
            )

        # Get valid design matrix.
        x, valid_cols, intercept_index = valid_design_matrix(x)

        self.valid_predictors = valid_cols

        if intercept_index is not None:
            self.has_intercept = True
            self.intercept_index = intercept_index
            warnings.warn(
                "An intercept was detected in the design matrix. Note that any prior "
                "given to the intercept will be ignored because the data are either "
                "standardized or centered (which eliminates the intercept) for estimation. "
                "Effectively, the intercept is treated with a vague Normal prior and "
                "derived after the model is fitted."
            )
        else:
            self.has_intercept = False
            self.intercept_index = None

        self.num_coeff = x.shape[1]
        self.predictors_names = [
            p
            for i, p in enumerate(self.predictors_names)
            if i in valid_cols
        ]
        self.num_predictors = self.num_coeff - int(self.has_intercept)

        # Warn about model stability if the number of predictors exceeds the number of observations
        if self.num_coeff > self.num_obs:
            warnings.warn(
                "The number of predictors exceeds the number of observations. "
                "Results will be sensitive to choice of priors."
            )

        return x

    def prepare_data(
            self,
            standardize_data: bool,
            fit_intercept: bool
    ):
        y = self._process_response(response=self.response)
        x = self._process_predictors(predictors=self.predictors)

        # Store untransformed versions of response and predictors
        self.response = y
        self.predictors = x

        if self.has_intercept:
            x = np.delete(x, self.intercept_index, axis=1)

        if fit_intercept:
            self.predictors = np.insert(x, self._intercept_index, 1., axis=1)

        if standardize_data:
            ss = StandardScaler(
                with_mean=True,
                with_std=True
            )
        else:
            ss = StandardScaler(
                with_mean=True,
                with_std=False
            )

        data = np.c_[y, x]
        data = ss.fit_transform(data)
        self.data_transformer = ss
        y, x = data[:, 0:1], data[:, 1:]

        # Capture means and standard deviations for back-transformations
        self._back_transform_params()

        return y, x

    def _back_transform_params(self):
        data_transformer = self.data_transformer
        with_sd = data_transformer.with_std
        scales = data_transformer.scale_
        means = data_transformer.mean_
        m_y, m_x = means[0], means[1:]

        if with_sd:
            sd_y, sd_x = scales[0], scales[1:]
        else:
            sd_y, sd_x = 1., np.ones_like(means[1:])

        self._back_transform_means = np.concatenate(([m_y], m_x))
        self._back_transform_sds = np.concatenate(([sd_y], sd_x))

        return

    def _reconfig_prior_coeff_mean(
            self,
            prior_coeff_mean: np.ndarray
    ):
        pcm = prior_coeff_mean.copy()

        if self.has_intercept:
            # Standardizing or centering reparameterizes the intercept.
            # Effectively, the implied intercept turns into a
            # parameter with a vague prior. This will be enforced
            # with a zero-mean and high variance.

            # Delete row associated with intercept index
            pcm = np.delete(pcm, self.intercept_index, axis=0)

        if self.fit_intercept:
            pcm = np.insert(pcm, self._intercept_index, 0, axis=0)
            self.prior = self.prior._replace(prior_coeff_mean=pcm)

            # Now delete the intercept prior for estimation
            pcm = np.delete(
                pcm,
                self._intercept_index,
                axis=0
            )

        if self.standardize_data:
            data_scales = self.data_transformer.scale_
            sd_y, sd_x = data_scales[0], data_scales[1:]
            W = np.diag(sd_x)
            pcm = W @ pcm / sd_y

        return pcm

    def _reconfig_prior_coeff_cov(
            self,
            prior_coeff_cov,
            is_custom
    ):
        pcc = prior_coeff_cov.copy()
        var_y = np.var(self.response)
        sd_x = self._back_transform_sds[1:]

        if is_custom:
            max_pcc_diag = np.max(np.diag(pcc))
            if self.has_intercept:
                pcc = np.delete(pcc, self.intercept_index, axis=0)
                pcc = np.delete(pcc, self.intercept_index, axis=1)
        else:
            W = np.diag(1 / sd_x)
            pcc = W @ pcc @ W
            max_pcc_diag = np.max(np.diag(pcc))

        if self.fit_intercept:
            pcc = np.insert(pcc, self._intercept_index, 0, axis=0)
            pcc = np.insert(pcc, self._intercept_index, 0, axis=1)

            # Define prior variance for intercept
            if var_y < max_pcc_diag:
                intercept_var = max_pcc_diag * 1e4
            else:
                intercept_var = np.min([var_y, max_pcc_diag * 1e4])

            pcc[self._intercept_index, self._intercept_index] = intercept_var
            self.prior = self.prior._replace(prior_coeff_cov=pcc)

            # Now delete the intercept prior for estimation
            pcc = np.delete(pcc, self._intercept_index, axis=0)
            pcc = np.delete(pcc, self._intercept_index, axis=1)

        if is_custom:
            if self.standardize_data:
                W = np.diag(sd_x)
                pcc = W @ pcc @ W
        else:
            W = np.diag(sd_x)
            pcc = W @ pcc @ W

        pcp = mat_inv(pcc)

        return pcc, pcp

    def _reconfig_prior_err_var_scale(
            self,
            prior_err_var_scale
    ):
        if self.standardize_data:
            sd_y = self.data_transformer.scale_[0]

            return prior_err_var_scale / sd_y ** 2
        else:
            return prior_err_var_scale

    def _back_transform_posterior(
            self,
            predictors: np.ndarray,
            post_coeff_mean,
            post_coeff_cov,
            post_coeff,
            post_err_var_shape,
            post_err_var_scale,
            post_err_var,
            ninvg_post_coeff_cov
    ):
        x = predictors.copy()

        if self.standardize_data:
            scales = self._back_transform_sds
            sd_y, sd_x = scales[0], scales[1:]
            W = np.diag(1 / sd_x)
            x = x @ mat_inv(W)
            post_coeff_mean = (W @ post_coeff_mean) * sd_y
            post_coeff_cov = (W @ post_coeff_cov @ W) * sd_y ** 2
            post_coeff = (post_coeff @ W) * sd_y
            post_err_var_scale = post_err_var_scale * sd_y ** 2
            post_err_var = post_err_var * sd_y ** 2
            ninvg_post_coeff_cov = W @ ninvg_post_coeff_cov @ W

        if self.fit_intercept:
            x = np.insert(x, self._intercept_index, 1., axis=1)
            means = self._back_transform_means
            m_y, m_x = means[0], means[1:]
            post_intercept_mean = (
                np.atleast_2d(
                    m_y - post_coeff_mean.flatten() @ m_x
                )
            )
            post_coeff_mean = np.insert(
                post_coeff_mean,
                self._intercept_index,
                post_intercept_mean,
                axis=0
            )
            post_intercept = m_y - post_coeff @ m_x
            post_coeff = np.insert(
                post_coeff,
                self._intercept_index,
                post_intercept,
                axis=1
            )
            prior_coeff_prec = mat_inv(self.prior.prior_coeff_cov)
            ninvg_post_coeff_cov = mat_inv(x.T @ x + prior_coeff_prec)
            post_coeff_cov = post_err_var_scale / post_err_var_shape * ninvg_post_coeff_cov

        return (
            post_coeff_mean,
            post_coeff_cov,
            post_coeff,
            post_err_var_scale,
            post_err_var,
            ninvg_post_coeff_cov
        )

    def fit(
            self,
            num_post_samp: int = 1000,
            standardize_data: bool = False,
            fit_intercept: bool = True,
            prior_coeff_mean: Union[list, tuple, np.ndarray] = None,
            prior_coeff_cov: Union[list, tuple, np.ndarray] = None,
            prior_err_var_shape: Union[int, float] = None,
            prior_err_var_scale: Union[int, float] = None,
            zellner_g: Union[int, float] = None,
            max_mat_cond_index: Union[int, float] = 30.,
    ) -> Posterior:
        """
        :param num_post_samp:
        :param standardize_data:
        :param fit_intercept:
        :param prior_coeff_mean:
        :param prior_coeff_cov:
        :param prior_err_var_shape:
        :param prior_err_var_scale:
        :param zellner_g:
        :param max_mat_cond_index:
        :return:
        """

        self.standardize_data = standardize_data
        self.fit_intercept = fit_intercept
        y, x = self.prepare_data(
            standardize_data=standardize_data,
            fit_intercept=fit_intercept
        )
        n, k = self.num_obs, self.num_coeff

        # Get SVD of design matrix
        U, S, Vt = svd(x)
        StS = S.T @ S

        # Check shape prior for error variance
        if prior_err_var_shape is not None:
            if (
                    isinstance(prior_err_var_shape, (int, float))
                    and prior_err_var_shape > 0
            ):
                pass
            else:
                raise ValueError(
                    "prior_err_var_shape must be a strictly positive integer or float."
                )
        else:
            prior_err_var_shape = 0.01

        # Check scale prior for error variance
        if prior_err_var_scale is not None:
            if (
                    isinstance(prior_err_var_scale, (int, float))
                    and prior_err_var_scale > 0
            ):
                pass
            else:
                raise ValueError(
                    "prior_err_var_scale must be a strictly positive integer or float."
                    "prior_err_var_scale must be a strictly positive integer or float."
                )
        else:
            ss = StandardScaler()
            ss.fit(y)
            sd_y = ss.scale_[0]
            prior_err_var_scale = (0.01 * sd_y) ** 2

        # Check prior mean for regression coefficients
        if prior_coeff_mean is not None:
            if not isinstance(prior_coeff_mean, (list, tuple, np.ndarray)):
                raise TypeError(
                    "prior_coeff_mean must be a list, tuple, or NumPy array."
                )
            else:
                if isinstance(prior_coeff_mean, (list, tuple)):
                    prior_coeff_mean = np.asarray(prior_coeff_mean, dtype=np.float64)
                else:
                    prior_coeff_mean = prior_coeff_mean.astype(float)

                prior_coeff_mean = prior_coeff_mean[self.valid_predictors]

                if prior_coeff_mean.ndim not in (1, 2):
                    raise ValueError("prior_coeff_mean must have dimension 1 or 2.")
                elif prior_coeff_mean.ndim == 1:
                    prior_coeff_mean = prior_coeff_mean.reshape(k, 1)
                else:
                    pass

                if not prior_coeff_mean.shape == (k, 1):
                    raise ValueError(
                        f"prior_coeff_mean must have shape ({k}, 1)."
                    )
                if np.any(np.isnan(prior_coeff_mean)):
                    raise ValueError("prior_coeff_mean cannot have NaN values.")
                if np.any(np.isinf(prior_coeff_mean)):
                    raise ValueError(
                        "prior_coeff_mean cannot have Inf and/or -Inf values."
                    )
        else:
            prior_coeff_mean = np.zeros((k, 1))

        # Check prior covariance matrix for regression coefficients
        if prior_coeff_cov is not None:
            is_custom_cov = True

            if not isinstance(prior_coeff_cov, (list, tuple, np.ndarray)):
                raise TypeError(
                    "prior_coeff_cov must be a list, tuple, or NumPy array."
                )
            else:
                if isinstance(prior_coeff_cov, (list, tuple)):
                    prior_coeff_cov = np.atleast_2d(
                        np.asarray(prior_coeff_cov, dtype=np.float64)
                    )
                else:
                    prior_coeff_cov = np.atleast_2d(prior_coeff_cov.astype(float))

                # noinspection PyTypeChecker
                prior_coeff_cov = prior_coeff_cov[
                    np.ix_(self.valid_predictors, self.valid_predictors)
                ]

                if prior_coeff_cov.ndim != 2:
                    raise ValueError("prior_coeff_cov must have dimension 2.")
                if not prior_coeff_cov.shape == (k, k):
                    raise ValueError(
                        f"prior_coeff_cov must have shape ({k}, {k})."
                    )
                if not is_positive_semidefinite(prior_coeff_cov):
                    raise ValueError(
                        "prior_coeff_cov must be a positive definite matrix."
                    )
                if not is_symmetric(prior_coeff_cov):
                    raise ValueError("prior_coeff_cov must be a symmetric matrix.")

        else:
            is_custom_cov = False
            """
            If predictors are specified without a precision prior, Zellner's g-prior will
            be enforced. Specifically, 1 / g * (w * dot(X.T, X) + (1 - w) * diag(dot(X.T, X))),
            where g = n / prior_obs, prior_obs is the number of prior observations given to the
            regression coefficient mean prior (i.e., it controls how much weight is given to the
            mean prior), n is the number of observations, X is the design matrix, and
            diag(dot(X.T, X)) is a diagonal matrix with the diagonal elements matching those of
            dot(X.T, X). The addition of the diagonal matrix to dot(X.T, X) is to guard against
            singularity (i.e., a design matrix that is not full rank). The weighting controlled
            by w is set to 0.5.
            """

            if zellner_g is not None:
                if isinstance(zellner_g, (int, float, np.int64)) and zellner_g > 0:
                    pass
                else:
                    raise ValueError(
                        "zellner_g must be a strictly positive integer or float."
                    )
            else:
                zellner_g = default_zellner_g(x)

            # Get weights for untransformed and diagonalized precision matrix.
            # Use the ratio of the average determinant to average trace
            # to measure the stability of the design matrix. The
            # lower this ratio is, the less stable the design matrix is,
            # in which case, more weight will be given to a diagonal precision
            # matrix.

            prior_coeff_cov = zellner_covariance(
                x=x,
                zellner_g=zellner_g,
                max_mat_cond_index=max_mat_cond_index
            )

        self.prior = Prior(
            prior_coeff_mean=prior_coeff_mean,
            prior_coeff_cov=prior_coeff_cov,
            prior_err_var_shape=prior_err_var_shape,
            prior_err_var_scale=prior_err_var_scale,
            zellner_g=zellner_g,
        )

        # Reconfigure certain priors depending on data standardization and intercept
        # treatment.
        prior_coeff_mean = self._reconfig_prior_coeff_mean(
            prior_coeff_mean=prior_coeff_mean
        )
        prior_coeff_cov, prior_coeff_prec = self._reconfig_prior_coeff_cov(
            prior_coeff_cov=prior_coeff_cov,
            is_custom=is_custom_cov
        )
        prior_err_var_scale = self._reconfig_prior_err_var_scale(
            prior_err_var_scale=prior_err_var_scale
        )

        # Posterior values for coefficient mean, coefficient covariance matrix,
        # error variance shape, and error variance scale.

        # Note: storing the normal-inverse-gamma precision and covariance matrices
        # could cause memory problems if the coefficient vector has high dimension.
        # May want to reconsider temporary storage of these matrices.
        ninvg_post_coeff_prec = Vt.T @ (StS + Vt @ prior_coeff_prec @ Vt.T) @ Vt
        ninvg_post_coeff_cov = Vt.T @ mat_inv(StS + Vt @ prior_coeff_prec @ Vt.T) @ Vt
        post_coeff_mean = ninvg_post_coeff_cov @ (
                x.T @ y + prior_coeff_prec @ prior_coeff_mean
        )
        post_err_var_shape = prior_err_var_shape + 0.5 * n
        post_err_var_scale = (
                prior_err_var_scale
                + 0.5
                * (
                        y.T @ y
                        + prior_coeff_mean.T @ prior_coeff_prec @ prior_coeff_mean
                        - post_coeff_mean.T @ ninvg_post_coeff_prec @ post_coeff_mean
                )
        )[0][0]

        if post_err_var_scale < 0:
            post_err_var_scale = (
                    prior_err_var_scale
                    + 0.5
                    * (
                            (y - x @ prior_coeff_mean).T
                            @ mat_inv(np.eye(n) + x @ prior_coeff_cov @ x.T)
                            @ (y - x @ prior_coeff_mean)
                    )
            )[0][0]

        # Marginal posterior distribution for variance parameter
        post_err_var = invgamma.rvs(
            post_err_var_shape,
            scale=post_err_var_scale,
            size=(num_post_samp, 1),
            random_state=self.seed,
        )

        # Joint posterior distribution for coefficients and variance parameter
        svd_post_coeff_mean = Vt @ post_coeff_mean.flatten()
        svd_ninvg_post_coeff_cov = Vt @ ninvg_post_coeff_cov @ Vt.T
        post_coeff_cov = (
                post_err_var_scale / post_err_var_shape * svd_ninvg_post_coeff_cov
        )

        post_coeff = np.empty((num_post_samp, x.shape[1]))
        rng = np.random.default_rng(self.seed)
        for s in range(num_post_samp):
            cond_post_coeff_cov = post_err_var[s][0] * svd_ninvg_post_coeff_cov
            post_coeff[s] = rng.multivariate_normal(
                mean=svd_post_coeff_mean, cov=cond_post_coeff_cov
            )

        # This will happen if the number of observations is 1
        if post_coeff.ndim == 1:
            post_coeff = post_coeff.reshape((num_post_samp, Vt.shape[0]))

        # Back-transform parameters from SVD to the original scale
        post_coeff_cov = Vt.T @ post_coeff_cov @ Vt
        post_coeff = post_coeff @ Vt

        # Back-transform posterior if applicable
        (
            post_coeff_mean,
            post_coeff_cov,
            post_coeff,
            post_err_var_scale,
            post_err_var,
            ninvg_post_coeff_cov
        ) = self._back_transform_posterior(
            predictors=x,
            post_coeff_mean=post_coeff_mean,
            post_coeff_cov=post_coeff_cov,
            post_coeff=post_coeff,
            post_err_var_shape=post_err_var_shape,
            post_err_var_scale=post_err_var_scale,
            post_err_var=post_err_var,
            ninvg_post_coeff_cov=ninvg_post_coeff_cov
        )

        self.posterior = Posterior(
            num_post_samp=num_post_samp,
            post_coeff_cov=post_coeff_cov,
            ninvg_post_coeff_cov=ninvg_post_coeff_cov,
            post_coeff_mean=post_coeff_mean,
            post_coeff=post_coeff,
            post_err_var_shape=post_err_var_shape,
            post_err_var_scale=post_err_var_scale,
            post_err_var=post_err_var,
        )

        if fit_intercept:
            self.predictors_names = [self._intercept_name] + self.predictors_names

        if not self.has_intercept and fit_intercept:
            self.num_coeff += 1

        return self.posterior

    def predict(
            self,
            predictors: Union[np.ndarray, list, tuple, pd.Series, pd.DataFrame, None] = None,
            mean_only: bool = False,
    ):
        """

        :param predictors:
        :param mean_only:
        :return:
        """
        if predictors is None:
            x = self.predictors
        else:
            # Check if the object type is valid
            if not isinstance(
                    predictors, (np.ndarray, list, tuple, pd.Series, pd.DataFrame)
            ):
                raise TypeError(
                    "The predictors array must be a NumPy array, list, tuple, Pandas Series, \n"
                    "or Pandas DataFrame."
                )
            else:
                if isinstance(predictors, (list, tuple)):
                    x = np.asarray(predictors, dtype=np.float64)
                else:
                    x = predictors.copy()

                # If Pandas type, grab index and column names
                if isinstance(x, (pd.Series, pd.DataFrame)):
                    x = x.to_numpy()

                # Check dimensions
                if x.ndim not in (1, 2):
                    raise ValueError("The predictors array must have dimension 1 or 2.")
                elif x.ndim == 1:
                    x = x.reshape(-1, 1)

                if np.isnan(x).any():
                    raise ValueError("The predictors array cannot have null values.")

                if np.isinf(x).any():
                    raise ValueError(
                        "The predictors array cannot have Inf and/or -Inf values."
                    )
                x = x[:, self.valid_predictors]

            if self.has_intercept:
                x = np.delete(x, self.intercept_index, axis=1)

            if self.fit_intercept:
                x = np.insert(x, self._intercept_index, 1., axis=1)

        self._posterior_exists_check()
        n = x.shape[0]

        # Marginal posterior predictive distribution
        post_coeff_mean = self.posterior.post_coeff_mean
        post_err_var_shape = self.posterior.post_err_var_shape
        post_err_var_scale = self.posterior.post_err_var_scale
        ninvg_post_coeff_cov = self.posterior.ninvg_post_coeff_cov

        V = (
                post_err_var_scale
                / post_err_var_shape
                * (1 + np.array([z.T @ ninvg_post_coeff_cov @ z for z in x]))
        )
        post_mean = x @ post_coeff_mean

        if not mean_only:
            post = t.rvs(
                df=2 * post_err_var_shape,
                loc=post_mean.flatten(),
                scale=V ** 0.5,
                size=(self.posterior.num_post_samp, n),
                random_state=self.seed,
            )
        else:
            post = None

        return post, post_mean

    def posterior_predictive_distribution(self):
        self._posterior_exists_check()
        self.post_pred_dist = self.predict()[0]

        return self.post_pred_dist

    def posterior_summary(self, cred_int_level: float = 0.05):
        self._posterior_exists_check()

        if not 0 < cred_int_level < 1:
            raise ValueError("The credible interval level must be a value in (0, 1).")

        posterior = self.posterior
        num_coeff = self.num_coeff
        pred_names = self.predictors_names
        lb, ub = 0.5 * cred_int_level, 1 - 0.5 * cred_int_level

        # Coefficients
        post_coeff = posterior.post_coeff
        post_coeff_mean = posterior.post_coeff_mean.flatten()
        post_coeff_std = np.sqrt(np.diag(posterior.post_coeff_cov))
        post_coeff_lb = np.quantile(post_coeff, lb, axis=0)
        post_coeff_ub = np.quantile(post_coeff, ub, axis=0)
        coeff_prob_pos_post = (
                np.sum((post_coeff > 0) * 1, axis=0) / posterior.num_post_samp
        )

        # Error variance
        post_err_var = posterior.post_err_var
        post_err_var_shape = posterior.post_err_var_shape
        post_err_var_scale = posterior.post_err_var_scale

        if post_err_var_shape > 1:
            post_err_var_mean = post_err_var_scale / (post_err_var_shape - 1)
        else:
            post_err_var_mean = None

        if post_err_var_shape > 2:
            post_err_var_std = np.sqrt(
                post_err_var_scale ** 2
                / ((post_err_var_shape - 1) ** 2 * (post_err_var_shape - 2))
            )
        else:
            post_err_var_std = None

        post_err_var_mode = post_err_var_scale / (post_err_var_shape + 1)
        post_err_var_lb = np.quantile(post_err_var, lb)
        post_err_var_ub = np.quantile(post_err_var, ub)

        summary = {}
        for k in range(num_coeff):
            summary[f"Post.Mean[Coeff.{pred_names[k]}]"] = post_coeff_mean[k]
            summary[f"Post.StdDev[Coeff.{pred_names[k]}]"] = post_coeff_std[k]
            summary[f"Post.CredInt.LB[Coeff.{pred_names[k]}]"] = post_coeff_lb[k]
            summary[f"Post.CredInt.UB[Coeff.{pred_names[k]}]"] = post_coeff_ub[k]
            summary[f"Post.Prob.Positive[Coeff.{pred_names[k]}]"] = coeff_prob_pos_post[
                k
            ]

        summary["Post.Mean[ErrorVariance]"] = post_err_var_mean
        summary["Post.Mode[ErrorVariance]"] = post_err_var_mode
        summary["Post.StdDev[ErrorVariance]"] = post_err_var_std
        summary["Post.CredInt.LB[ErrorVariance]"] = post_err_var_lb
        summary["Post.CredInt.UB[ErrorVariance]"] = post_err_var_ub

        return summary

    def waic(self):
        self._posterior_exists_check()
        x = self.predictors
        post_resp_mean = self.posterior.post_coeff @ x.T

        return watanabe_akaike(
            response=self.response.T,
            post_resp_mean=post_resp_mean,
            post_err_var=self.posterior.post_err_var,
        )

    def mspe(self):
        self._posterior_exists_check()
        self._post_pred_dist_exists_check()

        return mean_squared_prediction_error(
            response=self.response,
            post_pred_dist=self.post_pred_dist,
            post_err_var=self.posterior.post_err_var,
        )

    def r_sqr(self):
        self._posterior_exists_check()
        self._post_pred_dist_exists_check()

        return r_squared(
            post_pred_dist=self.post_pred_dist, post_err_var=self.posterior.post_err_var
        )

    def r_sqr_classic(self):
        self._posterior_exists_check()
        mean_prediction = self.predictors @ self.posterior.post_coeff_mean

        return r_squared_classic(
            response=self.response, mean_prediction=mean_prediction
        )

    def oos_error(self,
                  leverage_predictors: Union[np.ndarray, None] = None,
                  leverage_prior_coeff_cov: Union[np.ndarray, None] = None
                  ):
        """
        :param leverage_predictors: When a data transformation is made, such as centering,
        this affects the projection matrix, which in turn could affect computation of leverage.
        Because leverage is used to compute LOO error, it may be desirable to compute LOO
        on the untransformed data. This argument can be used for passing a modified design
        matrix that will adjust the leverage calculation and provide LOO error on the
        untransformed dependent variable. If provided, leverage_prior_coeff_cov must also be
        provided.
        :param leverage_prior_coeff_cov: When a data transformation is made, such as centering,
        this affects the projection matrix, which in turn could affect computation of leverage.
        Because leverage is used to compute LOO error, it may be desirable to compute LOO
        on the untransformed data. This argument is used to accommodate a modified design
        matrix passed to argument 'leverage_predictors'. If provided, leverage_predictors
        must also be provided.
        """
        self._posterior_exists_check()

        response = self.response
        predictors = self.predictors
        prior_coeff_prec = mat_inv(self.prior.prior_coeff_cov)

        if leverage_predictors is not None and leverage_prior_coeff_cov is None:
            raise ValueError(
                "leverage_predictors is not None but leverage_prior_coeff_cov is None. "
                "Both leverage_predictors and leverage_prior_coeff_cov must be provided."
            )

        if leverage_predictors is None and leverage_prior_coeff_cov is not None:
            raise ValueError(
                "leverage_predictors is None but leverage_prior_coeff_cov is not None. "
                "Both leverage_predictors and leverage_prior_coeff_cov must be provided."
            )

        if leverage_prior_coeff_cov is not None:
            leverage_prior_coeff_prec = mat_inv(leverage_prior_coeff_cov)
        else:
            leverage_prior_coeff_prec = None

        post_coeff_mean = self.posterior.post_coeff_mean

        return oos_error(
            response=response,
            predictors=predictors,
            mean_coeff=post_coeff_mean,
            prior_coeff_prec=prior_coeff_prec,
            leverage_predictors=leverage_predictors,
            leverage_prior_coeff_prec=leverage_prior_coeff_prec
        )
