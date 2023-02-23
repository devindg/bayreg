import numpy as np
import pandas as pd
import warnings
from scipy.stats import multivariate_t, invgamma
from linear_algebra.array_checks import is_symmetric, is_positive_definite
from linear_algebra.array_operations import mat_inv
from linear_algebra.vectorized import vec_norm
from model_assessment.in_sample_fit import watanabe_akaike, mean_squared_prediction_error, r_squared
from numba import njit
from typing import Union, NamedTuple


class Posterior(NamedTuple):
    num_post_samp: int
    coeff_mean_post: np.ndarray
    coeff_cov_post: np.ndarray
    coeff_post: np.ndarray
    err_var_shape_post: np.ndarray
    err_var_scale_post: np.ndarray
    err_var_post: np.ndarray


class Prior(NamedTuple):
    coeff_mean_prior: np.ndarray
    coeff_cov_prior: np.ndarray
    err_var_shape_prior: np.ndarray
    err_var_scale_prior: np.ndarray
    zellner_prior_obs: float


@njit
def _set_numba_seed(value):
    np.random.seed(value)


class ConjugateBayesianLinearRegression:
    """
    A Bayesian statistical procedure for linear regression. The model is:

        Estimating equation:
        y = X*beta + epsilon
        epsilon | X, beta ~ N(0, sigma^2*I)

        Likelihood:
        y | X, beta, sigma^2 ~ N(X*beta, sigma^2*I)

        Prior:
        beta | sigma^2 ~ N(psi, sigma^2*LAMBDA)
        sigma^2 ~ IG(alpha, tau)
        => (beta, sigma^2) ~ N-IG(psi, LAMBDA, alpha, tau)

    where

        - y is an n x 1 outcome vector;
        - X is an n x k design matrix, with k being the number of predictors;
        - epsilon is an n x 1 model error vector;
        - beta is a k x 1 vector of parameters;
        - sigma^2 is a scalar representing the homoskedastic variance of y;
        - I is the n x n identity matrix;
        - psi is a k x 1 vector representing a prior about the mean of beta;
        - LAMBDA is a k x k matrix representing a prior about the variance of beta;
        - alpha is a scalar representing a prior about the shape of sigma^2's distribution;
        - tau is a scalar representing a prior about the scale of sigma^2's distribution;

        - N(a,b) represents a normally distributed random variable with mean a and variance b;
        - IG(a,b) represents an inverse-gamma distributed random variable with shape a and scale b;
        - N-IG(a,b,c,d) represents a normal inverse-gamma distributed multivariate random variable
            with parameters a, b, c, and d

    The N-IG prior for the parameters beta and sigma^2 is a conjugate prior.
    Consequently, the posterior distribution for beta and sigma^2 is also N-IG.
    """

    def __init__(self,
                 response: Union[np.ndarray, pd.Series, pd.DataFrame],
                 predictors: Union[np.ndarray, pd.Series, pd.DataFrame],
                 seed: int = None):
        """

        :param response:
        :param predictors:
        :param seed:
        """

        self.response_index = None
        self.predictors_names = None

        # CHECK AND PREPARE RESPONSE DATA
        # -- dimension and null/inf checks
        if not isinstance(response, (np.ndarray, pd.Series, pd.DataFrame)):
            raise TypeError("The response array must be a Numpy array, Pandas Series, or Pandas DataFrame.")
        else:
            resp = response.copy()
            if isinstance(response, (pd.Series, pd.DataFrame)):
                if isinstance(response, pd.Series):
                    self.response_name = [response.name]
                else:
                    self.response_name = response.columns.values.tolist()

                self.response_index = response.index
                resp = resp.to_numpy()

        if resp.ndim not in (1, 2):
            raise ValueError('The response array must have dimension 1 or 2.')
        elif resp.ndim == 1:
            resp = resp.reshape(-1, 1)
        else:
            if all(i > 1 for i in resp.shape):
                raise ValueError('The response array must have shape (1, n) or (n, 1), '
                                 'where n is the number of observations. Both the row and column '
                                 'count exceed 1.')
            else:
                resp = resp.reshape(-1, 1)

        if np.any(np.isnan(resp)):
            raise ValueError('The response array cannot have null values.')
        if np.any(np.isinf(resp)):
            raise ValueError('The response array cannot have Inf and/or -Inf values.')

        if resp.shape[0] < 2:
            raise ValueError('At least two observations are required to fit a model.')

        # CHECK AND PREPARE PREDICTORS DATA
        if not isinstance(predictors, (np.ndarray, pd.Series, pd.DataFrame)):
            raise TypeError("The predictors array must be a Numpy array, Pandas Series, or Pandas DataFrame.")
        else:
            pred = predictors.copy()
            # -- check if response and predictors are same date type.
            if isinstance(response, np.ndarray) and not isinstance(predictors, np.ndarray):
                raise TypeError('The response array provided is a NumPy array, but the predictors '
                                'array is not. Object types must match.')

            if (isinstance(response, (pd.Series, pd.DataFrame)) and
                    not isinstance(predictors, (pd.Series, pd.DataFrame))):
                raise TypeError('The response array provided is a Pandas Series/DataFrame, but the predictors '
                                'array is not. Object types must match.')

            # -- get predictor names if a Pandas object and sort index
            if isinstance(predictors, (pd.Series, pd.DataFrame)):
                if not (predictors.index == response.index).all():
                    raise ValueError('The response and predictors indexes must match.')

                if isinstance(predictors, pd.Series):
                    self.predictors_names = [predictors.name]
                else:
                    self.predictors_names = predictors.columns.values.tolist()

                pred = pred.to_numpy()

        # -- dimension and null/inf checks
        if pred.ndim not in (1, 2):
            raise ValueError('The predictors array must have dimension 1 or 2. Dimension is 0.')
        elif pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        else:
            if 1 in pred.shape:
                pred = pred.reshape(-1, 1)

        if np.any(np.isnan(pred)):
            raise ValueError('The predictors array cannot have null values.')
        if np.any(np.isinf(pred)):
            raise ValueError('The predictors array cannot have Inf and/or -Inf values.')

        # -- conformable number of observations
        if pred.shape[0] != resp.shape[0]:
            raise ValueError('The number of observations in the predictors array must match '
                             'the number of observations in the response array.')

        # -- check if design matrix has a constant
        pred_column_mean = np.mean(pred, axis=0)
        pred_offset = pred - pred_column_mean[np.newaxis, :]
        diag_pred_offset_squared = np.diag(pred_offset.T @ pred_offset)
        if np.any(diag_pred_offset_squared == 0):
            self.has_constant = True
            if np.sum(diag_pred_offset_squared == 0) > 1:
                raise ValueError('More than one column is a constant value. Only one column can be constant.')
            self.constant_index = np.argwhere(diag_pred_offset_squared == 0)[0][0]
        else:
            self.has_constant = False

        # -- warn about model stability if the number of predictors exceeds number of observations
        if pred.shape[1] > pred.shape[0]:
            warnings.warn('The number of predictors exceeds the number of observations. '
                          'Results will be sensitive to choice of priors.')

        if seed is not None:
            if not isinstance(seed, int):
                raise TypeError('seed must be an integer.')
            if not 0 < seed < 2 ** 32 - 1:
                raise ValueError('seed must be an integer between 0 and 2**32 - 1.')
            _set_numba_seed(seed)  # for Numba JIT functions
            np.random.seed(seed)

        self.response = resp
        self.num_obs = resp.shape[0]
        self.predictors = pred
        self.num_coeff = pred.shape[1]
        self.num_predictors = self.num_coeff - self.has_constant
        self.prior = None
        self.posterior = None
        self.post_pred_dist = None

        if self.response_index is None:
            self.response_index = np.arange(resp.shape[0])

        # Create variable names for predictors, if applicable
        if self.predictors_names is None:
            self.predictors_names = [f"x{i + 1}" for i in range(self.num_coeff)]

    @staticmethod
    def _svd(predictors):
        _, s, Vt = np.linalg.svd(predictors, full_matrices=False)
        S = np.diag(s)
        return S, Vt

    def fit(self,
            num_post_samp=1000,
            coeff_mean_prior=None,
            coeff_cov_prior=None,
            err_var_shape_prior=None,
            err_var_scale_prior=None,
            zellner_prior_obs=None):
        """

        :param num_post_samp:
        :param coeff_mean_prior:
        :param coeff_cov_prior:
        :param err_var_shape_prior:
        :param err_var_scale_prior:
        :param zellner_prior_obs:
        :return:
        """

        y, x, n, k = self.response, self.predictors, self.num_obs, self.num_coeff
        S, Vt = self._svd(x)
        XtX = Vt.T @ (S ** 2) @ Vt

        # Check shape prior for error variance
        if err_var_shape_prior is not None:
            if not err_var_shape_prior > 0:
                raise ValueError('err_var_shape_prior must be strictly positive.')
        else:
            err_var_shape_prior = 1e-6

        # Check scale prior for error variance
        if err_var_scale_prior is not None:
            if not err_var_scale_prior > 0:
                raise ValueError('err_var_scale_prior must be strictly positive.')
        else:
            err_var_scale_prior = 1e-6

        # Check prior mean for regression coefficients
        if coeff_mean_prior is not None:
            if not coeff_mean_prior.shape == (self.num_coeff, 1):
                raise ValueError(f'coeff_mean_prior must have shape ({self.num_coeff}, 1).')
            if np.any(np.isnan(coeff_mean_prior)):
                raise ValueError('coeff_mean_prior cannot have NaN values.')
            if np.any(np.isinf(coeff_mean_prior)):
                raise ValueError('coeff_mean_prior cannot have Inf and/or -Inf values.')
        else:
            coeff_mean_prior = np.zeros((self.num_coeff, 1))

        # Check prior covariance matrix for regression coefficients
        if coeff_cov_prior is not None:
            if not coeff_cov_prior.shape == (self.num_coeff, self.num_coeff):
                raise ValueError(f'coeff_cov_prior must have shape ({self.num_coeff}, '
                                 f'{self.num_coeff}).')
            if not is_positive_definite(coeff_cov_prior):
                raise ValueError('coeff_cov_prior must be a positive definite matrix.')
            if not is_symmetric(coeff_cov_prior):
                raise ValueError('coeff_cov_prior must be a symmetric matrix.')

            coeff_prec_prior = mat_inv(coeff_cov_prior)
        else:
            '''
            If predictors are specified without a precision prior, Zellner's g-prior will
            be enforced. Specifically, 1 / g * (w * dot(X.T, X) + (1 - w) * diag(dot(X.T, X))), 
            where g = n / prior_obs, prior_obs is the number of prior observations given to the 
            regression coefficient mean prior (i.e., it controls how much weight is given to the 
            mean prior), n is the number of observations, X is the design matrix, and 
            diag(dot(X.T, X)) is a diagonal matrix with the diagonal elements matching those of
            dot(X.T, X). The addition of the diagonal matrix to dot(X.T, X) is to guard against 
            singularity (i.e., a design matrix that is not full rank). The weighting controlled 
            by w is set to 0.5.
            '''

            if zellner_prior_obs is None:
                zellner_prior_obs = 1e-6

            w = 0.5
            coeff_prec_prior = zellner_prior_obs / n * (w * XtX + (1 - w) * np.diag(np.diag(XtX)))
            coeff_cov_prior = mat_inv(coeff_prec_prior)

        self.prior = Prior(coeff_mean_prior=coeff_mean_prior,
                           coeff_cov_prior=coeff_cov_prior,
                           err_var_shape_prior=err_var_shape_prior,
                           err_var_scale_prior=err_var_scale_prior,
                           zellner_prior_obs=zellner_prior_obs)

        # Posterior values for coefficient mean, coefficient covariance matrix,
        # error variance shape, and error variance scale.

        # Note: storing the normal-inverse-gamma precision and covariance matrices
        # could cause memory problems if the coefficient vector has high dimension.
        # May want to reconsider temporary storage of these matrices.
        ninvg_coeff_prec_post = Vt.T @ (S**2 + Vt @ coeff_prec_prior @ Vt.T) @ Vt
        ninvg_coeff_cov_post = Vt.T @ mat_inv(S**2 + Vt @ coeff_prec_prior @ Vt.T) @ Vt
        coeff_mean_post = ninvg_coeff_cov_post @ (x.T @ y + coeff_prec_prior @ coeff_mean_prior)
        err_var_shape_post = err_var_shape_prior + 0.5 * n
        err_var_scale_post = (err_var_scale_prior +
                              0.5 * (y.T @ y
                                     + coeff_mean_prior.T @ coeff_prec_prior @ coeff_mean_prior
                                     - coeff_mean_post.T @ ninvg_coeff_prec_post @ coeff_mean_post))[0][0]

        # Marginal posterior distribution for variance parameter
        err_var_post = invgamma.rvs(err_var_shape_post,
                                    scale=err_var_scale_post,
                                    size=(num_post_samp, 1))

        # Marginal posterior distribution for coefficients
        coeff_cov_post = err_var_scale_post / err_var_shape_post * (Vt @ ninvg_coeff_cov_post @ Vt.T)

        # Check if the covariance matrix corresponding to the coefficients' marginal
        # posterior distribution is ill-conditioned.
        if not is_positive_definite(coeff_cov_post):
            raise ValueError("The covariance matrix corresponding to the coefficients' "
                             "marginal posterior distribution (multivariate Student-t) "
                             "is not positive definite. Try scaling the predictors, the "
                             "response, or both to eliminate the possibility of an "
                             "ill-conditioned matrix.")

        coeff_post = multivariate_t(df=2 * err_var_shape_post,
                                    loc=Vt @ coeff_mean_post.squeeze(),
                                    shape=coeff_cov_post,
                                    allow_singular=True).rvs(num_post_samp)

        # Back-transform parameters from SVD to original scale
        coeff_cov_post = Vt.T @ coeff_cov_post @ Vt
        coeff_post = (Vt.T @ coeff_post.T).T

        self.posterior = Posterior(num_post_samp=num_post_samp,
                                   coeff_cov_post=coeff_cov_post,
                                   coeff_mean_post=coeff_mean_post,
                                   coeff_post=coeff_post,
                                   err_var_shape_post=err_var_shape_post,
                                   err_var_scale_post=err_var_scale_post,
                                   err_var_post=err_var_post)

        self.post_pred_dist = self.predict(self.predictors)

        # # Computations without SVD
        # ninvg_coeff_prec_post = x.T @ x + coeff_prec_prior
        # ninvg_coeff_cov_post = mat_inv(ninvg_coeff_prec_post)
        # coeff_mean_post = np.linalg.solve(ninvg_coeff_prec_post, x.T @ y + coeff_prec_prior @ coeff_mean_prior)
        # err_var_shape_post = err_var_shape_prior + 0.5 * n
        # err_var_scale_post = (err_var_scale_prior +
        #                       0.5 * (y.T @ y
        #                              + coeff_mean_prior.T @ coeff_prec_prior @ coeff_mean_prior
        #                              - coeff_mean_post.T @ ninvg_coeff_prec_post @ coeff_mean_post))[0][0]
        #
        # # Marginal posterior distribution for variance parameter
        # err_var_post = invgamma.rvs(err_var_shape_post,
        #                             scale=err_var_scale_post,
        #                             size=(num_post_samp, 1))
        #
        # # Marginal posterior distribution for coefficients
        # coeff_cov_post = err_var_scale_post / err_var_shape_post * ninvg_coeff_cov_post

        # coeff_post = multivariate_t(df=2 * err_var_shape_post,
        #                             loc=coeff_mean_post.squeeze(),
        #                             shape=coeff_cov_post,
        #                             allow_singular=True).rvs(num_post_samp)

        return self.posterior

    def predict(self, predictors):
        """

        :param predictors:
        :return:
        """
        if predictors.shape[1] != self.num_coeff:
            raise ValueError("The number of columns in predictors must match the "
                             "number of columns in the predictor/design matrix "
                             "passed to the ConjugateBayesianLinearRegression class. "
                             "Ensure that the number and order of predictors matches "
                             "the number and order of predictors in the design matrix "
                             "used for model fitting.")

        if self.posterior is None:
            raise AttributeError("A posterior distribution has not been generated "
                                 "because no model has been fit to data. The predict() "
                                 "function is operational only if fit() has been used.")

        n = predictors.shape[0]
        x = predictors
        num_post_samp = self.posterior.num_post_samp

        # # Closed-form posterior predictive distribution.
        # # Sampling from this distribution is computationally
        # # expensive due to the n x n matrix V.
        # beta_mean = self.posterior.coeff_mean_post
        # alpha = self.posterior.err_var_shape_post
        # tau = self.posterior.err_var_scale_post
        # beta_cov = self.posterior.coeff_cov_post
        #
        # V = tau / alpha * (np.eye(n) + x @ beta_cov @ x.T)
        # M = x @ beta_mean
        # post_pred_dist = multivariate_t(df=2 * alpha,
        #                                 loc=M.squeeze(),
        #                                 shape=V).rvs(S)

        beta = self.posterior.coeff_post
        err_var = self.posterior.err_var_post
        post_pred_dist = np.empty((num_post_samp, n))
        for s in range(num_post_samp):
            post_pred_dist[s, :] = vec_norm(x @ beta[s],
                                            np.sqrt(err_var[s]))

        return post_pred_dist

class ModelSummary:
    def __init__(self, model: ConjugateBayesianLinearRegression):
        if not isinstance(model, ConjugateBayesianLinearRegression):
            raise ValueError("The model object must be of type ConjugateBayesianLinearRegression.")
        self.model = model

    def waic(self):
        return watanabe_akaike(response=self.model.response,
                               post_pred_dist=self.model.post_pred_dist,
                               err_var_post=self.model.posterior.err_var_post)

    def mspe(self):
        return mean_squared_prediction_error(response=self.model.response,
                                             post_pred_dist=self.model.post_pred_dist,
                                             err_var_post=self.model.posterior.err_var_post)

    def r_sqr(self):
        return r_squared(post_pred_dist=self.model.post_pred_dist,
                         err_var_post=self.model.posterior.err_var_post)

    def parameters(self, cred_int_level=0.05):
        posterior = self.model.posterior
        num_coeff = self.model.num_coeff
        pred_names = self.model.predictors_names
        lb, ub = 0.5 * cred_int_level, 0.5 * (1 - cred_int_level)

        # Coefficients
        coeff_post = posterior.coeff_post
        coeff_mean_post = posterior.coeff_mean_post.squeeze()
        coeff_std_post = np.sqrt(np.diag(posterior.coeff_cov_post))
        coeff_lb_post = np.quantile(coeff_post, lb, axis=0)
        coeff_ub_post = np.quantile(coeff_post, ub, axis=0)
        coeff_prob_pos_post = np.sum((coeff_post > 0) * 1, axis=0) / posterior.num_post_samp

        # Error variance
        err_var_post = posterior.err_var_post
        err_var_shape_post = posterior.err_var_shape_post
        err_var_scale_post = posterior.err_var_scale_post
        if err_var_shape_post > 1:
            err_var_mean_post = err_var_scale_post / (err_var_shape_post - 1)
        else:
            err_var_mean_post = None

        if err_var_shape_post > 2:
            err_var_std_post = np.sqrt(err_var_scale_post ** 2
                                       / ((err_var_shape_post - 1) ** 2 * (err_var_shape_post - 2)))
        else:
            err_var_std_post = None

        err_var_mode_post = err_var_scale_post / (err_var_shape_post + 1)
        err_var_lb_post = np.quantile(err_var_post, lb)
        err_var_ub_post = np.quantile(err_var_post, ub)

        summary = {}
        for k in range(num_coeff):
            summary[f"Post.Mean[Coeff.{pred_names[k]}]"] = coeff_mean_post[k]
            summary[f"Post.StdDev[Coeff.{pred_names[k]}]"] = coeff_std_post[k]
            summary[f"Post.CredInt.LB[Coeff.{pred_names[k]}]"] = coeff_lb_post[k]
            summary[f"Post.CredInt.UB[Coeff.{pred_names[k]}]"] = coeff_ub_post[k]
            summary[f"Post.Prob.Positive[Coeff.{pred_names[k]}]"] = coeff_prob_pos_post[k]

        summary["Post.Mean[ErrorVariance]"] = err_var_mean_post
        summary["Post.Mode[ErrorVariance]"] = err_var_mode_post
        summary["Post.StdDev[ErrorVariance]"] = err_var_std_post
        summary["Post.CredInt.LB[ErrorVariance]"] = err_var_lb_post
        summary["Post.CredInt.UB[ErrorVariance]"] = err_var_ub_post

        return summary
