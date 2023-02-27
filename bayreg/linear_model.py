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
    post_coeff_mean: np.ndarray
    post_coeff_cov: np.ndarray
    post_coeff: np.ndarray
    post_err_var_shape: np.ndarray
    post_err_var_scale: np.ndarray
    post_err_var: np.ndarray


class Prior(NamedTuple):
    prior_coeff_mean: np.ndarray
    prior_coeff_cov: np.ndarray
    prior_err_var_shape: np.ndarray
    prior_err_var_scale: np.ndarray
    zellner_prior_obs: float


@njit
def _set_numba_seed(value):
    np.random.seed(value)


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

            # -- get predictor names if a Pandas object
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
            raise ValueError('The predictors array must have dimension 1 or 2.')
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
            prior_coeff_mean=None,
            prior_coeff_cov=None,
            prior_err_var_shape=None,
            prior_err_var_scale=None,
            zellner_prior_obs=None):
        """

        :param num_post_samp:
        :param prior_coeff_mean:
        :param prior_coeff_cov:
        :param prior_err_var_shape:
        :param prior_err_var_scale:
        :param zellner_prior_obs:
        :return:
        """

        y, x, n, k = self.response, self.predictors, self.num_obs, self.num_coeff
        S, Vt = self._svd(x)
        XtX = Vt.T @ (S ** 2) @ Vt

        # Check shape prior for error variance
        if prior_err_var_shape is not None:
            if not prior_err_var_shape > 0:
                raise ValueError('prior_err_var_shape must be strictly positive.')
        else:
            prior_err_var_shape = 1e-6

        # Check scale prior for error variance
        if prior_err_var_scale is not None:
            if not prior_err_var_scale > 0:
                raise ValueError('prior_err_var_scale must be strictly positive.')
        else:
            prior_err_var_scale = 1e-6

        # Check prior mean for regression coefficients
        if prior_coeff_mean is not None:
            if not prior_coeff_mean.shape == (self.num_coeff, 1):
                raise ValueError(f'prior_coeff_mean must have shape ({self.num_coeff}, 1).')
            if np.any(np.isnan(prior_coeff_mean)):
                raise ValueError('prior_coeff_mean cannot have NaN values.')
            if np.any(np.isinf(prior_coeff_mean)):
                raise ValueError('prior_coeff_mean cannot have Inf and/or -Inf values.')
        else:
            prior_coeff_mean = np.zeros((self.num_coeff, 1))

        # Check prior covariance matrix for regression coefficients
        if prior_coeff_cov is not None:
            if not prior_coeff_cov.shape == (self.num_coeff, self.num_coeff):
                raise ValueError(f'prior_coeff_cov must have shape ({self.num_coeff}, '
                                 f'{self.num_coeff}).')
            if not is_positive_definite(prior_coeff_cov):
                raise ValueError('prior_coeff_cov must be a positive definite matrix.')
            if not is_symmetric(prior_coeff_cov):
                raise ValueError('prior_coeff_cov must be a symmetric matrix.')

            prior_coeff_prec = mat_inv(prior_coeff_cov)
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
            prior_coeff_prec = zellner_prior_obs / n * (w * XtX + (1 - w) * np.diag(np.diag(XtX)))
            prior_coeff_cov = mat_inv(prior_coeff_prec)

        self.prior = Prior(prior_coeff_mean=prior_coeff_mean,
                           prior_coeff_cov=prior_coeff_cov,
                           prior_err_var_shape=prior_err_var_shape,
                           prior_err_var_scale=prior_err_var_scale,
                           zellner_prior_obs=zellner_prior_obs)

        # Posterior values for coefficient mean, coefficient covariance matrix,
        # error variance shape, and error variance scale.

        # Note: storing the normal-inverse-gamma precision and covariance matrices
        # could cause memory problems if the coefficient vector has high dimension.
        # May want to reconsider temporary storage of these matrices.
        ninvg_post_coeff_prec = Vt.T @ (S ** 2 + Vt @ prior_coeff_prec @ Vt.T) @ Vt
        ninvg_post_coeff_cov = Vt.T @ mat_inv(S ** 2 + Vt @ prior_coeff_prec @ Vt.T) @ Vt
        post_coeff_mean = ninvg_post_coeff_cov @ (x.T @ y + prior_coeff_prec @ prior_coeff_mean)
        post_err_var_shape = prior_err_var_shape + 0.5 * n
        post_err_var_scale = (prior_err_var_scale +
                              0.5 * (y.T @ y
                                     + prior_coeff_mean.T @ prior_coeff_prec @ prior_coeff_mean
                                     - post_coeff_mean.T @ ninvg_post_coeff_prec @ post_coeff_mean))[0][0]

        # Marginal posterior distribution for variance parameter
        post_err_var = invgamma.rvs(post_err_var_shape,
                                    scale=post_err_var_scale,
                                    size=(num_post_samp, 1))

        # Marginal posterior distribution for coefficients
        post_coeff_cov = post_err_var_scale / post_err_var_shape * (Vt @ ninvg_post_coeff_cov @ Vt.T)

        # Check if the covariance matrix corresponding to the coefficients' marginal
        # posterior distribution is ill-conditioned.
        if not is_positive_definite(post_coeff_cov):
            raise ValueError("The covariance matrix corresponding to the coefficients' "
                             "marginal posterior distribution (multivariate Student-t) "
                             "is not positive definite. Try scaling the predictors, the "
                             "response, or both to eliminate the possibility of an "
                             "ill-conditioned matrix.")

        post_coeff = multivariate_t(df=2 * post_err_var_shape,
                                    loc=Vt @ post_coeff_mean.squeeze(),
                                    shape=post_coeff_cov,
                                    allow_singular=True).rvs(num_post_samp)

        # Back-transform parameters from SVD to original scale
        post_coeff_cov = Vt.T @ post_coeff_cov @ Vt
        post_coeff = (Vt.T @ post_coeff.T).T

        self.posterior = Posterior(num_post_samp=num_post_samp,
                                   post_coeff_cov=post_coeff_cov,
                                   post_coeff_mean=post_coeff_mean,
                                   post_coeff=post_coeff,
                                   post_err_var_shape=post_err_var_shape,
                                   post_err_var_scale=post_err_var_scale,
                                   post_err_var=post_err_var)

        # # Computations without SVD
        # ninvg_post_coeff_prec = x.T @ x + prior_coeff_prec
        # ninvg_post_coeff_cov = mat_inv(ninvg_post_coeff_prec)
        # post_coeff_mean = np.linalg.solve(ninvg_post_coeff_prec, x.T @ y + prior_coeff_prec @ prior_coeff_mean)
        # post_err_var_shape = prior_err_var_shape + 0.5 * n
        # post_err_var_scale = (prior_err_var_scale +
        #                       0.5 * (y.T @ y
        #                              + prior_coeff_mean.T @ prior_coeff_prec @ prior_coeff_mean
        #                              - post_coeff_mean.T @ ninvg_post_coeff_prec @ post_coeff_mean))[0][0]
        #
        # # Marginal posterior distribution for variance parameter
        # post_err_var = invgamma.rvs(post_err_var_shape,
        #                             scale=post_err_var_scale,
        #                             size=(num_post_samp, 1))
        #
        # # Marginal posterior distribution for coefficients
        # post_coeff_cov = post_err_var_scale / post_err_var_shape * ninvg_post_coeff_cov

        # post_coeff = multivariate_t(df=2 * post_err_var_shape,
        #                             loc=post_coeff_mean.squeeze(),
        #                             shape=post_coeff_cov,
        #                             allow_singular=True).rvs(num_post_samp)

        return self.posterior

    def predict(self, predictors, mean_only=False):
        """

        :param predictors:
        :param mean_only: 
        :return:
        """

        # -- check if object type is valid
        if not isinstance(predictors, (pd.Series, pd.DataFrame, np.ndarray)):
            raise TypeError("The predictors array must be a NumPy array, Pandas Series, "
                            "or Pandas DataFrame.")
        else:
            x = predictors.copy()
            # Check and prepare predictor data
            # -- data types match across instantiated predictors and predictors
            if not isinstance(predictors, type(self.predictors)):
                raise TypeError('Object type for predictors does not match the predictors '
                                'object type instantiated with ConjugateBayesianLinearRegression.')
            else:
                # -- if Pandas type, grab index and column names
                if isinstance(predictors, (pd.Series, pd.DataFrame)):
                    if not isinstance(predictors.index, type(self.response_index)):
                        raise TypeError('Index type for predictors does not match the predictors '
                                        'index type instantiated with ConjugateBayesianLinearRegression.')

                    if isinstance(predictors, pd.Series):
                        predictors_names = [predictors.name]
                    else:
                        predictors_names = predictors.columns.values.tolist()

                    if len(predictors_names) != self.num_coeff:
                        raise ValueError(
                            f'The number of predictors used for historical estimation {self.num_coeff} '
                            f'does not match the number of predictors specified for forecasting '
                            f'{len(predictors_names)}. The same set of predictors must be used.')
                    else:
                        if not all(self.predictors_names[i] == predictors_names[i]
                                   for i in range(self.num_coeff)):
                            raise ValueError('The order and names of the columns in predictors must match '
                                             'the order and names in the predictors array instantiated '
                                             'with the ConjugateBayesianLinearRegression class.')

                    x = x.to_numpy()

                # -- dimensions
                if x.ndim not in (1, 2):
                    raise ValueError('The predictors array must have dimension 1 or 2.')
                elif x.ndim == 1:
                    x = x.reshape(-1, 1)
                else:
                    if 1 in x.shape:
                        x = x.reshape(-1, 1)

                if np.isnan(x).any():
                    raise ValueError('The predictors array cannot have null values.')
                if np.isinf(x).any():
                    raise ValueError('The predictors array cannot have Inf and/or -Inf values.')

                # Final sanity checks
                if x.shape[1] != self.num_coeff:
                    raise ValueError("The number of columns in predictors must match the "
                                     "number of columns in the predictor/design matrix "
                                     "instantiated with the ConjugateBayesianLinearRegression class. "
                                     "Ensure that the number and order of predictors matches "
                                     "the number and order of predictors in the design matrix "
                                     "used for model fitting.")

        if self.posterior is None:
            raise AttributeError("A posterior distribution has not been generated "
                                 "because no model has been fit to data. The predict() "
                                 "method is operational only if fit() has been used.")

        n = predictors.shape[0]

        # # Closed-form posterior predictive distribution.
        # # Sampling from this distribution is computationally
        # # expensive due to the n x n matrix V.
        # beta_mean = self.posterior.post_coeff_mean
        # alpha = self.posterior.post_err_var_shape
        # tau = self.posterior.post_err_var_scale
        # beta_cov = self.posterior.post_coeff_cov
        #
        # V = tau / alpha * (np.eye(n) + x @ beta_cov @ x.T)
        # M = x @ beta_mean
        # post_pred_dist = multivariate_t(df=2 * alpha,
        #                                 loc=M.squeeze(),
        #                                 shape=V).rvs(S)

        posterior_prediction = np.empty((self.posterior.num_post_samp, n))

        if not mean_only:
            for s in range(self.posterior.num_post_samp):
                posterior_prediction[s, :] = vec_norm(x @ self.posterior.post_coeff[s],
                                                      np.sqrt(self.posterior.post_err_var[s]))
        else:
            posterior_prediction = x @ self.posterior.post_coeff_mean

        return posterior_prediction

    def posterior_predictive_distribution(self):
        if self.posterior is None:
            raise AttributeError("A posterior distribution has not been generated "
                                 "because no model has been fit to data. The "
                                 "posterior_predictive_distribution() method is operational "
                                 "only if fit() has been used.")
        self.post_pred_dist = self.predict(self.predictors)
        return self.post_pred_dist

    def posterior_summary(self, cred_int_level=0.05):
        if self.posterior is None:
            raise AttributeError("A posterior distribution has not been generated "
                                 "because no model has been fit to data. The "
                                 "posterior_summary() method is operational "
                                 "only if fit() has been used.")

        if not 0 < cred_int_level < 1:
            raise ValueError("The credible interval level must be a value in (0, 1).")

        posterior = self.posterior
        num_coeff = self.num_coeff
        pred_names = self.predictors_names
        lb, ub = 0.5 * cred_int_level, 1 - 0.5 * cred_int_level

        # Coefficients
        post_coeff = posterior.post_coeff
        post_coeff_mean = posterior.post_coeff_mean.squeeze()
        post_coeff_std = np.sqrt(np.diag(posterior.post_coeff_cov))
        post_coeff_lb = np.quantile(post_coeff, lb, axis=0)
        post_coeff_ub = np.quantile(post_coeff, ub, axis=0)
        coeff_prob_pos_post = np.sum((post_coeff > 0) * 1, axis=0) / posterior.num_post_samp

        # Error variance
        post_err_var = posterior.post_err_var
        post_err_var_shape = posterior.post_err_var_shape
        post_err_var_scale = posterior.post_err_var_scale
        if post_err_var_shape > 1:
            post_err_var_mean = post_err_var_scale / (post_err_var_shape - 1)
        else:
            post_err_var_mean = None

        if post_err_var_shape > 2:
            post_err_var_std = np.sqrt(post_err_var_scale ** 2
                                       / ((post_err_var_shape - 1) ** 2 * (post_err_var_shape - 2)))
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
            summary[f"Post.Prob.Positive[Coeff.{pred_names[k]}]"] = coeff_prob_pos_post[k]

        summary["Post.Mean[ErrorVariance]"] = post_err_var_mean
        summary["Post.Mode[ErrorVariance]"] = post_err_var_mode
        summary["Post.StdDev[ErrorVariance]"] = post_err_var_std
        summary["Post.CredInt.LB[ErrorVariance]"] = post_err_var_lb
        summary["Post.CredInt.UB[ErrorVariance]"] = post_err_var_ub

        return summary


class ModelPerformance:
    def __init__(self, model: ConjugateBayesianLinearRegression):
        if not isinstance(model, ConjugateBayesianLinearRegression):
            raise ValueError("The model object must be of type ConjugateBayesianLinearRegression.")
        self.model = model

        if self.model.posterior is None:
            raise AttributeError("No posterior distribution for the model's parameters was found. "
                                 "The ModelSummary class is not viable. Make sure to use the "
                                 "fit() method in ConjugateBayesianLinearRegression.")

        if self.model.post_pred_dist is None:
            raise AttributeError("No posterior predictive distribution was found. "
                                 "The ModelSummary class is not viable. Make sure to use the "
                                 "fit() and posterior_predictive_distribution() methods "
                                 "in ConjugateBayesianLinearRegression.")

    def waic(self):
        return watanabe_akaike(response=self.model.response,
                               post_pred_dist=self.model.post_pred_dist,
                               post_err_var=self.model.posterior.post_err_var)

    def mspe(self):
        return mean_squared_prediction_error(response=self.model.response,
                                             post_pred_dist=self.model.post_pred_dist,
                                             post_err_var=self.model.posterior.post_err_var)

    def r_sqr(self):
        return r_squared(post_pred_dist=self.model.post_pred_dist,
                         post_err_var=self.model.posterior.post_err_var)
