import numpy as np
import pandas as pd
import warnings
from scipy.stats import invgamma, multivariate_normal, t
from bayreg.linear_algebra.array_checks import is_symmetric, is_positive_definite
from bayreg.linear_algebra.array_operations import mat_inv
from bayreg.model_assessment.performance import watanabe_akaike, mean_squared_prediction_error, r_squared
from typing import Union, NamedTuple


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
    zellner_prior_obs: Union[int, float]


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
                 response: Union[np.ndarray, list, tuple, pd.Series, pd.DataFrame],
                 predictors: Union[np.ndarray, list, tuple, pd.Series, pd.DataFrame],
                 seed: int = None):
        """

        :param response:
        :param predictors:
        :param seed:
        """

        self.response_type = type(response)
        self.predictors_type = type(predictors)
        self.response_index = None
        self.predictors_names = None

        # CHECK AND PREPARE RESPONSE DATA
        # -- dimension and null/inf checks
        if not isinstance(response, (np.ndarray, list, tuple, pd.Series, pd.DataFrame)):
            raise TypeError("The response array must be a Numpy array, list, tuple, Pandas Series, \n"
                            "or Pandas DataFrame.")
        else:
            if isinstance(response, (list, tuple)):
                resp = np.asarray(response, dtype=np.float64)
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

        self.response = resp
        self.num_obs = resp.shape[0]
        if self.response_index is None:
            self.response_index = np.arange(resp.shape[0])

        # CHECK AND PREPARE PREDICTORS DATA
        if not isinstance(predictors, (np.ndarray, list, tuple, pd.Series, pd.DataFrame)):
            raise TypeError("The predictors array must be a Numpy array, list, tuple, Pandas Series, \n"
                            "or Pandas DataFrame.")
        else:
            if isinstance(predictors, (list, tuple)):
                pred = np.asarray(predictors, dtype=np.float64)
            else:
                pred = predictors.copy()

            # -- check if response and predictors are same data type.
            if (isinstance(response, (np.ndarray, list, tuple))
                    and not isinstance(pred, (np.ndarray, list, tuple))):
                raise TypeError('The response array provided is a NumPy array, list, or tuple, but the predictors '
                                'array is not. Object types must match.')

            if (isinstance(response, (pd.Series, pd.DataFrame)) and
                    not isinstance(pred, (pd.Series, pd.DataFrame))):
                raise TypeError('The response array provided is a Pandas Series/DataFrame, but the predictors '
                                'array is not. Object types must match.')

            # -- get predictor names if a Pandas object
            if isinstance(pred, (pd.Series, pd.DataFrame)):
                if not np.all(pred.index == response.index):
                    raise ValueError('The response and predictors indexes must match.')

                if isinstance(pred, pd.Series):
                    self.predictors_names = [pred.name]
                else:
                    self.predictors_names = pred.columns.values.tolist()

                pred = pred.to_numpy()

        # -- dimension and null/inf checks
        if pred.ndim not in (1, 2):
            raise ValueError('The predictors array must have dimension 1 or 2.')
        elif pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        else:
            pass

        if np.any(np.isnan(pred)):
            raise ValueError('The predictors array cannot have null values.')
        if np.any(np.isinf(pred)):
            raise ValueError('The predictors array cannot have Inf and/or -Inf values.')
        # -- conformable number of observations
        if pred.shape[0] != self.num_obs:
            raise ValueError('The number of observations in the predictors array must match '
                             'the number of observations in the response array.')

        # -- check if design matrix has a constant
        sd_pred = np.std(pred, axis=0)
        if np.any(sd_pred <= 1e-6):
            self.has_constant = True

            if np.sum(sd_pred <= 1e-6) > 1 and self.num_obs > 1:
                raise ValueError('More than one column in the regression matrix cannot be constant.')

            self.constant_index = np.argwhere(sd_pred == 0)[0][0]
        else:
            self.has_constant = False

        self.predictors = pred
        self.num_coeff = pred.shape[1]
        self.num_predictors = self.num_coeff - self.has_constant
        # Create variable names for predictors, if applicable
        if self.predictors_names is None:
            self.predictors_names = [f"x{i + 1}" for i in range(self.num_coeff)]

        # -- warn about model stability if the number of predictors exceeds number of observations
        if self.num_coeff > self.num_obs:
            warnings.warn('The number of predictors exceeds the number of observations. '
                          'Results will be sensitive to choice of priors.')

        if seed is not None:
            if not isinstance(seed, int):
                raise TypeError('seed must be an integer.')
            if not 0 < seed < 2 ** 32 - 1:
                raise ValueError('seed must be an integer between 0 and 2**32 - 1.')
            np.random.seed(seed)

        self.prior = None
        self.posterior = None
        self.post_pred_dist = None

    def _posterior_exists_check(self):
        if self.posterior is None:
            raise AttributeError("No posterior distribution was found. The fit() method must be called.")

        return

    def _post_pred_dist_exists_check(self):
        if self.post_pred_dist is None:
            raise AttributeError("No posterior predictive distribution was found. "
                                 "TThe fit() and posterior_predictive_distribution() methods "
                                 "must be called.")

        return

    def fit(self,
            num_post_samp: int = 1000,
            prior_coeff_mean: Union[list, tuple, np.ndarray] = None,
            prior_coeff_cov: Union[list, tuple, np.ndarray] = None,
            prior_err_var_shape: Union[int, float] = None,
            prior_err_var_scale: Union[int, float] = None,
            zellner_prior_obs: Union[int, float] = None):
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
        if n >= k:
            _, s, Vt = np.linalg.svd(x, full_matrices=False)
            S = np.diag(s)
            StS = S ** 2
        else:
            _, s, Vt = np.linalg.svd(x, full_matrices=True)
            S = np.zeros((n, k))
            S[:n, :n] = np.diag(s)
            StS = S.T @ S

        XtX = Vt.T @ StS @ Vt

        # Check shape prior for error variance
        if prior_err_var_shape is not None:
            if isinstance(prior_err_var_shape, (int, float)) and prior_err_var_shape > 0:
                pass
            else:
                raise ValueError('prior_err_var_shape must be a strictly positive integer or float.')
        else:
            prior_err_var_shape = 1e-6

        # Check scale prior for error variance
        if prior_err_var_scale is not None:
            if isinstance(prior_err_var_scale, (int, float)) and prior_err_var_scale > 0:
                pass
            else:
                raise ValueError('prior_err_var_scale must be a strictly positive integer or float.')
        else:
            prior_err_var_scale = 1e-6

        # Check prior mean for regression coefficients
        if prior_coeff_mean is not None:
            if not isinstance(prior_coeff_mean, (list, tuple, np.ndarray)):
                raise TypeError("prior_coeff_mean must be a list, tuple, or NumPy array.")
            else:
                if isinstance(prior_coeff_mean, (list, tuple)):
                    prior_coeff_mean = np.asarray(prior_coeff_mean, dtype=np.float64)
                else:
                    prior_coeff_mean = prior_coeff_mean.astype(float)

                if prior_coeff_mean.ndim not in (1, 2):
                    raise ValueError('prior_coeff_mean must have dimension 1 or 2.')
                elif prior_coeff_mean.ndim == 1:
                    prior_coeff_mean = prior_coeff_mean.reshape(self.num_coeff, 1)
                else:
                    pass

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
            if not isinstance(prior_coeff_cov, (list, tuple, np.ndarray)):
                raise TypeError("prior_coeff_cov must be a list, tuple, or NumPy array.")
            else:
                if isinstance(prior_coeff_cov, (list, tuple)):
                    prior_coeff_cov = np.atleast_2d(np.asarray(prior_coeff_cov, dtype=np.float64))
                else:
                    prior_coeff_cov = np.atleast_2d(prior_coeff_cov.astype(float))

                if prior_coeff_cov.ndim != 2:
                    raise ValueError('prior_coeff_cov must have dimension 2.')
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

            if zellner_prior_obs is not None:
                if isinstance(zellner_prior_obs, (int, float)) and zellner_prior_obs > 0:
                    pass
                else:
                    raise ValueError('zellner_prior_obs must be a strictly positive integer or float.')
            else:
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
        ninvg_post_coeff_prec = Vt.T @ (StS + Vt @ prior_coeff_prec @ Vt.T) @ Vt
        ninvg_post_coeff_cov = Vt.T @ mat_inv(StS + Vt @ prior_coeff_prec @ Vt.T) @ Vt
        post_coeff_mean = ninvg_post_coeff_cov @ (x.T @ y + prior_coeff_prec @ prior_coeff_mean)
        post_err_var_shape = prior_err_var_shape + 0.5 * n
        post_err_var_scale = (prior_err_var_scale +
                              0.5 * (y.T @ y
                                     + prior_coeff_mean.T @ prior_coeff_prec @ prior_coeff_mean
                                     - post_coeff_mean.T @ ninvg_post_coeff_prec @ post_coeff_mean))[0][0]

        if post_err_var_scale < 0:
            post_err_var_scale = (prior_err_var_scale +
                                  0.5 * ((y - x @ prior_coeff_mean).T
                                         @ mat_inv(np.eye(n) + x @ prior_coeff_cov @ x.T)
                                         @ (y - x @ prior_coeff_mean)))[0][0]

        # Marginal posterior distribution for variance parameter
        post_err_var = invgamma.rvs(post_err_var_shape,
                                    scale=post_err_var_scale,
                                    size=(num_post_samp, 1))

        # Joint posterior distribution for coefficients and variance parameter
        svd_post_coeff_mean = Vt @ post_coeff_mean.flatten()
        svd_ninvg_post_coeff_cov = Vt @ ninvg_post_coeff_cov @ Vt.T
        post_coeff_cov = post_err_var_scale / post_err_var_shape * svd_ninvg_post_coeff_cov

        # Check if the covariance matrix corresponding to the coefficients' marginal
        # posterior distribution is ill-conditioned.
        if not is_positive_definite(post_coeff_cov):
            raise ValueError("The covariance matrix corresponding to the coefficients' "
                             "marginal posterior distribution (multivariate Student-t) "
                             "is not positive definite. Try scaling the predictors, the "
                             "response, or both to eliminate the possibility of an "
                             "ill-conditioned matrix.")

        post_coeff = np.empty((num_post_samp, self.num_coeff))
        for s in range(num_post_samp):
            cond_post_coeff_cov = post_err_var[s] * svd_ninvg_post_coeff_cov
            post_coeff[s] = multivariate_normal(mean=svd_post_coeff_mean,
                                                cov=cond_post_coeff_cov,
                                                allow_singular=True).rvs(1)

        # This will happen if the number of observations is 1
        if post_coeff.ndim == 1:
            post_coeff = post_coeff.reshape((num_post_samp, Vt.shape[0]))

        # Back-transform parameters from SVD to original scale
        post_coeff_cov = Vt.T @ post_coeff_cov @ Vt
        post_coeff = post_coeff @ Vt  # Same as (Vt.T @ post_coeff.T).T

        self.posterior = Posterior(num_post_samp=num_post_samp,
                                   post_coeff_cov=post_coeff_cov,
                                   ninvg_post_coeff_cov=ninvg_post_coeff_cov,
                                   post_coeff_mean=post_coeff_mean,
                                   post_coeff=post_coeff,
                                   post_err_var_shape=post_err_var_shape,
                                   post_err_var_scale=post_err_var_scale,
                                   post_err_var=post_err_var)

        return self.posterior

    def predict(self,
                predictors: Union[np.ndarray, list, tuple, pd.Series, pd.DataFrame],
                mean_only: bool = False):
        """

        :param predictors:
        :param mean_only:
        :return:
        """

        # -- check if object type is valid
        if not isinstance(predictors, (np.ndarray, list, tuple, pd.Series, pd.DataFrame)):
            raise TypeError("The predictors array must be a NumPy array, list, tuple, Pandas Series, \n"
                            "or Pandas DataFrame.")
        else:
            if isinstance(predictors, (list, tuple)):
                x = np.asarray(predictors, dtype=np.float64)
            else:
                x = predictors.copy()
            # Check and prepare predictor data
            # -- check if data types match across instantiated predictors and predictors
            if x.shape == self.predictors.shape:
                if np.allclose(x, self.predictors):
                    pass
            else:
                if not isinstance(x, self.predictors_type):
                    raise TypeError('Object type for predictors does not match the predictors object type '
                                    'instantiated with ConjugateBayesianLinearRegression.')

            # -- if Pandas type, grab index and column names
            if isinstance(x, (pd.Series, pd.DataFrame)):
                if not isinstance(x.index, type(self.response_index)):
                    warnings.warn('Index type for predictors does not match the predictors index type '
                                  'instantiated with ConjugateBayesianLinearRegression.')

                if isinstance(x, pd.Series):
                    predictors_names = [x.name]
                else:
                    predictors_names = x.columns.values.tolist()

                if not all(self.predictors_names[i] == predictors_names[i]
                           for i in range(self.num_coeff)):
                    warnings.warn('The order and/or names of the columns in predictors do not match '
                                  'the order and names in the predictors array instantiated '
                                  'with the ConjugateBayesianLinearRegression class.')

                x = x.to_numpy()

            # -- dimensions
            if x.ndim not in (1, 2):
                raise ValueError('The predictors array must have dimension 1 or 2.')
            elif x.ndim == 1:
                x = x.reshape(-1, 1)
            else:
                pass

            if np.isnan(x).any():
                raise ValueError('The predictors array cannot have null values.')
            if np.isinf(x).any():
                raise ValueError('The predictors array cannot have Inf and/or -Inf values.')

            if x.shape[1] != self.num_coeff:
                raise ValueError("The number of columns in predictors must match the "
                                 "number of columns in the predictor/design matrix "
                                 "instantiated with the ConjugateBayesianLinearRegression class. "
                                 "Ensure that the number and order of predictors matches "
                                 "the number and order of predictors in the design matrix "
                                 "used for model fitting.")

        self._posterior_exists_check()
        n = predictors.shape[0]

        # Marginal posterior predictive distribution
        post_coeff_mean = self.posterior.post_coeff_mean
        post_err_var_shape = self.posterior.post_err_var_shape
        post_err_var_scale = self.posterior.post_err_var_scale
        ninvg_post_coeff_cov = self.posterior.ninvg_post_coeff_cov

        V = (post_err_var_scale / post_err_var_shape
             * (1 + np.array([z.T @ ninvg_post_coeff_cov @ z for z in x])))
        response_mean = x @ post_coeff_mean

        if not mean_only:
            posterior_prediction = t.rvs(df=2 * post_err_var_shape,
                                         loc=response_mean.flatten(),
                                         scale=V ** 0.5,
                                         size=(self.posterior.num_post_samp, n))
        else:
            posterior_prediction = response_mean

        # posterior_prediction = np.empty((self.posterior.num_post_samp, n))
        # if not mean_only:
        #     for s in range(self.posterior.num_post_samp):
        #         posterior_prediction[s, :] = vec_norm(x @ self.posterior.post_coeff[s],
        #                                               np.sqrt(self.posterior.post_err_var[s]))
        # else:
        #     posterior_prediction = x @ self.posterior.post_coeff_mean
        #
        return posterior_prediction

    def posterior_predictive_distribution(self):
        self._posterior_exists_check()
        self.post_pred_dist = self.predict(self.predictors)

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

    def waic(self):
        self._posterior_exists_check()
        x = self.predictors
        post_resp_mean = self.posterior.post_coeff @ x.T

        return watanabe_akaike(response=self.response.T,
                               post_resp_mean=post_resp_mean,
                               post_err_var=self.posterior.post_err_var)

    def mspe(self):
        self._posterior_exists_check()
        self._post_pred_dist_exists_check()

        return mean_squared_prediction_error(response=self.response,
                                             post_pred_dist=self.post_pred_dist,
                                             post_err_var=self.posterior.post_err_var)

    def r_sqr(self):
        self._posterior_exists_check()
        self._post_pred_dist_exists_check()

        return r_squared(post_pred_dist=self.post_pred_dist,
                         post_err_var=self.posterior.post_err_var)
