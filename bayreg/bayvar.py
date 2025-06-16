import warnings
from typing import Union
import numpy as np
from .linear_regression import ConjugateBayesianLinearRegression as CBLR
from .linear_regression import (
    default_zellner_g,
    zellner_covariance
)
from numba import njit, vectorize, float64, prange
from numpy.random import normal
from .transformations.data_utils import fourier_matrix, shift_array
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


@vectorize([float64(float64, float64)])
def vec_rand_norm(mean, sd):
    return normal(mean, sd)


@njit
def cumulative_sum(arr):
    cuml_sum = np.empty_like(arr, dtype=np.float64)
    current_sum = 0.0
    for i in prange(arr.size):
        current_sum += arr[i]
        cuml_sum[i] = current_sum

    return cuml_sum


@njit(cache=True)
def var_forecast(
        endog_lag: np.ndarray,
        exog: np.ndarray,
        ar_order: int,
        first_difference: bool,
        with_drift: bool,
        posterior: tuple,
        mean_only: bool,
        last_endog: np.ndarray,
        seed: int
):
    np.random.seed(seed)
    num_endog = len(posterior)
    horizon = endog_lag.shape[0]
    num_coeff = posterior[0].post_coeff_mean.size
    num_endog_lag = endog_lag.shape[1]
    endog_lag = endog_lag.T
    exog = exog.T

    # Initialize as if mean_only=True. This will get overwritten if not.
    num_samp = 1
    errors = np.zeros((num_endog, horizon))
    coeffs = np.empty(
        shape=(num_endog, num_coeff),
        dtype=np.float64
    )
    for i in prange(num_endog):
        coeffs[i, :] = posterior[i].post_coeff_mean.T

    if not mean_only:
        num_samp = posterior[0].post_err_var.size

    y_fcst = np.full(
        shape=(num_samp, num_endog, horizon),
        dtype=np.float64,
        fill_value=np.nan
    )

    for s in prange(num_samp):
        if not mean_only:
            error_vars = np.empty(
                shape=(num_endog,),
                dtype=np.float64
            )
            for i in prange(num_endog):
                coeffs[i, :] = posterior[i].post_coeff[s, :]
                error_vars[i] = posterior[i].post_err_var[s, 0]

            errors = np.empty(
                shape=(num_endog, horizon),
                dtype=np.float64
            )
            for t in prange(horizon):
                errors[:, t] = vec_rand_norm(
                    np.zeros(num_endog),
                    error_vars ** 0.5
                )

        if not with_drift:
            drift = np.zeros(num_endog)
            endog_lag_coeffs = coeffs[:, :num_endog_lag]
            exog_coeffs = coeffs[:, num_endog_lag:]
        else:
            drift = coeffs[:, 0]
            endog_lag_coeffs = coeffs[:, 1:num_endog_lag + 1]
            exog_coeffs = coeffs[:, 1 + num_endog_lag:]

        for t in prange(horizon):
            if t == 0:
                pass
            else:
                ys = np.concatenate((y_fcst[s, :, t - 1], endog_lag[:, t - 1]))
                if 0 < t < ar_order:
                    endog_lag[:num_endog * t, t] = ys[:num_endog * t]
                else:
                    endog_lag[:num_endog * ar_order, t] = ys[:num_endog * ar_order]

            y_fcst[s, :, t] = (
                    drift
                    + endog_lag_coeffs @ endog_lag[:, t]
                    + exog_coeffs @ exog[:, t]
                    + errors[:, t]
            )

    if first_difference:
        for s in prange(num_samp):
            for i in prange(num_endog):
                y_fcst[s, i] = last_endog[i] + cumulative_sum(y_fcst[s, i])

    return y_fcst


class BayesianVAR:
    def __init__(
            self,
            ar_order: int = 1,
            first_difference: bool = False,
            add_intercept: bool = True,
            add_trend: bool = False,
            add_seasonal: bool = False,
            seasonal_periodicity: int = 1,
            num_seasonal_harmonics: int = 0
    ):
        self.ar_order = ar_order
        self.first_difference = first_difference
        self.add_intercept = add_intercept
        self.add_trend = add_trend
        self.add_seasonal = add_seasonal
        self.seasonal_periodicity = seasonal_periodicity
        self.num_seasonal_harmonics = num_seasonal_harmonics
        self.standardize_data = None

        """ Standardizing data simultaneously wipes out all constants and
        implicitly re-introduces them. For example, if the model being
        estimated is (error term omitted for exposition)

        y_t = a + b * x_t,

        then the standardized model is

        (y_t - mean(y)) / sd(y) = b * (x_t - mean(x)) / sd(x).

        The intercept 'a' gets wiped out, but the standardized model
        implies

        y_t = [mean(y) - b * sd(y) / sd(x) * mean(x)] + [b * sd(y) / sd(x)] * x_t
        iff y_t = A + B * x_t,

        where A = [mean(y) - b * sd(y) / sd(x) * mean(x)] and B = [b * sd(y) / sd(x)]

        The standardized model is, therefore, isomorphic up to a constant, implying
        that

        y_t = a + b * x_t (with drift) OR y_t = b * x_t (without drift)

        are mathematically equivalent after standardization.

        This means it's important to track if the original (non-standardized) model
        has drift, especially for forecasting.

        As mentioned above, standardization implicitly introduces an intercept into the
        model. Thus, if the model to be estimated has no intercept/drift, standardization
        of the data will not include mean-centering of the variables.
        """

        self.has_drift = False
        if not first_difference:
            if add_intercept:
                self.has_drift = True
        else:
            if add_trend:
                self.has_drift = True

        # Initialize other class attributes
        self.num_endog = None
        self.orig_num_obs = None
        self.num_obs = None
        self.valid_predictors = None
        self.train_data_prepared = False
        self.last_data = None
        self.fit_seed = None
        self.prior = None
        self.posterior = None
        self.fit_with_exog = None
        self.zellner_g = None
        self.cross_endog_zellner_g = None

    def prepare_data(
            self,
            endog: np.ndarray,
            exog: Union[None, np.ndarray],
            for_forecasting: bool
    ):

        ar_order = self.ar_order
        first_difference = self.first_difference
        add_trend = self.add_trend
        add_seasonal = self.add_seasonal
        periodicity = self.seasonal_periodicity
        num_seasonal_harmonics = self.num_seasonal_harmonics
        orig_num_obs = self.orig_num_obs
        last_data = self.last_data
        train_data_prepared = self.train_data_prepared

        if for_forecasting and not train_data_prepared:
            raise AssertionError(
                "Training data needs to be prepared before a forecasting data set "
                "can be assembled. The fit() method automatically performs preparation "
                "of the training data set, but prepare_data() can also be used explicitly "
                "to prepare the training data set."
            )

        y = np.asarray(endog)
        if y.ndim != 2:
            raise ValueError(
                "The array of endogenous variables needs to be a 2D array."
            )
        else:
            num_rows, num_endog = y.shape
            if num_endog < 2:
                raise ValueError(
                    "The number of endogenous variables needs to be at least 2."
                )
            if not for_forecasting:
                if num_rows <= num_endog:
                    raise ValueError(
                        "The number of endogenous variables cannot exceed the number "
                        "of observations."
                    )

        num_lag_vars = ar_order * num_endog

        if not for_forecasting:
            time_offset = 0
        else:
            time_offset = orig_num_obs

        if exog is not None:
            x = np.asarray(exog)
            x = (
                np.atleast_2d(x)
                .reshape(num_rows, int(x.size / num_rows))
            )
        else:
            x = None

        if not for_forecasting:
            y_lags = []
            for p in range(1, ar_order + 1):
                y_lags.append(shift_array(y, p, fill_value=np.nan))

            y_lags = np.concatenate(y_lags, axis=1)
        else:
            y_lags = np.full(
                shape=(num_rows, num_lag_vars),
                dtype=np.float64,
                fill_value=np.nan
            )
            for p in range(ar_order):
                if p >= num_rows:
                    break
                w = p * num_endog
                y_lags[p, w:] = last_data[0, :num_lag_vars - w]

        if add_trend or add_seasonal:
            time_polynomial = []
            if add_trend:
                if not first_difference:
                    (
                        time_polynomial
                        .append(
                            np.arange(num_rows)
                            .reshape(num_rows, 1) + time_offset
                        )
                    )
            if add_seasonal:
                if periodicity > 1:
                    if num_seasonal_harmonics == 0:
                        h = int(periodicity / 2)
                    else:
                        h = num_seasonal_harmonics

                    ft = fourier_matrix(
                        time_index=np.arange(num_rows) + time_offset,
                        periodicity=periodicity,
                        num_harmonics=h
                    )

                    if periodicity % 2 == 0 and 2 * h == periodicity:
                        ft = ft[:, :-1]

                    self.num_seasonal_harmonics = h
                    time_polynomial.append(ft)

            if time_polynomial:
                time_polynomial = np.concatenate(time_polynomial, axis=1)
            else:
                time_polynomial = None
        else:
            time_polynomial = None

        if x is not None and time_polynomial is not None:
            data = np.concatenate((y, y_lags, x, time_polynomial), axis=1)
        elif x is None and time_polynomial is not None:
            data = np.concatenate((y, y_lags, time_polynomial), axis=1)
        elif x is not None and time_polynomial is None:
            data = np.concatenate((y, y_lags, x), axis=1)
        else:
            data = np.concatenate((y, y_lags), axis=1)

        if not for_forecasting:
            self.last_data = data[-1, :].reshape(1, -1)
            self.orig_num_obs = data.shape[0]
            self.num_endog = num_endog

        if first_difference:
            if for_forecasting:
                data = np.concatenate((last_data, data), axis=0)
            data = np.diff(data, n=1, axis=0)

        if not for_forecasting:
            data = data[~np.any(np.isnan(data), axis=1)]
            self.num_obs = data.shape[0]
            self.train_data_prepared = True

        if not for_forecasting:
            endog = data[:, :num_endog]
        else:
            endog = None

        endog_lag = data[:, num_endog:num_endog + num_lag_vars]
        exog = data[:, num_endog + num_lag_vars:]

        return endog, endog_lag, exog

    def fit(self,
            endog: np.ndarray,
            exog: Union[None, np.ndarray] = None,
            num_post_samp: int = 1000,
            standardize_data: bool = True,
            prior_coeff_mean: Union[tuple, None] = None,
            prior_coeff_cov: Union[tuple, None] = None,
            prior_err_var_shape: Union[tuple, None] = None,
            prior_err_var_scale: Union[tuple, None] = None,
            zellner_g: Union[float, None] = None,
            cross_lag_endog_zellner_g_factor: Union[int, float] = 1.,
            max_mat_cond_index=30,
            seed: int = 123
            ):

        if cross_lag_endog_zellner_g_factor > 1:
            raise ValueError("cross_lag_endog_zellner_g_factor must be in (0, 1].")

        self.standardize_data = standardize_data

        if exog is None:
            self.fit_with_exog = False
        else:
            self.fit_with_exog = True

        self.fit_seed = seed
        self.cross_endog_zellner_g = cross_lag_endog_zellner_g_factor

        endog, endog_lag, exog = self.prepare_data(
            endog=endog,
            exog=exog,
            for_forecasting=False
        )

        ar_order = self.ar_order
        num_endog = self.num_endog
        num_lag_vars = ar_order * num_endog
        x = np.c_[endog_lag, exog]
        num_pred_vars = x.shape[1]

        if prior_coeff_mean is None:
            prior_coeff_mean = [np.zeros(num_pred_vars)] * num_endog

        if prior_coeff_cov is None:
            """
            The default covariance prior is a modification of the Zellner-g
            prior. Instead of using a fixed g value to scale
            the sample covariance matrix, different g values are used
            for certain features. This is implemented as follows:

            PriorCovariance = G @ V @ G,

            where @ is the dot product, V is the sample covariance matrix,
            and G = sqrt(diag(g_1, g_2, ..., g_k)), where k is the number of
            predictors. By default, g_1 = g_2 = ... = g_k. However, there is
            the option of shrinking cross-endogenous predictor variables more
            so than own-endogenous predictor variables toward zero. For example,
            assume the model is

            y_1,t = a11 * y_1,t-1 + a12 * y_2,t-1 + b1 * x_t + e_1,t
            y_2,t = a21 * y_1,t-1 + a22 * y_2,t-1 + b2 * x_t + e_2,t

            The cross-endogenous coefficients are a12 and a21; the own-endogenous
            coefficients are a11 and a22; and the exogenous coefficients are
            b1 and b2. The sample covariance matrix is

            V = (X.T @ X) ^ (-1),

            where X = [vec(y_1,t-1), vec(y,2,t-1), vec(x_t)], t=1,...,T

            The standard Zellner-g prior takes the form g * V. But it may be
            desirable to shrink cross-endogenous predictor variables more than
            the other predictor variables in an attempt to be more conservative,
            especially if there is doubt about the degree of endogeneity (i.e.,
            the extent of dynamic feedback loops). In other words, it's plausible
            in the given model that y_1 and y_2 are weakly related.

            Taking g as given, and assuming endogeneity is believed to be weak, a
            reasonable prior for the covariance might be

            PriorCovariance_1 = sqrt(diag(g, g * h, g)) @ V @ sqrt(diag(g, g * h, g))
            PriorCovariance_2 = sqrt(diag(g * h, g, g)) @ V @ sqrt(diag(g * h, g, g))

            where h is some factor that scales g. In the context of weak endogeneity, The
            0 < h <= 1.

            This prior will more aggressively shrink a12 and a21 closer to 0, assuming
            the mean prior is a vector of zeros, than a11, a22, b1, and b2.

            Effectively, this prior casts doubt about the degree of feedback
            between the modeled endogenous variables, which will give more weight
            to the possibility of a standard AR(p) model with exogenous variables, also
            known as a dynamic regression model.

            """
            if zellner_g is None:
                zellner_g = default_zellner_g(x=x)

            pcc = zellner_covariance(
                x=StandardScaler(with_std=standardize_data).fit_transform(x),
                zellner_g=zellner_g,
                max_mat_cond_index=max_mat_cond_index
            )
            if standardize_data:
                W = np.diag(1 / np.std(x, axis=0))
            else:
                W = np.eye(num_pred_vars)

            prior_coeff_cov = []
            for k in range(num_endog):
                if standardize_data:
                    var_y = np.var(endog[:, k])
                else:
                    var_y = 1.

                zell_g_k = np.ones(num_pred_vars) * zellner_g
                mask = np.array([False] * num_pred_vars)
                mask[:num_lag_vars][k::num_endog] = True
                mask[num_lag_vars:] = True
                zell_g_k[~mask] = zellner_g * cross_lag_endog_zellner_g_factor
                prior_coeff_cov.append(W @ (
                        np.diag(zell_g_k ** 0.5)
                        @ (1 / zellner_g * pcc)
                        @ np.diag(zell_g_k ** 0.5)
                        @ W) * var_y
                                       )

                if cross_lag_endog_zellner_g_factor < 1:
                    pcm = np.asarray(prior_coeff_mean[k])
                    pcm[~mask] = 0.
                    prior_coeff_mean[k] = pcm.tolist()
        else:
            if zellner_g is None:
                zellner_g = default_zellner_g(x=x)

        self.zellner_g = zellner_g

        if prior_err_var_shape is None:
            prior_err_var_shape = [None] * num_endog

        if prior_err_var_scale is None:
            prior_err_var_scale = [None] * num_endog

        # Fit the model for each endogenous variable, equation by equation.
        posteriors = []
        priors = []

        for j in range(num_endog):
            mod = CBLR(
                response=endog[:, j],
                predictors=x,
                seed=seed
            )
            fit = mod.fit(
                num_post_samp=num_post_samp,
                prior_coeff_mean=prior_coeff_mean[j],
                prior_coeff_cov=prior_coeff_cov[j],
                prior_err_var_shape=prior_err_var_shape[j],
                prior_err_var_scale=prior_err_var_scale[j],
                zellner_g=zellner_g,
                max_mat_cond_index=max_mat_cond_index,
                standardize_data=standardize_data,
                fit_intercept=self.has_drift
            )
            if j == 0:
                self.valid_predictors = mod.valid_predictors

            prior = mod.prior
            posteriors.append(fit)
            priors.append(prior)

        self.posterior = posteriors
        self.prior = priors

        return self.posterior

    def forecast(self,
                 horizon: int,
                 exog: Union[np.ndarray, None],
                 mean_only=False,
                 ):

        first_difference = self.first_difference
        num_endog = self.num_endog
        ar_order = self.ar_order
        posterior = self.posterior
        fit_with_exog = self.fit_with_exog
        last_endog = self.last_data[0, :num_endog]
        fit_seed = self.fit_seed

        if posterior is None:
            raise AssertionError(
                "No model was fit. Fit a model with method fit() "
                "before calling forecast()."
            )

        if fit_with_exog is False and exog is not None:
            raise ValueError(
                "The model was not fit with exogenous variables. "
                "Make sure the 'exog' argument is None."
            )

        if fit_with_exog is True and exog is None:
            raise ValueError(
                "The model was fit with exogenous variables. "
                "Make sure to pass an array of exogenous variables "
                " to the 'exog' argument."
            )

        _, endog_lag, exog = self.prepare_data(
            endog=np.full(
                shape=(horizon, num_endog),
                dtype=np.float64,
                fill_value=np.nan
            ),
            exog=exog,
            for_forecasting=True
        )
        num_endog_lag = endog_lag.shape[1]
        num_exog = exog.shape[1]

        # Get valid predictors based on the fitted model
        endog_lag_index = [j for j in range(num_endog_lag)]
        exog_index = [j for j in range(num_endog_lag, num_endog_lag + num_exog)]
        endog_lag = endog_lag[:, list(set(endog_lag_index) & set(self.valid_predictors))]
        exog = exog[:, [j - num_endog_lag for j in list(set(exog_index) & set(self.valid_predictors))]]

        y_fcst = var_forecast(
            endog_lag=endog_lag,
            exog=exog,
            ar_order=ar_order,
            first_difference=first_difference,
            with_drift=self.has_drift,
            posterior=tuple(posterior),
            mean_only=mean_only,
            last_endog=last_endog,
            seed=fit_seed
        )

        return y_fcst
