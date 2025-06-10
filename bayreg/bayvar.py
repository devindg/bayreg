import warnings
from typing import Union
import numpy as np
from .linear_regression import ConjugateBayesianLinearRegression as CBLR
from .linear_regression import (
    valid_design_matrix,
    default_zellner_g,
    zellner_covariance
)
from ..bayreg.linear_algebra.array_operations import mat_inv
from numba import njit, vectorize, float64, prange
from numpy.random import normal

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
        data: np.ndarray,
        ar_order: int,
        first_difference: bool,
        intercept_index: int,
        posterior: tuple,
        mean_only: bool,
        last_endog: np.ndarray,
        seed: int
):
    np.random.seed(seed)
    num_endog = len(posterior)
    num_lag_vars = num_endog * ar_order
    horizon = data.shape[0]
    num_coeff = posterior[0].post_coeff_mean.size

    if intercept_index == -1:
        y_lags = data[:, :num_lag_vars].T
        z = data[:, num_lag_vars:].T
    else:
        y_lags = data[:, 1:num_lag_vars + 1].T
        z = data[:, 1 + num_lag_vars:].T

    if not mean_only:
        num_samp = posterior[0].post_err_var.size
        y_fcst = np.full(
            shape=(num_samp, num_endog, horizon),
            dtype=np.float64,
            fill_value=np.nan
        )

        for s in prange(num_samp):
            coeffs = np.empty(
                shape=(num_endog, num_coeff),
                dtype=np.float64
            )
            error_vars = np.empty(
                shape=(num_endog,),
                dtype=np.float64
            )
            for i in prange(num_endog):
                coeffs[i, :] = posterior[i].post_coeff[s, :]
                error_vars[i] = posterior[i].post_err_var[s, 0]

            if intercept_index == -1:
                y_lag_coeffs = coeffs[:, :num_lag_vars]
                z_coeffs = coeffs[:, num_lag_vars:]
                drift = np.zeros(num_samp)
            elif intercept_index == 0:
                y_lag_coeffs = coeffs[:, 1:num_lag_vars + 1]
                z_coeffs = coeffs[:, 1 + num_lag_vars:]
                drift = coeffs[:, intercept_index]
            else:
                y_lag_coeffs = coeffs[:, :num_lag_vars]
                z_coeffs = coeffs[:, 1 + num_lag_vars:]
                drift = coeffs[:, intercept_index]

            for t in prange(horizon):
                errors = vec_rand_norm(np.zeros_like(error_vars), error_vars ** 0.5)
                if t == 0:
                    pass
                else:
                    ys = np.concatenate((y_fcst[s, :, t - 1], y_lags[:, t - 1]))
                    if 0 < t < ar_order:
                        y_lags[:num_endog * t, t] = ys[:num_endog * t]
                    else:
                        y_lags[:num_endog * ar_order, t] = ys[:num_endog * ar_order]

                y_fcst[s, :, t] = drift + y_lag_coeffs @ y_lags[:, t] + z_coeffs @ z[:, t] + errors

    else:
        num_samp = 1
        y_fcst = np.full(
            shape=(num_samp, num_endog, horizon),
            dtype=np.float64,
            fill_value=np.nan
        )
        coeffs = np.empty(
            shape=(num_endog, num_coeff),
            dtype=np.float64
        )
        for i in prange(num_endog):
            coeffs[i, :] = posterior[i].post_coeff_mean.T

        if intercept_index == -1:
            y_lag_coeffs = coeffs[:, :num_lag_vars]
            z_coeffs = coeffs[:, num_lag_vars:]
            drift = np.zeros(num_samp)
        elif intercept_index == 0:
            y_lag_coeffs = coeffs[:, 1:num_lag_vars + 1]
            z_coeffs = coeffs[:, 1 + num_lag_vars:]
            drift = coeffs[:, intercept_index]
        else:
            y_lag_coeffs = coeffs[:, :num_lag_vars]
            z_coeffs = coeffs[:, 1 + num_lag_vars:]
            drift = coeffs[:, intercept_index]

        for t in prange(horizon):
            if t == 0:
                pass
            else:
                ys = np.concatenate((y_fcst[0, :, t - 1], y_lags[:, t - 1]))
                if 0 < t < ar_order:
                    y_lags[:num_endog * t, t] = ys[:num_endog * t]
                else:
                    y_lags[:num_endog * ar_order, t] = ys[:num_endog * ar_order]

            y_fcst[0, :, t] = drift + y_lag_coeffs @ y_lags[:, t] + z_coeffs @ z[:, t]

    if first_difference:
        for s in prange(num_samp):
            for i in prange(num_endog):
                y_fcst[s, i] = last_endog[i] + cumulative_sum(y_fcst[s, i])

    return y_fcst


def shift_array(a, shift, fill_value=np.nan):
    a_shift = np.empty_like(a)
    if shift > 0:
        a_shift[:shift] = fill_value
        a_shift[shift:] = a[:-shift]
    elif shift < 0:
        a_shift[shift:] = fill_value
        a_shift[:shift] = a[-shift:]
    else:
        a_shift[:] = a

    return a_shift


def fourier_matrix(
        time_index,
        periodicity,
        num_harmonics
):
    """
    Creates a Fourier representation of a periodic/oscillating function of time.
    An ordered integer array of values capturing time is mapped to a
    matrix, where in general row R is a Fourier series represenation of some
    unknown function f(t) evaluated at t=R. The matrix returned is F. It takes
    the following form:

    t=1: [cos(2 * pi * n * 1 / p), sin(2 * pi * n * 1 / p)], n = 1, 2, ..., N
    t=2: [cos(2 * pi * n * 2 / p), sin(2 * pi * n * 2 / p)], n = 1, 2, ..., N
    t=3: [cos(2 * pi * n * 3 / p), sin(2 * pi * n * 3 / p)], n = 1, 2, ..., N
    .
    .
    .
    t=T: [cos(2 * pi * n * T / p), sin(2 * pi * 1 * T / p)], n = 1, 2, ..., N

    Each row in F is of length 2N. Assuming a cycle of length P, row
    R is the same as row (P+1)R.

    The matrix F is intended to be augmented to a design matrix for regression,
    where the outcome variable is measured over time.


    Parameters
    ----------
    time_index : array
        Sequence of ordered integers representing the evolution of time.
        For example, t = [0,1,2,3,4,5, ..., T], where T is the terminal period.

    periodicity: float
        The amount of time it takes for a period/cycle to end. For example,
        if the frequency of data is monthly, then a period completes in 12
        months. If data is daily, then there could conceivably be two period
        lengths, one for every week (a period of length 7) and one for every
        year (a period of length 365.25). Must be positive.

    num_harmonics : integer
        The number of cosine-sine pairs to approximate oscillations in the
        variable t.


    Returns
    -------
    A T x 2N matrix of cosine-sine pairs that take the form

        cos(2 * pi * n * t / p), sin(2 * pi * n * t / p),
        t = 1, 2, ..., T
        n = 1, 2, ..., N

    """

    # Create cosine and sine input scalar factors, 2 * pi * n / (p / s), n=1,...,N
    c = 2 * np.pi * np.arange(1, num_harmonics + 1) / periodicity
    # Create cosine and sine input values, 2 * pi * n / (p / s) * t, t=1,...,T and n=1,...,N
    x = c * time_index[:, np.newaxis]
    # Pass X to cos() and sin() functions to create Fourier series
    fft_mat = np.c_[np.cos(x), np.sin(x)]

    return fft_mat


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
        self.num_pred_vars = None
        self.orig_num_obs = None
        self.num_obs = None
        self.train_data_prepared = False
        self.last_data = None
        self.fit_seed = None
        self.prior = None
        self.posterior = None
        self.fit_with_exog = None
        self.zellner_g = None
        self.cross_endog_zellner_g = None
        self._intercept_index = None

    def prepare_data(
            self,
            endog: np.ndarray,
            exog: Union[None, np.ndarray],
            for_forecasting: bool
    ):

        ar_order = self.ar_order
        first_difference = self.first_difference
        add_intercept = self.add_intercept
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
            if not for_forecasting:
                x = valid_design_matrix(x)[0]
        else:
            x = None

        if not for_forecasting:
            y_lags = []
            for p in range(1, ar_order + 1):
                y_lags.append(shift_array(y, p, fill_value=np.nan))

            y_lags = np.concatenate(y_lags, axis=1)
        else:
            num_lag_vars = int(num_endog * ar_order)
            y_lags = np.full(
                shape=(num_rows, num_lag_vars),
                dtype=np.float64,
                fill_value=np.nan
            )
            for p in range(ar_order):
                w = p * num_endog
                y_lags[p, w:] = last_data[0, :num_lag_vars - w]

        if add_intercept or add_trend or add_seasonal:
            time_polynomial = []
            if add_intercept:
                if first_difference:
                    pass
                else:
                    time_polynomial.append(np.ones((num_rows, 1)))
            if add_trend:
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
            self.num_pred_vars = data.shape[1] - num_endog
            self.train_data_prepared = True
        else:
            data = data[:, num_endog:]

        return data

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

        data = self.prepare_data(
            endog=endog,
            exog=exog,
            for_forecasting=False
        )

        ar_order = self.ar_order
        num_endog = self.num_endog
        num_pred_vars = self.num_pred_vars
        endog_vars = data[:, :num_endog]
        pred_vars = data[:, num_endog:]
        num_lag_vars = ar_order * num_endog

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
                zellner_g = default_zellner_g(x=pred_vars)

            pcc = zellner_covariance(
                x=pred_vars,
                zellner_g=zellner_g,
                max_mat_cond_index=max_mat_cond_index
            )
            prior_coeff_cov = []
            for k in range(num_endog):
                zell_g_k = np.ones(num_pred_vars) * zellner_g
                mask = np.array([False] * num_pred_vars)
                mask[:num_lag_vars][k::num_endog] = True
                mask[num_lag_vars:] = True
                zell_g_k[~mask] = zellner_g * cross_lag_endog_zellner_g_factor
                prior_coeff_cov.append(
                        np.diag(zell_g_k ** 0.5)
                        @ (1 / zellner_g * pcc)
                        @ np.diag(zell_g_k ** 0.5)
                )
                if cross_lag_endog_zellner_g_factor < 1:
                    pcm = np.asarray(prior_coeff_mean[k])
                    pcm[~mask] = 0.
                    prior_coeff_mean[k] = pcm.tolist()
        else:
            if zellner_g is None:
                zellner_g = default_zellner_g(x=pred_vars)

        self.zellner_g = zellner_g

        if prior_err_var_shape is None:
            prior_err_var_shape = [None] * num_endog

        if prior_err_var_scale is None:
            prior_err_var_scale = [None] * num_endog

        # Fit the model for each endogenous variable, equation by equation.
        posteriors = []
        priors = []
        self._intercept_index = valid_design_matrix(pred_vars)[-1]
        for j in range(num_endog):
            mod = CBLR(
                response=endog_vars[:, j],
                predictors=pred_vars,
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
            )
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
        has_drift = self.has_drift
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

        data = self.prepare_data(
            endog=np.full(
                shape=(horizon, num_endog),
                dtype=np.float64,
                fill_value=np.nan
            ),
            exog=exog,
            for_forecasting=True
        )

        if self.standardize_data:
            if self._intercept_index is not None:
                data = np.delete(data, self._intercept_index, axis=1)
                data = np.insert(data, 0, 1., axis=1)
                intercept_index = 0
            else:
                intercept_index = -1
        else:
            if self._intercept_index is not None:
                intercept_index = self._intercept_index
            else:
                intercept_index = -1

        y_fcst = var_forecast(
            data=data,
            ar_order=ar_order,
            first_difference=first_difference,
            intercept_index=intercept_index,
            posterior=tuple(posterior),
            mean_only=mean_only,
            last_endog=last_endog,
            seed=fit_seed
        )

        return y_fcst