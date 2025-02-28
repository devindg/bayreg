import warnings
from typing import Union
import numpy as np
from .linear_regression import ConjugateBayesianLinearRegression as CBLR
from .linear_regression import valid_design_matrix

warnings.filterwarnings("ignore")


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
    X = c * time_index[:, np.newaxis]
    # Pass X to cos() and sin() functions to create Fourier series
    fft_mat = np.c_[np.cos(X), np.sin(X)]

    return fft_mat


class BayesianVAR:
    def __init__(
            self,
            ar_order: int = 1,
            first_difference: bool = False,
            add_intercept: bool = True,
            add_trend: bool = False,
            add_seasonal: bool = False,
            periodicity: int = 1,
            num_seasonal_harmonics: int = 0,
    ):
        self.ar_order = ar_order
        self.first_difference = first_difference
        self.add_intercept = add_intercept
        self.add_trend = add_trend
        self.add_seasonal = add_seasonal
        self.periodicity = periodicity
        self.num_seasonal_harmonics = num_seasonal_harmonics
        self.num_endog = None
        self.orig_num_obs = None
        self.num_obs = None
        self.last_data = None
        self.posterior = None
        self.fit_with_exog = None

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
        periodicity = self.periodicity
        num_seasonal_harmonics = self.num_seasonal_harmonics
        orig_num_obs = self.orig_num_obs

        y = np.array(endog)
        num_rows, num_endog = y.shape

        if not for_forecasting:
            time_offset = 0
        else:
            time_offset = orig_num_obs

        if exog is not None:
            x = np.array(exog)
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
                y_lags[p, w:] = self.last_data[0, :num_lag_vars - w]

        time_polynomial = []
        if add_intercept:
            if not first_difference:
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

        if len(time_polynomial) > 0:
            time_polynomial = np.concatenate(time_polynomial, axis=1)
            if x is not None:
                data = np.concatenate((y, y_lags, x, time_polynomial), axis=1)
            else:
                data = np.concatenate((y, y_lags, time_polynomial), axis=1)
        else:
            if x is not None:
                data = np.concatenate((y, y_lags, x), axis=1)
            else:
                data = np.concatenate((y, y_lags), axis=1)

        if not for_forecasting:
            self.last_data = data[-1, :].reshape(1, -1)
            self.orig_num_obs = data.shape[0]
            self.num_endog = num_endog

        if first_difference and for_forecasting:
            data = np.concatenate((self.last_data, data), axis=0)

        if first_difference:
            data = np.diff(data, n=1, axis=0)

        if not for_forecasting:
            data = data[~np.any(np.isnan(data), axis=1)]
            self.num_obs = data.shape[0]

        return data

    def fit(self,
            endog: np.ndarray,
            exog: Union[None, np.ndarray] = None,
            num_post_samp=1000,
            prior_coeff_mean=None,
            prior_coeff_cov=None,
            prior_err_var_shape=None,
            prior_err_var_scale=None,
            zellner_g=None,
            max_mat_cond_index=30,
            seed: int = 123
            ):

        if exog is None:
            self.fit_with_exog = False
        else:
            self.fit_with_exog = True

        data = self.prepare_data(
            endog=endog,
            exog=exog,
            for_forecasting=False
        )

        num_endog = self.num_endog
        dep_vars = data[:, :num_endog]
        pred_vars = data[:, num_endog:]

        posterior = []
        for j in range(num_endog):
            cblr = CBLR(
                response=dep_vars[:, j],
                predictors=pred_vars,
                seed=seed
            )
            cblr_fit = cblr.fit(
                num_post_samp=num_post_samp,
                prior_coeff_mean=prior_coeff_mean,
                prior_coeff_cov=prior_coeff_cov,
                prior_err_var_shape=prior_err_var_shape,
                prior_err_var_scale=prior_err_var_scale,
                zellner_g=zellner_g,
                max_mat_cond_index=max_mat_cond_index
            )
            posterior.append(cblr_fit)
            self.posterior = posterior

        return self.posterior

    def forecast(self,
                 horizon: int,
                 exog: Union[np.ndarray, None],
                 ):

        if self.posterior is None:
            raise AssertionError("No model was fit. Fit a model with method fit() "
                                 "before calling forecast()."
                                 )

        if self.fit_with_exog is False and exog is not None:
            raise ValueError("The model was not fit with exogenous variables. "
                             "Make sure the 'exog' argument is None")

        if self.fit_with_exog is True and exog is None:
            raise ValueError("The model was fit with exogenous variables. "
                             "Make sure to pass an array of exogenous variables "
                             " to the 'exog' argument.")

        first_difference = self.first_difference
        num_endog = self.num_endog
        ar_order = self.ar_order
        num_lag_vars = num_endog * ar_order
        posterior = self.posterior

        endog = np.full(
            shape=(horizon, num_endog),
            dtype=np.float64,
            fill_value=np.nan
        )

        data = self.prepare_data(
            endog=endog,
            exog=exog,
            for_forecasting=True
        )

        coeffs = np.concatenate([c.post_coeff_mean for c in posterior], axis=1).T
        y_fcst = data[:, :num_endog].T
        y_lags = data[:, num_endog:num_endog + num_lag_vars]
        y_lag_coeffs = coeffs[:, :num_lag_vars]
        z = data[:, num_endog + num_lag_vars:]
        z_coeffs = coeffs[:, num_lag_vars:]

        for t in range(horizon):
            if t == 0:
                pass
            else:
                ys = np.concatenate((y_fcst[:, t - 1], y_lags[t - 1, :]))
                if 0 < t < ar_order:
                    y_lags[t, :num_endog * t] = ys[:num_endog * t]
                else:
                    y_lags[t, :num_endog * ar_order] = ys[:num_endog * ar_order]

            y_fcst[:, t] = y_lag_coeffs @ y_lags[t] + z_coeffs @ z[t]

        y_fcst = y_fcst.T

        if first_difference:
            last_ys = self.last_data[0, :num_endog]
            y_fcst = np.cumsum(y_fcst, axis=0) + last_ys[np.newaxis, :]

        return y_fcst
