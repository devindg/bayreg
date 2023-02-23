import numpy as np
from scipy.special import logsumexp
from typing import Union, NamedTuple


class MSPE(NamedTuple):
    mean_squared_prediction_error: np.ndarray
    prediction_bias: np.ndarray
    prediction_variance: np.ndarray


class WAIC(NamedTuple):
    waic: np.ndarray
    eff_num_params: np.ndarray


def pointwise_loglike_norm(response, predicted_response, error_variance):
    return -0.5 * (np.log(2. * np.pi * error_variance)
                   + (response.T - predicted_response) ** 2 / error_variance)


def watanabe_akaike(response, post_pred_dist, err_var_post):
    """
    Source: Understanding predictive information criteria for Bayesian models
            Gelman, Hwang, and Vehtari (2013)

    Formulas for log pointwise predictive density (lppd) and the effective
    number of parameters (p_waic) are in Section 3.4, pages 8-9.

    lppd = SUM[log(1/S * SUM[p(y_i | theta(s)), s=1,...,S]), i=1,...,n],

    where S is the number posterior samples, n is the number of observed
    data points, theta(s) = [beta_0(s), beta_1(s), ..., beta_k(s), sigma2(s)]
    is the s-th posterior sample of the parameters, and p(y_i | theta(s))
    is the pointwise predictive density for observation i given the posterior
    parameter sample theta(s). More specifically,

    p(y_i | theta(s))   = exp[-(1/2 * log(2 * PI * sigma2(s))
                            + (y_i - x_i @ beta(s))**2 / (2 * sigma2(s)))]
                        = exp[-(1/2 * log(2 * PI * sigma2(s))
                            + (y_i - y_post_i)**2 / (2 * sigma2(s)))]

    p_waic = SUM[VAR[log(p(y_i | theta(s))), s=1,...,S], i=1,...,n],

    where the expression VAR[] is the sample variance of the log pointwise
    predictive density for observation i over the posterior samples
    s = 1,...,S.

    To efficiently compute lppd and p_waic, the pointwise log likelihood
    (i.e., the log likelihood evaluated for each observation and posterior
    sample) can be computed by first creating repetitions of the response
    and posterior error variance arrays. Specifically,

        resp_rep = [repeat(y_1), repeat(y_2), ..., repeat(y_n)],

    where repeat(y_i) is a column vector of dimension S x 1. Thus, resp_rep
    is matrix of dimension S x n. Similarly,

        err_var_rep = [repeat(sig2_1), repeat(sig2_2), ..., repeat(sig2_S)].T

    where repeat(sig2_s) is a column vector of dimension S x 1, and 'T' denotes
    the transpose operator. Thus, err_var_rep is a matrix of dimension S x n.

    This configuration allows fast computation of the pointwise log likelihood
    function, where in this case

        lppd = -0.5 * (np.log(2. * np.pi * err_var) + (resp - pred_resp) ** 2 / err_var)

    When resp_rep, err_var_rep, and post_pred_dist are passed to lppd, S x n
    elements are operated on using fast matrix calculations.

    """
    log_prob = pointwise_loglike_norm(response,
                                      post_pred_dist,
                                      err_var_post)

    log_ppd = np.sum(logsumexp(log_prob, axis=0, b=1 / post_pred_dist.shape[0]))
    eff_num_params = np.sum(np.var(log_prob, axis=0, ddof=1))
    waic = -2. * (log_ppd - eff_num_params)

    return WAIC(waic=waic, eff_num_params=eff_num_params)


def mean_squared_prediction_error(response, post_pred_dist, err_var_post):
    prediction_mean = np.mean(post_pred_dist, axis=0).reshape(1, -1)
    prediction_variance = np.mean((post_pred_dist - prediction_mean) ** 2, axis=1).reshape(-1, 1)
    prediction_bias = np.mean(response.T - prediction_mean)
    mspe = prediction_variance + prediction_bias ** 2 + err_var_post

    return MSPE(mspe, prediction_bias, prediction_variance)


def r_squared(post_pred_dist, err_var_post):
    predicted_variance = np.var(post_pred_dist, axis=1, ddof=1).reshape(-1, 1)
    r2 = predicted_variance / (predicted_variance + err_var_post)

    return r2
