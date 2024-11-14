import warnings
from itertools import repeat
from typing import Literal
from multiprocessing import Pool
from typing import NamedTuple, Union
from operator import itemgetter
from threadpoolctl import ThreadpoolController
import numpy as np
import pandas as pd
from ..bayreg.linear_regression import ConjugateBayesianLinearRegression as CBLR
from ..bayreg.linear_regression import Prior, Posterior, drop_zero_cols
from ..bayreg.model_assessment.performance import WAIC, OOS_Error
from ..bayreg.model_assessment.residual_diagnostics import studentized_residuals as stud_resid
from ..bayreg.model_assessment.residual_diagnostics import Outliers
from ..bayreg.linear_algebra.array_operations import mat_inv
from ..bayreg.transformations.data_utils import ProcessPanelRegressionData, PreparedPanelData

thread_controller = ThreadpoolController()


class UniqueIDLengthMismatch(Exception):
    def __init__(self, arg):
        self.message = f"""If {arg} has length greater than 0, 
                        then its length must be equal to the number of 
                        unique ID's present in the prepared dataset. 
                        Note that raw data, in the form of a Pandas DataFrame, 
                        should be passed to method PrepareData() before 
                        any model fitting methods are used. Specifically, 
                        the object returned from PrepareData() should be 
                        passed to group_fit() and member_fit() as argument 
                        'data_context'.
                       """
        super().__init__(self.message)


class FitResults(NamedTuple):
    fit: Posterior
    fit_vals: np.ndarray
    fit_vals_back_transform: Union[np.ndarray, None]
    prior: Prior
    waic: WAIC
    oos_error: OOS_Error
    student_resid: Outliers
    rmspe: float
    r_sqr: float
    r_sqr_classic: float
    pred_vars: list


class BayesianPanelRegression(ProcessPanelRegressionData):
    def __init__(
            self,
            unique_id_var: str,
            date_var: str,
            dep_var: str,
            predictor_vars: list,
            transform: Union[Literal["center", "difference"], None] = None,
            make_dynamic: bool = False,
            seed: Union[int, None] = None,
    ):
        super(BayesianPanelRegression, self).__init__(
            unique_id_var=unique_id_var,
            date_var=date_var,
            dep_var=dep_var,
            predictor_vars=predictor_vars,
            transform=transform,
            make_dynamic=make_dynamic
        )

        if seed is not None:
            if isinstance(seed, int) and 0 < seed < 2 ** 32 - 1:
                pass
            else:
                raise ValueError(
                    "seed must be an integer between 0 and 2**32 - 1."
                )
            self.seed = seed
        else:
            self.seed = 123

    def _fit(
            self,
            data: pd.DataFrame,
            dep_var: str,
            predictor_vars: list,
            num_post_samp: int,
            prior_coeff_mean: Union[np.ndarray, list, tuple] = None,
            prior_coeff_cov: Union[np.ndarray, list, tuple] = None,
            prior_err_var_shape: Union[int, float] = None,
            prior_err_var_scale: Union[int, float] = None,
            zellner_g: Union[int, float] = None,
            leverage_predictors_adj: Union[np.ndarray, None] = None,
            offset_var: Union[str, None] = None
    ) -> FitResults:
        y = data[dep_var]
        x = data[predictor_vars]

        with thread_controller.limit(limits=1, user_api="blas"):
            mod = CBLR(response=y, predictors=x, seed=self.seed)
            fit = mod.fit(
                num_post_samp=num_post_samp,
                prior_coeff_mean=prior_coeff_mean,
                prior_coeff_cov=prior_coeff_cov,
                prior_err_var_shape=prior_err_var_shape,
                prior_err_var_scale=prior_err_var_scale,
                zellner_g=zellner_g,
            )

        x = (
            x[mod.predictors_names]
            .to_numpy()
            .reshape(-1, len(mod.predictors_names))
        )
        fit_vals = x @ fit.post_coeff_mean

        if offset_var is not None:
            fit_vals_back_transform = fit_vals + data[[offset_var]].to_numpy()
        else:
            fit_vals_back_transform = None

        prior = mod.prior
        pred_vars = mod.predictors_names
        mod.posterior_predictive_distribution()
        student_resid = stud_resid(
            response=mod.response,
            predictors=mod.predictors,
            prior_coeff_prec=mat_inv(prior.prior_coeff_cov),
            prior_coeff_mean=prior.prior_coeff_mean
        )
        waic = mod.waic()

        # Adjust design and projection matrix if a transform to the data was made
        # to get correct LOO error estimates.
        if leverage_predictors_adj is None:
            oos_error = mod.oos_error()
        else:
            num_pred_add = leverage_predictors_adj.shape[1]
            lev_predictors = np.c_[leverage_predictors_adj, x]
            num_lev_pred = lev_predictors.shape[1]
            lev_prior_coeff_cov = np.zeros((num_lev_pred, num_lev_pred))

            # Adjust prior coefficient covariance matrix to reflect new
            # design matrix. Make the prior vague.
            for m in range(num_pred_add):
                lev_prior_coeff_cov[m, m] = 1e6

            lev_prior_coeff_cov[num_pred_add:, num_pred_add:] = prior.prior_coeff_cov

            oos_error = mod.oos_error(
                leverage_predictors=lev_predictors,
                leverage_prior_coeff_cov=lev_prior_coeff_cov
            )

        rmspe = np.sqrt(np.mean(mod.mspe().mean_squared_prediction_error))
        r_sqr = np.mean(mod.r_sqr())
        r_sqr_classic = np.mean(mod.r_sqr_classic())

        return FitResults(
            fit,
            fit_vals,
            fit_vals_back_transform,
            prior,
            waic,
            oos_error,
            student_resid,
            rmspe,
            r_sqr,
            r_sqr_classic,
            pred_vars
        )

    def _predict(self,
                 fit: FitResults,
                 data: pd.DataFrame,
                 prediction_type: Literal["standard", "forecast"]
                 ) -> np.ndarray:

        if prediction_type not in ("standard", "forecast"):
            raise ValueError(
                "prediction_type must be either 'standard' or 'forecast'"
            )

        b = fit.fit.post_coeff_mean
        x = data[fit.pred_vars]
        coeff_map = dict(zip(fit.pred_vars, b.flatten()))
        num_rows = x.shape[0]

        if self.dep_var_offset is not None:
            offset = (data[self.dep_var_offset]
                      .to_numpy()
                      .reshape(-1, 1)
                      )
        else:
            offset = np.zeros((num_rows, 1))

        def dynamic_forecast(dv_last_obs,
                             dv_ar_coeff,
                             iv_pred_design_mat,
                             iv_pred_coeff):

            # Need offset for lagged, centered DV when centering is used
            dv_lag_offset = 0.
            if self.transform == "center":
                dv_lag_offset = (data[f"{self.dep_var_lag}"]
                                 - data[f"{self.dep_var_lag}_"
                                        f"{self._var_center_postfix}"]
                                 )
                dv_lag_offset = dv_lag_offset.iloc[0]

            h = iv_pred_design_mat.shape[0]
            pred = np.zeros(h)
            dv_part = np.zeros(h)
            iv_part = np.zeros(h)
            iv_part[0] = iv_pred_design_mat[0, :] @ iv_pred_coeff
            for r in range(1, h + 1):
                dv_part[r - 1] = dv_ar_coeff ** r * dv_last_obs
                if self.transform == "center":
                    dv_part[r - 1] += (
                            dv_ar_coeff
                            * (1 - dv_ar_coeff ** (r - 1))
                            / (1 - dv_ar_coeff ** r)
                            * (offset[0] - dv_lag_offset)
                    )
                if r > 1:
                    iv_part[r - 1] = (
                            iv_pred_design_mat[r - 1, :] @ iv_pred_coeff
                            + dv_ar_coeff * iv_part[r - 2]
                    )

                pred[r - 1] = dv_part[r - 1] + iv_part[r - 1]

            if self.transform == "center":
                pred = offset[0] + pred
            elif self.transform == "difference":
                dv_last_lvl_obs = data[(f"{self.dep_var}_"
                                        f"{self._var_last_postfix}"
                                        )].iloc[0]
                pred = dv_last_lvl_obs + np.cumsum(pred)

            return pred.reshape(-1, 1)

        if prediction_type == "standard":
            x = x.to_numpy()
            prediction = x @ b + offset
        else:
            if self.transform in ("center", None) and not self.make_dynamic:
                x = x.to_numpy()
                prediction = x @ b + offset
            elif self.transform is None and self.make_dynamic:
                dv_lag_var = f"{self.dep_var_lag}"
                dv_lag_coeff = coeff_map[dv_lag_var]
                dv_last_var = f"{self.dep_var}_{self._var_last_postfix}"
                dv_last = data[dv_last_var].iloc[0]
                x_iv = x[[c for c in x.columns if c != dv_lag_var]].to_numpy()
                x_iv_coeff = np.array([v for k, v in coeff_map.items()
                                       if k != dv_lag_var]
                                      )

                prediction = dynamic_forecast(
                    dv_last_obs=dv_last,
                    dv_ar_coeff=dv_lag_coeff,
                    iv_pred_design_mat=x_iv,
                    iv_pred_coeff=x_iv_coeff
                )

            elif self.transform == "center" and self.make_dynamic:
                dv_lag_cent_var = (f"{self.dep_var_lag}_"
                                   f"{self._var_center_postfix}"
                                   )
                dv_lag_cent_coeff = coeff_map[dv_lag_cent_var]
                dv_last_cent_var = (f"{self.dep_var}_"
                                    f"{self._var_center_postfix}_"
                                    f"{self._var_last_postfix}"
                                    )
                dv_last_cent = data[dv_last_cent_var].iloc[0]
                x_iv = x[[c for c in x.columns if c != dv_lag_cent_var]].to_numpy()
                x_iv_coeff = np.array([v for k, v in coeff_map.items()
                                       if k != dv_lag_cent_var]
                                      )

                prediction = dynamic_forecast(
                    dv_last_obs=dv_last_cent,
                    dv_ar_coeff=dv_lag_cent_coeff,
                    iv_pred_design_mat=x_iv,
                    iv_pred_coeff=x_iv_coeff
                )

            elif self.transform == "difference" and not self.make_dynamic:
                x = x.to_numpy()
                pred_diff = x @ b
                dv_last_var = (f"{self.dep_var}_"
                               f"{self._var_last_postfix}"
                               )
                dv_last = data[dv_last_var].iloc[0]
                prediction = dv_last + np.cumsum(pred_diff)
            else:
                dv_lag_diff_var = (f"{self.dep_var_lag}_"
                                   f"{self._var_diff_postfix}"
                                   )
                dv_lag_diff_coeff = coeff_map[(f"{self.dep_var_lag}_"
                                               f"{self._var_diff_postfix}"
                                               )]
                dv_last_diff_var = (f"{self.dep_var}_"
                                    f"{self._var_diff_postfix}_"
                                    f"{self._var_last_postfix}"
                                    )
                dv_last_diff = data[dv_last_diff_var].iloc[0]
                x_iv = x[[c for c in x.columns if c != dv_lag_diff_var]].to_numpy()
                x_iv_coeff = np.array([v for k, v in coeff_map.items()
                                       if k != dv_lag_diff_var]
                                      )

                prediction = dynamic_forecast(
                    dv_last_obs=dv_last_diff,
                    dv_ar_coeff=dv_lag_diff_coeff,
                    iv_pred_design_mat=x_iv,
                    iv_pred_coeff=x_iv_coeff
                )

        return prediction

    def group_fit(
            self,
            data: PreparedPanelData,
            num_post_samp: int,
            prior_coeff_mean: Union[np.ndarray, list, tuple] = None,
            prior_coeff_cov: Union[np.ndarray, list, tuple] = None,
            prior_err_var_shape: Union[int, float] = None,
            prior_err_var_scale: Union[int, float] = None,
            zellner_g: Union[int, float] = None,
    ) -> FitResults:

        if data.data_type == "out-of-sample":
            warnings.warn(
                "PreparedPanelData object is tagged as out-of-sample "
                "(see 'data_type' attribute). Ignore if the intention is "
                "to fit a model on an out-of-sample set.")

        if data.transform is None:
            dep_var = data.dep_var
            predictor_vars = data.predictor_vars
            offset_var = None
        else:
            transform_vars = data.transform_vars
            dep_var = transform_vars[0]
            predictor_vars = transform_vars[1:]
            offset_var = self.dep_var_offset

        df = (pd.concat([j[1] for j in data.member_dfs])
              .reset_index(drop=True)
              )

        # Data transformation can affect the leverage matrix for the LOO CV calculation.
        # An adjustment needs to be applied to the design matrix and prior coefficient
        # precision matrix in some cases.
        if self.transform is None:
            lev_x_grp_adj = None
        elif self.transform == "center":
            lev_x_grp_adj = (
                    pd
                    .get_dummies(df[self.unique_id_var])
                    .to_numpy() * 1.
            )
        else:
            lev_x_grp_adj = None

        group_res = self._fit(
            data=df,
            dep_var=dep_var,
            predictor_vars=predictor_vars,
            num_post_samp=num_post_samp,
            prior_coeff_mean=prior_coeff_mean,
            prior_coeff_cov=prior_coeff_cov,
            prior_err_var_shape=prior_err_var_shape,
            prior_err_var_scale=prior_err_var_scale,
            zellner_g=zellner_g,
            leverage_predictors_adj=lev_x_grp_adj,
            offset_var=offset_var
        )

        return group_res

    def member_fit(
            self,
            data: Union[PreparedPanelData, list],
            num_post_samp: int,
            prior_coeff_mean: list = None,
            prior_coeff_cov: list = None,
            prior_err_var_shape: list = None,
            prior_err_var_scale: list = None,
            zellner_g: list = None,
    ) -> list:

        if self.transform is None:
            dep_var = self.dep_var
            predictor_vars = self.predictor_vars
            offset_var = None
        else:
            transform_vars = self.transform_vars
            dep_var = transform_vars[0]
            predictor_vars = transform_vars[1:]
            offset_var = self.dep_var_offset

        if isinstance(data, PreparedPanelData):
            data = data.member_dfs

        num_members = len(data)
        mem_ids = [j[0] for j in data]
        df_m = [j[1] for j in data]
        y_m = [j[dep_var] for j in df_m]
        pred_vars_m = [[c for c in predictor_vars if c in j.columns] for j in df_m]

        # Data transformation can affect the leverage matrix for the LOO CV calculation.
        # An adjustment needs to be applied to the design matrix and prior coefficient
        # precision matrix in some cases.
        if self.transform is None:
            lev_x_grp_adj = repeat(None)
        elif self.transform == "center":
            lev_x_grp_adj = [np.ones((j.size, 1)) for j in y_m]
        else:
            lev_x_grp_adj = repeat(None)

        if prior_coeff_mean is not None:
            if len(prior_coeff_mean) == 1:
                prior_coeff_mean = repeat(prior_coeff_mean[0])
            else:
                if len(prior_coeff_mean) != num_members:
                    raise UniqueIDLengthMismatch("prior_coeff_mean")
        else:
            prior_coeff_mean = repeat(prior_coeff_mean)

        if prior_coeff_cov is not None:
            if len(prior_coeff_cov) == 1:
                prior_coeff_cov = repeat(prior_coeff_cov[0])
            else:
                if len(prior_coeff_cov) != num_members:
                    raise UniqueIDLengthMismatch("prior_coeff_cov")
        else:
            prior_coeff_cov = repeat(prior_coeff_cov)

        if prior_err_var_shape is not None:
            if len(prior_err_var_shape) == 1:
                prior_err_var_shape = repeat(prior_err_var_shape[0])
            else:
                if len(prior_err_var_shape) != num_members:
                    raise UniqueIDLengthMismatch("prior_err_var_shape")
        else:
            prior_err_var_shape = repeat(prior_err_var_shape)

        if prior_err_var_scale is not None:
            if len(prior_err_var_scale) == 1:
                prior_err_var_scale = repeat(prior_err_var_scale[0])
            else:
                if len(prior_err_var_scale) != num_members:
                    raise UniqueIDLengthMismatch("prior_err_var_scale")
        else:
            prior_err_var_scale = repeat(prior_err_var_scale)

        if zellner_g is not None:
            if len(zellner_g) == 1:
                zellner_g = repeat(zellner_g[0])
            else:
                if len(zellner_g) != num_members:
                    raise UniqueIDLengthMismatch("zellner_g")
        else:
            zellner_g = repeat(zellner_g)

        with Pool() as pool:
            mem_res = pool.starmap(
                self._fit,
                zip(
                    df_m,
                    repeat(dep_var),
                    pred_vars_m,
                    repeat(num_post_samp),
                    prior_coeff_mean,
                    prior_coeff_cov,
                    prior_err_var_shape,
                    prior_err_var_scale,
                    zellner_g,
                    lev_x_grp_adj,
                    repeat(offset_var)
                ),
            )
            pool.close()
            pool.join()

        # Map each member's results to their ID
        mem_res = [(g, mem_res[i])
                   for i, g in enumerate(mem_ids)
                   ]

        return mem_res

    def hierarchical_fit(
            self,
            data: Union[pd.DataFrame, PreparedPanelData],
            group_fit_results: FitResults,
            member_num_post_samp: int = 500,
            group_post_cov_shrink_factor: Union[int, float] = 1.0,
    ) -> FitResults:
        if not isinstance(data, (pd.DataFrame, PreparedPanelData)):
            raise TypeError(
                "data must be a Pandas Dataframe or a PreparedPanelData object."
            )

        if group_post_cov_shrink_factor < 0:
            raise ValueError(
                "group_posterior_shrink_factor must be a non-negative number."
            )

        if isinstance(data, pd.DataFrame):
            data = self.prepare_data(data)

        if data.data_type == "out-of-sample":
            warnings.warn(
                "PreparedPanelData object is tagged as out-of-sample "
                "(see 'data_type' attribute). Ignore if the intention is "
                "to fit a model on an out-of-sample set."
            )

        if data.transform is not None:
            pred_vars = data.transform_vars[1:]
        else:
            pred_vars = data.predictor_vars

        num_pred_vars = len(pred_vars)

        # Retrieve the group-level, coefficient posterior mean and covariance matrix
        grp_fit = group_fit_results.fit
        grp_post_coeff_mean = grp_fit.post_coeff_mean
        grp_post_coeff_cov_diag = np.diag(np.diag(grp_fit.post_coeff_cov))

        def shrinkage_factor(cov_mat):
            grp_r_sqr = group_fit_results.r_sqr
            grp_post_err_var = grp_fit.post_err_var_scale / grp_fit.post_err_var_shape
            num_pred = cov_mat.shape[0]
            # Grab trace of the posterior covariance matrix.
            # This will be used to scale the covariance matrix.

            # De-scale the covariance matrix by dividing out the
            # posterior variance of the error term. This is done
            # before taking the trace and determinant to help avoid
            # issues with numerical overflow. The posterior variance
            # will be re-introduced later.
            trace_grp_post_coeff_cov = np.trace(cov_mat / grp_post_err_var)

            # Re-introduce posterior variance.
            tr = trace_grp_post_coeff_cov / num_pred * grp_post_err_var

            # Shrinkage factors based on group posterior confidence
            if group_post_cov_shrink_factor == 0:
                g = 1.0
            else:
                g = (group_post_cov_shrink_factor
                     * (1 / tr * (1 - grp_r_sqr) / grp_r_sqr))

                # The trace and/or the determinant of the covariance matrix
                # can be a very large number, resulting in shrinkage factors
                # that are excessive. If a shrinkage factor exceeds
                # the group-level posterior error variance, then fall back
                # to the inverse of the posterior variance.
                if 1 / (g * grp_post_err_var) > 1:
                    g = 1 / grp_post_err_var

            return g

        """
        Get shrinkage factors for each member in the group.
        Shrinkage is based on the group-level posterior
        coefficient covariance matrix. Because the group level
        regression is pooled, it's possible that some member's
        in the group could have predictors without variation.
        For example, Member 1 one might have values [1, 2, 3]
        for predictor X, but Member 2 might have values [0, 0, 0]
        or [3, 3, 3] (note that [1, 1, 1] is valid as that
        represents an intercept).

        For members that have predictors without variation,
        their shrinkage factor will be based on a slice of
        the group-level covariance matrix. For example, if the 
        covariance matrix is

                        1 0 0
                        0 5 0
                        0 0 8

        and a member has no variation for predictor 2, then 
        shrinkage will be based on the sliced covariance matrix

                        1 0
                        0 8

        That is, the 2nd row and 2nd column are deleted from 
        the full covariance matrix.
        """

        mem_ids = [j[0] for j in data.member_dfs]
        dfs = [j[1].copy() for j in data.member_dfs]
        mem_prior_coeff_mean = []
        mem_prior_coeff_cov = []
        new_dfs = []
        for k, df_m in enumerate(dfs):
            x_m = df_m.loc[:, pred_vars]
            x_m_new, valid_cols = drop_zero_cols(x_m.to_numpy())
            n_m, k_m = x_m_new.shape

            if len(valid_cols) == num_pred_vars:
                new_df = df_m.copy()
            else:
                new_df = (
                    df_m
                    .copy()
                    .drop(columns=[pred_vars[i] for i in range(num_pred_vars) if i not in valid_cols])
                )
            new_dfs.append(new_df)

            if k_m != num_pred_vars:
                cov = grp_post_coeff_cov_diag[
                    np.ix_(valid_cols, valid_cols)
                ]
            else:
                cov = grp_post_coeff_cov_diag

            g = shrinkage_factor(cov)
            mem_prior_coeff_cov.append(g * cov)
            mem_prior_coeff_mean.append(grp_post_coeff_mean[valid_cols, :])

        new_data = list(zip(mem_ids, new_dfs))
        hierarchical_mem_res = self.member_fit(
            data=new_data,
            num_post_samp=member_num_post_samp,
            prior_coeff_mean=mem_prior_coeff_mean,
            prior_coeff_cov=mem_prior_coeff_cov,
            prior_err_var_shape=None,
            prior_err_var_scale=None,
            zellner_g=None,
        )

        return hierarchical_mem_res

    def predict(self,
                fit: list,
                data: PreparedPanelData
                ) -> tuple:

        # Reconstruct member_fit and predictor_data based on matching keys.
        mem_fit_keys = [j[0] for j in fit]
        mem_data_keys = [j[0] for j in data.member_dfs]
        matching_keys = [k for k in mem_fit_keys if k in mem_data_keys]

        if len(matching_keys) == 0:
            raise ValueError(
                "No keys match between mem_fit and predictor_data objects."
            )

        # Record non-matching keys.
        non_matching_keys = {
            "exclusive_fit_keys": [k for k in mem_fit_keys
                                   if k not in mem_data_keys],
            "exclusive_predictor_data_keys": [k for k in mem_data_keys
                                              if k not in mem_fit_keys],
        }

        # Align fit and data across member keys
        mem_fit = sorted([j for j in fit
                          if j[0] in matching_keys],
                         key=itemgetter(0))

        mem_data = sorted(
            [j for j in data.member_dfs
             if j[0] in matching_keys],
            key=itemgetter(0)
        )

        # Get dataframes and fits for each member
        dfs = [j[1] for j in mem_data]
        fits = [j[1] for j in mem_fit]

        # Get predictions
        with Pool() as pool:
            predictions = pool.starmap(
                self._predict,
                zip(fits, dfs, repeat(data.prediction_type)),
            )

        # Collect results into a dataframe
        num_members = len(mem_fit)
        panel_date = [mem_data[i][1][[self.unique_id_var, self.date_var]]
                      for i in range(num_members)
                      ]

        predictions_df = []
        for i, p in enumerate(predictions):
            df = panel_date[i].copy()
            df["PREDICTION"] = p
            predictions_df.append(df)

        predictions_df = pd.concat(predictions_df).reset_index(drop=True)

        return predictions_df, non_matching_keys
