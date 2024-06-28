import numpy as np
from typing import NamedTuple, Union
import warnings
import pandas as pd

pd.set_option("display.float_format", lambda x: "%.2f" % x)


def fourier_transform(t, p, N):
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
    t : array
        Sequence of ordered integers representing the evolution of time.
        For example, t = [0,1,2,3,4,5, ..., T], where T is the terminal period.

    p: float
        The amount of time it takes for a period/cycle to end. For example,
        if the frequency of data is monthly, then a period completes in 12
        months. If data is daily, then there could conceivably be two period
        lengths, one for every week (a period of length 7) and one for every
        year (a period of length 365.25). Must be positive.

    N : integer
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
    c = 2 * np.pi * np.arange(1, N + 1) / p
    # Create cosine and sine input values, 2 * pi * n / (p / s) * t, t=1,...,T and n=1,...,N
    X = c * t[:, np.newaxis]
    # Pass X to cos() and sin() functions to create Fourier series
    F = np.c_[np.cos(X), np.sin(X)]

    return F


def forecast_cv_data_split(
        data: Union[pd.Series, pd.DataFrame],
        forecast_horizon: int,
        num_cv_test_sets: int,
        num_cv_val_sets: int,
        roll_back_cv_test_step_size: int = 1) -> tuple:
    if num_cv_val_sets > 0:
        val_set_size = forecast_horizon + num_cv_val_sets - 1
    else:
        val_set_size = 0

    num_obs = data.shape[0]
    cv_test_sets = []
    cv_val_sets = []

    if (
            num_obs - val_set_size
            <= (num_cv_test_sets - 1) * roll_back_cv_test_step_size + forecast_horizon
    ):
        warnings.warn(
            "The number of observations reserved for training, after accounting "
            "for the size of the validation set, is not sufficient given the "
            "desired number of CV test sets, roll-back step size, and forecast "
            "horizon. No CV sets will be created."
        )
        return cv_test_sets, cv_val_sets

    if val_set_size > 0:
        df_test = data.iloc[:-val_set_size]
    else:
        df_test = data

    test_num_obs = df_test.shape[0]
    k = 0
    for j in range(num_cv_test_sets):
        test_hist_start, test_hist_end = 0, test_num_obs - forecast_horizon - k
        test_fut_start, test_fut_end = (
            test_num_obs - forecast_horizon - k,
            test_num_obs - k,
        )
        df_test_hist = df_test.iloc[test_hist_start:test_hist_end]
        df_test_fut = df_test.iloc[test_fut_start:test_fut_end]
        cv_test_sets.append((df_test_hist, df_test_fut))

        k += roll_back_cv_test_step_size

    if val_set_size > 0:
        k = 0
        for j in range(num_cv_val_sets):
            val_hist_start, val_hist_end = 0, num_obs - val_set_size + k
            val_fut_start, val_fut_end = (
                num_obs - val_set_size + k,
                num_obs - val_set_size + k + forecast_horizon,
            )

            df_val_hist = data.iloc[val_hist_start:val_hist_end]
            df_val_fut = data.iloc[val_fut_start:val_fut_end]
            cv_val_sets.append((df_val_hist, df_val_fut))

            k += 1

    return cv_test_sets, cv_val_sets


class PreparedPanelData(NamedTuple):
    member_dfs: list
    unique_id_var: str
    date_var: str
    dep_var: str
    predictor_vars: list
    transform: Union[None, str]
    transform_vars: Union[None, list]
    dynamic: bool
    data_type: str
    num_obs: int
    num_members: int


class ProcessPanelRegressionData:
    def __init__(
            self,
            unique_id_var: str,
            date_var: str,
            dep_var: str,
            predictor_vars: list,
            transform: Union[None, str] = None,
            make_dynamic: bool = False
    ):
        if transform not in (None, "difference", "center"):
            raise ValueError(
                "Valid options for 'transform' are "
                "'difference', 'center', and None."
            )

        self.unique_id_var = unique_id_var
        self.date_var = date_var
        self.dep_var = dep_var
        self.predictor_vars = predictor_vars
        self.make_dynamic = make_dynamic
        self.transform = transform
        self.has_constant = None
        self.constant_index = None
        self.constant_name = None
        self.dep_var_offset = None
        self.dep_var_lag = None
        self.transform_vars = None
        self._var_center_postfix = "CENT"
        self._var_diff_postfix = "DIFF"
        self._var_last_postfix = "LAST"

    def prepare_data(
            self,
            data: pd.DataFrame,
            forecast_dates: Union[pd.DatetimeIndex, None] = None
    ) -> PreparedPanelData:
        df = data.copy()
        unique_id_var = self.unique_id_var
        date_var = self.date_var
        dep_var = self.dep_var
        predictor_vars = self.predictor_vars
        data_cols = [dep_var] + predictor_vars
        df[data_cols] = df[data_cols].astype(float)
        predictor_data = df[predictor_vars].to_numpy().flatten()
        data_type = "in-sample"

        if self.transform == "difference":
            self.dep_var_offset = f"{dep_var}_{self._var_diff_postfix}_OFFSET"
        elif self.transform == "center":
            self.dep_var_offset = f"{dep_var}_{self._var_center_postfix}_OFFSET"

        # Sort data
        df = (
            df
            .sort_values(by=[unique_id_var, date_var])
        )

        # Check if predictor data has non-numeric values
        if np.any(np.isnan(predictor_data)) or np.any(np.isinf(predictor_data)):
            raise ValueError(
                "None of the predictor data can have null or non-finite values."
            )

        # Check if target data has non-numeric values
        target_data = df[[dep_var, date_var]].set_index(date_var)
        if forecast_dates is None:
            target_data = target_data.to_numpy().flatten()

            if np.any(np.isnan(target_data)) or np.any(np.isinf(target_data)):
                raise ValueError(
                    "None of the target data can have null or non-finite values."
                )
        else:
            sorted_forecast_dates = sorted(list(forecast_dates))
            first_forecast_date = sorted_forecast_dates[0]
            target_data_hist = target_data[target_data.index < first_forecast_date]
            target_data_hist = target_data_hist.to_numpy().flatten()

            if np.any(np.isnan(target_data_hist)) or np.any(np.isinf(target_data_hist)):
                raise ValueError(
                    "For dates preceding the specified forecast dates, none of "
                    "the target data can have null or non-finite values."
                )

            # Enforce NaN values for dependent variable at forecast dates
            df.loc[df[date_var] >= first_forecast_date, dep_var] = np.nan

        # Check if design matrix has one or more constant values
        const_x = np.all(df[predictor_vars] == df[predictor_vars].iloc[0, :], axis=0)
        has_const_ = np.all(df[predictor_vars] == 1, axis=0)
        if np.any(has_const_):
            self.has_constant = True
            self.constant_index = np.argwhere(has_const_)[0][0]

            if np.sum(const_x) > 1:
                warnings.warn(
                    "More than one column in the predictor matrix is constant."
                )

            self.constant_name = predictor_vars[self.constant_index]
        else:
            self.has_constant = False

        if self.make_dynamic:
            self.dep_var_lag = f"{dep_var}_LAG"
            self.predictor_vars = [self.dep_var_lag] + predictor_vars
            df[self.dep_var_lag] = (df
                                    .groupby(unique_id_var)[dep_var]
                                    .transform(lambda x: x.shift(1))
                                    )

            # Drop null values from 1-period shift
            first_row = df.groupby(unique_id_var, as_index=False).nth(0)
            df = pd.concat([df, first_row]).drop_duplicates(keep=False)

            if self.transform is None:
                transform_vars = None
            else:
                df, transform_vars = self.groupby_transform(
                    data=df,
                    transform=self.transform
                )
        else:
            if self.transform is None:
                transform_vars = None
            else:
                df, transform_vars = self.groupby_transform(
                    data=df,
                    transform=self.transform
                )

        # Organize columns
        if self.transform is None:
            df = df[[unique_id_var, date_var, dep_var]
                    + self.predictor_vars
                    ]
        else:
            self.transform_vars = transform_vars
            df = df[[unique_id_var, date_var, dep_var]
                    + self.predictor_vars
                    + [self.dep_var_offset]
                    + transform_vars
                    ]

        df = df.reset_index(drop=True)
        num_obs = df.shape[0]

        members = df.groupby(unique_id_var)
        num_members = len(members)
        member_dfs = [(m, df) for m, df in members]

        if forecast_dates is not None:
            forecast_horizon = len(forecast_dates)
            return self.forecast_data_split(
                data=member_dfs,
                forecast_horizon=forecast_horizon,
                num_cv_test_sets=1
            )
        else:
            return PreparedPanelData(
                member_dfs,
                unique_id_var,
                date_var,
                dep_var,
                self.predictor_vars,
                self.transform,
                transform_vars,
                self.make_dynamic,
                data_type,
                num_obs,
                num_members
            )

    def _transform_var_names(self,
                             transform: str):
        if self.has_constant:
            data_vars = [self.dep_var] + [j for j in self.predictor_vars
                                          if j != self.constant_name
                                          ]
        else:
            data_vars = [self.dep_var] + self.predictor_vars

        if transform == "difference":
            postfix = self._var_diff_postfix
        else:
            postfix = self._var_center_postfix

        transform_vars = [f"{v}_{postfix}" for v in data_vars]

        return transform_vars, data_vars

    def groupby_transform(self,
                          data: Union[pd.DataFrame, PreparedPanelData],
                          transform: Union[None, str]
                          ) -> tuple:

        if transform not in ("center", "difference"):
            raise ValueError(
                f"Valid data transformations are 'center' and 'difference'. "
                f"Received {transform}."
            )

        if not isinstance(data, (pd.DataFrame, PreparedPanelData)):
            raise TypeError(
                "Data must be a Pandas Dataframe or a PreparedPanelData"
                "object, where the latter is generated by calling method "
                "prepare_date()."
            )

        if isinstance(data, PreparedPanelData):
            df = pd.concat([j[1] for j in data.member_dfs])
        else:
            df = data.copy()

        transform_vars, data_vars = self._transform_var_names(transform)

        # Drop groups with one observation
        mask = (
            df
            .groupby(self.unique_id_var)[self.unique_id_var]
            .transform("count")
        )
        df = df.loc[mask > 1].reset_index(drop=True)

        # Transform
        if transform == "difference":
            # Sort data
            df = (
                df
                .sort_values(by=[self.unique_id_var, self.date_var])
            )

            df[transform_vars] = (
                df
                .groupby(self.unique_id_var)[data_vars]
                .transform(lambda x: x - x.shift(1))
            )

            df[self.dep_var_offset] = (
                df
                .groupby(self.unique_id_var)[self.dep_var]
                .transform(lambda x: x.shift(1))
            )

            # Drop null values from 1-period shift
            first_row = df.groupby(self.unique_id_var, as_index=False).nth(0)
            df = pd.concat([df, first_row]).drop_duplicates(keep=False)
        else:
            df[transform_vars] = (
                df
                .groupby(self.unique_id_var)[data_vars]
                .transform(lambda x: x - x.mean())
            )

            df[self.dep_var_offset] = (
                df
                .groupby(self.unique_id_var)[self.dep_var]
                .transform(lambda x: x.mean())
            )

        return df, transform_vars

    def forecast_data_split(
            self,
            data: Union[PreparedPanelData, list],
            forecast_horizon: int,
            num_cv_test_sets: int,
            # num_cv_val_sets: int,
            roll_back_cv_test_step_size: int = 1
    ) -> Union[PreparedPanelData, list]:

        if isinstance(data, PreparedPanelData):
            member_dfs = data.member_dfs
        else:
            member_dfs = data

        # Collect CV sets for each member
        cv_sets = []
        for j in member_dfs:
            m, df_m = j

            test_sets, _ = (forecast_cv_data_split(
                data=df_m,
                forecast_horizon=forecast_horizon,
                num_cv_test_sets=num_cv_test_sets,
                num_cv_val_sets=0,
                roll_back_cv_test_step_size=roll_back_cv_test_step_size)
            )

            if len(test_sets) == 0:
                # This happens when there is not enough observations for CV
                cv_sets.append(None)
            else:
                if self.transform in ("difference", None):
                    adj_test_sets = []
                    # To generate forecasts, need last recorded
                    # training value for the dependent variable.
                    for s in test_sets:
                        df_in_samp = s[0].copy()
                        df_out_samp = s[1].copy()
                        dep_var_last = df_in_samp[self.dep_var].iloc[-1]
                        df_out_samp[f"{self.dep_var}_{self._var_last_postfix}"] = dep_var_last

                        # To generate forecasts under a difference transform,
                        # need last recorded training value for the differenced
                        # dependent variable.
                        if self.transform == "difference":
                            dep_var_diff = self.transform_vars[0]
                            dep_var_diff_last = df_in_samp[dep_var_diff].iloc[-1]
                            df_out_samp[f"{dep_var_diff}_{self._var_last_postfix}"] = dep_var_diff_last

                        adj_test_sets.append((df_in_samp, df_out_samp))

                    cv_sets.append(adj_test_sets)

                else:
                    postfix = self._var_center_postfix
                    data_vars = [self.dep_var] + [c for c in self.predictor_vars
                                                  if c != self.constant_name]
                    # Centering before splitting introduces out-of-sample
                    # data leakage in the test sets. Need to correct the
                    # transformed data.
                    adj_test_sets = []
                    for s in test_sets:
                        df_in_samp = s[0].copy()
                        df_out_samp = s[1].copy()
                        df_in_samp_means = df_in_samp[data_vars].mean()
                        df_in_samp[self.dep_var_offset] = df_in_samp_means[self.dep_var]
                        df_out_samp[self.dep_var_offset] = df_in_samp_means[self.dep_var]

                        for i in df_in_samp_means.index:
                            df_in_samp[f"{i}_{postfix}"] = (
                                    df_in_samp[i] - df_in_samp_means[i]
                            )
                            df_out_samp[f"{i}_{postfix}"] = (
                                    df_out_samp[i] - df_in_samp_means[i]
                            )

                        # To generate forecasts, need last recorded
                        # training value for the dependent variable.
                        dep_var_last = df_in_samp[self.dep_var].iloc[-1]
                        df_out_samp[f"{self.dep_var}_{self._var_last_postfix}"] = dep_var_last
                        dep_var_cent = self.transform_vars[0]
                        dep_var_cent_last = df_in_samp[dep_var_cent].iloc[-1]
                        df_out_samp[f"{dep_var_cent}_{self._var_last_postfix}"] = dep_var_cent_last

                        adj_test_sets.append((df_in_samp, df_out_samp))

                    cv_sets.append(adj_test_sets)

        # Create pairs of (train, test) PreparedPanelData objects
        cv_splits = []
        for k in range(num_cv_test_sets):
            in_samp = []
            out_samp = []
            num_obs_in_samp = 0
            num_obs_out_samp = 0
            for i, j in enumerate(member_dfs):
                if cv_sets[i] is None:
                    pass
                else:
                    hist, fut = cv_sets[i][k]
                    in_samp.append((j[0], hist))
                    out_samp.append((j[0], fut))
                    num_obs_in_samp += len(hist)
                    num_obs_out_samp += len(fut)

            cv_in_samp = PreparedPanelData(
                in_samp,
                self.unique_id_var,
                self.date_var,
                self.dep_var,
                self.predictor_vars,
                self.transform,
                self.transform_vars,
                self.make_dynamic,
                "in-sample",
                num_obs_in_samp,
                len(member_dfs)
            )

            cv_out_samp = PreparedPanelData(
                out_samp,
                self.unique_id_var,
                self.date_var,
                self.dep_var,
                self.predictor_vars,
                self.transform,
                self.transform_vars,
                self.make_dynamic,
                "out-of-sample",
                num_obs_out_samp,
                len(member_dfs)
            )

            cv_splits.append((cv_in_samp, cv_out_samp))

        return cv_splits
