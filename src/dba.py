import pandas as pd
import numpy as np
from datetime import datetime, time
from tslearn.barycenters import dtw_barycenter_averaging
from tslearn.utils import to_time_series_dataset
from src.helper import get_night_start_date


class DBAAverager:
    def __init__(self, df: pd.DataFrame,
                 night_start_hour: int = None,
                 morning_end_hour: int = None):
        """
        :param df: (pd.DataFrame) df with datetime index and time series columns
        :param night_start_hour: (int) Hour when night starts (0-23)
        :param morning_end_hour: (int) Hour when morning ends (0-23)
        """
        if night_start_hour is None or morning_end_hour is None:
            raise ValueError("Both night_start_hour and morning_end_hour must "
                             "be provided.")
        self.df = df
        self.cols = df.columns.tolist()
        self.night_start_hour = night_start_hour
        self.morning_end_hour = morning_end_hour
        self.full_cycle_times = self._generate_full_cycle_times()
        self.dba_averaged_dataframe = None

        night_profiles = self._get_night_profiles()
        if night_profiles:
            night_profiles = [
                np.where(pd.isna(profile), np.nan, profile) for profile in
                night_profiles
            ]
            X = to_time_series_dataset(night_profiles)
            X_imputed = self._impute_missing(X)
            dba_avg = dtw_barycenter_averaging(X_imputed,
                                               max_iter=100,
                                               tol=1e-3,
                                               verbose=False)
            if dba_avg.shape[0] == len(self.full_cycle_times):
                dba_avg_df = pd.DataFrame(dba_avg, columns=self.cols,
                                          index=self.full_cycle_times)
                dba_avg_df.index.name = 'time'
                self.dba_averaged_dataframe = dba_avg_df
            else:
                print(f"Shape mismatch: {dba_avg.shape[0]} vs "
                      f"{len(self.full_cycle_times)}")

    def night_hour(self, hour):
        return (hour - self.night_start_hour if hour >= self.night_start_hour
                else 24 - self.night_start_hour + hour)

    def get_dba_averaged_dataframe(self):
        """Return the DBA averaged DataFrame."""
        return self.dba_averaged_dataframe

    def get_dba_and_variance_dataframe(self, rolling_window=None):
        """
        Return a DataFrame with MultiIndex columns (variable, 'dba' or 'var'),
        containing both the DBA mean and variance for each time point. It also
        includes a 'night_hour' column indicating the hour of the night (for
        ordering), and an 'hour' column for the hour of the day (useful in
        plotting)
        """
        night_profiles = self._get_night_profiles()
        if not night_profiles:
            return None
        X = np.stack(night_profiles)
        X_imputed = self._impute_missing(X)
        dba_avg = self.dba_averaged_dataframe.values

        if rolling_window is not None:
            # Compute rolling variance for each profile
            rolling_vars = []
            for profile in X_imputed:
                df = pd.DataFrame(profile, columns=self.cols,
                                  index=self.full_cycle_times)
                roll_var = df.rolling(window=rolling_window, min_periods=1,
                                      center=True).var()
                rolling_vars.append(roll_var.values)
            var = np.nanmean(np.stack(rolling_vars), axis=0)
        else:
            var = np.nanvar(X_imputed, axis=0)

        # Build a dict of Series for each (variable, stat) column
        data = {}
        for i, col in enumerate(self.cols):
            data[(col, 'dba')] = (
                pd.Series(dba_avg[:, i], index=self.full_cycle_times))
            data[(col, 'var')] = (
                pd.Series(var[:, i], index=self.full_cycle_times))

        result_df = pd.DataFrame(data)
        result_df.index.name = 'time'
        result_df['hour'] = [t.hour for t in result_df.index]
        result_df['night_hour'] = result_df['hour'].apply(self.night_hour)
        return result_df.reset_index()

    def _generate_full_cycle_times(self):
        """Generate a list of time points for the full night cycle."""
        evening = (pd.to_timedelta(np.arange(self.night_start_hour, 24, 0.5),
                                   unit='hour'))
        morning = (pd.to_timedelta(np.arange(0, self.morning_end_hour, 0.5),
                                   unit='hour'))
        return [(datetime.min + td).time() for td in evening.append(morning)]

    def _get_night_profiles(self):
        """
        Extract and align night profiles from the DataFrame.
        :return: (list) List of arrays of variable values for a night dataframe
        """
        df_reset = self.df.reset_index()
        df_reset['night_start_date'] = (
            get_night_start_date(df_reset['datetime'], self.night_start_hour))
        df_reset['time'] = df_reset['datetime'].dt.time

        return [
            night_df.set_index('time')[self.cols].
            reindex(self.full_cycle_times).values
            for _, night_df in df_reset.groupby(['id','night_start_date'])
        ]

    def _impute_missing(self, X):
        """Impute missing values in a 3D array using linear interpolation."""
        if X.dtype == object:
            X = np.where(pd.isna(X), np.nan, X)
        X = X.astype(float)
        for i in range(X.shape[0]):
            for j in range(X.shape[2]):
                series = pd.Series(X[i, :, j]).astype('float64')
                if series.isna().any():
                    X[i, :, j] = (
                        series.interpolate(method='linear',
                                           limit_direction='both').values)
        return X

def get_dba_and_variance(df: pd.DataFrame,
                         night_start: time = None,
                         morning_end: time = None,
                         rolling_window: int = None) -> pd.DataFrame:
    """
    Create dataframe that averages using DBA and produces variance either
    point-in-time or using a rolling window.
    :param df: (pd.DataFrame) DataFrame with datetime index and time series
        columns
    :param night_start: (datetime.time) Night start time
    :param morning_end: (datetime.time) Night end time
    :param rolling_window: (int) Number of intervals for the rolling window or
        do not set to leave as point-in-time
    :return: (pd.DataFrame) DataFrame
    """
    # print(df.index.dtype)
    # if not np.issubdtype(df.index.dtype, np.datetime64):
    #     raise TypeError('Index must be single level DatetimeIndex')
    dba = DBAAverager(df, night_start, morning_end)
    return dba.get_dba_and_variance_dataframe(rolling_window=rolling_window)

