import pandas as pd
import numpy as np
from datetime import datetime
from tslearn.barycenters import dtw_barycenter_averaging
from tslearn.utils import to_time_series_dataset

class DBAAverager:
    def __init__(self, df: pd.DataFrame, night_start_hour: int = 17, morning_end_hour: int = 11):
        """
        :param df: DataFrame with datetime index and time series columns.
        :param night_start_hour: Hour when night starts (0-23).
        :param morning_end_hour: Hour when morning ends (0-23).
        """
        self.df = df
        self.cols = df.columns.tolist()
        self.night_start = night_start_hour
        self.morning_end = morning_end_hour
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
            dba_avg = dtw_barycenter_averaging(X_imputed, max_iter=100, tol=1e-3, verbose=False)
            if dba_avg.shape[0] == len(self.full_cycle_times):
                dba_avg_df = pd.DataFrame(dba_avg, columns=self.cols, index=self.full_cycle_times)
                dba_avg_df.index.name = 'time'
                self.dba_averaged_dataframe = dba_avg_df
            else:
                print(f"Shape mismatch: {dba_avg.shape[0]} vs {len(self.full_cycle_times)}")

    def night_hour(self, hour):
        return hour - self.night_start if hour >= self.night_start else 24 - self.night_start + hour

    def get_dba_averaged_dataframe(self):
        """Return the DBA averaged DataFrame."""
        return self.dba_averaged_dataframe

    def get_dba_and_variance_dataframe(self):
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
        var = np.nanvar(X_imputed, axis=0)

        # Build a dict of Series for each (variable, stat) column
        data = {}
        for i, col in enumerate(self.cols):
            data[(col, 'dba')] = pd.Series(dba_avg[:, i],
                                           index=self.full_cycle_times)
            data[(col, 'var')] = pd.Series(var[:, i],
                                           index=self.full_cycle_times)

        result_df = pd.DataFrame(data)
        result_df.index.name = 'time'
        result_df['hour'] = [t.hour for t in result_df.index]
        result_df['night_hour'] = result_df['hour'].apply(self.night_hour)
        return result_df.reset_index()

    def _generate_full_cycle_times(self):
        """Generate a list of time points for the full night cycle."""
        evening = pd.to_timedelta(np.arange(self.night_start, 24, 0.5), unit='hour')
        morning = pd.to_timedelta(np.arange(0, self.morning_end, 0.5), unit='hour')
        return [(datetime.min + td).time() for td in evening.append(morning)]

    def get_night_start_date(self, timestamp, start_hour=None):
        """Get the date corresponding to the start of the night for a timestamp."""
        start_hour = start_hour or self.night_start
        return timestamp.date() if timestamp.hour >= start_hour else (timestamp - pd.Timedelta(days=1)).date()

    def _get_night_profiles(self):
        """Extract and align night profiles from the DataFrame."""
        df_reset = self.df.reset_index()
        df_reset['night_start_date'] = df_reset['datetime'].apply(self.get_night_start_date)
        df_reset['time'] = df_reset['datetime'].dt.time
        return [
            night_df.set_index('time')[self.cols].reindex(self.full_cycle_times).values
            for _, night_df in df_reset.groupby('night_start_date')
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
                    X[i, :, j] = series.interpolate(method='linear', limit_direction='both').values
        return X


if __name__ == "__main__":
    from src.helper import load_final_filtered_csv
    from src.configurations import Configuration

    config = Configuration()
    df = load_final_filtered_csv(config, interpolate_cob=True)

    grouped = df.groupby('id')
    key, df = next(iter(grouped))
    print(df.head())
    dbaa = DBAAverager(df[['iob mean','cob mean','bg mean']], night_start_hour=17, morning_end_hour=11)
    dbaa_avg_df = dbaa.get_dba_averaged_dataframe()
    if dbaa_avg_df is not None:
        print(dbaa_avg_df.head())
    else:
        print("No DBA averaged DataFrame available.")

    print('----------- with variance------------')
    print(dbaa.get_dba_and_variance_dataframe())
