import pandas as pd
import numpy as np
import joblib

from scipy.signal import find_peaks
from pathlib import Path
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler

from src.configurations import Configuration
from src.helper import check_df_index


class FeatureSet:

    def __init__(self, df: pd.DataFrame=None, input_path: Path=None,
                 sample_rate: int=30):
        """
        Class to process pre-processed dataset into features to be used in training
        the models.
        :param df: DataFrame containing pre-processed data.
        :param input_path: Path to the pre-processed data file.
        :param sample_rate: Sample rate in minutes for the data.
        """
        self.df = df
        self.scaler = None
        self.sample_rate = sample_rate
        self.mean_cols = ['iob mean', 'cob mean', 'bg mean']
        self.minmax_cols = ['iob min', 'cob min', 'bg min',
                            'iob max', 'cob max', 'bg max']
        self.new_feature_cols = []
        self.all_feature_cols = []
        config = Configuration()
        if input_path is not None:
            self.filetype = input_path.suffix
            self.input_path = input_path
        else:
            self.input_path = config.final_filtered_csv
        self.output_path = config.feature_set_csv_file

        if self.df is None:
            self.load_preprocessed_data()
        elif self.df is not None:
            self.df = check_df_index(self.df)

    def load_preprocessed_data(self):
        try:
            if self.filetype == '.parquet':
                df = pd.read_parquet(self.input_path)
            elif self.filetype == '.csv':
                df = pd.read_csv(self.input_path)
            else:
                raise ValueError(f'Unsupported file type: {self.filetype}')
        except FileNotFoundError:
            raise FileNotFoundError(f'File {self.input_path} not found.')
        except Exception as e:
            raise e

        df = check_df_index(df)

        if 'system' in df.columns:
            df.drop('system', axis=1, inplace=True)

        self.df = df.sort_index(level=['id', 'datetime'])

    def add_day_type(self):
        """
        Identify whether the day is a weekend or a weekday and onehot encode as
        features.
        :return:
        """
        self.df['day_type'] = \
            (self.df.index.
             get_level_values('datetime').
             weekday.
             map(lambda x: 'weekend' if x >= 5 else 'weekday').
             astype('category'))
        onehot_cols = pd.get_dummies(self.df['day_type'], prefix='day_type')
        self.df = pd.concat([self.df.drop(columns=['day_type']),
                             onehot_cols], axis=1)
        self.new_feature_cols.extend(onehot_cols.columns.tolist())

    def add_rate_of_change(self, columns=None):
        """
        Add columns that provide the rate of change for intervals and
        NaNs where the criteria of sequential intervals isn't met.
        :param columns: Columns to have the RoC applied.
        """
        if columns is None:
            columns = self.mean_cols
        self.df['time_diff'] = (self.df.index.
                                     get_level_values('datetime').diff())
        first_idx = ~self.df.index.get_level_values('id').duplicated()
        self.df.loc[first_idx, 'time_diff'] = np.nan

        interval = pd.Timedelta(f'{self.sample_rate}min')
        for col in columns:
            value_diff = (self.df[col].
                          groupby(self.df.index.get_level_values('id')).
                          diff())
            rate_of_change = (value_diff.
                              where(self.df['time_diff'] == interval))
            col_name = f'{col}_rate_of_change'
            self.df[col_name] = rate_of_change
            self.new_feature_cols.append(col_name)

    def add_hourly_mean(self, columns=None):
        """
        Adds columns with the hourly mean for specified columns.
        If columns is None, computes for all numeric columns.
        :param columns: Columns to have the hourly mean.
        """
        if columns is None:
            columns = self.mean_cols

        hour_index = self.df.index.get_level_values('datetime').hour
        id_index = self.df.index.get_level_values('id')

        for col in columns:
            col_name = f'{col} hourly_mean'
            self.df[col_name] = (
                self.df[col]
                .groupby([id_index, hour_index])
                .transform('mean')
            )
            self.new_feature_cols.append(col_name)

    def add_peaks_above_mean(self):
        """
        Identify each peak that is above the mean for the individual
        (grouped by id) for any max variable and update the original df with the
        new features.
        """
        for col in [col for col in self.df.columns if col.endswith('max')]:
            col_name = f'{col}_peaks_above_mean'
            for id_, group in self.df.groupby('id'):
                mean = group[col].mean()
                peaks = find_peaks(group[col], height=mean, distance=2)[0]
                peak_indices = group.index[peaks]
                self.df.loc[group.index, col_name] = 0
                self.df.loc[peak_indices, col_name] = 1
            self.new_feature_cols.append(col_name)

    def add_time_based_features(self):
        """
        Adds three time-based features: the hour of the day and then
        trigonometric features of sin and cos of the hour.
        """
        def sin_transformer(period):
            return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))

        def cos_transformer(period):
            return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))

        self.df['hour_of_day'] = (self.df.index.
                                       get_level_values('datetime').
                                       hour.astype(float))
        self.df['hour_sin'] = (
            sin_transformer(24).fit_transform(self.df['hour_of_day']))
        self.df['hour_cos'] =(
            cos_transformer(24).fit_transform(self.df['hour_of_day']))
        self.new_feature_cols.extend(['hour_of_day', 'hour_sin', 'hour_cos'])

    def get_all_features(self):
        """
        Adds all features to the DataFrame and returns the updated DataFrame
        with the new features, including the original variables, ready for
        model training.
        :return: pd.DataFrame with all features.
        """
        self.add_day_type()
        self.add_rate_of_change(columns=self.mean_cols)
        self.add_hourly_mean(columns=self.mean_cols)
        self.add_peaks_above_mean()
        self.add_time_based_features()
        self.scale_features()
        return self.df[self.get_all_feature_columns()].copy()

    def read_feature_set_from_file(self):
        try:
            self.df = (pd.read_csv(self.output_path).
                                set_index(['id', 'datetime']).
                                drop(columns='Unnamed: 0').
                                sort_index(level=['id', 'datetime']))
        except FileNotFoundError:
            raise FileNotFoundError(f'File {self.output_path} not found.')

    def export_features_file(self):
        self.df[self.get_all_features()].to_csv(self.output_path, index=False)

    def scale_features(self):
        cols = self.get_all_feature_columns()
        print(f'Scaling {cols} columns')
        self.scaler = MinMaxScaler()
        self.df[cols] = self.scaler.fit_transform(self.df[cols])

    def inverse_scale_features(self):
        cols = self.get_all_feature_columns()
        if self.scaler is None:
            raise ValueError("Scaler has not been fitted.")
        self.df[cols] = self.scaler.inverse_transform(
            self.df[cols])

    def save_scaler(self, path):
        joblib.dump(self.scaler, path)

    def load_scaler(self, path):
        self.scaler = joblib.load(path)

    def get_all_feature_columns(self):
        """
        Returns all feature columns including original variables.
        :return: List of all feature columns.
        """
        return (self.mean_cols + self.minmax_cols + self.new_feature_cols)
