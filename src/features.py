import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler

from src.configurations import Configuration
from src.helper import check_df_index


class FeatureSet:

    def __init__(self, dataset=None, input_path=None):
        """
        Class to process pre-processed dataset into features to be used in training
        the models.
        """
        self.dataset = dataset
        self.scaler = None
        config = Configuration()
        if input_path is not None:
            self.input_path = input_path
        else:
            self.input_path = config.dedup_flat_device_status_parquet_file
        self.output_path = config.feature_set_csv_file

        if self.dataset is None:
            self.load_preprocessed_data()

    def load_preprocessed_data(self):
        try:
            df = pd.read_parquet(self.input_path)
        except FileNotFoundError:
            raise FileNotFoundError(f'File {self.input_path} not found.')
        except Exception as e:
            raise e

        df = check_df_index(df)

        if 'system' in df.columns:
            df.drop('system', axis=1, inplace=True)

        self.dataset = df.sort_index(level=['id', 'datetime'])

    def add_day_type(self):
        self.dataset['day_type'] = \
            (self.dataset.index.
             get_level_values('datetime').
             weekday.
             map(lambda x: 'weekend' if x >= 5 else 'weekday').
             astype('category'))
        self.dataset = pd.get_dummies(self.dataset, columns=['day_type'],
                                      prefix='day_type')

    def add_rate_of_change(self, columns=None):
        """
        Add columns that provide the rate of change for 15min intervals and
        NaNs where the criteria of sequential 15 min intervals isn't met.
        :param columns: Columns to have the RoC applied.
        """
        self.dataset['time_diff'] = (self.dataset.index.
                                     get_level_values('datetime').diff())
        first_idx = ~self.dataset.index.get_level_values('id').duplicated()
        self.dataset.loc[first_idx, 'time_diff'] = np.nan

        interval = pd.Timedelta('15min')
        for col in columns:
            value_diff = (self.dataset[col].
                          groupby(self.dataset.index.get_level_values('id')).
                          diff())
            rate_of_change = (value_diff.
                              where(self.dataset['time_diff'] == interval))
            self.dataset[f'{col} rate_of_change'] = rate_of_change

    def add_hourly_mean(self, columns=None):
        """
        Adds columns with the hourly mean for specified columns.
        If columns is None, computes for all numeric columns.
        :param columns: Columns to have the hourly mean.
        """
        if columns is None:
            columns = self.dataset.select_dtypes(include=[np.number]).columns

        hour_index = self.dataset.index.get_level_values('datetime').hour
        id_index = self.dataset.index.get_level_values('id')

        for col in columns:
            hourly_mean = (
                self.dataset[col]
                .groupby([id_index, hour_index])
                .transform('mean')
            )
            self.dataset[f'{col} hourly_mean'] = hourly_mean

    def add_last_cob_peak(self):
        pass

    def add_last_iob_peak(self):
        pass

    def add_time_based_features(self):
        """
        Adds three time-based features: the hour of the day and then
        trigonometric features of sin and cos of the hour.
        """
        def sin_transformer(period):
            return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))

        def cos_transformer(period):
            return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))

        self.dataset['hour_of_day'] = (self.dataset.index.
                                       get_level_values('datetime').
                                       hour.astype(float))

        self.dataset['hour_sin'] = (
            sin_transformer(24).fit_transform(self.dataset)['hour_of_day'])
        self.dataset['hour_cos'] =(
            cos_transformer(24).fit_transform(self.dataset)['hour_of_day'])

    def read_feature_set_from_file(self):
        try:
            self.dataset = (pd.read_csv(self.output_path).
                                set_index(['id', 'datetime']).
                                drop(columns='Unnamed: 0').
                                sort_index(level=['id', 'datetime']))
        except FileNotFoundError:
            raise FileNotFoundError(f'File {self.output_path} not found.')

    def export_features_file(self):
        pd.to_csv(self.output_path, index=False)

    def scale_features(self, columns):
        self.scaler = MinMaxScaler()
        self.dataset[columns] = self.scaler.fit_transform(self.dataset[columns])

    def inverse_scale_features(self, columns):
        if self.scaler is None:
            raise ValueError("Scaler has not been fitted.")
        self.dataset[columns] = self.scaler.inverse_transform(
            self.dataset[columns])

    def save_scaler(self, path):
        joblib.dump(self.scaler, path)

    def load_scaler(self, path):
        self.scaler = joblib.load(path)
