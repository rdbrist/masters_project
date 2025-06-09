import pandas as pd
import numpy as np
from sklearn.preprocessing import FunctionTransformer

from src.configurations import Configuration

class FeatureSet:
    """
    Class to process pre-processed dataset into features to be used in training
    the Hidden Markov Model.
    """
    def __init__(self, dataset=None, input_path=None):
        self.dataset = dataset
        config = Configuration()
        if input_path is not None:
            self.input_path = input_path
        else:
            self.input_path = config.dedup_flat_device_status_parquet_file
        self.output_path = config.feature_set_csv_file

        if self.dataset is None:
            self.load_preprocessed_data()

        # Add all features
        self.add_time_based_features()
        self.add_day_type()
        self.add_rate_of_change()

        # Export file for reference and use
        self.export_features_file()

    def load_preprocessed_data(self):
        try:
            df = pd.read_parquet(self.input_path)
        except FileNotFoundError:
            raise FileNotFoundError(f'File {self.input_path} not found.')
        except Exception as e:
            raise e
        if not isinstance(df.index, pd.MultiIndex):
            raise ValueError("DataFrame index must be a MultiIndex")
        if list(df.index.names) != ["id", "datetime"]:
            raise ValueError(
                "DataFrame index must be a MultiIndex with levels "
                "['id', 'datetime'].")
        id_level = df.index.get_level_values('id')
        datetime_level = df.index.get_level_values('datetime')
        if not pd.api.types.is_integer_dtype(id_level):
            raise ValueError("Index level 'id' must be of integer dtype.")
        if not pd.api.types.is_datetime64_any_dtype(datetime_level):
            raise ValueError(
                "Index level 'datetime' must be of datetime dtype.")
        self.dataset = df.sort_index(level=['id', 'datetime'])

    def add_day_type(self):
        self.dataset.index.get_level_values('datetime').weekday.map(
            lambda x: 'weekend' if x >= 5 else 'weekday').astype('category')
        self.dataset = pd.get_dummies(self.dataset, columns=['day_type'],
                                      prefix='day_type')

    def add_rate_of_change(self):
        self.dataset['time_diff'] = (self.dataset.index.
                                     get_level_values('datetime').diff())
        first_idx = ~self.dataset.index.get_level_values('id').duplicated()
        self.dataset.loc[first_idx, 'time_diff'] = np.nan
        self.dataset.head()

        interval = pd.Timedelta('15min')
        # Then add rate columns
        for col in ['iob mean', 'cob mean', 'bg mean']:
            value_diff = (self.dataset[col].
                          groupby(self.dataset.index.get_level_values('id')).
                          diff())
            rate_of_change = (value_diff.
                              where(self.dataset['time_diff'] == interval))
            self.dataset[f'{col} rate_of_change'] = rate_of_change

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
                                       get_level_values('datetime').hour)
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
