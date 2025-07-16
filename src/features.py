import pandas as pd
import numpy as np
import joblib

from scipy.signal import find_peaks
from pathlib import Path
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.configurations import Configuration
from src.helper import check_df_index


class FeatureSet:

    def __init__(self, df: pd.DataFrame = None, input_path: Path = None,
                 sample_rate: int = None):
        """
        Class to process pre-processed dataset into features to be used in
        training models.
        :param df: DataFrame containing pre-processed data.
        :param input_path: Path to the pre-processed data file.
        :param sample_rate: Sample rate in minutes for the data.
        """
        if sample_rate is None:
            raise ValueError("Sample rate must be provided.")
        self.df = df
        self.info_cols = ['night_start_date', 'cluster']
        if any(col not in self.df.columns for col in self.info_cols):
            print(f'Given the DataFrame does not contain all of '
                  f'["night_start_date", "cluster"] columns, these will not be '
                  f'returned with the features.')
        self.scaler = None
        self.sample_rate = sample_rate
        self.mean_cols = ['iob mean', 'cob mean', 'bg mean']
        minmax_columns = ['iob min', 'cob min', 'bg min',
                            'iob max', 'cob max', 'bg max']
        self.minmax_cols = [col for col in minmax_columns
                            if col in self.df.columns]
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
            self.df[col_name] = (self.df[col].groupby([id_index, hour_index]).
                                 transform('mean'))
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
        self.df['hour_cos'] = (
            cos_transformer(24).fit_transform(self.df['hour_of_day']))
        self.new_feature_cols.extend(['hour_of_day', 'hour_sin', 'hour_cos'])

    def add_level_excursion_features(self):
        """
        Adds features for excursions based on level 1 and level 2 hypo/hyper
        glycaemic thresholds. Level 1 is defined as a low threshold of 70 mg/dL
        and a high threshold of 180 mg/dL. Level 2 is defined as a low threshold
        of 54 mg/dL and a high threshold of 250 mg/dL.
        """
        l1_low, l1_high = 70, 180
        l2_low, l2_high = 54, 250
        self.df['l1_hypo'] = (self.df['bg min'] < l1_low).astype(int)
        self.df['l1_hyper'] = (self.df['bg max'] > l1_high).astype(int)
        self.df['l2_hypo'] = (self.df['bg min'] < l2_low).astype(int)
        self.df['l2_hyper'] = (self.df['bg max'] > l2_high).astype(int)

        self.new_feature_cols.extend(['l1_hypo', 'l1_hyper',
                                      'l2_hypo', 'l2_hyper'])

    def add_sd_excursion_features(self, mode='turning_point_max_amplitude') \
            -> pd.DataFrame:
        """
        Calculates the amplitude of qualifying glycaemic excursions at data
        points within specified 'night' periods for each individual. A
        qualifying excursion is defined as a change in 'bg mean' between a peak-
        nadir or nadir-peak, that exceeds one standard deviation of the
        'bg mean' values for that specific 'night'. The function adds a new
        column 'excursion_amplitude' to the input DataFrame. This
        column will contain the amplitude of the qualifying excursion if the
        data point is a peak or nadir of such an excursion. If a point is part
        of multiple qualifying excursions (e.g., a peak is the end of one upward
        and the start of a downward excursion), the maximum amplitude associated
        with it is assigned. For data points not part of any qualifying
        excursion, the value will be 0.0.
        :param mode: (str) Determines how the 'excursion_amplitude'
            column is populated.
            Options:
            - 'turning_point_max_amplitude' (default): Assigns the maximum
            amplitude of any qualifying excursion to its turning points
            (peaks/nadirs). Other points are 0.0.
            - 'excursion_amplitude_filled': Assigns the amplitude of a
            qualifying excursion to all 30-minute intervals within its duration.
            If intervals overlap, the maximum amplitude is taken.
        :returns: (pd.DataFrame) The input DataFrame with the added
            'excursion_amplitude' column

        """
        required_cols = ['bg mean', 'time', 'night_start_date']
        if (not isinstance(self.df.index, pd.MultiIndex) or
                list(self.df.index.names) != ['id', 'datetime']):
            raise ValueError(
                "DataFrame must have a MultiIndex ['id', 'datetime'].")
        if not all(col in self.df.columns for col in required_cols):
            missing_cols = [col for col in required_cols if
                            col not in self.df.columns]
            raise ValueError(
                f"DataFrame must contain {', '.join(missing_cols)} columns.")

        df_working = self.df[required_cols].reset_index()

        df_working['datetime'] = pd.to_datetime(df_working['datetime'])
        df_working = df_working.sort_values(by=['id', 'datetime'])

        # SD of the 'bg mean' values within each defined 'night'
        df_working['sd_threshold'] = (
            df_working.groupby(['id', 'night_start_date'])['bg mean'].
            transform('std'))

        # Initialise the new column for excursion amplitudes with zeros
        df_working[['excursion_amplitude', 'excursion_flag']] = 0.0


        # Group by individual then by night for processing each sequence
        for (individual_id, night_date_val), group in df_working.groupby(
                ['id', 'night_start_date']):
            glucose_series = group['bg mean']
            current_sd_threshold = group['sd_threshold'].iloc[
                0]  # SD is constant for the night

            # Skip calculation if insufficient data points or SD is invalid
            if len(glucose_series) < 2 or pd.isna(
                    current_sd_threshold) or current_sd_threshold == 0:
                continue

            # Step 4: Identify Peaks and Nadirs within each Night
            shifted_prev = glucose_series.shift(1)
            shifted_next = glucose_series.shift(-1)

            is_peak = (glucose_series > shifted_prev) & (
                        glucose_series > shifted_next)
            is_nadir = (glucose_series < shifted_prev) & (
                        glucose_series < shifted_next)

            # Handle boundary conditions for peaks/nadirs
            if glucose_series.iloc[0] > glucose_series.iloc[1]:
                is_peak.iloc[0] = True
            elif glucose_series.iloc[0] < glucose_series.iloc[1]:
                is_nadir.iloc[0] = True

            if glucose_series.iloc[-1] > glucose_series.iloc[-2]:
                is_peak.iloc[-1] = True
            elif glucose_series.iloc[-1] < glucose_series.iloc[-2]:
                is_nadir.iloc[-1] = True

            turning_points_indices = group.index[is_peak | is_nadir].tolist()

            # Store qualifying excursions for 'excursion_amplitude_filled' mode
            qualifying_excursions_data = []

            # Step 5: Identify Significant Excursions and Populate Column
            for i in range(len(turning_points_indices) - 1):
                idx1 = turning_points_indices[i]
                idx2 = turning_points_indices[i + 1]

                val1 = df_working.loc[idx1, 'bg mean']
                val2 = df_working.loc[idx2, 'bg mean']

                is_upward_excursion = is_nadir.loc[idx1] and is_peak.loc[idx2]
                is_downward_excursion = is_peak.loc[idx1] and is_nadir.loc[idx2]

                if is_upward_excursion or is_downward_excursion:
                    amplitude = abs(val2 - val1)

                    if amplitude > current_sd_threshold:
                        if mode == 'turning_point_max_amplitude':
                            df_working.loc[
                                idx1, 'excursion_amplitude'] = max(
                                df_working.loc[
                                    idx1, 'excursion_amplitude'],
                                amplitude)
                            df_working.loc[
                                idx2, 'excursion_amplitude'] = max(
                                df_working.loc[
                                    idx2, 'excursion_amplitude'],
                                amplitude)
                            df_working.loc[
                                idx1, 'excursion_flag'] = 1.0
                            df_working.loc[
                                idx2, 'excursion_flag'] = 1.0
                        elif mode == 'excursion_amplitude_filled':
                            # Store info for later filling to avoid overwriting
                            # issues in loop
                            qualifying_excursions_data.append({
                                'start_idx': idx1,
                                'end_idx': idx2,
                                'amplitude': amplitude
                            })

            # For 'excursion_amplitude_filled' mode, fill values after
            # identifying all excursions
            if mode == 'excursion_amplitude_filled':
                # Create temporary series for this group to apply filling logic
                temp_amplitude_series = pd.Series(0.0, index=group.index)
                for exc in qualifying_excursions_data:
                    start_loc = group.index.get_loc(exc['start_idx'])
                    end_loc = group.index.get_loc(exc['end_idx'])

                    # Get the slice of indices for this excursion
                    indices_in_excursion = group.index[start_loc: end_loc + 1]

                    # Apply the amplitude, taking the max if already set by
                    # another excursion
                    for idx in indices_in_excursion:
                        temp_amplitude_series.loc[idx] = max(
                            temp_amplitude_series.loc[idx], exc['amplitude'])

                # Update the main df_working DataFrame for this group
                df_working.loc[group.index, 'excursion_amplitude'] \
                    = temp_amplitude_series

        df_working.set_index(['id', 'datetime'], inplace=True)
        self.df = (
            self.df.join(df_working[['excursion_amplitude','excursion_flag']],
                         on=['id', 'datetime'], how='left'))
        self.new_feature_cols.extend(['excursion_amplitude', 'excursion_flag'])

        return df_working

    def add_cob_peaks(self, height=None, distance=1):
        """
        Identify peaks in the 'cob max' column and add a new column
        indicating whether each row is a peak.
        """
        self.df['cob_peaks'] = 0
        for (id_, night_start_date), group in (
                self.df.groupby(['id', 'night_start_date'])):
            peaks, _ = find_peaks(group['cob mean'], height=height,
                                  distance=distance)
            peak_indices = group.index[peaks]
            self.df.loc[peak_indices, 'cob_peaks'] = 1
        self.new_feature_cols.append('cob_peaks')

    def get_all_features(self, scale=True):
        """
        Adds all features to the DataFrame and returns the updated DataFrame
        with the new features, including the original variables, ready for
        model training.
        :param scale: (bool) Whether to scale the features using StandardScaler.
        :return: pd.DataFrame with all features.
        """
        self.add_day_type()
        self.add_rate_of_change(columns=self.mean_cols)
        self.add_hourly_mean(columns=self.mean_cols)
        self.add_peaks_above_mean()
        self.add_time_based_features()
        self.add_level_excursion_features()
        self.add_sd_excursion_features(mode='turning_point_max_amplitude')
        self.add_cob_peaks(height=None, distance=1)
        if scale:
            self.scale_features()
        return (self.df[self.get_all_feature_columns_only() + self.info_cols]
                .copy())

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
        cols = self.get_all_feature_columns_only()
        print(f'Scaling {cols} columns')
        self.scaler = StandardScaler()
        self.df[cols] = self.scaler.fit_transform(self.df[cols])

    def inverse_scale_features(self):
        cols = self.get_all_feature_columns_only()
        if self.scaler is None:
            raise ValueError("Scaler has not been fitted.")
        self.df[cols] = self.scaler.inverse_transform(
            self.df[cols])

    def save_scaler(self, path):
        joblib.dump(self.scaler, path)

    def load_scaler(self, path):
        self.scaler = joblib.load(path)

    def get_all_feature_columns_only(self):
        """
        Returns all feature columns including original variables.
        :return: List of all feature columns.
        """
        return (self.mean_cols + self.minmax_cols + self.new_feature_cols)

    def get_full_df(self):
        """
        Returns the full DataFrame with all features.
        :return: pd.DataFrame with all features.
        """
        return self.df.copy()
