from datetime import time
from math import sqrt
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.features import FeatureSet
from src.nights import Nights
from test_read import input_file


class ObjectiveFunction:
    """
    Calculates the objective function value for a given DataFrame.
    """
    def __init__(self, df: pd.DataFrame, weights: list,
                 sample_rate: int = None):
        """
        Initialises the ObjectiveFunction with a DataFrame, checks for the
        required columns, and raises an error if they are not present.
        :param df:
        """
        required_columns = ['bg mean', 'bg std', 'bg count', 'cob max']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"DataFrame must contain the following columns: "
                             f"{required_columns}")
        if sample_rate is None:
            raise ValueError('sample_rate must be provided')

        self.sample_rate = sample_rate
        self.features = FeatureSet(df, sample_rate=sample_rate)
        self.df = self.features.get_all_features(scale=False)
        self.feature_cols = self.features.new_feature_cols
        self.night_features = None
        self.aggregate_features()
        self.weights = self.assign_weights(weights) if weights else None

    def aggregate_features(self):
        """
        Aggregates the features for each night period in the DataFrame:
            - Mean of the background
            - Standard deviation of the background
            - Coefficient of variation of the background
            - L1 excursions
            - L2 excursions
            - Mean amplitude of glycaemic excursions
            - COB peaks
        :return: 
        """
        agg_list = []
        for (id_, night_start_date), night_df in self.df.groupby(
                ['id', 'night_start_date']):
            bg_night_mean, bg_night_std = (
                ObjectiveFunction._calculate_overall_std(night_df))
            agg_dict = {
                'id': id_,
                'night_start_date': night_start_date,
                'bg_night_mean': bg_night_mean,
                'bg_night_std': bg_night_std,
                'bg_night_cv':
                    ObjectiveFunction.
                    _calculate_overall_coefficient_of_variation(night_df),
                'l1_excursions': night_df[['l1_hyper', 'l1_hypo']].sum().sum(),
                'l2_excursions': night_df[['l2_hyper', 'l2_hypo']].sum().sum(),
                'mage': night_df['excursion_amplitude'].mean(),
                # 'cob_peaks': night_df['cob_peaks'].sum()
            }
            agg_list.append(agg_dict)
        self.night_features = (pd.DataFrame(agg_list).
                               set_index(['id','night_start_date']))
        print('Feature columns and order created, now apply weights using '
              'assign_weights:')

        return self.night_features

    @staticmethod
    def _calculate_mean_background(df):
        """
        Calculates the mean background of the night period.
        :return: Mean background of the night period.
        """
        sum_n_g = (df['bg count'] * df['bg mean']).sum()
        sum_n = df['bg count'].sum()
        return sum_n_g / sum_n if sum_n != 0 else float('nan'), sum_n

    @staticmethod
    def _calculate_overall_std(night_df):
        """
        Calculates the overall standard deviation of the night period.
        :param night_df: DataFrame containing the night period data.
        :return: Overall standard deviation of the background. and G_bar (mean)
        """
        df = night_df.copy()
        if df['bg std'].isnull().any():
            df['bg std'] = df['bg std'].fillna(0)

        if df['bg count'].isnull().any():
            df['bg count'] = df['bg count'].fillna(1).astype(int)
        else:
            df['bg count'] = df['bg count'].astype(int)
    
        df['bg count'] = df['bg count'].astype(int)
        g_mean, sum_n = ObjectiveFunction._calculate_mean_background(df)

        # Step 2: 1st component (Mean of Within-Interval Variances, Weighted)
        # Sum(n_i * s_i^2) / Sum(n_i)
        df['bg_variance'] = df['bg std'] ** 2
        sum_n_s_squared = (df['bg count'] * df[
            'bg_variance']).sum()
        component1 = sum_n_s_squared / sum_n

        # Step 3: 2nd component (Variance of Interval Means, Weighted)
        # Sum(n_i * (g_i - g_mean)^2) / Sum(n_i)
        df['diff_squared'] = (df['bg mean'] - g_mean) ** 2
        sum_n_diff_squared = (df['bg count'] * df[
            'diff_squared']).sum()
        component2 = sum_n_diff_squared / sum_n

        return sqrt(component1 + component2), g_mean

    @staticmethod
    def _calculate_overall_coefficient_of_variation(night_df):
        """
        Calculates the overall coefficient of variation of the night period.
        :param night_df: DataFrame containing the night period data.
        :return: Overall coefficient of variation of the background.
        """
        overall_std, g_mean = (
            ObjectiveFunction._calculate_overall_std(night_df))

        return overall_std / g_mean if g_mean != 0 else float('inf')

    def scale_features(self):
        """
        Scales the features in the DataFrame.
        :return: Scaled DataFrame.
        """
        if self.weights:
            df = self.night_features * self.weights
        return StandardScaler().fit_transform(df)

    def get_objective_function_scores(self):
        """
        Calculates the objective function scores for each night period.
        :return: DataFrame with the objective function scores.
        """
        if self.night_features is None:
            self.aggregate_features()

        scaled = self.scale_features()
        scores = scaled.sum(axis=1) / scaled.shape[1]

        return pd.DataFrame(scores, columns=['score'],
                            index=self.night_features.index)

    def get_scaled_features(self):
        """
        Calculates the objective function scores for each night period.
        :return: DataFrame with the objective function scores.
        """
        if self.night_features is None:
            self.aggregate_features()

        df = self.night_features.copy()
        df[self.night_features.columns] = self.scale_features()

        return df

    def assign_weights(self, weights=None):
        """
        Set weights for the objective function components between 0 and 2.
        Default to 1 for all (keeping the values the same).
        :param weights: (list) values between 0 and 2
        :return:
        """
        if weights is None:
            weights = []
        if not weights:
            weights = [1] * len(self.night_features.columns)
        if len(weights) != len(self.night_features.columns):
            raise Exception('Number of weights does not match number of '
                            'features.')
        if any(w < 0 or w > 2 for w in weights):
            raise Exception('Weights must be between 0 and 2.')

        print('Weights being set as follows:')
        for i, col in enumerate(self.night_features.columns):
            print(f'{col}: {weights[i]}')

        return weights