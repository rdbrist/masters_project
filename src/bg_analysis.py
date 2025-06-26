import seaborn as sns
import pandas as pd
from scipy.stats import skew, kurtosis
from matplotlib import pyplot as plt
from datetime import time

from src.candidate_selection import provide_data_statistics, \
    create_nights_objects
from src.nights import Nights


class BGAnalyser:
    """
    Class for analysing blood glucose data from nights.
    """
    def __init__(self, separated_dfs: (int, pd.DataFrame)=None,
                 night_start: time=None,
                 morning_end: time=None,
                 sample_rate: int=None):
        if any(p is None for p in
               [separated_dfs, night_start, morning_end, sample_rate]):
            raise ValueError('Parameters all need to be set to instantiate '
                             'the BGAnalysis class.')

        self.separated_dfs = separated_dfs
        self.night_start = night_start
        self.morning_end = morning_end
        self.sample_rate = sample_rate
        self.nights_objects = create_nights_objects(separated_dfs,
                                                    night_start=night_start,
                                                    morning_end=morning_end,
                                                    sample_rate=sample_rate)


    def plot_bg_variability(self):
        """
        Plot the blood glucose variability measures (SD, IQR, Range) for each
        individual.
        """
        cols = ['bg_sd_median', 'bg_range_median', 'bg_iqr_median']
        melted = (self.population_stats
                  .melt(value_vars=cols, var_name='Variability Measure',
                        value_name='Value'))

        fig, ax1 = plt.subplots(figsize=(10, 6))

        sns.boxplot(x='Variability Measure', y='Value', data=melted[
            melted['Variability Measure'].isin(['bg_sd_median', 'bg_iqr_median'])],
                    ax=ax1)
        ax1.set_ylabel('SD / IQR (mg/dL)')
        ax1.set_xlabel('Variability Measure')
        ax1.set_title(
            'Distribution of Blood Glucose Variability Measures (Median per '
            'Individual)')

        ax2 = ax1.twinx()
        sns.boxplot(x='Variability Measure', y='Value',
                    data=melted[melted['Variability Measure'] == 'bg_range_median'],
                    ax=ax2, boxprops=dict(facecolor='lightcoral', alpha=0.5))
        ax2.set_ylabel('Range (mg/dL)')

        ax1.set_xticks([0, 1, 2])
        ax1.set_xticklabels(['SD', 'IQR', 'Range'])
        plt.tight_layout()
        plt.show()

def calculate_skew_kurtosis(df: pd.DataFrame,
                            variables: list = None) -> pd.DataFrame:
    """
    Calculate skewness and kurtosis for the variables included.
    :param df: (DataFrame) DataFrame containing the variables.
    :param variables: (list) List of variables to calculate skewness and
        kurtosis for.
    :return: (DataFrame) DataFrame with skewness and kurtosis values.
    """
    skewness = df[variables].apply(skew)
    kurt = df[variables].apply(kurtosis)

    return pd.DataFrame({'Skewness': skewness, 'Kurtosis': kurt})
