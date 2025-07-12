import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from datetime import time

from src.candidate_selection import (create_nights_objects,
                                     provide_data_statistics)
from src.nights import Nights


class BGAnalyser:
    """
    Class for analysing blood glucose data from nights.
    """
    def __init__(self, separated_dfs: (int, pd.DataFrame) = None,
                 night_start: time = None,
                 morning_end: time = None,
                 sample_rate: int = None):
        if any(p is None for p in
               [separated_dfs, night_start, morning_end, sample_rate]):
            raise ValueError('Parameters all need to be set to instantiate '
                             'the BGAnalysis class.')

        self.separated_dfs = separated_dfs
        self.zip_ids = [p[0] for p in separated_dfs]
        self.night_start = night_start
        self.morning_end = morning_end
        self.sample_rate = sample_rate
        self.nights_objects = create_nights_objects(separated_dfs,
                                                    night_start=night_start,
                                                    morning_end=morning_end,
                                                    sample_rate=sample_rate)
        self.population_stats = provide_data_statistics(self.nights_objects)

    def get_nights_objects(self):
        return self.nights_objects

    def _return_z_scores(self):
        zscore_data = []
        for nights in self.nights_objects:
            bg_zscore = [stats['bg_zscore'] for stats in
                         nights.get_stats_per_night()]
            night_dates = [stats['night_start_date'] for stats in
                           nights.get_stats_per_night()]
            for i, z in enumerate(bg_zscore):
                zscore_data.append({'zip_id': nights.zip_id,
                                    'night_start_date': night_dates[i],
                                    'bg_zscore': z})

        return pd.DataFrame(zscore_data)

    def plot_z_scores_per_individual(self):
        df_zscore = self._return_z_scores()
        counts = df_zscore.groupby('zip_id').size().to_dict()

        g = sns.FacetGrid(df_zscore, col='zip_id', col_wrap=3, height=2.5,
                          aspect=1)
        g.map(sns.histplot, 'bg_zscore', kde=True)

        # Set custom titles for each facet
        for zip_id, ax in g.axes_dict.items():
            count = counts.get(zip_id, 0)
            ax.set_title(f'Person {zip_id}\nN={count}', fontsize=10)

        g.set_axis_labels('Z-Score', 'Count')
        g.tight_layout()
        g.fig.subplots_adjust(top=1.01)
        g.fig.suptitle(
            'Blood Glucose Z-Scores by Individual for individuals\n'
            'with => 30 complete nights for the period 22:00-06:00',
            y=1.05)

        plt.show()

    def plot_bg_variability(self):
        """
        Plot the blood glucose variability measures (SD, IQR, Range) for each
        individual.
        """
        cols = ['bg_sd_median', 'bg_range_median', 'bg_iqr_median']
        melted = (self.population_stats
                  .melt(value_vars=cols, var_name='Variability Measure',
                        value_name='Value'))

        fig, ax1 = plt.subplots(figsize=(6, 4))
        data = melted[
            melted['Variability Measure'].
            isin(['bg_sd_median', 'bg_iqr_median'])]
        sns.boxplot(x='Variability Measure', y='Value', data=data, ax=ax1)
        ax1.set_ylabel('SD / IQR (mg/dL)')
        ax1.set_xlabel('Variability Measure')
        ax1.set_title(
            'Distribution of Blood Glucose Variability Measures (Median per '
            'Individual)')
        ax1.grid(False)

        ax2 = ax1.twinx()
        sns.boxplot(x='Variability Measure', y='Value',
                    data=melted[melted['Variability Measure'] ==
                                'bg_range_median'],
                    ax=ax2, boxprops=dict(facecolor='coral', alpha=0.5))
        ax2.set_ylabel('Range (mg/dL)')
        ax2.grid(False)

        ax1.set_xticks([0, 1, 2])
        ax1.set_xticklabels(['SD', 'IQR', 'Range'])
        plt.tight_layout()
        plt.show()


def plot_zscores_individual_boxplot(zip_id: int = None, nights: Nights = None):
    """
    Plot the z-scores of the blood glucose values for an individual in a boxplot
    :param zip_id: (int) ID of the zip code to filter the nights.
    :param nights: (Nights) Nights object containing the nights data.
    :return:
    """
    if zip_id is None or nights is None:
        raise ValueError('zip_id and nights parameters need to be set.')

    stats = nights.get_stats_per_night()
    zscores = stats['bg_zscore']

    sns.boxplot(zscores, color='lightblue')
