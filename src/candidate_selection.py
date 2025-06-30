from typing import List, Tuple

import pandas as pd
from loguru import logger
from matplotlib import pyplot as plt
from datetime import time

from src.helper import check_df_index
from src.config import FIGURES_DIR
from src.nights import Nights


def remove_null_variable_individuals(df: pd.DataFrame,
                                     logging: bool = False) -> pd.DataFrame:
    """
    Removes people from the dataset that have null or zero values for any
    IOB, COB or BG variable across their dataset.
    :param df: (pd.Dataframe) Dataframe to process
    :param logging: (bool) Whether to log the ids of individuals removed
    :return: Dataframe with individuals removed
    """
    df = check_df_index(df)
    check_cols = ['iob count', 'cob count', 'bg count']
    ids_with_only_nans_or_zeros = {}
    for col in check_cols:
        mask = (
            df.groupby(level='id')[col]
            .apply(lambda x: x.isna().all() or (x.fillna(0) == 0).all())
        )
        ids_with_only_nans_or_zeros[col] = mask[mask].index.tolist()
    ids = set()
    for key, val in ids_with_only_nans_or_zeros.items():
        ids.update(set(val))
    if logging:
        logger.info(f'Following individuals have one or more variables missing:'
                    f' {ids}')
    return df[~df.index.get_level_values('id').isin(ids)]


def get_all_individuals_night_stats(separated: (int, pd.DataFrame) = None,
                                    sample_rate: int = None,
                                    night_start: time = None,
                                    morning_end: time = None) -> pd.DataFrame:
    """
     Creates a list of Night objects for each individual and returns the stats
     produced.
     :param separated: (int, pd.DataFrame) zip_id and df of nights
     :param sample_rate: (int) sample rate in minutes
     :param night_start: (time) start time of night
     :param morning_end: (time) end time of night
     :return:
     """
    nights_objects = create_nights_objects(separated, sample_rate,
                                           night_start, morning_end)
    return provide_data_statistics(nights_objects)


def provide_data_statistics(night_objects: [Nights] = None) -> pd.DataFrame:
    """
    Creates statistics from the analysis of the nights of an individual through
    iteration of the Nights class. Useful in assessing the level of
    completeness of the data sought, plus other important statistics for
    analysis.
    :param night_objects: List of tuples of id and dataframe, where the df is
        the time series data for an individual
    :return: Dataframe with the statistics calculated for individuals
    """
    if night_objects is None or night_objects == []:
        raise TypeError('night_objects cannot be None')
    overall_stats_list = []
    for nights in night_objects:
        stats = nights.overall_stats
        if stats:
            stats['id'] = nights.zip_id
            stats['period_total_intervals'] = nights.total_intervals()
            stats['period_total_minutes'] = nights.total_minutes()
            overall_stats_list.append(stats)
    df_overall_stats = pd.DataFrame(overall_stats_list)
    df_overall_stats = df_overall_stats.set_index('id')
    df_overall_stats.sort_values('complete_nights', ascending=False)

    return df_overall_stats


def create_nights_objects(separated: List[Tuple[int, pd.DataFrame]] = None,
                          sample_rate: int = None,
                          night_start: time = None,
                          morning_end: time = None) -> [Nights]:
    """
    Creates Nights objects from list of separated dataframes for each
    patient
    :param separated: Groupby iterable of zip_id and df of separated patients
    :param sample_rate: Sample rate in minutes
    :param night_start: Time when the night starts
    :param morning_end: Time when the night ends
    :return: List of Nights objects
    """
    if separated is None or separated == []:
        raise TypeError('separated argument cannot be None or empty')
    nights_objects = []
    for id_val, df_individual in separated:
        nights = Nights(zip_id=id_val,
                        df=df_individual,
                        sample_rate=sample_rate,
                        night_start=night_start,
                        morning_end=morning_end)
        nights_objects.append(nights)
    return nights_objects


def plot_nights_vs_avg_intervals(df_overall_stats: pd.DataFrame):
    """
    Plot the number of nights vs average intervals with markers showing the
    length of average total length of missing intervals in minutes.
    :param df_overall_stats: Dataframe of individual (id index) and stat columns
    """
    marker_sizes = (df_overall_stats['avg_total_break_duration'] + 1) / 4

    max_y = df_overall_stats['period_total_intervals'].max()

    plt.figure(figsize=(8, 5))
    plt.scatter(
        df_overall_stats['count_of_nights'],
        df_overall_stats['avg_num_intervals'],
        s=marker_sizes,
        alpha=0.7,
        c='tab:blue'
    )

    # Marker size legend
    for size in [20, 120, 220, 320]:
        plt.scatter([], [], s=(size + 1) / 2, c='tab:blue', alpha=0.7,
                    label=size)
    plt.legend(title='Avg Total Break Length')

    plt.axhline(y=max_y, color='grey', linestyle='dotted', linewidth=1)
    plt.text(
        plt.xlim()[1] * 0.98, max_y,
        'Max possible intervals',
        color='grey', fontsize=9, ha='right', va='bottom'
    )

    plt.xlabel('Count of Nights')
    plt.ylabel('Average Number of Intervals')
    plt.title('Count of Nights vs Average Number of Intervals\n'
              '(Marker size = Average Total Break Length in Minutes)')
    plt.ylim(top=max_y * 1.05)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'nights_vs_avg_intervals.png', dpi=400)
    plt.show()
