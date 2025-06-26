import pandas as pd
import numpy as np
from loguru import logger
from typing import Tuple, List
from matplotlib import pyplot as plt
from datetime import time

from src.helper import check_df_index
from src.config import FIGURES_DIR
from src.nights import Nights


def remove_null_variable_individuals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes people from the dataset that have null or zero values for any
    IOB, COB or BG variable across their dataset.
    :param df: Dataframe to process
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
    logger.info(f'Following individuals have one or more variables missing: '
                f'{ids}')
    return df[~df.index.get_level_values('id').isin(ids)]

def provide_data_statistics(separated: list[Tuple[int, pd.DataFrame]],
                            sample_rate: int=None,
                            night_start: time=time(19, 0),
                            morning_end: time=time(11, 0)) -> pd.DataFrame:
    """
    Creates statistics from the analysis of the nights of an individual through
    iteration of the Nights class. Useful in assessing the level of
    completeness of the data sought, plus other important statistics for
    analsis.
    :param separated: List of tuples of id and dataframe, where the df is the
        time series data for an individual
    :param sample_rate: Sample rate in minutes, default is 15
    :param night_start: Time when the night starts, default is 19:00
    :param morning_end: Time when the night ends, default is 11:00
    :return: Dataframe with the statistics calculated for individuals
    """
    overall_stats_list = create_nights_objects(separated=separated,
                            sample_rate=sample_rate,
                            night_start=night_start,
                            morning_end=morning_end)

    df_overall_stats = pd.DataFrame(overall_stats_list)
    df_overall_stats = df_overall_stats.set_index('id')
    df_overall_stats.sort_values('complete_nights', ascending=False)

    return df_overall_stats

def create_nights_objects(separated: (int, pd.DataFrame)=None,
                          sample_rate: int=None,
                          night_start: time=None,
                          morning_end: time=None) -> [Nights]:
    """
    Creates Nights objects from list of separated dataframes for each
    patient
    :param separated: 
    :param sample_rate: Sample rate in minutes
    :param night_start: Time when the night starts
    :param morning_end: Time when the night ends
    :return: List of Nights objects
    """
    for id_val, df_individual in separated:
        nights = Nights(zip_id=id_val,
                        df=df_individual,
                        sample_rate=sample_rate,
                        night_start=night_start,
                        morning_end=morning_end)
    return nights

def plot_nights_vs_avg_intervals(df_overall_stats: pd.DataFrame):
    """
    Plot the number of nights vs average intervals with markers showing the
    length of average total length of missing intervals in minutes.
    :param df_overall_stats: Dataframe of individual (id index) and stat columns
    """
    marker_sizes = (df_overall_stats['avg_total_break_length'] + 1) / 4

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
    plt.title('Count of Nights vs Average Number of Intervals\n(Marker size = Average Total Break Length in Minutes)')
    plt.ylim(top=max_y * 1.05)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'nights_vs_avg_intervals.png', dpi=400)
    plt.show()

def get_complete_nights_only(all_nights_objects: List[Nights]) -> np.ndarray:
    """
    Returns a list of only the nights that are complete, i.e. have no missing intervals.
    :param all_nights_objects: List of Nights objects
    :return: List of objects with shape (id, [night_df, ...])
    """
    processed_individuals = []
    for nights in all_nights_objects:
        complete_nights_indices = nights.stats_per_night[nights.stats_per_night['missed_intervals'] == 0].index
        print(complete_nights_indices)
    return processed_individuals

def reconsolidate_flat_file_from_nights(
        nights_objects: List[Nights]) -> pd.DataFrame:
    """
    Reconstructs a flat file from the list of Nights objects in the common
    format, with a multi-index of id and datetime.
    :param nights_objects: List of Nights objects
    :return: DataFrame with the reconstructed flat file
    """
    flat_file = pd.DataFrame()
    for nights in nights_objects:
        for night_start_date, night_df in nights.nights:
            # Ensure the index is a DatetimeIndex
            if not isinstance(night_df.index, pd.DatetimeIndex):
                raise ValueError("Night DataFrame index must be a DatetimeIndex.")
            df = night_df.copy()
            df['id'] = nights.zip_id
            df = df.reset_index().set_index(['id', 'datetime'])
            flat_file = pd.concat([flat_file, df])
    return flat_file
