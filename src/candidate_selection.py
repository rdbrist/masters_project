import pandas as pd
import numpy as np
from loguru import logger
from typing import Tuple, List
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
from datetime import time, datetime

from src.helper import check_df_index

class Nights:
    def __init__(self, zip_id: int,
                 df: pd.DataFrame,
                 night_start=time(19, 0),
                 morning_end=time(12, 0),
                 sample_rate = 15):
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex.")
        self.zip_id = zip_id
        self.sample_rate = sample_rate
        self.df = df.sort_index()
        self.night_start = night_start
        self.morning_end = morning_end
        self.nights = self._split_nights()
        self.stats_per_night = self._create_stats_per_night()
        self.overall_stats = self._calculate_overall_stats()

    def get_night_period(self, date):
        """
        Given a date, night_start (datetime.time), and morning_end (datetime.time),
        returns the corresponding start and end pd.Timestamp for the nightly period.
        """
        date_obj = pd.Timestamp(date).date()
        night_start_dt = pd.Timestamp.combine(date_obj, self.night_start)
        if self.morning_end <= self.night_start:
            morning_end_dt = pd.Timestamp.combine(
                date_obj + pd.Timedelta(days=1), self.morning_end)
        else:
            morning_end_dt = pd.Timestamp.combine(date_obj, self.morning_end)
        return night_start_dt, morning_end_dt

    def _get_total_minutes(self):
        """
        Returns the total number of minutes between night_start and morning_end.
        """
        base_date = datetime(2000, 1, 1)
        night_start_dt, morning_end_dt = self.get_night_period(base_date)
        return int((morning_end_dt - night_start_dt).total_seconds() // 60)

    def total_minutes(self):
        """
        Returns the number of minutes in the time period.
        """
        return self._get_total_minutes()

    def total_intervals(self):
        """
        Returns the number of intervals in the time period based on the sample rate.
        """
        return self._get_total_minutes() // self.sample_rate


    def _split_nights(self):
        """
        Split full time series dataframe for individual into separate timeseries dataframes covering each
        nightly period, based on the night_start and morning_end times.
        :return: List of dataframes, one pertaining to each night period
        """
        nights = []
        dates = pd.to_datetime(self.df.index.date)
        for date in pd.unique(dates):
            night_start_dt, morning_end_dt = self.get_night_period(date)
            mask = (self.df.index >= night_start_dt) & (self.df.index < morning_end_dt)
            night_df = self.df.loc[mask]
            if not night_df.empty:
                nights.append(night_df)

        return nights

    def _calculate_overall_stats(self):
        """
        Calculates stats for the overall dataset to determine the number of gaps in
        data (determined by missing intervals) to inform the level of completeness.
        :return: dict with average stats across all nights
        """
        count_of_nights = len(self.nights)
        if not self.stats_per_night:
            return print(f'No stats per night have been calculated for {self.zip_id}. Returning no output.')
        avg_num_intervals = sum(d['num_intervals'] for d in self.stats_per_night) / count_of_nights
        avg_num_breaks = sum(d['num_breaks'] for d in self.stats_per_night) / count_of_nights
        avg_missed_intervals = sum(d['missed_intervals'] for d in self.stats_per_night) / count_of_nights
        avg_break_length = sum(d['avg_break_length'] for d in self.stats_per_night) / count_of_nights
        avg_max_break_length = sum(d['max_break_length'] for d in self.stats_per_night) / count_of_nights
        avg_total_break_length = sum(d['total_break_length'] for d in self.stats_per_night) / count_of_nights
        complete_nights =  sum(1 for d in self.stats_per_night if d['missed_intervals'] == 0)
        missed_interval_vectors = [d['missed_interval_vector'] for d in self.stats_per_night]

        return {
            'count_of_nights': count_of_nights,
            'complete_nights': complete_nights,
            'avg_num_intervals': avg_num_intervals,
            'avg_missed_intervals': avg_missed_intervals,
            'avg_num_breaks': avg_num_breaks,
            'avg_break_length': avg_break_length,
            'avg_max_break_length': avg_max_break_length,
            'avg_total_break_length': avg_total_break_length,
            'missed_interval_vectors': missed_interval_vectors,
        }

    def _create_stats_per_night(self):
        """
        Produces a dictionary of stats per night that can then be used for selection or further aggregation in overall stats.
        :return: List of dicts, one per night
        """
        stats = []
        for night_df in self.nights:
            date = night_df.index[0].date()
            night_start_dt, morning_end_dt = self.get_night_period(date)
            expected_index = pd.date_range(
                start=night_start_dt,
                end=morning_end_dt - pd.Timedelta(minutes=self.sample_rate),
                freq=f'{self.sample_rate}min'
            )
            vector = np.where(expected_index.isin(night_df.index), 0, 1)
            times = night_df.index.sort_values()
            # Calculate time differences in minutes between consecutive timestamps
            diffs = times.to_series().diff().dropna().dt.total_seconds() / 60
            # A break is any gap greater than the sample rate
            breaks = diffs[diffs > self.sample_rate]
            num_intervals = len(times)
            num_breaks = len(breaks)
            avg_break_length = breaks.mean() if num_breaks > 0 else 0
            max_break_length = breaks.max() if num_breaks > 0 else 0
            total_break_length = breaks.sum() if num_breaks > 0 else 0
            stats.append({
                'num_intervals': num_intervals,
                'missed_intervals': np.sum(vector),
                'num_breaks': num_breaks,
                'avg_break_length': avg_break_length,
                'max_break_length': max_break_length,
                'total_break_length': total_break_length,
                'missed_interval_vector': vector.tolist(),
            })

        return stats

    def remove_incomplete_nights(self):
        """
        Removes nights that have missing intervals, i.e. where the number of missed intervals is greater than 0.
        """
        self.stats_per_night = [s for s in self.stats_per_night if s['missed_intervals'] == 0]
        self.nights = [self.nights[i] for i, s in enumerate(self.stats_per_night) if s['missed_intervals'] == 0]
        self.overall_stats = self._calculate_overall_stats()

    def plot_break_distribution(self):
        """
        Plots the total break length per night as bars (left axis) and the number of breaks as a line (right axis),
        sorted by total break length (ascending).
        """
        if not self.stats_per_night:
            print("No stats to plot.")
            return

        # Extract and sort by total break length
        stats = sorted(self.stats_per_night, key=lambda d: d["total_break_length"])
        total_break_length = [d["total_break_length"] for d in stats]
        num_breaks = [d["num_breaks"] for d in stats]

        fig, ax1 = plt.subplots(figsize=(10, 5))

        # Bar plot for total break length (left y-axis)
        ax1.bar(range(len(total_break_length)), total_break_length, color='skyblue', label='Total Break Length')
        ax1.set_xlabel('Night (sorted by total break length)')
        ax1.set_ylabel('Total Break Length (minutes)', color='skyblue')
        ax1.tick_params(axis='y', labelcolor='skyblue')

        # Line plot for number of breaks (right y-axis)
        ax2 = ax1.twinx()
        ax2.plot(range(len(num_breaks)), num_breaks, color='orange', marker='o', label='Number of Breaks')
        ax2.set_ylabel('Number of Breaks', color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')

        plt.title('Break Distribution per Night (Sorted by Total Break Length)')
        fig.tight_layout()
        plt.show()

    def plot_break_histograms(self, bins=10):
        """
        Plots two separate histograms:
        1. Number of breaks per night.
        2. Total break length per night (in minutes).
        """
        if not self.stats_per_night:
            print("No stats to plot.")
            return

        num_breaks = [d["num_breaks"] for d in self.stats_per_night]
        total_break_length = [d["total_break_length"] for d in self.stats_per_night]

        # Plot histogram for number of breaks
        plt.figure(figsize=(8, 4))
        plt.hist(num_breaks, bins=bins, color='skyblue', alpha=0.8)
        plt.xlabel('Number of Breaks')
        plt.ylabel('Count (Nights)')
        plt.title('Histogram of Number of Breaks per Night')
        plt.tight_layout()
        plt.show()

        # Plot histogram for total break length
        plt.figure(figsize=(8, 4))
        plt.hist(total_break_length, bins=bins, color='orange', alpha=0.8)
        plt.xlabel('Total Break Length (minutes)')
        plt.ylabel('Count (Nights)')
        plt.title('Histogram of Total Break Lengths per Night')
        plt.tight_layout()
        plt.show()

    def plot_breaks_scatter(self):
        """
        Plots a scatter plot of number of breaks (x-axis) vs total break length (y-axis) per night.
        """
        if not self.stats_per_night:
            print("No stats to plot.")
            return

        num_breaks = [d["num_breaks"] for d in self.stats_per_night]
        total_break_length = [d["total_break_length"] for d in self.stats_per_night]

        plt.figure(figsize=(8, 5))
        plt.scatter(num_breaks, total_break_length, color='purple', alpha=0.7)
        plt.xlabel('Number of Breaks')
        plt.ylabel('Total Break Length (minutes)')
        plt.title('Scatter Plot of Breaks vs Total Break Length per Night')
        plt.tight_layout()
        plt.show()

    def get_nights(self):
        return self.nights
    
    def plot_missing_intervals_histogram(self):
        # Stack vectors: shape (n_nights, n_intervals)
        vectors_arr = np.vstack(self.overall_stats['missed_interval_vectors'])
        missing_counts = vectors_arr.sum(axis=0)
        n_nights = vectors_arr.shape[0]

        # Get time labels for x-axis
        date = self.nights[0].index[0].date()
        night_start_dt, morning_end_dt = self.get_night_period(date)
        expected_index = pd.date_range(
            start=night_start_dt,
            end=morning_end_dt - pd.Timedelta(minutes=self.sample_rate),
            freq=f'{self.sample_rate}min'
        )
        time_labels = expected_index.time

        plt.figure(figsize=(12, 4))
        plt.bar(range(len(missing_counts)), missing_counts, color='skyblue')
        plt.axhline(n_nights, color='red', linestyle='--', label='Total nights')
        plt.xticks(ticks=range(0, len(time_labels), 4),
                   labels=[t.strftime('%H:%M') for t in time_labels[::4]],
                   rotation=45)
        plt.xlabel('Time Interval')
        plt.ylabel('Count of Missing Intervals')
        plt.title('Missing Intervals by Time Across All Nights')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_missingness_heatmap(self, cell_size=0.75, max_width=6,
                                 max_height=10):
        arr = np.vstack(self.overall_stats['missed_interval_vectors'])
        n_nights, n_intervals = arr.shape

        width = min(n_intervals * cell_size, max_width)
        height = min(n_nights * cell_size, max_height)

        # Define custom colormap: 0 -> white, 1 -> light blue
        cmap = ListedColormap(['white', '#add8e6'])

        plt.figure(figsize=(width, height))
        im = plt.imshow(arr, aspect='auto', cmap=cmap, interpolation='none',
                        vmin=0, vmax=1)
        plt.xlabel('Time Interval')
        plt.ylabel('Night')
        plt.title('Heatmap of Missing Intervals')

        # Custom legend
        legend_handles = [
            Patch(facecolor='white', edgecolor='#cccccc', linewidth=1.5, label='Present'),
            Patch(facecolor='#add8e6', edgecolor='#cccccc', linewidth=1.5, label='Missing')
        ]
        plt.legend(handles=legend_handles, loc='upper right', frameon=True)

        plt.tight_layout()
        plt.show()


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

def provide_data_statistics(
        separated: list[Tuple[int, pd.DataFrame]], sample_rate: int=15) -> pd.DataFrame:
    """
    Creates statsistics useful in assessing the level of completeness of the
    data saught.
    :param separated: List of tuples of id and dataframe, where the df is the
        time series data for an individual
    :return: Dataframe with the statistics calculated for individuals
    """
    overall_stats_list = []
    for id_val, df_individual in separated:

        nights = Nights(zip_id=id_val, df=df_individual, sample_rate=sample_rate)
        stats = nights.overall_stats
        if stats:  # skip if stats is None
            stats['id'] = id_val
            stats['period_total_intervals'] = nights.total_intervals()
            stats['period_total_minutes'] = nights.total_minutes()
            overall_stats_list.append(stats)

    df_overall_stats = pd.DataFrame(overall_stats_list)
    df_overall_stats = df_overall_stats.set_index('id')
    df_overall_stats.sort_values('complete_nights', ascending=False)

    return df_overall_stats

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
    plt.show()

def get_complete_nights_only(all_nights_objects: List[Nights]) -> np.ndarray:
    """
    Returns a list of only the nights that are complete, i.e. have no missing intervals.
    :param nights_array: List of Nights objects
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
        for night_df in nights.get_nights():
            # Ensure the index is a DatetimeIndex
            if not isinstance(night_df.index, pd.DatetimeIndex):
                raise ValueError("Night DataFrame index must be a DatetimeIndex.")
            night_df['id'] = nights.zip_id
            night_df = night_df.reset_index().set_index(['id', 'datetime'])
            flat_file = pd.concat([flat_file, night_df])
    return flat_file
