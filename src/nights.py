from datetime import time, datetime
from typing import List, Tuple
from loguru import logger
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from src.helper import get_night_start_date


class Nights:
    """
    Class for analysing and managing nightly periods of data for a given
    individual. The class takes a DataFrame with a DatetimeIndex and splits it
    into individual nights based on the specified night start and morning end
    times.
    """
    def __init__(self,
                 df: pd.DataFrame,
                 zip_id: int = None,
                 night_start: time = None,
                 morning_end: time = None,
                 sample_rate: int = None):
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex.")
        if zip_id is None and sample_rate is None:
            raise ValueError("zip_id and sample_rate must be provided.")
        self.zip_id = zip_id
        self.sample_rate = sample_rate
        self.df = df.sort_index()
        self.night_start = night_start
        self.morning_end = morning_end
        self.df['night_start_date'] = (
            get_night_start_date(self.df.index, night_start.hour))
        self.nights = self._split_nights()
        self._calculate_stats()

    def _calculate_stats(self):
        self.bg_mean, self.bg_std = self._calculate_bg_mean_and_sd()
        self.stats_per_night = self._create_stats_per_night()
        self.overall_stats = self._calculate_overall_stats()

    def update_nights(self, nights: List[Tuple[datetime.date, pd.DataFrame]]):
        """
        Updates the nights attribute with a new list of nights.
        :param nights: List of tuples (night_start_date, night_df)
        """
        self.nights = nights
        self._calculate_stats()

        return self

    def _calculate_bg_mean_and_sd(self):
        """
        Calculates the mean blood glucose across all night periods.
        :return: Tuple of floats representing the mean and standard deviation
            of blood glucose for all nights
        """
        bg_mean = None
        for _, night_df in self.nights:
            if bg_mean is None:
                bg_mean = night_df['bg mean'].astype(float).copy()
            else:
                bg_mean = (
                    pd.concat([bg_mean, night_df['bg mean'].astype(float)]))
        if bg_mean is None or bg_mean.empty:
            return np.nan, np.nan
        return bg_mean.mean(), bg_mean.std()

    def get_night_period(self, date):
        """
        Given a date, night_start (datetime.time), and morning_end
        (datetime.time), returns the corresponding start and end pd.Timestamp
        for the nightly period.
        """
        if self.night_start is None or self.morning_end is None:
            raise ValueError("night_start and morning_end must be set to "
                             "valid times for get_night_period to work.")
        date_obj = pd.Timestamp(date).date()
        night_start_dt = pd.Timestamp.combine(date_obj, self.night_start)
        if self.morning_end <= self.night_start:
            morning_end_dt = pd.Timestamp.combine(
                date_obj + pd.Timedelta(days=1), self.morning_end)
        else:
            morning_end_dt = pd.Timestamp.combine(date_obj, self.morning_end)
        return night_start_dt, morning_end_dt

    def get_df_for_night(self, date):
        """
        Returns the DataFrame for the night period defined by the date from the
        nights attribute of tuples (night_start_date, night_df).
        :param date: Date for which to get the night DataFrame
        :return: DataFrame for the night period
        """
        for nights_start_date, night_df in self.nights:
            if nights_start_date == date:
                return night_df

    def get_stats_per_night(self):
        return self.stats_per_night

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
        Returns the number of intervals in the time period based on the sample
        rate.
        """
        return self._get_total_minutes() // self.sample_rate

    def _split_nights(self):
        """
        Split full time series dataframe for individual into separate
        timeseries dataframes covering each nightly period, based on the
        night_start and morning_end times.
        :return: List of dataframes, one pertaining to each night period
        """
        nights = []
        dates = pd.to_datetime(self.df.index.date)
        for date in pd.unique(dates):
            night_start_dt, morning_end_dt = self.get_night_period(date)
            mask = ((self.df.index >= night_start_dt) &
                    (self.df.index < morning_end_dt))
            night_df = self.df.loc[mask]
            if not night_df.empty:
                nights.append((night_start_dt.date(), night_df))

        return nights

    def _calculate_overall_stats(self):
        """
        Calculates stats for the overall dataset to determine the number of gaps
        in data (determined by missing intervals) to inform the level of
        completeness.
        :return: dict with average stats across all nights
        """
        count_of_nights = len(self.nights)
        if not self.stats_per_night:
            logger.info(f'No stats per night have been calculated for '
                        f'{self.zip_id}. Returning no output.')
            return

        bg_sd_values = [d['bg_sd'] for d in self.stats_per_night if
                        pd.notna(d['bg_sd'])]
        bg_range_values = [d['bg_range'] for d in self.stats_per_night if
                           pd.notna(d['bg_range'])]
        bg_iqr_values = [d['bg_iqr'] for d in self.stats_per_night if
                         pd.notna(d['bg_iqr'])]
        cob_nans = sum(d['cob_nans'] for d in self.stats_per_night)
        iob_nans = sum(d['iob_nans'] for d in self.stats_per_night)
        bg_nans = sum(d['bg_nans'] for d in self.stats_per_night)
        return {
            'count_of_nights': count_of_nights,
            'complete_nights':
                sum(1 for d in self.stats_per_night if
                    d['missed_intervals'] == 0),
            'single_interval_nights':
                sum(1 for d in self.stats_per_night if
                    d['missed_intervals'] == 1),
            'avg_num_intervals':
                sum(d['num_intervals'] for d in
                    self.stats_per_night) / count_of_nights,
            'avg_missed_intervals':
                sum(d['missed_intervals'] for d in
                    self.stats_per_night) / count_of_nights,
            'avg_num_breaks':
                sum(d['num_breaks'] for d in
                    self.stats_per_night) / count_of_nights,
            'avg_break_duration':
                sum(d['avg_break_duration'] for d in
                    self.stats_per_night) / count_of_nights,
            'avg_max_break_duration':
                sum(d['max_break_duration'] for d in
                    self.stats_per_night) / count_of_nights,
            'avg_total_break_duration':
                sum(d['total_break_duration'] for d in
                    self.stats_per_night) / count_of_nights,
            'bg_sd_median':
                np.nanmedian(bg_sd_values) if bg_sd_values else np.nan,
            'bg_range_median':
                np.nanmedian(bg_range_values) if bg_range_values else np.nan,
            'bg_iqr_median':
                np.nanmedian(bg_iqr_values) if bg_iqr_values else np.nan,
            'total_cob_nans': cob_nans,
            'total_iob_nans': iob_nans,
            'total_bg_nans': bg_nans,
            'cob_nan_ratio':
                cob_nans / count_of_nights * self.total_intervals(),
            'iob_nan_ratio':
                iob_nans / count_of_nights * self.total_intervals(),
            'bg_nan_ratio':
                bg_nans / count_of_nights * self.total_intervals(),
            'missed_interval_vectors':
                [d['missed_interval_vector'] for d in self.stats_per_night],
        }

    def _create_stats_per_night(self):
        """
        Produces a dictionary of stats per night that can then be used for
        selection or further aggregation in overall stats.
        :return: List of dicts, one per night
        """
        if self.night_start is None:
            raise ValueError("night_start must be set to valid times for "
                             "_create_stats_per_night to work.")

        def max_run_of_ones(lst):  # For max run of missing intervals
            max_run = current_run = 0
            for val in lst:
                if val == 1:
                    current_run += 1
                    max_run = max(max_run, current_run)
                else:
                    current_run = 0
            return max_run

        stats = []
        for night_start_date, night_df in self.nights:
            if night_df.index[0].hour < self.night_start.hour:
                date = night_df.index[0].date() - pd.Timedelta(days=1)
            else:
                date = night_df.index[0].date()
            night_start_dt, morning_end_dt = self.get_night_period(date)
            expected_index = pd.date_range(
                start=night_start_dt,
                end=morning_end_dt - pd.Timedelta(minutes=self.sample_rate),
                freq=f'{self.sample_rate}min'
            )
            vector = np.where(expected_index.isin(night_df.index), 0, 1)
            times = night_df.index.sort_values()
            # Calculate time differences in minutes between timestamps
            diffs = times.to_series().diff().dropna().dt.total_seconds() / 60
            # A break is any gap greater than the sample rate
            breaks = diffs[diffs > self.sample_rate]
            num_intervals = len(times)
            num_breaks = len(breaks)
            avg_break_duration = breaks.mean() if num_breaks > 0 else 0
            max_break_duration = breaks.max() if num_breaks > 0 else 0
            total_break_duration = breaks.sum() if num_breaks > 0 else 0

            if night_df['bg mean'].dtype != 'Float32':
                print(self.zip_id)
            bg = night_df['bg mean'].astype(float)
            bg_night_mean = bg.mean()
            cob_nans = night_df['cob mean'].isna().sum()
            iob_nans = night_df['iob mean'].isna().sum()
            bg_nans = night_df['bg mean'].isna().sum()
            stats.append({
                'night_date': date,
                'num_intervals': num_intervals,
                'missed_intervals': np.sum(vector),
                'num_breaks': num_breaks,
                'max_break_run': max_run_of_ones(vector),
                'avg_break_duration': avg_break_duration,
                'max_break_duration': max_break_duration,
                'total_break_duration': total_break_duration,
                'missed_interval_vector': vector.tolist(),
                'bg_sd': bg.std(),
                'bg_range': bg.max() - bg.min(),
                'bg_mean': bg_night_mean,
                'bg_iqr': bg.quantile(0.75) - bg.quantile(0.25),
                'bg_zscore': (bg_night_mean - self.bg_mean) / self.bg_std,
                'cob_nans': cob_nans,
                'iob_nans': iob_nans,
                'bg_nans': bg_nans,
                'cob_nan_ratio': cob_nans / num_intervals,
                'iob_nan_ratio': iob_nans / num_intervals,
                'bg_nan_ratio': bg_nans / num_intervals,
            })

        return stats

    def remove_incomplete_nights(self):
        """
        Removes nights that have missing intervals, i.e. where the number of
        missed intervals is greater than 0.
        :return: self with incomplete nights removed
        """
        self.stats_per_night = [s for s in self.stats_per_night if
                                s['missed_intervals'] == 0]
        desired_dates = {s['night_date'] for s in self.stats_per_night if
                         s['missed_intervals'] == 0}
        # Filter self.nights to only those with a matching date
        self.nights = [night for night in self.nights if
                       night[0] in desired_dates]
        self.overall_stats = self._calculate_overall_stats()

        return self

    def plot_break_distribution(self):
        """
        Plots the total break length per night as bars (left axis) and the
        number of breaks as a line (right axis), sorted by total break length
        (ascending).
        """
        if not self.stats_per_night:
            print("No stats to plot.")
            return

        # Extract and sort by total break length
        stats = sorted(self.stats_per_night,
                       key=lambda d: d["total_break_duration"])
        total_break_duration = [d["total_break_duration"] for d in stats]
        num_breaks = [d["num_breaks"] for d in stats]

        fig, ax1 = plt.subplots(figsize=(10, 5))

        # Bar plot for total break length (left y-axis)
        ax1.bar(range(len(total_break_duration)), total_break_duration,
                color='skyblue', label='Total Break Length')
        ax1.set_xlabel('Night (sorted by total break length)')
        ax1.set_ylabel('Total Break Length (minutes)', color='skyblue')
        ax1.tick_params(axis='y', labelcolor='skyblue')

        # Line plot for number of breaks (right y-axis)
        ax2 = ax1.twinx()
        ax2.plot(range(len(num_breaks)), num_breaks, color='orange',
                 marker='o', label='Number of Breaks')
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
        total_break_duration =\
            [d["total_break_duration"] for d in self.stats_per_night]

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
        plt.hist(total_break_duration, bins=bins, color='orange', alpha=0.8)
        plt.xlabel('Total Break Length (minutes)')
        plt.ylabel('Count (Nights)')
        plt.title('Histogram of Total Break Lengths per Night')
        plt.tight_layout()
        plt.show()

    def plot_breaks_scatter(self):
        """
        Plots a scatter plot of number of breaks (x-axis) vs total break length
        (y-axis) per night.
        """
        if not self.stats_per_night:
            print("No stats to plot.")
            return

        num_breaks = [d["num_breaks"] for d in self.stats_per_night]
        total_break_duration = \
            [d["total_break_duration"] for d in self.stats_per_night]

        plt.figure(figsize=(8, 5))
        plt.scatter(num_breaks, total_break_duration, color='purple', alpha=0.7)
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
        date = self.nights[0][0]
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
            Patch(facecolor='white', edgecolor='#cccccc',
                  linewidth=1.5, label='Present'),
            Patch(facecolor='#add8e6', edgecolor='#cccccc',
                  linewidth=1.5, label='Missing')
        ]
        plt.legend(handles=legend_handles, loc='upper right', frameon=True)

        plt.tight_layout()
        plt.show()


def filter_nights(nights: Nights, missed_intervals: int,
                  max_break_run: float,
                  cob_nan_min: float,
                  iob_nan_min: float,
                  bg_nan_min: float) -> (
        List[Tuple[datetime.date, pd.DataFrame]]):
    """
    Returns a list of only the nights within the number of missing intervals
    provided.
    intervals.
    :param nights: Nights object
    :param missed_intervals: Number of nights intervals missing to filter by
    :param max_break_run: Maximum string of missing intervals
    :param cob_nan_min: Minimum percentage of COB NaN allowed
    :param iob_nan_min: Minimum percentage of IOB NaN allowed
    :param bg_nan_min: Minimum percentage of BG NaN allowed
    :return: List of objects with shape (id, [night_df, ...])
    """
    night_dates = [s['night_date'] for s in nights.stats_per_night
                   if s['missed_intervals'] <= missed_intervals and
                   s['max_break_run'] <= max_break_run and
                   s['cob_nan_ratio'] <= cob_nan_min and
                   s['iob_nan_ratio'] <= iob_nan_min and
                   s['bg_nan_ratio'] <= bg_nan_min]
    filtered_nights = [
        (night_date, night_df) for night_date, night_df in nights.nights
        if night_date in night_dates]

    return filtered_nights


def nights_with_missed_intervals(
        nights_objects: List[Nights], missed_intervals: int) -> List[Nights]:
    """
    Returns a list of Nights objects that have the specified number of
    missed intervals.
    :param nights_objects: List of Nights objects
    :param missed_intervals: Number of missed intervals to filter by
    :return: List of Nights objects with the specified number of missed
        intervals
    """
    return [nights for nights in nights_objects if
            any(s['missed_intervals'] <= missed_intervals
                for s in nights.stats_per_night)]


def consolidate_df_from_nights(
        nights_objects: List[Nights]) -> pd.DataFrame:
    """
    Reconstructs a flat file from the list of Nights objects in the common
    format, with a multi-index of id and datetime.
    :param nights_objects: List of Nights objects
    :return: DataFrame with the reconstructed flat file
    """
    consolidated_df = pd.DataFrame()
    for nights in nights_objects:
        for night_start_date, night_df in nights.nights:
            # Ensure the index is a DatetimeIndex
            if not isinstance(night_df.index, pd.DatetimeIndex):
                raise ValueError("Night DataFrame index must be a "
                                 "DatetimeIndex.")
            df = night_df.copy()
            df['id'] = nights.zip_id
            df = df.reset_index().set_index(['id', 'datetime'])
            consolidated_df = pd.concat([consolidated_df, df])
    return consolidated_df
