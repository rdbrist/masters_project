from datetime import time, datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


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
                 night_start=time(19, 0),
                 morning_end=time(12, 0),
                 sample_rate = 15):
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex.")
        if zip_id is None:
            raise ValueError("zip_id must be provided.")
        self.zip_id = zip_id
        self.sample_rate = sample_rate
        self.df = df.sort_index()
        self.night_start = night_start
        self.morning_end = morning_end
        self.df['night_start_date'] = (self.df.index.
                                       map(self.get_night_start_date))
        self.nights = self._split_nights()
        self.stats_per_night = self._create_stats_per_night()
        self.overall_stats = self._calculate_overall_stats()

    def get_night_start_date(self, timestamp, night_start_hour=None):
        """Determine the start date of the night period based on the timestamp."""
        if night_start_hour is None:
            night_start_hour = self.night_start.hour
        if timestamp.hour >= night_start_hour:
            return timestamp.date()
        else:
            return (timestamp - pd.Timedelta(days=1)).date()

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

    def get_df_for_night(self, date):
        """
        Returns the DataFrame for the night period defined by the date from the
        nights attribute of tuples (night_start_date, night_df).
        :param date: Date for which to get the night DataFrame
        :return: DataFrame for the night period
        """
        for nights_start_date, night_df in self.nights:
            if nights_start_date.date() == date:
                return night_df

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
        Split full time series dataframe for individual into separate
        timeseries dataframes covering each nightly period, based on the
        night_start and morning_end times.
        :return: List of dataframes, one pertaining to each night period
        """
        nights = []
        dates = pd.to_datetime(self.df.index.date)
        for date in pd.unique(dates):
            night_start_dt, morning_end_dt = self.get_night_period(date)
            mask = (self.df.index >= night_start_dt) & (self.df.index < morning_end_dt)
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
            return print(f'No stats per night have been calculated for '
                         f'{self.zip_id}. Returning no output.')

        bg_sd_values = [d['bg_sd'] for d in self.stats_per_night if
                        pd.notna(d['bg_sd'])]
        bg_range_values = [d['bg_range'] for d in self.stats_per_night if
                           pd.notna(d['bg_range'])]
        bg_iqr_values = [d['bg_iqr'] for d in self.stats_per_night if
                         pd.notna(d['bg_iqr'])]
        return {
            'count_of_nights': count_of_nights,
            'complete_nights':
                sum(1 for d in self.stats_per_night if
                    d['missed_intervals'] == 0),
            'avg_num_intervals':
                sum(d['num_intervals'] for d in
                    self.stats_per_night) / count_of_nights,
            'avg_missed_intervals':
                sum(d['missed_intervals'] for d in
                    self.stats_per_night) / count_of_nights,
            'avg_num_breaks':
                sum(d['num_breaks'] for d in
                    self.stats_per_night) / count_of_nights,
            'avg_break_length':
                sum(d['avg_break_length'] for d in
                    self.stats_per_night) / count_of_nights,
            'avg_max_break_length':
                sum(d['max_break_length'] for d in
                    self.stats_per_night) / count_of_nights,
            'avg_total_break_length':
                sum(d['total_break_length'] for d in
                    self.stats_per_night) / count_of_nights,
            'bg_sd_median':
                np.nanmedian(bg_sd_values) if bg_sd_values else np.nan,
            'bg_range_median':
                np.nanmedian(bg_range_values) if bg_range_values else np.nan,
            'bg_iqr_median':
                np.nanmedian(bg_iqr_values) if bg_iqr_values else np.nan,
            'missed_interval_vectors':
                [d['missed_interval_vector'] for d in self.stats_per_night],
        }

    def _create_stats_per_night(self):
        """
        Produces a dictionary of stats per night that can then be used for selection or further aggregation in overall stats.
        :return: List of dicts, one per night
        """
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
            avg_break_length = breaks.mean() if num_breaks > 0 else 0
            max_break_length = breaks.max() if num_breaks > 0 else 0
            total_break_length = breaks.sum() if num_breaks > 0 else 0
            if night_df['bg mean'].dtype != 'Float32':
                print(self.zip_id)
            bg = night_df['bg mean'].astype(float)
            stats.append({
                'night_date': date,
                'num_intervals': num_intervals,
                'missed_intervals': np.sum(vector),
                'num_breaks': num_breaks,
                'avg_break_length': avg_break_length,
                'max_break_length': max_break_length,
                'total_break_length': total_break_length,
                'missed_interval_vector': vector.tolist(),
                'bg_sd': bg.std(),
                'bg_range': bg.max() - bg.min(),
                'bg_mean': bg.mean(),
                'bg_iqr': bg.quantile(0.75) - bg.quantile(0.25),
            })

        return stats

    def remove_incomplete_nights(self):
        """
        Removes nights that have missing intervals, i.e. where the number of missed intervals is greater than 0.
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
            Patch(facecolor='white', edgecolor='#cccccc', linewidth=1.5, label='Present'),
            Patch(facecolor='#add8e6', edgecolor='#cccccc', linewidth=1.5, label='Missing')
        ]
        plt.legend(handles=legend_handles, loc='upper right', frameon=True)

        plt.tight_layout()
        plt.show()
