import pandas as pd
from datetime import timedelta
from loguru import logger
from typing import Tuple
from matplotlib import pyplot as plt
from datetime import timedelta, time, datetime

class Nights:
    def __init__(self, zip_id, df, night_start=time(19, 0), morning_end=time(12, 0), sample_rate = 15):
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

    def _get_total_minutes(self):
        """
        Returns the total number of minutes between night_start and morning_end.
        """
        base_date = datetime(2000, 1, 1)
        start_dt = datetime.combine(base_date, self.night_start)
        # Handle overnight periods
        if self.morning_end <= self.night_start:
            end_dt = datetime.combine(base_date + timedelta(days=1), self.morning_end)
        else:
            end_dt = datetime.combine(base_date, self.morning_end)
        return int((end_dt - start_dt).total_seconds() // 60)

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
            date_obj = pd.Timestamp(date).date()
            night_start_dt = pd.Timestamp.combine(date_obj, self.night_start)
            morning_end_dt = pd.Timestamp.combine(date_obj + timedelta(days=1), self.morning_end)
            mask = (self.df.index >= night_start_dt) & (self.df.index < morning_end_dt)
            night_df = self.df.loc[mask]
            if not night_df.empty:
                nights.append(night_df)

        return nights

    def remove_incomplete_nights(self):
        """
        Removes any nights that do not have a timestamp at each 15 minute interval.
        """
        complete_nights = []
        for night_df in self.nights:
            if night_df.index.nunique() == self.total_intervals:
                complete_nights.append(night_df)
        self.nights = complete_nights

    def _calculate_overall_stats(self):
        """
        Calculates stats for the overall dataset to determine the number of gaps in
        data (determined by missing intervals) to inform the level of completeness.
        :return: dict with average stats across all nights
        """
        if not self.stats_per_night:
            return print(f'No stats per night have been calculated for {self.zip_id}. Returning no output.')
        avg_num_intervals = sum(d['num_intervals'] for d in self.stats_per_night) / len(self.stats_per_night)
        avg_num_breaks = sum(d['num_breaks'] for d in self.stats_per_night) / len(self.stats_per_night)
        avg_break_length = sum(d['avg_break_length'] for d in self.stats_per_night) / len(self.stats_per_night)
        avg_max_break_length = sum(d['max_break_length'] for d in self.stats_per_night) / len(self.stats_per_night)
        avg_total_break_length = sum(d['total_break_length'] for d in self.stats_per_night) / len(self.stats_per_night)
        complete_nights =  sum(1 for d in self.stats_per_night if d['num_breaks'] == 0)

        return {
            'count_of_nights': len(self.nights),
            'complete_nights': complete_nights,
            'avg_num_intervals': avg_num_intervals,
            'avg_num_breaks': avg_num_breaks,
            'avg_break_length': avg_break_length,
            'avg_max_break_length': avg_max_break_length,
            'avg_total_break_length': avg_total_break_length,
        }

    def _create_stats_per_night(self):
        """
        Produces a dictionary of stats per night that can then be used for selection or further aggregation in overall stats.
        :return: List of dicts, one per night
        """
        stats = []
        for night_df in self.nights:
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
                'num_breaks': num_breaks,
                'avg_break_length': avg_break_length,
                'max_break_length': max_break_length,
                'total_break_length': total_break_length
            })

        return stats

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


def apply_and_filter_by_offsets(
        offsets_df: pd.DataFrame = None,
        interim_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Applies the offsets from the offsets_df to the
    :param offsets_df: Dataframe of offsets with id as index and an integer for
        the offset to apply to all timestamps for that person.
    :param interim_df: Dataframe to which the offsets have to be applied.
    :return: Dataframe with the same shape, with timestamps offset, and limited
        to only those ids that exist in both.
    """
    if offsets_df.index.duplicated().any():
        raise ValueError("Profile offsets DataFrame contains duplicate IDs."
                         " Please ensure each ID is unique such that only"
                         " one offset exists.")

    # Check for missing ids before mapping
    missing_ids = (
            set(interim_df.index.get_level_values('id')) -
            set(offsets_df.index))
    if missing_ids:
        raise ValueError(f"IDs missing in offsets_df: {missing_ids}")

    interim_df = interim_df.reset_index()
    interim_df['offset'] = interim_df['id'].map(offsets_df['offset'])
    interim_df['datetime'] += interim_df['offset'].apply(timedelta)
    interim_df['day'] = interim_df['datetime'].dt.date
    interim_df['time'] = interim_df['datetime'].dt.time
    return interim_df.set_index(['id', 'datetime']).sort_index()

def remove_null_variable_individuals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes peoples from the dataset that have null or zero values for any
    IOB, COB or BG variable across their dataset.
    :param df: Dataframe to process
    :return: Dataframe with individuals removed
    """
    check_cols = ['iob count', 'cob count', 'bg count']
    ids_with_only_nans_or_zeros = {}
    for col in check_cols:
        mask = (
            df.groupby('id')[col]
            .apply(lambda x: x.isna().all() or (x.fillna(0) == 0).all())
        )
        ids_with_only_nans_or_zeros[col] = mask[mask].index.tolist()
    ids = set()
    for key, val in ids_with_only_nans_or_zeros.items():
        ids.update(set(val))
    logger.info(f'Following individuals have one or more variables missing: '
                f'{ids}')
    return df[~df['id'].isin(ids)]

def provide_statistics(
        separated: list(Tuple[int, pd.DataFrame])) -> pd.DataFrame:
    """
    Creates statsistics useful in assessing the level of completeness of the
    data saught.
    :param separated: List of tuples of id and dataframe, where the df is the
        time series data for an individual
    :return: Dataframe with the statistics calculated for individuals
    """
    overall_stats_list = []
    for id_val, df_individual in separated:
        nights = Nights(zip_id=id_val, df=df_individual, sample_rate=15)
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
    

