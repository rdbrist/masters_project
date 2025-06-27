from datetime import time
from loguru import logger

from src.candidate_selection import remove_null_variable_individuals, \
    get_all_individuals_night_stats, create_nights_objects, \
    provide_data_statistics
from src.configurations import ThirtyMinute, Resampling
from src.data_processing.read import read_profile_offsets_csv
from src.data_processing.read_preprocessed_df import ReadPreprocessedDataFrame, \
    apply_and_filter_by_offsets
from src.helper import separate_flat_file, filter_separated_by_ids
from src.nights import Nights, get_filtered_nights
from src.time_series_analysis import return_count_intervals
from src.configurations import Configuration


class SampleFilter:
    """Class to filter samples based on offsets and other criteria."""
    def __init__(self, night_start: time=None, morning_end: time=None,
                 sampling: Resampling=None, missed_intervals: int=None,
                 min_nights: int=None):
        """
        Initialises the SampleFilter class.
        :param night_start: (time) Time of night start, e.g. 17:00
        :param morning_end: (time) End of the morning, e.g. 11:00
        :param sampling: (Resampling) Resampling class, e.g. Thirty().
        :param missed_intervals: (int) Allowable missed intervals for nights.
        :param min_nights: (int) Minimum number of nights required for candidate
        """
        if any(p is None for p in [night_start, morning_end, sampling,
                                   missed_intervals]):
            raise ValueError("All parameters must be provided: "
                             "night_start, morning_end, sample_rate, "
                             "and missed_intervals.")
        self.raw_df = (
            ReadPreprocessedDataFrame(sampling, file_type='parquet').df)
        self.night_start = night_start
        self.morning_end = morning_end
        self.sample_rate = sampling.minutes
        self.count_of_intervals = (
            return_count_intervals(night_start, morning_end,
                                   minute_interval=self.sample_rate))

        config = Configuration()
        df_offsets = read_profile_offsets_csv(config)
        df_processed = apply_and_filter_by_offsets(offsets_df=df_offsets,
                                                   interim_df=self.raw_df,
                                                   verbose=False)
        df_processed = remove_null_variable_individuals(df_processed)
        separated = separate_flat_file(df_processed)
        self.nights_objects = create_nights_objects(separated,
                                                   night_start=night_start,
                                                   morning_end=morning_end,
                                                   sample_rate=self.sample_rate)
        self.stats = provide_data_statistics(self.nights_objects,)
        self._filter_candidates(missed_intervals, min_nights)

    def _filter_candidates(self, missed_intervals: int,
                           min_nights: int):
        """
        Filters the candidates based on the specified constraint.
        :param missed_intervals: (str) Allowable missed intervals for nights
        :param min_nights: (int) Minimum number of nights required for
            candidate.
        """
        filtered = self.stats[self.stats['missed_intervals'] <= missed_intervals]
        filtered = filtered[filtered['complete_nights'] >= min_nights]
        elif filter_constraint == 'single_interval':
            filtered = (
                self.stats[['complete_nights', 'single_interval_nights']]
                .copy())
            filtered['total_nights'] = (filtered['single_interval_nights'] +
                                        filtered['complete_nights'])
            filtered = filtered[filtered['total_nights'] != 0]

        else:
            raise ValueError("Invalid filter constraint. "
                             "Use 'single_interval' or 'complete_night'.")

        self.candidates = filtered.index.unique().tolist()
        print(f'Filter Constraint: {filter_constraint}, produces '
              f'{len(self.candidates)} candidates.')

    def _filter_nights_by_constraint(self, filter_constraint: str=None):
        """
        Filters the nights by the applied constraint for each Nights object,
        such that only valid nights are retained.
        :param filter_constraint: (str) Constraint to filter candidates.
        """
        new_nights_objs = []
        filtered_nights_objs = [nights_obj for nights_obj in self.nights_objects
                           if nights_obj.zip_id in self.candidates]
        if filter_constraint == 'complete_nights':
            for nights_obj in filtered_nights_objs:
                new_nights_objs.append(
                    get_filtered_nights(nights_obj, missed_intervals=0))
        elif filter_constraint == 'single_interval':
            for nights_obj in filtered_nights_objs:
                new_nights_objs.append(
                    get_filtered_nights(nights_obj, missed_intervals=1))
        self.new_nights_objects = new_nights_objs


    def apply_filter(self, filter_constraint: str,
                 min_nights: int = None):
        """
        Applies the filter to the candidates based on the constraint.
        :param filter_constraint: (str) Constraint to filter candidates.
        :param min_nights: (int) Minimum number of nights required for
            candidate.
        """
        if not isinstance(min_nights, int) or min_nights < 1:
            raise ValueError("min_nights must be a positive integer.")

        logger.info(f'Applying filter with constraint: {filter_constraint}, '
                    f'minimum nights: {min_nights}')

        self._filter_candidates(filter_constraint, min_nights)

    def get_filtered_candidates(self):
        """
        Returns the filtered candidates.
        :return: List of candidate IDs.
        """
        return self.candidates

    def get_filtered_nights(self):
        """
        Returns the nights objects for the filtered candidates.
        :return: List of Nights objects for the candidates.
        """
        nights = [nights for nights in self.nights_objects
                if nights.zip_id in self.candidates]
        if
        return

    def get_filtered_stats(self):
        """
        Returns the statistics for the filtered candidates.
        :return: DataFrame with statistics for the candidates.
        """
        return self.stats[self.stats.index.isin(self.candidates)]