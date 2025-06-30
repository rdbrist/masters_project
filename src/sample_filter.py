from datetime import time
from loguru import logger

from src.candidate_selection import (remove_null_variable_individuals,
                                     create_nights_objects,
                                     provide_data_statistics)
from src.configurations import Resampling
from src.data_processing.read import read_profile_offsets_csv
from src.data_processing.read_preprocessed_df import (
    ReadPreprocessedDataFrame, apply_and_filter_by_offsets)
from src.helper import separate_flat_file
from src.nights import filter_nights, consolidate_df_from_nights
from src.configurations import Configuration


class SampleFilter:
    """Class to filter samples based on offsets and other criteria."""
    def __init__(self, night_start: time = None, morning_end: time = None,
                 sampling: Resampling = None, missed_intervals: int = None,
                 max_break_run: float = None,
                 min_nights: int = None,
                 cob_nan_min: float = 1,
                 iob_nan_min: float = 1,
                 bg_nan_min: float = 1):
        """
        Initialises the SampleFilter class.
        :param night_start: (time) Time of night start, e.g. 17:00
        :param morning_end: (time) End of the morning, e.g. 11:00
        :param sampling: (Resampling) Resampling class, e.g. Thirty()
        :param missed_intervals: (int) Allowable missed intervals for nights
        :param max_break_run: (float) Max string of missing intervals for nights
        :param min_nights: (int) Minimum number of nights required for candidate
        :param cob_nan_min: (float) Minimum percentage of COB NaN allowed
        :param iob_nan_min: (float) Minimum percentage of IOB NaN allowed
        :param bg_nan_min: (float) Minimum percentage of BG NaN allowed
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
        self.min_nights = min_nights
        self.missed_intervals = missed_intervals
        self.max_avg_break = max_break_run

        config = Configuration()
        # TODO: Add obfuscation to the offset ids or carry that out post merge
        df_offsets = read_profile_offsets_csv(config)
        df_processed = apply_and_filter_by_offsets(offsets_df=df_offsets,
                                                   interim_df=self.raw_df,
                                                   verbose=False)
        df_processed = remove_null_variable_individuals(df_processed)
        separated = separate_flat_file(df_processed)
        self.candidates = []
        self.nights_objects = None
        self.night_count = 0
        self.stats = None
        self.nights_objects = (
            create_nights_objects(separated, night_start=night_start,
                                  morning_end=morning_end,
                                  sample_rate=self.sample_rate))
        print(f"Number of nights objects created: {len(self.nights_objects)}")
        self.apply_constraints(missed_intervals, max_break_run, min_nights,
                               cob_nan_min, iob_nan_min, bg_nan_min)

    def apply_constraints(self, missed_intervals: int = None,
                          max_break_run: float = None,
                          min_nights: int = None,
                          cob_nan_min: float = None,
                          iob_nan_min: float = None,
                          bg_nan_min: float = None):
        """
        Filters the nights by the applied constraint for each Nights object,
        such that only valid nights are retained.
        """
        new_nights_objs = []
        night_count = 0
        for nights_obj in self.nights_objects:
            new_nights_list = (
                filter_nights(nights_obj, missed_intervals=missed_intervals,
                              max_break_run=max_break_run,
                              cob_nan_min=cob_nan_min,
                              iob_nan_min=iob_nan_min,
                              bg_nan_min=bg_nan_min))
            if len(new_nights_list) >= min_nights:
                self.candidates.append(nights_obj.zip_id)
                (new_nights_objs.
                 append(nights_obj.update_nights(new_nights_list)))
                night_count += len(nights_obj.nights)
        self.nights_objects = new_nights_objs
        self.night_count = night_count
        self.stats = provide_data_statistics(self.nights_objects)

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
        return self.nights_objects

    def get_filtered_stats(self):
        """
        Returns the statistics for the filtered candidates.
        :return: DataFrame with statistics for the candidates.
        """
        return self.stats[self.stats.index.isin(self.candidates)]

    def get_consolidated_df(self):
        """
        Returns a consolidated DataFrame for the filtered candidates.
        :return: DataFrame with consolidated data for the candidates.
        """
        return consolidate_df_from_nights(self.nights_objects)

    def return_counts(self, logging: bool = True):
        """
        Prints the counts of candidates and nights.
        """
        if logging:
            logger.info(f'For sample rate of {self.sample_rate} minutes:'
                        f'\n  min_nights={self.min_nights}'
                        f'\n  missed_intervals={self.missed_intervals}')
            logger.info(f"Number of candidates: {len(self.candidates)}")
            logger.info(f"Number of nights: {self.night_count}")
        return len(self.candidates), self.night_count
