from datetime import time
from loguru import logger

from src.candidate_selection import (remove_null_variable_individuals,
                                     create_nights_objects,
                                     provide_data_statistics)
from src.configurations import Resampling
from src.data_processing.read import read_profile_offsets_csv
from src.data_processing.read_preprocessed_df import ReadPreprocessedDataFrame, \
    apply_and_filter_by_offsets
from src.helper import separate_flat_file
from src.nights import filter_nights
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
        self.stats = provide_data_statistics(self.nights_objects)
        self.candidates = []
        self._apply_constraints(missed_intervals, min_nights)


    def _apply_constraints(self, missed_intervals: int=None, min_nights: int=None):
        """
        Filters the nights by the applied constraint for each Nights object,
        such that only valid nights are retained.
        :param missed_intervals: (int) Number of missed intervals allowed
        """
        new_nights_objs = []
        for nights_obj in self.nights_objects:
            new_nights_list = (
                filter_nights(nights_obj, missed_intervals=missed_intervals))
            if len(new_nights_list) >= min_nights:
                self.candidates.append(nights_obj.zip_id)
                new_nights_objs.append(nights_obj.update_nights(new_nights_list))
        self.nights_objects = new_nights_objs

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