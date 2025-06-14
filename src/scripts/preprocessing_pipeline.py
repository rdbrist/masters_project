import time
import pandas as pd
import os
from pathlib import Path
from datetime import timedelta, time
from loguru import logger

from src.candidate_selection import remove_null_variable_individuals, \
    provide_data_statistics, Nights, reconsolidate_flat_file_from_nights, \
    plot_nights_vs_avg_intervals
from src.configurations import Configuration, Irregular, ThirtyMinute
from src.data_processing.read import (read_all_device_status,
                                      get_all_offsets_df_from_profiles)
from src.data_processing.read_preprocessed_df import (
    apply_and_filter_by_offsets, ReadPreprocessedDataFrame)
from src.data_processing.write import write_read_record
from src.data_processing.format import as_flat_dataframe
from src.data_processing.preprocess import dedup_device_status_dataframes
from src.data_processing.resampling import ResampleDataFrame
from src.config import INTERIM_DATA_DIR
from src.helper import separate_flat_file, filter_separated_by_ids
from src.time_series_analysis import plot_night_means_for_individual


def main():
    start_time = time.time()
    config = Configuration()
    sampling = ThirtyMinute()
    night_start = time(17, 0)  # 5 PM
    morning_end = time(11, 0) # 11 AM
    resampled_parquet_file = (INTERIM_DATA_DIR /
                              sampling.file_name('parquet'))

    # STAGE 1 : Write consolidated flat file
    # -------------------------------------------------------------------------
    as_flat_file = True

    if not resampled_parquet_file.exists():
        result = read_all_device_status(config)
        write_read_record(result,
                          as_flat_file,
                          INTERIM_DATA_DIR,
                          config.flat_device_status_csv_file_name,
                          file_type='csv')
        write_read_record(result,
                          as_flat_file,
                          INTERIM_DATA_DIR,
                          config.flat_device_status_parquet_file_name,
                          file_type='parquet')
        print(f'Completed writing device status flat files at '
              f'{timedelta(seconds=(time.time() - start_time))}')

        # STAGE 2 : Write processed irregular file
        # ---------------------------------------------------------------------
        de_dup_result = dedup_device_status_dataframes(result)

        # write irregular
        write_read_record(de_dup_result,
                          as_flat_file,
                          INTERIM_DATA_DIR,
                          Irregular.file_name(),
                          keep_cols=config.keep_columns)
        print(f'Completed writing processed (irregular) flat file at '
              f'{timedelta(seconds=(time.time() - start_time))}')

        # STAGE 3 : Write resampled files for chosen intervals
        # ---------------------------------------------------------------------
        resampled_dfs = []

        df = as_flat_dataframe(de_dup_result, drop_na=False,
                               keep_cols=config.keep_columns)

        for zip_id, group in df.groupby('id'):
            resampler = ResampleDataFrame(group)
            resampled_dfs.append(
                resampler.resample_to(sampling).dropna(how='all', axis=1))

        # Concatenate and write resampled DataFrame
        df_resampled = pd.concat(resampled_dfs)
        df_resampled.to_parquet(resampled_parquet_file)

        print(f'Completed writing resampled flat file(s) at '
              f'{timedelta(seconds=(time.time() - start_time))}')

    # STAGE 4 : Adjust timestamps by offsets to localise times
    # -------------------------------------------------------------------------

    if df_resampled not in locals():
        df_resampled = ReadPreprocessedDataFrame(sampling,
                                                 file_type='parquet')

    profile_offsets = get_all_offsets_df_from_profiles(config)
    profile_offsets = (profile_offsets[
                           ~profile_offsets['id'].duplicated(keep=False) &
                           profile_offsets['offset'].notnull()].
                       set_index('id'))
    profile_offsets.to_csv(INTERIM_DATA_DIR / 'profile_offsets.csv')

    df_localised = apply_and_filter_by_offsets(profile_offsets, df_resampled)
    logger.info(f'After filtering by single timezone, '
                f'n={len(df_localised.index.get_level_values("id").unique())}')

    # STAGE 5 : Final candidate selection, analysis and filtering
    # -------------------------------------------------------------------------
    # 1. Remove individuals with null variables
    df_processed = remove_null_variable_individuals(df)

    # 2. Separate the data into individual dataframes
    separated = separate_flat_file(df_processed)

    # 3. Process the data through the Nights class
    df_overall_stats = provide_data_statistics(separated,
                                               sample_rate=sampling.minutes,
                                               night_start=night_start,
                                               morning_end=morning_end)

    # 4. Aggregate stats and visualise the data
    plot_nights_vs_avg_intervals(df_overall_stats)

    # 5. Identify individuals with satisfactory level of completeness
    df_filtered = df_overall_stats[df_overall_stats['complete_nights'] > 30]
    candidates = df_filtered.index.tolist()
    logger.info(f'Final candidate: {candidates}')

    # 6. Get only the complete nights for the candidates for further analysis
    filtered_separated = filter_separated_by_ids(separated, candidates)
    nights_objects = []
    for id_, df in filtered_separated:
        nights = Nights(zip_id=id_, df=df,
                        night_start=night_start, morning_end=morning_end,
                        sample_rate=sampling.minutes)
        nights_objects.append(nights.remove_incomplete_nights())
        logger.info(f'Candidate: {id_}, Complete Nights: '
                    f'{nights.overall_stats["complete_nights"]}')

    df_all_selected = reconsolidate_flat_file_from_nights(nights_objects)

    for zip_id in candidates:
        plot_night_means_for_individual(df_all_selected, zip_id,
                                        night_start=night_start.hour,
                                        morning_end=morning_end.hour)


if __name__ == "__main__":
    main()
