import time
import pandas as pd
from datetime import timedelta
from loguru import logger

from src.configurations import Configuration, Irregular, FifteenMinute
from src.data_processing.read import (read_all_device_status,
                                      get_all_offsets_df_from_profiles)
from src.data_processing.write import write_read_record
from src.data_processing.format import as_flat_dataframe
from src.data_processing.preprocess import dedup_device_status_dataframes
from src.data_processing.resampling import ResampleDataFrame
from src.config import INTERIM_DATA_DIR
from src.cob_analysis import Cob


def main():
    start_time = time.time()
    config = Configuration()
    keep_cols = config.keep_columns
    fifteen_minute = FifteenMinute()
    resampled_parquet_file = (INTERIM_DATA_DIR /
                              fifteen_minute.file_name('parquet'))
    cob = Cob()

    # ----------------------Write consolidated flat file------------------------
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
        print(f'Completed writing device status flat file in '
              f'{timedelta(seconds=(time.time() - start_time))}')

        # ---------------------Write processed irregular file-------------------
        de_dup_result = dedup_device_status_dataframes(result)

        # write irregular
        write_read_record(de_dup_result,
                          as_flat_file,
                          INTERIM_DATA_DIR,
                          Irregular.file_name(),
                          keep_cols=config.keep_columns)
        print(f'Completed writing processed (irregular) flat file in '
              f'{timedelta(seconds=(time.time() - start_time))}')

        # -----------Write resampled files for 15 min intervals-----------------
        fifteen_min_dfs = []

        df = as_flat_dataframe(de_dup_result, drop_na=False,
                               keep_cols=config.keep_columns)

        for zip_id, group in df.groupby('id'):
            resampler = ResampleDataFrame(group)
            fifteen_min_dfs.append(
                resampler.resample_to(fifteen_minute).dropna(how='all', axis=1))

        # Concatenate and write resampled DataFrame
        pd.concat(fifteen_min_dfs).to_parquet(resampled_parquet_file)

        print(f'Completed writing resampled flat file(s) in '
              f'{timedelta(seconds=(time.time() - start_time))}')

    # ----------------Adjust timestamps by offsets to localise times------------
    cob.read_interim_data(file_name='15min_iob_cob_bg',
                                  file_type='parquet')

    # Get offsets from profiles - limited to individuals with one timezone
    profile_offsets = get_all_offsets_df_from_profiles(config)
    profile_offsets = (profile_offsets[
                           ~profile_offsets['id'].duplicated(keep=False) &
                           profile_offsets['offset'].notnull()].
                       set_index('id'))
    profile_offsets.to_csv(INTERIM_DATA_DIR / 'profile_offsets.csv')

    # Adjust timestamps in the resampled DataFrames
    args = {'height': 15, 'distance': 5, 'suppress': True}
    df_cob = cob.process_one_tz_individuals(profile_offsets, args)
    logger.info(f'Processed COB data for one timezone individuals: '
                f'{len(df_cob)} records remaining after processing.')


if __name__ == "__main__":
    main()
