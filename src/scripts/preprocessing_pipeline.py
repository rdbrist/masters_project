import time
import pandas as pd
from datetime import timedelta

from src.configurations import (Configuration,
                                Irregular, FifteenMinute, FiveMinute)
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

    # ----------------------Write consolidated flat file------------------------
    as_flat_file = True

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

    # ---------------------Write processed irregular file-----------------------
    de_dup_result = dedup_device_status_dataframes(result)

    # write irregular
    write_read_record(de_dup_result,
                      as_flat_file,
                      INTERIM_DATA_DIR,
                      Irregular.csv_file_name(),
                      keep_cols=config.keep_columns)
    print(f'Completed writing processed (irregular) flat file in '
          f'{timedelta(seconds=(time.time() - start_time))}')

    # -----------Write resampled files for 5 & 15 min intervals-----------------
    fifteen_minute = FifteenMinute()
    five_minute = FiveMinute()

    resampled_dfs = {
        'fifteen_minute': [],
        'five_minute': []
    }

    df = as_flat_dataframe(de_dup_result, drop_na=False, keep_cols=keep_cols)

    for zip_id, group in df.groupby('id'):
        resampler = ResampleDataFrame(group)
        resampled_dfs['fifteen_minute'].append(
            resampler.resample_to(fifteen_minute))
        resampled_dfs['five_minute'].append(
            resampler.resample_to(five_minute))

    # Concatenate and write each resampled DataFrame
    (pd.concat(resampled_dfs['fifteen_minute']).
     reset_index(drop=True).
     to_csv(INTERIM_DATA_DIR / fifteen_minute.csv_file_name(), index=False))
    (pd.concat(resampled_dfs['five_minute']).
     reset_index(drop=True).
     to_csv(INTERIM_DATA_DIR / five_minute.csv_file_name(), index=False))

    print(f'Completed writing resampled flat file(s) in '
          f'{timedelta(seconds=(time.time() - start_time))}')

    # ----------------Adjust datetimes by offsets to localise times-------------

    # Get offsets from profiles - limited to individuals with one timezone
    profile_offsets = get_all_offsets_df_from_profiles(config)
    profile_offsets = (profile_offsets[
                           ~profile_offsets['id'].duplicated(keep=False) &
                           profile_offsets['offset'].notnull()].
                       set_index('id'))
    profile_offsets.to_csv(INTERIM_DATA_DIR / 'profile_offsets.csv')

    # Adjust datetimes in the resampled DataFrames
    cob_fifteen = Cob()
    cob_fifteen.read_interim_data(file_name='15min_iob_cob_bg', file_type='parquet')
    args = {'height': 15, 'distance': 5, 'suppress': True}
    df_cob = cob_fifteen.process_one_tz_individuals(profile_offsets, args)

if __name__ == "__main__":
    main()
