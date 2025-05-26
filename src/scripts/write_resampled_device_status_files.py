import os
from glob import glob
from os.path import normpath, basename
from pathlib import Path
import pandas as pd
import time

from src.configurations import (Configuration, Irregular, Daily, Hourly,
                                FifteenMinute, FiveMinute, GeneralisedCols)
from src.helper import preprocessed_file_for
from src.data_processing.read_preprocessed_df import ReadPreprocessedDataFrame
from src.data_processing.resampling import ResampleDataFrame
from src.config import INTERIM_DATA_DIR


def resample_irregular_flat_file(flat_file_path, output_folder):
    daily = Daily()
    hourly = Hourly()
    fifteen_minute = FifteenMinute()
    five_minute = FiveMinute()

    df = pd.read_csv(flat_file_path)
    df[GeneralisedCols.datetime] = (
        pd.to_datetime(df[GeneralisedCols.datetime], format='ISO8601'))
    # Assuming 'zip_id' is the column to group by
    resampled_dfs = {
        'daily': [],
        'hourly': [],
        'fifteen_minute': [],
        'five_minute': []
    }

    for zip_id, group in df.groupby('id'):
        resampler = ResampleDataFrame(group)
        resampled_dfs['daily'].append(resampler.resample_to(daily))
        resampled_dfs['hourly'].append(resampler.resample_to(hourly))
        resampled_dfs['fifteen_minute'].append(resampler.
                                               resample_to(fifteen_minute))
        resampled_dfs['five_minute'].append(resampler.resample_to(five_minute))

    # Concatenate and write each resampled DataFrame
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    pd.concat(resampled_dfs['daily']).reset_index(drop=True).to_csv(
        Path(output_folder, daily.csv_file_name()), index=False)
    pd.concat(resampled_dfs['hourly']).reset_index(drop=True).to_csv(
        Path(output_folder, hourly.csv_file_name()), index=False)
    pd.concat(resampled_dfs['fifteen_minute']).reset_index(drop=True).to_csv(
        Path(output_folder, fifteen_minute.csv_file_name()), index=False)
    pd.concat(resampled_dfs['five_minute']).reset_index(drop=True).to_csv(
        Path(output_folder, five_minute.csv_file_name()), index=False)


def main():
    start_time = time.time()
    # reads irregular sampled file (create first!) and writes daily and hourly
    # sampled files per id and as flat file
    flat_file_folder = Path(INTERIM_DATA_DIR)
    config = Configuration()
    per_id_folder = flat_file_folder / 'perid'
    irregular = Irregular()
    hourly = Hourly()
    daily = Daily()
    fifteen_minute = FifteenMinute()
    five_minute = FiveMinute()

    missing_zipids = []
    big_hourly_df = None
    big_daily_df = None
    big_fifteen_minute_df = None
    big_five_minute_df = None

    # write hourly and daily dfs per id
    zip_id_dirs = glob(os.path.join(per_id_folder, "*", ""))
    zip_ids = [basename(normpath(path_str)) for path_str in zip_id_dirs]
    for zip_id in zip_ids:
        # check that irregular file exists otherwise print error
        file = preprocessed_file_for(config.perid_data_folder,
                                     zip_id,
                                     irregular)
        if file is None:
            missing_zipids.append(zip_id)
            print("No irregular sampled file for zip id: " + zip_id)
            if len(missing_zipids) == len(zip_ids):
                try:
                    resample_irregular_flat_file(
                        flat_file_folder / irregular.csv_file_name(),
                        flat_file_folder)
                    return
                except Exception as e:
                    print(
                        f"No Per ID files and error resampling flat file: {e}")
                    return
            continue

        # read the irregular file into df
        df = ReadPreprocessedDataFrame(sampling=irregular, zip_id=zip_id).df

        # resample to hourly and daily df
        resampler = ResampleDataFrame(df)
        daily_df = resampler.resample_to(daily)
        hourly_df = resampler.resample_to(hourly)
        fifteen_minute_df = resampler.resample_to(fifteen_minute)
        five_minute_df = resampler.resample_to(five_minute)

        # write pre id
        daily_resampled_file_name = (
            Path(per_id_folder, zip_id, daily.csv_file_name()))
        daily_df.to_csv(daily_resampled_file_name)
        hourly_resampled_file_name = (
            Path(per_id_folder, zip_id, hourly.csv_file_name()))
        hourly_df.to_csv(hourly_resampled_file_name)
        fifteen_resampled_file_name = (
            Path(per_id_folder, zip_id, fifteen_minute.csv_file_name()))
        fifteen_minute_df.to_csv(fifteen_resampled_file_name)
        five_resampled_file_name = (
            Path(per_id_folder, zip_id, five_minute.csv_file_name()))
        five_minute_df.to_csv(five_resampled_file_name)

        # add to overall dataframe
        if big_hourly_df is None:
            big_hourly_df = hourly_df
            big_daily_df = daily_df
            big_fifteen_minute_df = fifteen_minute_df
            big_five_minute_df = five_minute_df
        else:
            if not hourly_df.empty and not hourly_df.isna().all().all():
                big_hourly_df = pd.concat([big_hourly_df, hourly_df])
            if not daily_df.empty and not daily_df.isna().all().all():
                big_daily_df = pd.concat([big_daily_df, daily_df])
            if (not fifteen_minute_df.empty and
                    not fifteen_minute_df.isna().all()):
                big_fifteen_minute_df = (
                    pd.concat([big_fifteen_minute_df, fifteen_minute_df]))
            if not five_minute_df.empty and not five_minute_df.isna().all():
                big_five_minute_df = (
                    pd.concat([big_five_minute_df, five_minute_df]))

    # reset index for big dfs
    big_hourly_df.reset_index(inplace=True, drop=True)
    big_daily_df.reset_index(inplace=True, drop=True)
    big_fifteen_minute_df.reset_index(inplace=True, drop=True)
    big_five_minute_df.reset_index(inplace=True, drop=True)

    # write flat_file dfs
    daily_resampled_file_name = (
        Path(flat_file_folder, daily.csv_file_name()))
    big_daily_df.to_csv(daily_resampled_file_name)
    hourly_resampled_file_name = (
        Path(flat_file_folder, hourly.csv_file_name()))
    big_hourly_df.to_csv(hourly_resampled_file_name)
    fifteen_minute_resampled_file_name = (
        Path(flat_file_folder, fifteen_minute.csv_file_name()))
    big_fifteen_minute_df.to_csv(fifteen_minute_resampled_file_name)
    five_minute_resampled_file_name = (
        Path(flat_file_folder, five_minute.csv_file_name()))
    big_five_minute_df.to_csv(five_minute_resampled_file_name)

    print('Number of zip ids without irregular device status files: '
          + str(len(missing_zipids)))
    print(time.time() - start_time)


if __name__ == "__main__":
    main()
