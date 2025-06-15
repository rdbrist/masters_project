from dataclasses import dataclass
from src.config import (PROJ_ROOT, INTERIM_DATA_DIR, RAW_DATA_DIR,
                        PROCESSED_DATA_DIR)

import pandas as pd
import yaml
from os import path


def load_private_yaml():
    private_file = path.join(PROJ_ROOT, 'private.yaml')
    assert (path.exists(private_file))
    with open(private_file, "r") as f:
        config = yaml.safe_load(f)
    return config


@dataclass
class OpenAPSConfigs:
    # values for openAPS
    iob = 'openaps/enacted/IOB'
    cob = 'openaps/enacted/COB'
    bg = 'openaps/enacted/bg'
    datetime = 'openaps/enacted/timestamp'
    system_name = 'OpenAPS'


@dataclass
class Aggregators:
    # colum name and name of aggregation function
    min = 'min'
    max = 'max'
    mean = 'mean'
    std = 'std'
    count = 'count'


@dataclass
class GeneralisedCols:
    # generalised configs across systems
    iob = 'iob'
    cob = 'cob'
    bg = 'bg'
    id = 'id'
    mean_iob = iob + ' ' + Aggregators.mean
    mean_cob = cob + ' ' + Aggregators.mean
    mean_bg = bg + ' ' + Aggregators.mean
    min_iob = iob + ' ' + Aggregators.min
    min_cob = cob + ' ' + Aggregators.min
    min_bg = bg + ' ' + Aggregators.min
    max_iob = iob + ' ' + Aggregators.max
    max_cob = cob + ' ' + Aggregators.max
    max_bg = bg + ' ' + Aggregators.max
    std_iob = iob + ' ' + Aggregators.std
    std_cob = cob + ' ' + Aggregators.std
    std_bg = bg + ' ' + Aggregators.std
    count_iob = iob + ' ' + Aggregators.count
    count_cob = cob + ' ' + Aggregators.count
    count_bg = bg + ' ' + Aggregators.count
    datetime = 'datetime'
    system = 'system'


@dataclass
class Resampling:
    minutes = None
    max_gap_in_min = None
    # how big the gap between two datetime stamps can be
    sample_rule = None
    # the frequency of the regular time series after resampling: 1H a reading
    # every hour, 1D a reading every day

    description = 'None'
    needs_max_gap_checking = False
    agg_cols = [Aggregators.min, Aggregators.max, Aggregators.mean,
                Aggregators.std, Aggregators.count]

    general_agg_cols_dictionary = {GeneralisedCols.id: 'first',
                                   GeneralisedCols.system: 'first',
                                   }
    @staticmethod
    def file_name():
        return ''


@dataclass
class Irregular(Resampling):
    description = 'None'

    @staticmethod
    def file_name(filetype: str = 'csv'):
        if filetype == 'csv':
            return 'irregular_iob_cob_bg.csv'
        elif filetype == 'parquet':
            return 'irregular_iob_cob_bg.parquet'


@dataclass
class Hourly(Resampling):
    minutes = 60
    max_gap_in_min = minutes
    # there needs to be a reading at least every hour for the data points to
    # be resampled for that hour
    sample_rule = '1h'
    needs_max_gap_checking = False
    description = 'Hourly'

    @staticmethod
    def file_name(filetype: str = 'csv'):
        if filetype == 'csv':
            return 'hourly_iob_cob_bg.csv'
        elif filetype == 'parquet':
            return 'hourly_iob_cob_bg.parquet'

@dataclass
class ThirtyMinute(Resampling):
    minutes = 30
    max_gap_in_min = minutes
    # there needs to be a reading at least every 30min for the data points to
    # be resampled for that period
    sample_rule = f'{str(minutes)}min'
    needs_max_gap_checking = False
    description = 'ThirtyMinute'

    @staticmethod
    def file_name(filetype: str = 'csv'):
        if filetype == 'csv':
            return '30min_iob_cob_bg.csv'
        elif filetype == 'parquet':
            return '30min_iob_cob_bg.parquet'

@dataclass
class FifteenMinute(Resampling):
    minutes = 15
    max_gap_in_min = minutes
    # there needs to be a reading at least every 15min for the data points to
    # be resampled for that period
    sample_rule = f'{str(minutes)}min'
    needs_max_gap_checking = False
    description = 'FifteenMinute'

    @staticmethod
    def file_name(filetype: str = 'csv'):
        if filetype == 'csv':
            return '15min_iob_cob_bg.csv'
        elif filetype == 'parquet':
            return '15min_iob_cob_bg.parquet'


@dataclass
class FiveMinute(Resampling):
    minutes = 5
    max_gap_in_min = minutes
    # there needs to be a reading at least every 5 min for the data points to
    # be resampled for that period
    sample_rule = f'{str(minutes)}min'
    needs_max_gap_checking = False
    description = 'FiveMinute'

    @staticmethod
    def file_name(filetype: str = 'csv'):
        if filetype == 'csv':
            return '5min_iob_cob_bg.csv'
        elif filetype == 'parquet':
            return '5min_iob_cob_bg.parquet'


@dataclass
class Daily(Resampling):
    minutes = 24 * 60
    max_gap_in_min = 180
    # a reading every three hours for a daily resampling to be created
    sample_rule = '1D'
    needs_max_gap_checking = True
    description = 'Daily'

    @staticmethod
    def file_name(stage: str = '', filetype: str = 'csv'):
        if filetype == 'csv':
            return 'daily_iob_cob_bg.csv'
        elif filetype == 'parquet':
            return 'daily_iob_cob_bg.parquet'


@dataclass
class Configuration:
    config = load_private_yaml()
    # READ CONFIGURATIONS
    data_dir = str(RAW_DATA_DIR)
    as_flat_file = config['flat_file']
    treat_timezone = config['treat_timezone']
    limit_to_2023_subset = config['limit_to_2023_subset']
    if limit_to_2023_subset:
        subset_ids = pd.read_csv(INTERIM_DATA_DIR / "15min_iob_cob_bg_insulin_need.csv",
                         usecols=['id'])
        zip_ids_2023_subset = subset_ids['id'].unique().tolist()
    perid_data_folder = INTERIM_DATA_DIR / 'perid'
    csv_extension = '.csv'
    parquet_extension = '.parquet'

    # bg files
    bg_csv_file_extension = '.json.csv'
    bg_csv_file_start = '_entries'
    bg_csv_file_android = 'BgReadings.csv'
    android_upload_info = 'UploadInfo.csv'
    bg_csv_file = 'bg_df.csv'
    bg_parquet_file = 'bg_df.parquet'

    # device status files
    device_status_csv_file_start = '_devicestatus'
    device_file = 'device_status_deduped.csv'
    device_status_col_type = {
        'id': str,
        'created_at': str,
        'device': str,
        'pump/clock': str,
        'pump/status/timestamp': str,
        'pump/status/suspended': str,
        'pump/status/status': str,
        'pump/status/bolusing': str,
        'pump/iob/timestamp': str,
        'pump/iob/iob': str,
        'openaps/enacted/deliverAt': str,
        'openaps/enacted/timestamp': str,
        'openaps/enacted/rate': pd.Float32Dtype(),
        'openaps/enacted/duration': pd.Float32Dtype(),
        'openaps/enacted/insulinReq': str,
        'openaps/enacted/COB': pd.Float32Dtype(),
        'openaps/enacted/IOB': pd.Float32Dtype(),
        'openaps/enacted/bg': pd.Float32Dtype(),
        'openaps/enacted/eventualBG': pd.Float32Dtype(),
        'openaps/enacted/minPredBG': pd.Float32Dtype(),
        'openaps/enacted/sensitivityRatio': pd.Float32Dtype(),
        'openaps/enacted/reason': str,
        'openaps/enacted/units': str,
        'openaps/iob/iob': pd.Float32Dtype(),
        'openaps/iob/bolusinsulin': pd.Float64Dtype(),
        'openaps/iob/microBolusInsulin': pd.Float32Dtype(),
        'openaps/iob/lastBolusTime': str,
        'openaps/iob/timestamp': str,
        'openaps/iob/lastTemp/rate': pd.Float32Dtype(),
        'openaps/iob/basaliob': str,
        'openaps/iob/netbasalinsulin': pd.Float32Dtype(),
        'openaps/iob/activity': str,
    }

    # profile files
    profile_csv_file_start = '_profile'

    # Output filename definitions
    device_status_prefix = 'device_status_df'
    tz_suffix_dict = {'keep': '_tz_aware',
                      'utc': '_tz_utc',
                      'localise': '_tz_naive'}
    tz_suffix = tz_suffix_dict[treat_timezone]
    device_file_prefix = device_status_prefix + tz_suffix

    flat_device_status_csv_file_name = device_file_prefix + csv_extension
    flat_device_status_parquet_file_name = (device_file_prefix +
                                            parquet_extension)
    flat_device_status_csv_file = (
            INTERIM_DATA_DIR / flat_device_status_csv_file_name)
    flat_device_status_parquet_file = (
            INTERIM_DATA_DIR / flat_device_status_parquet_file_name)
    dedup_flat_device_status_csv_file_name = (
            device_file_prefix + '_deduped' + csv_extension)
    dedup_flat_device_status_parquet_file_name = (
            device_file_prefix + '_deduped' + parquet_extension)
    dedup_flat_device_status_csv_file = (
            INTERIM_DATA_DIR / dedup_flat_device_status_csv_file_name)
    dedup_flat_device_status_parquet_file = (
            INTERIM_DATA_DIR / dedup_flat_device_status_parquet_file_name)
    profile_regions_csv_file = INTERIM_DATA_DIR / 'profile_regions.csv'
    profile_offsets_csv_file = INTERIM_DATA_DIR / 'profile_offsets.csv'
    final_filtered_csv = 'final_filtered_set.csv'
    feature_set_csv_file = PROCESSED_DATA_DIR / 'feature_set.csv'
    scaler_file = PROCESSED_DATA_DIR / 'scaler.pkl'

    # columns to keep
    # TODO use generalised cols instead
    keep_columns = [OpenAPSConfigs.datetime,
                    OpenAPSConfigs.iob,
                    OpenAPSConfigs.bg,
                    OpenAPSConfigs.cob]

    # Android APS has different format
    android_aps_zip = 'AndroidAPS Uploader.zip'

    @staticmethod
    def common_cols():
        # this should probably move to OpenAPS as it is OpenAPS specific
        return ['id', 'created_at', 'device']

    @staticmethod
    def info_columns():
        # returns the columns that have other info but not values to resample
        return [GeneralisedCols.datetime,
                GeneralisedCols.id,
                GeneralisedCols.system]

    @staticmethod
    def value_columns_to_resample():
        # returns all columns with values that need resampling
        return [GeneralisedCols.iob, GeneralisedCols.cob, GeneralisedCols.bg]

    @staticmethod
    def resampled_value_columns():
        # returns the columns for resampled values
        return [GeneralisedCols.mean_iob,
                GeneralisedCols.mean_cob,
                GeneralisedCols.mean_bg,
                GeneralisedCols.min_iob,
                GeneralisedCols.min_cob,
                GeneralisedCols.min_bg,
                GeneralisedCols.max_iob,
                GeneralisedCols.max_cob,
                GeneralisedCols.max_bg,
                GeneralisedCols.std_iob,
                GeneralisedCols.std_cob,
                GeneralisedCols.std_bg]

    @staticmethod
    def resampling_count_columns():
        return [GeneralisedCols.count_iob,
                GeneralisedCols.count_cob,
                GeneralisedCols.count_bg
                ]

    @staticmethod
    def resampled_mean_columns():
        return [GeneralisedCols.mean_iob,
                GeneralisedCols.mean_cob,
                GeneralisedCols.mean_bg,
                ]

    def enacted_cols(self):
        return [k for k in self.device_status_col_type.keys()
                if 'enacted' in k]

    def iob_cols(self):
        return [k for k in self.device_status_col_type.keys()
                if 'openaps/iob/' in k]

    def pump_cols(self):
        return [k for k in self.device_status_col_type.keys() if 'pump/' in k]

    def time_cols(self):
        profile_col = ['startDate']
        device_status_cols_a = \
            [k for k in self.device_status_col_type.keys()
             if 'time' in str(k).lower()]
        device_status_cols_b = ['created_at',
                                'openaps/enacted/deliverAt',
                                'pump/clock']
        entries_col = ['time']
        return (profile_col + device_status_cols_a +
                device_status_cols_b + entries_col)

# Configuration to use for unit tests. This turns Wandb logging off.
@dataclass
class TestConfiguration(Configuration):
    wandb_mode = 'todo'
