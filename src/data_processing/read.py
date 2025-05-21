# reads BG entries into a list of ReadRecords
import glob
import zipfile
from dataclasses import dataclass
from io import TextIOWrapper
from pathlib import Path
import logging
import pandas as pd
import re
from datetime import datetime, timezone
from typing import Union

from pandas.core import series

from src.configurations import Configuration, GeneralisedCols, OpenAPSConfigs


# Data object that keeps the information from reading each data zip file
@dataclass
class ReadRecord:
    zip_id: str = None  # Subject id
    is_android_upload: bool = False  # True if from android upload, False if not
    system: str = None  # For system specific files to indicate the system
    df: pd.DataFrame = None  # dataframe
    has_no_files: bool = False  # True if file to read was empty or not exist

    # calculated fields
    number_of_entries_files: int = 0  # number of entries files found
    number_of_rows: int = 0  # number of rows in total
    number_of_rows_without_nan: int = 0
    number_of_rows_with_nan: int = 0
    earliest_date: str = ''  # oldest date in series
    newest_date: str = ''  # newest date in series

    # helper method to set read records if there are no files
    def zero_files(self):
        self.has_no_files = True

    # return its own dataframe with the id added. Keep cols is a list, if None
    # keep all column, otherwise only specified column
    def df_with_id(self, keep_cols=None):
        if self.df is None:
            return None
        if keep_cols is None:
            keep_cols = self.df.columns

        missing_cols = [col for col in keep_cols if col not in self.df.columns]
        if missing_cols:
            print(f"Columns not in file for zip {self.zip_id}: {missing_cols}")
            self.df[missing_cols] = None

        result = self.df[keep_cols].copy()
        # drop row if all columns are empty
        result.dropna(how='all', inplace=True)
        result.drop_duplicates(inplace=True, ignore_index=True)
        result.insert(loc=0, column=GeneralisedCols.id, value=self.zip_id)
        result[GeneralisedCols.id] = result[GeneralisedCols.id].astype("string")
        if self.system is not None:
            result.insert(loc=0,
                          column=GeneralisedCols.system,
                          value=self.system)
            columns = {OpenAPSConfigs.iob: GeneralisedCols.iob,
                       OpenAPSConfigs.cob: GeneralisedCols.cob,
                       OpenAPSConfigs.bg: GeneralisedCols.bg,
                       OpenAPSConfigs.datetime: GeneralisedCols.datetime}
            if self.system == OpenAPSConfigs.system_name:
                result.rename(columns=columns, inplace=True)
        return result

    def add(self, df):
        if self.df is None:
            self.df = df
        else:
            self.df = pd.concat([self.df, df])

    def calculate_stats(self, time_col_name='time'):
        if self.has_no_files:
            return
        if self.df is None:
            return
        self.number_of_rows = self.df.shape[0]
        self.number_of_rows_without_nan = self.df.dropna().shape[0]
        self.number_of_rows_with_nan = (self.df.shape[0] -
                                        self.df.dropna().shape[0])
        self.earliest_date = str(self.df[time_col_name].min())
        self.newest_date = str(self.df[time_col_name].max())




# reads flat device data csv and does preprocessing
# allows path for file to read
def read_flat_device_status_df_from_file(file: Path, config: Configuration):
    return read_device_status_file_and_convert_date(headers_in_file(file),
                                                    config,
                                                    file)


# reads all BG files from each zip files without extracting the zip
def read_all_bg(config: Configuration):
    return read_all(config, read_bg_from_zip)


# reads all device status files into a list of read records
def read_all_device_status(config):
    """
    Reads all device status files from each zip file without extracting the zip.
    Return is a list of ReadRecords with the dataframes, consolidating the
    device status files from each zip file.
    :param config:
    :return: ReadRecord list with the dataframes
    """
    return read_all(config, read_device_status_from_zip)


# reads all files using function
def read_all(config, function):
    data = config.data_dir
    # get all zip files in folder
    filepaths = glob.glob(str(data) + "/*.zip")
    read_records = []
    for file in filepaths:
        # Android read below
        if file.endswith(config.android_aps_zip):
            continue
        read_record = function(file, config)
        read_records.append(read_record)
    return read_records


# reads BGs into df from the entries csv file in the given zip file without
# extracting the zip
def read_bg_from_zip(file_name, config):
    return read_zip_file(config,
                         file_name,
                         is_a_bg_csv_file,
                         read_entries_file_into_df)


# generic zip file read method
def read_zip_file(config,
                  file_name,
                  file_check_function,
                  read_file_into_df_function):
    read_record = ReadRecord()
    read_record.zip_id = Path(file_name).stem
    print(read_record.zip_id)
    # find bg files in the zip file
    with zipfile.ZipFile(file_name, mode="r") as archive:
        files_and_folders = archive.namelist()

        # finds all the .csv files that match the file check function
        files_to_read = [x for x in files_and_folders
                         if file_check_function(config, read_record.zip_id, x)]

        # check number of files
        number_of_files = len(files_to_read)
        # stop reading if there are no files
        if number_of_files == 0:
            read_record.zero_files()
            return read_record
        read_record.number_of_entries_files = number_of_files

        # read all the entries files into dataframes
        for file in files_to_read:
            info = archive.getinfo(file)

            # skip files that are zero size, but log them
            if info.file_size == 0:
                logging.info('Found empty file: ' + file +
                             ' for id: ' + read_record.zip_id)
                continue

            # read entries into pandas dataframe
            read_file_into_df_function(archive, file, read_record, config)

        # calculate some information from the dataframe
        time_col_name_for_stats = 'time'
        if 'device_status' in read_file_into_df_function.__name__:
            time_col_name_for_stats = 'created_at'
        read_record.calculate_stats(time_col_name_for_stats)
        return read_record


# reads BG data from entries file into df and adds it to read_record,
# config is there for consistency
def read_entries_file_into_df(archive, file, read_record, config):
    with archive.open(file, mode="r") as bg_file:
        try:
            df = pd.read_csv(TextIOWrapper(bg_file, encoding="utf-8"),
                             header=None,
                             dtype={
                                 'time': str,
                                 'bg': pd.Float64Dtype()
                             },
                             names=['time', 'bg'],
                             na_values=[' null', '', " "])
            df[['time']] = parse_date_columns(config.treat_timezone, df[['time']])
            read_record.add(df)
        except ValueError:  # A few files have headers that need cleansing first
            try:
                df = (pd.read_csv(TextIOWrapper(bg_file, encoding="utf-8"),
                                  header=0,
                                  usecols=['dateString', 'sgv'],
                                  na_values=[' null', '', " "],
                                  dtype={
                                      'dateString': str,
                                      'sgv': pd.Float64Dtype()
                                  }).
                      rename(columns={'dateString': 'time', 'sgv': 'bg'}))
                df[['time']] = parse_date_columns(config.treat_timezone, df[['time']])
                read_record.add(df)
            except Exception as e:
                print(f'ID {read_record.zip_id}: Could not read file: {file}')
                print(e)


# reads device status file into df and adds it to read_record
def read_device_status_file_into_df(archive, file, read_record, config):
    read_record.system = OpenAPSConfigs.system_name
    specific_cols_dic = config.device_status_col_type

    if specific_cols_dic:  # preprocess reading
        with archive.open(file, mode="r") as header_context:
            text_io_wrapper = TextIOWrapper(header_context, encoding="utf-8")
            actual_headers = headers_in_file(text_io_wrapper)
            missing_headers = [ele for ele in (specific_cols_dic.keys())
                               if ele not in list(actual_headers)]

            if missing_headers:
                if not any("enacted" in h for h in actual_headers):
                    # this is not a device status file from a looping period
                    return

                if not any("openaps" in h for h in actual_headers):
                    # this is likely a loop file and won't have bolus
                    # information in it, skip for now
                    return

        # read file for those headers
        with archive.open(file, mode="r") as file_context:
            file_to_read = TextIOWrapper(file_context, encoding="utf-8")
            df = read_device_status_file_and_convert_date(actual_headers,
                                                          config,
                                                          file_to_read)
    else:  # read file into one big dat file no encoding
        with archive.open(file, mode="r") as file_context:
            io_wrapper = TextIOWrapper(file_context, encoding="utf-8")
            df = pd.read_csv(io_wrapper)
    read_record.add(df)


def headers_in_file(file):
    header = pd.read_csv(file, nrows=0)
    return header.columns

# --------------------- Date Parsing Functions --------------------- #
def parse_date_columns(treat_timezone, df_time_cols: Union[pd.Series, pd.DataFrame]) \
        -> pd.DataFrame:
    """
    Takes in a dataframe of date columns and returns parsed columns
    :param treat_timezone: To determine the timezone treatment
    :param df_time_cols: pd.Series or pd.DataFrame
    :return parsed_cols: pd.DataFrame of parsed columns
    """
    # Only attempts one format, parse_date_string will attempt others
    dt_format = 'ISO8601'

    if isinstance(df_time_cols, pd.Series):
        parsed_cols = pd.Series()
        try:
            parsed_cols = (
                pd.to_datetime(df_time_cols, format=dt_format, utc=False))
        except ValueError:
            parsed_cols = (df_time_cols.
                           apply(lambda x: parse_date_string(treat_timezone, x)))
    else:
        parsed_cols = pd.DataFrame()
        for col_name in df_time_cols.columns:
            try:
                parsed_cols[col_name] = pd.to_datetime(df_time_cols[col_name],
                                                       format=dt_format,
                                                       utc=False)
            except ValueError:
                parsed_cols[col_name] = (
                    df_time_cols[col_name].
                    apply(lambda x: parse_date_string(treat_timezone, x)))

    if treat_timezone == 'localise':
        if isinstance(parsed_cols, pd.Series):
            parsed_cols = parsed_cols.dt.tz_localize(None)
        else:
            for col_name in parsed_cols.columns:
                parsed_cols[col_name] = parsed_cols[col_name].dt.tz_localize(None)
    elif treat_timezone == 'utc':
        parsed_cols = ensure_utc(parsed_cols)

    return parsed_cols

def parse_date_string(treat_timezone, date_str):
    if pd.isna(date_str):
        return pd.NaT

    parsed_dt = parse_int_and_standard_date(treat_timezone, date_str)



    if parsed_dt != pd.NaT:
        return parsed_dt

    raise ValueError(f'Could not parse date {date_str}')

def parse_standard_date(treat_timezone, date_str):
    formats = [
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%d %H:%M:%S%z',
        '%Y-%m-%d %H:%M:%S.%f%z',
        '%Y-%m-%d %H:%M:%S %z',
        '%Y-%m-%dT%H:%M:%S.%fZ',
        '%Y-%m-%dT%H:%M:%SZ',
        '%Y-%m-%dT%H:%M:%S%z',
        '%Y-%m-%dT%H:%M:%S.%f%z',
        '%a %b %d %H:%M:%S %Z %Y'
    ]

    for fmt in formats:
        try:
            # Replace any z or Z as the datetime becomes naive of tz otherwise
            date_str = re.sub(r'[Zz]$', '+00:00', date_str)
            new_dt = pd.to_datetime(date_str, format=fmt, utc=False)
            if treat_timezone == 'localise':
                # new_dt = new_dt.replace(tzinfo=None)
                new_dt = new_dt.tz_localize(None)
            elif treat_timezone == 'utc':
                new_dt = ensure_utc(new_dt)
            return new_dt
        except ValueError:
            logging.info(f'Could not parse date {date_str}: {date_str}')
            continue

    return pd.NaT


def parse_int_date(treat_timezone, date_str):
    try:
        numeric_dt = pd.to_numeric(date_str)
        if abs(numeric_dt) > 1e12:  # Threshold for milliseconds (1 trillion)
            unit = 'ms'
        elif numeric_dt < 0:
            return pd.NaT
        else:
            unit = 's'
        result = pd.to_datetime(numeric_dt, unit=unit, errors='raise')
        if treat_timezone == 'localise':
            result = result.tz_localize(None)
        elif treat_timezone == 'utc':
            result = ensure_utc(result)
        return result
    except ValueError:
        return pd.NaT


def parse_int_and_standard_date(treat_timezone, date_str):

    def is_integer(s):
        try:
            int(s)
            return True
        except ValueError:
            return False

    if pd.isna(date_str):
        return pd.NaT
    elif is_integer(date_str):  # Checks for int type epoch timestamps
        parsed_dt = parse_int_date(treat_timezone, date_str)
    else:
        parsed_dt = parse_standard_date(treat_timezone, date_str)

    # Last resort for odd timezones
    if pd.isna(parsed_dt):
        parsed_dt = correct_odd_tz(date_str)
        parsed_dt = parse_standard_date(treat_timezone, parsed_dt)

    if parsed_dt != pd.NaT:
        return parsed_dt

    return pd.NaT


def correct_odd_tz(date_str):  # Checks for untranslatable timezones
    tz_translate = {' CEST': ' +0200',
                    ' EDT': ' -0400',
                    ' EST': ' -0500',
                    ' CDT': ' -0500',
                    ' UTC': ' +0000'}

    for key, value in tz_translate.items():
        date_str = date_str.replace(key, value)

    return date_str

#------------------------------------------------------------------------------#

# reads OpenAPS device status file
def read_device_status_file_and_convert_date(actual_headers,
                                             config,
                                             file_to_read):
    # First check columns that are in this file
    time_cols = [k for k in config.time_cols() if k in actual_headers]
    cols = config.device_status_col_type.keys()
    df = pd.read_csv(file_to_read,
                     usecols=lambda c: c in set(cols),
                     dtype=config.device_status_col_type,
                     )

    df[time_cols] = parse_date_columns(config.treat_timezone, df[time_cols])

    return df


# checks if a file from zip namelist is a bg csv file
def is_a_bg_csv_file(config, patient_id, file_path):
    # file starts with patient id and _entries
    start_string = patient_id + config.bg_csv_file_start
    startswith = Path(file_path).name.startswith(start_string)

    # has right file ending
    endswith = file_path.endswith(config.bg_csv_file_extension)
    return startswith and endswith


# checks if a file from zip namelist is a device status csv file
def is_a_device_status_csv_file(config, patient_id, file_path):
    # file starts with patient id and _entries
    start_string = patient_id + config.device_status_csv_file_start
    startswith = Path(file_path).name.startswith(start_string)

    # has right file ending
    endswith = file_path.endswith(config.csv_extension)
    return startswith and endswith


# reads a device status file
def read_device_status_from_zip(file, config):
    return read_zip_file(config,
                         file,
                         is_a_device_status_csv_file,
                         read_device_status_file_into_df)

def ensure_utc(val):
    """
    Ensures input (Series or scalar) is timezone-aware and in UTC.
    """
    def series_to_utc(ser):
        if ser.dt.tz is None:
            return ser.dt.tz_localize('UTC')
        else:
            return ser.dt.tz_convert('UTC')

    if isinstance(val, pd.Series):
        val = series_to_utc(val)
    elif isinstance(val, pd.DataFrame):
        for col in val.columns:
            val[col] = series_to_utc(val[col])
    else:
        if val.tzinfo is None:
            val = val.tz_localize('UTC')
        else:
            val = val.tz_convert('UTC')

    return val