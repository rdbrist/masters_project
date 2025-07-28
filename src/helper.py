# various convenience methods

# calculates a df of all the different read records
import dataclasses
import glob
from datetime import time, datetime, timedelta

import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from typing import Tuple, List, Union
from pathlib import Path

from src.configurations import Configuration, Resampling


def dataframe_of_read_record_stats(read_records: []):
    result = list(map(dataclasses.asdict, read_records))
    data = {
        'zip_id':
            list(map(lambda x: x.get('zip_id'), result)),
        'rows':
            list(map(lambda x: x.get('number_of_rows'), result)),
        'rows_without_nan':
            list(map(lambda x: x.get('number_of_rows_without_nan'),
                     result)),
        'rows_with_nan':
            list(map(lambda x: x.get('number_of_rows_with_nan'), result)),
        'earliest_date':
            list(map(lambda x: x.get('earliest_date'), result)),
        'newest_date':
            list(map(lambda x: x.get('newest_date'), result)),
        'is_android_upload':
            list(map(lambda x: x.get('is_android_upload'), result))
    }
    return pd.DataFrame(data)


def files_for_id(folder, zip_id):
    return glob.glob(str(Path(folder, zip_id).resolve()) + "/*.csv")


def bg_file_path_for(folder, zip_id):
    files = files_for_id(folder, zip_id)
    return Path(
        list(filter(lambda x: Configuration().bg_file in x, files))[0])


def device_status_file_path_for(folder, zip_id):
    files = files_for_id(folder, zip_id)
    return Path(
        list(filter(lambda x: Configuration().device_file in x, files))[0])


def preprocessed_file_for(folder, zip_id: str, sampling: Resampling):
    files = files_for_id(folder, zip_id)
    name = sampling.file_name()
    files_matching_name = list(filter(lambda x: name in x, files))
    return Path(files_matching_name[0]) if files_matching_name else None


def flat_preprocessed_file_for(folder, sampling: Resampling,
                               file_type: str = 'csv'):
    return Path(folder / sampling.file_name(file_type))


def check_df_index(df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Carries out checks on the dataframe index to ensure that it complies to both
    naming and datatype, i.e. {'id':int, 'datetime':datetime}
    :param df: Dataframe to be checked
    :return: Dataframe following correction, or return the same dataframe.
    """
    if df is None:
        raise TypeError('No dataframe provided in check_df_index()')
    if not isinstance(df.index, pd.MultiIndex):
        try:
            df['id'] = df['id'].astype(int)
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index(['id', 'datetime'], inplace=True)
        except ValueError:
            raise ValueError("DataFrame index must be a MultiIndex")
    if list(df.index.names) != ["id", "datetime"]:
        raise ValueError(
            "DataFrame index must be a MultiIndex with levels "
            "['id', 'datetime'].")
    id_level = df.index.get_level_values('id')
    datetime_level = df.index.get_level_values('datetime')
    if not pd.api.types.is_integer_dtype(id_level):
        raise ValueError("Index level 'id' must be of integer dtype.")
    if not pd.api.types.is_datetime64_any_dtype(datetime_level):
        raise ValueError(
            "Index level 'datetime' must be of datetime dtype.")

    return df


def separate_flat_file(df: pd.DataFrame) -> List[Tuple[int, pd.DataFrame]]:
    """
    Accepts a dataframe with ['id', 'datetime'] index and returns a list of
    tuples, holding the id as the first term and the dataframe (with only a
     datetime index) as the second term.
    :param df: Dataframe with ['id', 'datetime'] multi level index
    :return: List of tuples (int, pd.DataFrame)
    """
    separated = []
    for group in df.groupby('id'):
        separated.append(group)

    for i, (id_val, df) in enumerate(separated):
        df = df.reset_index().drop(columns='id').set_index('datetime')
        separated[i] = (id_val, df)

    return separated


def get_dfs_from_separated(separated, zip_ids: Union[int, list]) -> (
        Union)[pd.DataFrame, List[pd.DataFrame]]:
    """
    Get the dataframe(s) for the specified id(s) from the separated list.
    :param separated: List of tuples with id and df
    :param zip_ids: Int of single zip is or list of zip ids
    :return: DataFrame for the specified id or list of DataFrames for all ids in
        the list.
    """
    if isinstance(zip_ids, (int, np.integer)):
        for id_, df in separated:
            if id_ == zip_ids:
                return df
    elif isinstance(zip_ids, list):
        result = []
        for id_, df in separated:
            if id_ in zip_ids:
                result.append(df)
        return result
    return None


def filter_separated_by_ids(separated: List[Tuple[int, pd.DataFrame]],
                            ids: List[int]) -> List[Tuple[int, pd.DataFrame]]:
    """
    Filters the separated list by the specified ids.
    :param separated: List of tuples with id and df
    :param ids: List of ids to filter by
    :return: Filtered list of tuples with id and df
    """
    return [(id_, df) for id_, df in separated if id_ in ids]


def load_final_filtered_csv(config: Configuration,
                            interpolate_cob: bool = True) -> pd.DataFrame:
    """
    Loads the final filtered CSV file into a DataFrame.
    :param config: Configuration object containing the path to the final
        filtered CSV.
    :param interpolate_cob: (bool) Whether to interpolate the 'cob mean',
        'cob min', and 'cob max' columns.
    :return: DataFrame containing the data from the final filtered CSV.
    """
    df = pd.read_csv(config.final_filtered_csv)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index(['id', 'datetime'])
    if interpolate_cob:
        df[['cob mean', 'cob min', 'cob max']] = (
            df.groupby('id')[['cob mean', 'cob min', 'cob max']].
            transform(lambda x: x.interpolate(method='linear')))
    return check_df_index(df)  # Ensure the index is correct


def calculate_skew_kurtosis(df: pd.DataFrame,
                            variables: list = None) -> pd.DataFrame:
    """
    Calculate skewness and kurtosis for the variables included.
    :param df: (DataFrame) DataFrame containing the variables.
    :param variables: (list) List of variables to calculate skewness and
        kurtosis for.
    :return: (DataFrame) DataFrame with skewness and kurtosis values.
    """
    skewness = df[variables].apply(skew)
    kurt = df[variables].apply(kurtosis)

    return pd.DataFrame({'Skewness': skewness, 'Kurtosis': kurt})


def obfuscate_ids(series: pd.Series) -> np.ndarray:
    """
    Obfuscate the id in the series by converting it to a list of ids starting
    from 1000
    :param series: (pd.Series) Series containing the ids to obfuscate.
    :return: (list) List of obfuscated ids starting from 1000.
    """
    return pd.Categorical(series).codes.astype(int) + 1000


def get_night_start_date(
        x: Union[pd.DatetimeIndex, pd.Series, np.array, list] = None,
        night_start_hour: int = None) -> pd.Series:
    """
    Get the night start date for each timestamp in the input series or array and
    return it as a series of the same length as the input, where a DatetimeIndex
    or Series is expected.
    :param x: Union[pd.DatetimeIndex, pd.Series, np.array, list]
        Input timestamps from which to derive the night start date.
    :param night_start_hour: (int) Hour of the night start (0-23).
    :return:
    """
    if x is None or night_start_hour is None:
        print(night_start_hour)
        print(x)
        raise ValueError("Input timestamps and night start hour must not be "
                         "None.")

    if isinstance(x, pd.Series):
        s = pd.to_datetime(x)
    elif isinstance(x, pd.DatetimeIndex):
        s = pd.Series(x, index=x)
    elif isinstance(x, (np.ndarray, list)):
        s = pd.Series(pd.to_datetime(x))
    else:
        raise ValueError(f"Input is type {type(x)} and must be a pd.Series, "
                         f"pd.DatetimeIndex, np.ndarray, or list.")

    night_start = s.dt.floor('D') + pd.Timedelta(hours=night_start_hour)
    night_start[s.dt.hour < night_start_hour] -= pd.Timedelta(days=1)
    return pd.Series(night_start.dt.date, index=s.index)

def minutes_since_night_start(t, night_start):
    """
    Calculate the number of minutes since the night start time for a given
    datetime object, t.
    :param t: (datetime) The datetime object for which to calculate the minutes
    :param night_start: (datetime.time) The time at which the night starts
    :return: (int) The number of minutes since the night start time
    """
    t_minutes = t.hour * 60 + t.minute
    start_minutes = night_start.hour * 60 + night_start.minute
    if t < night_start:
        t_minutes += 24 * 60
    return t_minutes - start_minutes

def rank_minutes_series(series: pd.Series, night_start: time) -> pd.Series:
    """
    Rank the minutes intervals for the purpose of using the ranking for ordering
    of times in plot axes.
    :param series: (pd.Series) Series containing datetime objects.
    :param night_start: (datetime.time) The time at which the night starts.
    :return: (pd.Series) Series containing datetime objects.
    """
    minutes_since_start = (
        series.apply(lambda t: minutes_since_night_start(t, night_start)))
    rank = minutes_since_start.rank(method='dense').astype(int)
    return pd.Series(rank, index=series.index)

def normalise_overnight_time(dt_obj, split_hour=6):
    """
    Normalises a datetime object to a single reference date, handling overnight
    periods. Times before `split_hour` (e.g., 6 AM) are moved to the next day
    relative to the reference.
    """
    reference_date = datetime(1900, 1, 1)
    # If input is a time object, use as is; if datetime, extract time
    if isinstance(dt_obj, time):
        t = dt_obj
    elif isinstance(dt_obj, datetime):
        t = dt_obj.time()
    else:
        raise TypeError(
            "Input must be a datetime.datetime or datetime.time object")
    if t > time(split_hour, 0, 0):
        return datetime.combine(reference_date, t)
    else:
        return datetime.combine(reference_date + timedelta(days=1), t)

def format_xticks_as_hhmm(ax, unique_times_str_list):
    """
    Helper function to format x-axis ticks to HH:MM.
    :param ax: (matplotlib.axes.Axes) The axes object of the plot.
    :param unique_times_str_list: (list) A list of unique time strings ('%H:%M')
        that represent the x-axis tick positions.
    """
    # Set the tick locations to be the numerical indices of the unique times
    ax.set_xticks(range(len(unique_times_str_list)))
    # Set the tick labels to be the HH:MM strings
    ax.set_xticklabels(unique_times_str_list, rotation=45, ha='right')

import pandas as pd
import numpy as np

def generate_alphabetical_aliases(ids_input):
    unique_ids_sorted_array = np.unique(ids_input)
    unique_ids = unique_ids_sorted_array.tolist()

    if len(unique_ids) > 26:
        raise ValueError(f"Too many unique IDs ({len(unique_ids)}). Can only generate aliases for up to 26 unique IDs.")

    aliases_map = {}
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    for i, unique_id in enumerate(unique_ids):
         aliases_map[unique_id] = alphabet[i]

    return aliases_map

def cluster_colours():
    """
    Returns a list of distinct colours for clustering purposes.
    :return: List of colour hex codes.
    """
    import seaborn as sns
    tab10_colors = sns.color_palette('tab10')
    return {
        0: tab10_colors[0],
        1: tab10_colors[1],
        2: tab10_colors[2],
        3: tab10_colors[3]
    }