# various convenience methods

# calculates a df of all the different read records
import dataclasses
import glob
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
    Get the night start date for each timestamp in the input series or array.
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
    if isinstance(x, pd.DatetimeIndex):
        x = pd.Series(x)
    elif isinstance(x, np.ndarray):
        x = pd.Series(pd.to_datetime(x))
    elif isinstance(x, list):
        x = pd.Series(pd.to_datetime(x))
    if not isinstance(x, pd.Series):
        raise ValueError(f"Input is type {type(x)} and must be a pd.Series, "
                         f"pd.DatetimeIndex, np.ndarray, or list.")
    night_start = x.dt.floor('D') + pd.Timedelta(hours=night_start_hour)
    night_start[x.dt.hour < night_start_hour] -= pd.Timedelta(days=1)
    return night_start.dt.date
