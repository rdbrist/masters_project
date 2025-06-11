# various convenience methods

# calculates a df of all the different read records
import dataclasses
import glob
import pandas as pd
from typing import Tuple, List
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


def flat_preprocessed_file_for(folder, sampling: Resampling):
    return Path(folder / sampling.file_name())


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
            df.set_index(['id', 'datetime'], inplace=True)
        except:
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
