from datetime import timedelta
from pathlib import Path
import pandas as pd
from loguru import logger

from src.configurations import Configuration, Resampling
from src.helper import flat_preprocessed_file_for, check_df_index, obfuscate_ids
from src.config import INTERIM_DATA_DIR


class ReadPreprocessedDataFrame:
    def __init__(self, sampling: Resampling,
                 file_path: Path = None,  # Includes the file name
                 file_type: str = 'csv'):
        """
        Class for reading preprocessed csv files into pandas df and configure
        sampling resolution for other classes to use.
        :param sampling: Resampling object: What the time series are resampled
            as this defines which file is being read
        :param file_type: str, Type of file to read, default 'csv'
        """
        self.__sampling = sampling
        self.__config = Configuration()
        self.__file_type = file_type
        if file_path is None:
            self.__file_path = flat_preprocessed_file_for(INTERIM_DATA_DIR,
                                                          self.__sampling,
                                                          self.__file_type)
        else:
            self.__file_path = file_path
        self.df = self.__read_df()

    def __read_df(self):
        try:
            dtypes = {'id': 'int', 'system': 'category',
                      'iob count': int, 'cob count': int, 'bg count': int}
            if self.__file_type == 'csv':
                df = (pd.read_csv(self.__file_path,
                                  parse_dates=['datetime'],
                                  dtype=dtypes,
                                  index_col=['id', 'datetime']))
                if 'Unnamed: 0' in df.columns:
                    df.drop(columns=['Unnamed: 0'], inplace=True)
                df = check_df_index(df)
            elif self.__file_type == 'parquet':
                df = pd.read_parquet(self.__file_path)
                df = check_df_index(df)
            else:
                raise ValueError("Invalid file type. "
                                 "Must be 'csv' or 'parquet'.")
            # Obfuscate IDs
            if self.__config.obfuscate_ids:
                ids = df.index.get_level_values('id')
                obfuscated_ids = obfuscate_ids(ids)
                df = df.copy()
                df = df.reset_index()
                df['id'] = obfuscated_ids
                df = df.set_index(['id', 'datetime'])
                (pd.DataFrame({'zip_id': ids, 'obfuscated_id': obfuscated_ids}).
                 to_csv(INTERIM_DATA_DIR / 'zip_ids_obfuscated_ids.csv',
                        index=False))
            df.sort_index(level=['id', 'datetime'], inplace=True)
        except FileNotFoundError:
            raise FileNotFoundError(f'File not found in path: '
                                    f'{self.__file_path}')
        except pd.errors.EmptyDataError as e:
            print(f"No data: {e}")
        except Exception as e:
            raise e

        if 'system' in df.columns:
            df.drop(columns=['system'], inplace=True)
        return df


def apply_and_filter_by_offsets(offsets_df: pd.DataFrame = None,
                                interim_df: pd.DataFrame = None,
                                verbose: bool = True) -> pd.DataFrame:
    """
    Applies the offsets from the offsets_df to the datetime column in the
    interim_df, such that they are adjusted. The offsets_df is assumed to
    be limited to single-timezone people only, i.e. all ids will be unique.
    :param offsets_df: Dataframe of offsets with id as index and an integer for
        the offset to apply to all timestamps for that person.
    :param interim_df: Dataframe to which the offsets have to be applied.
    :param verbose: If True, logs the ids that are missing in the offsets_df.
    :return: Dataframe with the same shape, with timestamps offset, and limited
        to only those ids that exist in both.
    """
    if offsets_df.index.duplicated().any():
        raise ValueError("Profile offsets DataFrame contains duplicate IDs."
                         " Please ensure each ID is unique such that only"
                         " one offset exists.")
    if interim_df is None or interim_df.empty:
        raise ValueError("interim_df must not be None or empty.")
    if offsets_df.index.name != 'id':
        raise ValueError("Profile offsets DataFrame must have 'id' index.")

    interim_df = check_df_index(interim_df)

    # Check for missing ids before mapping
    missing_ids = (
            set(interim_df.index.get_level_values('id')) -
            set(offsets_df.index))
    if missing_ids and verbose:
        logger.info(f"IDs missing in offsets_df: {missing_ids}")
    interim_df = (
        interim_df[~interim_df.index.get_level_values('id').isin(missing_ids)])
    interim_df = interim_df.reset_index()
    interim_df['offset'] = interim_df['id'].map(offsets_df['offset'])
    interim_df['datetime'] += interim_df['offset'] * pd.Timedelta(hours=1)
    interim_df['day'] = interim_df['datetime'].dt.date
    interim_df['time'] = interim_df['datetime'].dt.time
    return interim_df.set_index(['id', 'datetime']).sort_index()
