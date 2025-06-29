from pathlib import Path
from typing import Union

from src.data_processing.format import as_flat_dataframe
from src.data_processing.read import ReadRecord


def write_read_record(records: [ReadRecord],
                      as_flat_file: bool,
                      folder: Path,
                      file_name: str,
                      keep_cols: Union[list, None] = None,
                      file_type: str = 'csv'):
    """
    Writes either a flat file (multiple ids) or a file in a per id folder
    :param records: list of ReadRecords to write
    :param as_flat_file: flag to indicate if a flat file should be written
    :param folder: Path to folder where the file should be written
    :param file_name: name of the file to write, if as_flat_file is True this
        will be the name of the flat file
    :param keep_cols: list of columns to keep, if None all columns will be kept
    :param file_type: str, type of file to write, either 'csv' or 'parquet'
    :return: pd.DataFrame
    """
    if as_flat_file:
        # turn read records into a flat dataframe
        df = as_flat_dataframe(records, False, keep_cols=keep_cols)

        file = Path(folder, file_name)
        if file_type == 'csv':
            df.to_csv(file)
        elif file_type == 'parquet':
            df.to_parquet(file)
    else:
        # create folder
        for record in records:
            df = record.df_with_id(keep_cols=keep_cols)
            if df is None:
                continue

            file = Path(folder, record.zip_id, file_name)
            # create folders if not exist
            file.parent.mkdir(parents=True, exist_ok=True)
            # write df
            if file_type == 'csv':
                df.to_csv(file)
            elif file_type == 'parquet':
                df.to_parquet(file)

    return df
