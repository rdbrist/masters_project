import glob
import pandas as pd
import pytest
from pathlib import Path

from src.data_processing.write import write_read_record
from tests.helper.BgDfBuilder import BgDfBuilder
from tests.helper.ReadRecordBuilder import ReadRecordBuilder


folder = Path(__file__).parent / 'test_data/'
per_id_folder = folder / 'per_id'
flat_file_name = 'some_flat_file.csv'
flat_file_path = Path(folder, flat_file_name)
# build test ReadRecords
id1 = '123466'
df1 = BgDfBuilder().build()
test_record1 = ReadRecordBuilder().with_id(id1).with_df(df1).build()

id2 = 'ab'
df2 = BgDfBuilder().build()
test_record2 = ReadRecordBuilder().with_id(id2).with_df(df2).build()

id3 = '456'
df3 = BgDfBuilder().build()
test_record3 = ReadRecordBuilder().with_id(id3).with_df(df3).build()
records = [test_record1, test_record2, test_record3]


def per_id_files():
    return glob.glob(str(Path(per_id_folder).resolve()) + "/*/*.csv")

def sub_df_for_id(df, ids):
    result = df.loc[df['id'] == ids].drop(columns=['id'])
    result.reset_index(inplace=True, drop=True)
    return result

def read_df_from_csv(file):
    time_col = 'time'
    df = pd.read_csv(file, index_col=[0])
    df[time_col] = pd.to_datetime(df[time_col])
    df['id'] = df['id'].astype("string")
    df.sort_values(by=time_col, inplace=True)
    return df


@pytest.fixture(autouse=True)
def run_before_and_after_tests(tmpdir):
    # Setup:

    yield  # this is where the testing happens
    # Teardown:
    flat_file_path.unlink(True)
    # delete csv files
    for file in per_id_files():
        Path(file).unlink(True)
    directory = Path(per_id_folder)
    # delete id directories
    if directory.exists():
        for item in directory.iterdir():
            item.rmdir()
        # delete directory itself
        directory.rmdir()


def test_writes_multiple_read_records_as_flat_dataframe():
    # write as flat csv
    write_read_record(records, True, folder, flat_file_name)

    # read from csv
    df = read_df_from_csv(flat_file_path)
    # all three ids were written
    assert df.shape == (30, 3)
    pd.testing.assert_frame_equal(sub_df_for_id(df, id1), df1, check_dtype=False)
    pd.testing.assert_frame_equal(sub_df_for_id(df, id2), df2, check_dtype=False)
    pd.testing.assert_frame_equal(sub_df_for_id(df, id3), df3, check_dtype=False)


def test_writes_flat_dataframe_containing_only_columns_kept():
    # write csv files
    keep_columns = ['time']
    write_read_record(records, True, folder, flat_file_name, keep_cols=keep_columns)

    # read from written file
    df = read_df_from_csv(flat_file_path)

    assert set(df.columns) == {'id', 'time'}


def test_writes_csv_per_id():
    # write csv files
    write_read_record(records, False, per_id_folder, flat_file_name)

    # read
    filepaths = per_id_files()
    assert len(filepaths) == 3

    for record in records:
        files_for_id = [file for file in filepaths if record.zip_id in file]
        df = read_df_from_csv(files_for_id[0])
        pd.testing.assert_frame_equal(record.df_with_id(), df,
                                      check_dtype=False)


def test_writes_csv_per_id_only_containing_columns_kept():
    # write csv files
    keep_columns = ['time']
    write_read_record(records, False, per_id_folder, flat_file_name, keep_cols=keep_columns)

    # read from written files
    filepaths = per_id_files()

    for record in records:
        files_for_id = [file for file in filepaths if record.zip_id in file]
        df = read_df_from_csv(files_for_id[0])
        assert set(df.columns) == {'id', 'time'}

