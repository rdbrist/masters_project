import pytest
import pandas as pd
from pathlib import Path
from src.configurations import Configuration
import zipfile
from src.data_processing.read import (
    parse_standard_date,
    parse_int_date,
    parse_int_and_standard_date,
    correct_odd_tz,
    parse_date_string,
    parse_date_columns,
    read_zip_file,
    is_a_bg_csv_file,
    is_a_device_status_csv_file,
    read_entries_file_into_df,
    ReadRecord,
    read_bg_from_zip,
    read_all,
    read_all_device_status,
    read_all_bg,
    read_device_status_file_into_df,
    read_device_status_from_zip
)
from datetime import datetime

@pytest.fixture
def input_file():
    # Define the common input file path
    return Path(__file__).parent / "test_data" / "0001.zip"

@pytest.mark.parametrize("input_date, expected_output", [
    (1633046400, datetime(2021, 10, 1, 0, 0, 0)),  # Valid seconds timestamp
    (1633046400000, datetime(2021, 10, 1, 0, 0, 0)),  # Valid milliseconds timestamp
    ("1633046400", datetime(2021, 10, 1, 0, 0, 0)),  # Valid seconds as string
    ("1633046400000", datetime(2021, 10, 1, 0, 0, 0)),  # Valid milliseconds as string
    (int(1e9), datetime(2001, 9, 9, 1, 46, 40)),  # Large valid seconds timestamp
    ("invalid", pd.NaT),  # Invalid string input
    (-1, pd.NaT),  # Negative timestamp
    (None, pd.NaT),  # None input
    ("", pd.NaT),  # Empty string
])
def test_parse_int_date(input_date, expected_output):
    result = parse_int_date(input_date)
    if expected_output is pd.NaT:
        assert result is pd.NaT
    else:
        assert result == expected_output

@pytest.mark.parametrize("input_date, expected_output", [
    ("2023-10-01 12:00:00", datetime(2023, 10, 1, 12, 0, 0)),  # '%Y-%m-%d %H:%M:%S'
    ("2023-10-01 12:00:00+0200", datetime(2023, 10, 1, 12, 0, 0)),  # '%Y-%m-%d %H:%M:%S%z'
    ("2023-10-01 12:00:00 +0200", datetime(2023, 10, 1, 12, 0, 0)),  # '%Y-%m-%d %H:%M:%S %z'
    ("2023-10-01 12:00:00.123456+0200", datetime(2023, 10, 1, 12, 0, 0, 123456)),  # '%Y-%m-%d %H:%M:%S.%f%z'
    ("2023-10-01T12:00:00.123456Z", datetime(2023, 10, 1, 12, 0, 0, 123456)),  # '%Y-%m-%dT%H:%M:%S.%fZ'
    ("2023-10-01T12:00:00Z", datetime(2023, 10, 1, 12, 0, 0)),  # '%Y-%m-%dT%H:%M:%SZ'
    ("2023-10-01T12:00:00+0200", datetime(2023, 10, 1, 12, 0, 0)),  # '%Y-%m-%dT%H:%M:%S%z'
    ("2023-10-01T12:00:00.123456+0200", datetime(2023, 10, 1, 12, 0, 0, 123456)),  # '%Y-%m-%dT%H:%M:%S.%f%z'
    ("Sun Oct 01 12:00:00 UTC 2023", datetime(2023, 10, 1, 12, 0, 0)),  # '%a %b %d %H:%M:%S %Z %Y'
    ("Invalid Date", None),  # Invalid format
])
def test_parse_standard_date(input_date, expected_output):
    result = parse_standard_date(input_date)
    if expected_output is None:
        assert result is None or result is pd.NaT
    else:
        assert result == expected_output

@pytest.mark.parametrize("input_date, expected_output", [
    # Valid standard date formats
    ("2023-10-01 12:00:00+0200", datetime(2023, 10, 1, 12, 0, 0)),  # '%Y-%m-%d %H:%M:%S%z'
    ("2023-10-01T12:00:00Z", datetime(2023, 10, 1, 12, 0, 0)),  # '%Y-%m-%dT%H:%M:%SZ'
    ("Sun Oct 01 12:00:00 UTC 2023", datetime(2023, 10, 1, 12, 0, 0)),  # '%a %b %d %H:%M:%S %Z %Y'

    # Valid integer timestamps
    (1633046400, datetime(2021, 10, 1, 0, 0, 0)),  # Seconds timestamp
    ("1633046400000", datetime(2021, 10, 1, 0, 0, 0)),  # Milliseconds timestamp as string

    # Invalid inputs
    ("Invalid Date", pd.NaT),  # Invalid date string
    (-1, pd.NaT),  # Negative timestamp
    (None, pd.NaT),  # None input
    ("", pd.NaT),  # Empty string
])
def test_parse_int_and_standard_date(input_date, expected_output):
    result = parse_int_and_standard_date(input_date)
    if expected_output is pd.NaT:
        assert result is pd.NaT
    else:
        assert result == expected_output

@pytest.mark.parametrize("input_date, expected_output", [
    ("2023-10-01 12:00:00 CEST", datetime(2023, 10, 1, 12, 0)),  # Test for 'CEST'
    ("2023-10-01 12:00:00 EDT", datetime(2023, 10, 1, 12, 0)),  # Test for 'EDT'
    ("2023-10-01 12:00:00 EST", datetime(2023, 10, 1, 12, 0)),   # Test for 'EST'
    ("2023-10-01 12:00:00 CDT", datetime(2023, 10, 1, 12, 0)),   # Test for 'CDT'
    ("2023-10-01 12:00:00 UTC", datetime(2023, 10, 1, 12, 0)),     # Test for no translation needed
    ("2023-10-01 12:00:00", datetime(2023, 10, 1, 12, 0)),             # Test for no timezone
])
def test_correct_odd_tz(input_date, expected_output):
    assert correct_odd_tz(input_date) == expected_output

@pytest.mark.parametrize("input_date, expected_output", [
    # Valid standard date formats
    ("2023-10-01 12:00:00+0200", datetime(2023, 10, 1, 12, 0, 0)),  # '%Y-%m-%d %H:%M:%S%z'
    ("2023-10-01T12:00:00Z", datetime(2023, 10, 1, 12, 0, 0)),  # '%Y-%m-%dT%H:%M:%SZ'
    ("Sun Oct 01 12:00:00 UTC 2023", datetime(2023, 10, 1, 12, 0, 0)),  # '%a %b %d %H:%M:%S %Z %Y'

    # Valid odd timezone formats
    ("2023-10-01 12:00:00 CEST", datetime(2023, 10, 1, 12, 0, 0)),  # Corrected to '+0200'

    # Valid integer timestamps
    ("1633046400", datetime(2021, 10, 1, 0, 0, 0)),  # Seconds timestamp

    # Invalid inputs
    ("Invalid Date", pd.NaT),  # Invalid date string
    (None, pd.NaT),  # None input
    (float("nan"), pd.NaT),  # NaN input
    ("", pd.NaT),  # Empty string
])
def test_parse_date_string(input_date, expected_output):
    result = parse_date_string(input_date)
    if pd.isna(expected_output):
        assert pd.isna(result)
    else:
        assert result == expected_output


@pytest.mark.parametrize("input_data, expected_output", [
    # Test with a pandas Series
    (pd.Series(["2023-10-01 12:00:00+0200", "2023-10-01T12:00:00Z", "Invalid Date"]),
     pd.Series([datetime(2023, 10, 1, 12, 0, 0), datetime(2023, 10, 1, 12, 0, 0), pd.NaT])),

    # Test with a pandas DataFrame
    (pd.DataFrame({"col1": ["2023-10-01 12:00:00+0200", "Invalid Date"],
                   "col2": ["2023-10-01T12:00:00Z", None]}),
     pd.DataFrame({"col1": [datetime(2023, 10, 1, 12, 0, 0), pd.NaT],
                   "col2": [datetime(2023, 10, 1, 12, 0, 0), pd.NaT]})),
])
def test_parse_date_columns(input_data, expected_output):
    result = parse_date_columns(input_data)
    if isinstance(expected_output, pd.Series):
        pd.testing.assert_series_equal(result, expected_output)
    elif isinstance(expected_output, pd.DataFrame):
        pd.testing.assert_frame_equal(result, expected_output)


def test_read_zip_file(input_file):
    # Setup
    config = Configuration()  # Ensure this is properly initialized for the test
    read_record = read_zip_file(config, input_file, is_a_bg_csv_file, read_entries_file_into_df)

    # Assertions
    assert read_record.zip_id == "0001"
    assert read_record.number_of_entries_files > 0  # Ensure files were found
    assert read_record.df is not None  # Ensure DataFrame is populated
    assert not read_record.has_no_files  # Ensure files were read successfully
    assert read_record.number_of_rows > 0  # Ensure rows were read

def test_read_entries_file_into_df(input_file):
    # Setup
    config = Configuration()  # Ensure this is properly initialized for the test
    read_record = ReadRecord()
    read_record.zip_id = input_file.stem

    with zipfile.ZipFile(input_file, mode="r") as archive:
        # Identify an appropriate file using is_a_bg_csv_file
        bg_file = next(
            (f for f in archive.namelist() if is_a_bg_csv_file(config, read_record.zip_id, f)),
            None
        )
        assert bg_file is not None, "No valid BG CSV file found in the zip archive"

        # Call the function
        read_entries_file_into_df(archive, bg_file, read_record, config)

    # Assertions
    assert read_record.df is not None, "DataFrame should not be None"
    assert not read_record.df.empty, "DataFrame should not be empty"
    assert "time" in read_record.df.columns, "DataFrame should contain 'time' column"
    assert "bg" in read_record.df.columns, "DataFrame should contain 'bg' column"
    assert pd.api.types.is_datetime64_any_dtype(read_record.df["time"]), "'time' column should be datetime"
    assert pd.api.types.is_float_dtype(read_record.df["bg"]), "'bg' column should be float"

def test_read_bg_from_zip(input_file):
    # Setup
    config = Configuration()  # Ensure this is properly initialized for the test

    # Call the function
    read_record = read_bg_from_zip(input_file, config)

    # Assertions
    assert isinstance(read_record, ReadRecord), "The result should be a ReadRecord instance"
    assert read_record.zip_id == input_file.stem, "The zip_id should match the file name"
    assert read_record.number_of_entries_files > 0, "There should be at least one entries file"
    assert read_record.df is not None, "The DataFrame should not be None"
    assert not read_record.df.empty, "The DataFrame should not be empty"
    assert "time" in read_record.df.columns, "The DataFrame should contain a 'time' column"
    assert "bg" in read_record.df.columns, "The DataFrame should contain a 'bg' column"
    assert read_record.number_of_rows > 0, "The number of rows should be greater than 0"

def assert_from_bg_records(records):
    assert isinstance(records, list), "The result should be a list of ReadRecord instances"
    assert len(records) > 0, "The list of records should not be empty"

    for record in records:
        assert isinstance(record, ReadRecord), "Each item in the list should be a ReadRecord instance"
        assert record.zip_id is not None, "Each record should have a zip_id"
        assert record.df is not None, "Each record should have a DataFrame"
        assert not record.df.empty, "The DataFrame should not be empty"
        assert "time" in record.df.columns, "The DataFrame should contain a 'time' column"
        assert "bg" in record.df.columns, "The DataFrame should contain a 'bg' column"
        assert record.number_of_rows > 0, "The number of rows should be greater than 0"

def assert_from_device_status_records(records):
    assert isinstance(records, list), "The result should be a list of ReadRecord instances"
    assert len(records) > 0, "The list of records should not be empty"

    for record in records:
        assert isinstance(record, ReadRecord), "Each item in the list should be a ReadRecord instance"
        assert record.zip_id is not None, "Each record should have a zip_id"
        assert record.df is not None, "Each record should have a DataFrame"
        assert not record.df.empty, "The DataFrame should not be empty"
        assert "created_at" in record.df.columns, "The DataFrame should contain a 'created_at' column"
        assert record.number_of_rows > 0, "The number of rows should be greater than 0"

def test_read_all(input_file):
    # Setup
    config = Configuration()  # Ensure this is properly initialized for the test
    config.data_dir = str(input_file.parent)  # Set the data directory

    # Use read_bg_from_zip as the function to read from the zip file
    records = read_all(config, read_bg_from_zip)
    assert_from_bg_records(records)

    # Use read_device_status_from_zip as the function to read device status
    records = read_all(config, read_device_status_from_zip)
    assert_from_device_status_records(records)

def test_read_all_device_status(input_file):
    # Setup
    config = Configuration()  # Ensure this is properly initialized for the test
    config.data_dir = str(input_file.parent)  # Set the data directory
    config.device_status_csv_file_start = ""  # Adjust based on your file naming convention
    config.device_status_csv_file_extension = ".csv"  # Adjust based on your file extension

    # Call the function
    records = read_all_device_status(config)

    # Assertions
    assert_from_device_status_records(records)

def test_read_all_bg(input_file):
    # Setup
    config = Configuration()  # Ensure this is properly initialized for the test
    config.data_dir = str(input_file.parent)  # Set the data directory

    # Call the function
    records = read_all_bg(config)

    # Assertions
    assert_from_bg_records(records)

def test_read_device_status_file_into_df(input_file):
    # Setup
    config = Configuration()  # Ensure this is properly initialized for the test
    read_record = ReadRecord()
    read_record.zip_id = input_file.stem

    with zipfile.ZipFile(input_file, mode="r") as archive:
        # Identify an appropriate file using is_a_device_status_csv_file
        device_status_file = next(
            (f for f in archive.namelist() if is_a_device_status_csv_file(config, read_record.zip_id, f)),
            None
        )
        assert device_status_file is not None, "No valid device status CSV file found in the zip archive"

        # Call the function
        read_device_status_file_into_df(archive, device_status_file, read_record, config)

    # Assertions
    assert read_record.df is not None, "DataFrame should not be None"
    assert not read_record.df.empty, "DataFrame should not be empty"
    assert "created_at" in read_record.df.columns, "DataFrame should contain 'created_at' column"
    assert pd.api.types.is_datetime64_any_dtype(read_record.df["created_at"]), "'created_at' column should be datetime"