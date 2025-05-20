import pytest
import pandas as pd
from pathlib import Path
from src.configurations import Configuration
import zipfile
from src.data_processing.read import (
    parse_standard_date,
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
from datetime import datetime, timezone, timedelta

@pytest.fixture
def input_file():
    # Define the common input file path
    return Path(__file__).parent / "test_data" / "0001.zip"

@pytest.mark.parametrize("treat_timezone, input_date, expected_output", [
    # --- localise: remove tz info, keep local time (naive) ---
    ('localise', "2023-10-01 12:00:00", pd.Timestamp("2023-10-01 12:00:00")),
    ('localise', "2023-10-01 12:00:00+0200", pd.Timestamp("2023-10-01 12:00:00")),
    ('localise', "2023-10-01 12:00:00 +0200", pd.Timestamp("2023-10-01 12:00:00")),
    ('localise', "2023-10-01 12:00:00.123456+0200", pd.Timestamp("2023-10-01 12:00:00.123456")),
    ('localise', "2023-10-01T12:00:00.123456Z", pd.Timestamp("2023-10-01 12:00:00.123456")),
    ('localise', "2023-10-01T12:00:00Z", pd.Timestamp("2023-10-01 12:00:00")),
    ('localise', "2023-10-01T12:00:00+0200", pd.Timestamp("2023-10-01 12:00:00")),
    ('localise', "2023-10-01T12:00:00.123456+0200", pd.Timestamp("2023-10-01 12:00:00.123456")),
    ('localise', "Sun Oct 01 12:00:00 UTC 2023", pd.Timestamp("2023-10-01 12:00:00")),
    ('localise', 1633046400, pd.Timestamp(1633046400, unit="s")),
    ('localise', 1633046400000, pd.Timestamp(1633046400000, unit="ms")),
    ('localise', "1633046400", pd.Timestamp(1633046400, unit="s")),
    ('localise', "1633046400000", pd.Timestamp(1633046400000, unit="ms")),
    ('localise', int(1e9), pd.Timestamp(int(1e9), unit="s")),

    # --- keep: keep tz-aware datetime ---
    ('keep', "2023-10-01 12:00:00", pd.Timestamp("2023-10-01 12:00:00")),
    ('keep', "2023-10-01 12:00:00+0200", pd.Timestamp("2023-10-01 12:00:00+02:00")),
    ('keep', "2023-10-01 12:00:00 +0200", pd.Timestamp("2023-10-01 12:00:00+02:00")),
    ('keep', "2023-10-01 12:00:00.123456+0200", pd.Timestamp("2023-10-01 12:00:00.123456+02:00")),
    ('keep', "2023-10-01T12:00:00.123456Z", pd.Timestamp("2023-10-01 12:00:00.123456+00:00")),
    ('keep', "2023-10-01T12:00:00Z", pd.Timestamp("2023-10-01 12:00:00+00:00")),
    ('keep', "2023-10-01T12:00:00+0200", pd.Timestamp("2023-10-01 12:00:00+02:00")),
    ('keep', "2023-10-01T12:00:00.123456+0200", pd.Timestamp("2023-10-01 12:00:00.123456+02:00")),
    ('keep', "Sun Oct 01 12:00:00 UTC 2023", pd.Timestamp("2023-10-01 12:00:00+00:00")),
    ('keep', "Invalid Date", pd.NaT),
    ('keep', 1633046400, pd.Timestamp(1633046400, unit="s")),
    ('keep', 1633046400000, pd.Timestamp(1633046400000, unit="ms")),
    ('keep', "1633046400", pd.Timestamp(1633046400, unit="s")),
    ('keep', "1633046400000", pd.Timestamp(1633046400000, unit="ms")),
    ('keep', int(1e9), pd.Timestamp(int(1e9), unit="s")),

    # --- utc: convert to UTC ---
    ('utc', "2023-10-01 12:00:00", pd.Timestamp("2023-10-01 12:00:00", tz="UTC")),
    ('utc', "2023-10-01 12:00:00+0200", pd.Timestamp("2023-10-01 10:00:00+00:00")),
    ('utc', "2023-10-01 12:00:00 +0200", pd.Timestamp("2023-10-01 10:00:00+00:00")),
    ('utc', "2023-10-01 12:00:00.123456+0200", pd.Timestamp("2023-10-01 10:00:00.123456+00:00")),
    ('utc', "2023-10-01T12:00:00.123456Z", pd.Timestamp("2023-10-01 12:00:00.123456+00:00")),
    ('utc', "2023-10-01T12:00:00Z", pd.Timestamp("2023-10-01 12:00:00+00:00")),
    ('utc', "2023-10-01T12:00:00+0200", pd.Timestamp("2023-10-01 10:00:00+00:00")),
    ('utc', "2023-10-01T12:00:00.123456+0200", pd.Timestamp("2023-10-01 10:00:00.123456+00:00")),
    ('utc', "Sun Oct 01 12:00:00 UTC 2023", pd.Timestamp("2023-10-01 12:00:00+00:00")),
    ('utc', "Invalid Date", pd.NaT),
    ('utc', 1633046400, pd.Timestamp(1633046400, unit="s", tz="UTC")),
    ('utc', 1633046400000, pd.Timestamp(1633046400000, unit="ms", tz="UTC")),
    ('utc', "1633046400", pd.Timestamp(1633046400, unit="s", tz="UTC")),
    ('utc', "1633046400000", pd.Timestamp(1633046400000, unit="ms", tz="UTC")),
    ('utc', int(1e9), pd.Timestamp(int(1e9), unit="s", tz="UTC")),
])
def test_parse_int_and_standard_date(treat_timezone, input_date, expected_output, monkeypatch):
    monkeypatch.setattr(Configuration, "treat_timezone", treat_timezone)
    result = parse_int_and_standard_date(treat_timezone, input_date)
    if expected_output is None or expected_output is pd.NaT:
        assert result is None or result is pd.NaT
    elif getattr(expected_output, 'tzinfo', None):
        # For tz-aware, check both value and tzinfo
        assert result == expected_output and result.tzinfo == expected_output.tzinfo
    else:
        assert result == expected_output

@pytest.mark.parametrize("input_date, expected_output", [
    ("2023-10-01 12:00:00 CEST", datetime(2023, 10, 1, 12, 0, tzinfo=timezone(timedelta(hours=2)))),  # CEST is UTC+2
    ("2023-10-01 12:00:00 EDT", datetime(2023, 10, 1, 12, 0, tzinfo=timezone(timedelta(hours=-4)))),  # EDT is UTC-4
    ("2023-10-01 12:00:00 EST", datetime(2023, 10, 1, 12, 0, tzinfo=timezone(timedelta(hours=-5)))),  # EST is UTC-5
    ("2023-10-01 12:00:00 CDT", datetime(2023, 10, 1, 12, 0, tzinfo=timezone(timedelta(hours=-5)))),  # CDT is UTC-5
    ("2023-10-01 12:00:00 UTC", datetime(2023, 10, 1, 12, 0, tzinfo=timezone.utc)),                   # UTC
    ("2023-10-01 12:00:00", datetime(2023, 10, 1, 12, 0)),                                            # naive datetime
])
def test_correct_odd_tz(input_date, expected_output):
    result = correct_odd_tz(input_date)
    # Convert both to pandas.Timestamp for comparison
    assert pd.Timestamp(result) == pd.Timestamp(expected_output)

@pytest.mark.parametrize("treat_timezone, input_date, expected_output",  [
    # Valid standard date formats
    ('localise', "2023-10-01 12:00:00+0200", datetime(2023, 10, 1, 12, 0, 0)),  # '%Y-%m-%d %H:%M:%S%z'

    # Valid odd timezone formats
    ('utc', "2023-10-01 12:00:00 CEST", datetime(2023, 10, 1, 10, 0, 0, tzinfo=timezone.utc)),

    # Valid integer timestamps
    ('keep', "1633046400", datetime(2021, 10, 1, 0, 0, 0)),  # Seconds timestamp

    # Invalid inputs
    ('keep', "Invalid Date", pd.NaT),  # Invalid date string
    ('keep', None, pd.NaT),  # None input
    ('keep', float("nan"), pd.NaT),  # NaN input
    ('keep', "", pd.NaT),  # Empty string
])
def test_parse_date_string(treat_timezone, input_date, expected_output, monkeypatch):
    monkeypatch.setattr(Configuration, "treat_timezone", treat_timezone)
    result = parse_date_string(treat_timezone, input_date)
    if expected_output is None or expected_output is pd.NaT:
        assert result is None or result is pd.NaT
    elif getattr(expected_output, 'tzinfo', None):
        # For tz-aware, check both value and tzinfo
        assert result == expected_output and result.tzinfo == expected_output.tzinfo
    else:
        assert result == expected_output

@pytest.mark.parametrize("treat_timezone, input_series, expected", [
    # Mixed formats: ISO, with/without tz, and invalid
    (
        'keep',
        pd.Series(["2023-10-01 12:00:00+0200", "2023-10-01T12:00:00Z", "Invalid"]),
        pd.Series([
            datetime(2023, 10, 1, 12, 0, 0, tzinfo=timezone(timedelta(hours=2))),
            datetime(2023, 10, 1, 12, 0, 0, tzinfo=timezone.utc),
            pd.NaT
        ])
    ),
    (
        'localise',
        pd.Series(["2023-10-01 12:00:00+0200", "2023-10-01T12:00:00Z", "Invalid"]),
        pd.Series([
            datetime(2023, 10, 1, 12, 0, 0),
            datetime(2023, 10, 1, 12, 0, 0),
            pd.NaT
        ])
    ),
    (
        'utc',
        pd.Series(["2023-10-01 12:00:00+0200", "2023-10-01T12:00:00Z", "Invalid"]),
        pd.Series([
            datetime(2023, 10, 1, 10, 0, 0, tzinfo=timezone.utc),
            datetime(2023, 10, 1, 12, 0, 0, tzinfo=timezone.utc),
            pd.NaT
        ])
    ),
])
def test_parse_date_columns_series_treat_timezone(monkeypatch, treat_timezone, input_series, expected):
    monkeypatch.setattr(Configuration, "treat_timezone", treat_timezone)
    result = parse_date_columns(treat_timezone, input_series)
    pd.testing.assert_series_equal(result, expected, check_dtype=False)

@pytest.mark.parametrize("treat_timezone, input_df, expected", [
    (
        'keep',
        pd.DataFrame({
            "a": ["2023-10-01 12:00:00+0200", "Invalid"],
            "b": ["2023-10-01T12:00:00Z", "2023-10-01 12:00:00"]
        }),
        pd.DataFrame({
            "a": [
                datetime(2023, 10, 1, 12, 0, 0, tzinfo=timezone(timedelta(hours=2))),
                pd.NaT
            ],
            "b": [
                datetime(2023, 10, 1, 12, 0, 0, tzinfo=timezone.utc),
                datetime(2023, 10, 1, 12, 0, 0, tzinfo=timezone.utc)
            ]
        })
    ),
    (
        'localise',
        pd.DataFrame({
            "a": ["2023-10-01 12:00:00+0200", "Invalid"],
            "b": ["2023-10-01T12:00:00Z", "2023-10-01 12:00:00"]
        }),
        pd.DataFrame({
            "a": [
                datetime(2023, 10, 1, 12, 0, 0),
                pd.NaT
            ],
            "b": [
                datetime(2023, 10, 1, 12, 0, 0),
                datetime(2023, 10, 1, 12, 0, 0)
            ]
        })
    ),
    (
        'utc',
        pd.DataFrame({
            "a": ["2023-10-01 12:00:00+0200", "Invalid"],
            "b": ["2023-10-01T12:00:00Z", "2023-10-01 12:00:00"]
        }),
        pd.DataFrame({
            "a": [
                datetime(2023, 10, 1, 10, 0, 0, tzinfo=timezone.utc),
                pd.NaT
            ],
            "b": [
                datetime(2023, 10, 1, 12, 0, 0, tzinfo=timezone.utc),
                datetime(2023, 10, 1, 12, 0, 0, tzinfo=timezone.utc)
            ]
        })
    ),
])
def test_parse_date_columns_dataframe_mixed(monkeypatch, treat_timezone, input_df, expected):
    monkeypatch.setattr(Configuration, "treat_timezone", treat_timezone)
    result = parse_date_columns(treat_timezone, input_df)
    pd.testing.assert_frame_equal(result, expected, check_dtype=False)

@pytest.mark.parametrize("treat_timezone, input_df, expected", [
    (
        'keep',
        pd.DataFrame({"a": ["2023-10-01 12:00:00+0200", "2023-10-02 12:00:00+0200"]}),
        pd.DataFrame({"a": [
            datetime(2023, 10, 1, 12, 0, 0, tzinfo=timezone(timedelta(hours=2))),
            datetime(2023, 10, 2, 12, 0, 0, tzinfo=timezone(timedelta(hours=2)))
        ]})
    ),
    (
        'localise',
        pd.DataFrame({"a": ["2023-10-01 12:00:00+0200", "2023-10-02 12:00:00+0200"]}),
        pd.DataFrame({"a": [datetime(2023, 10, 1, 12, 0, 0),
                            datetime(2023, 10, 2, 12, 0, 0)]
        })
    ),
    (
        'utc',
        pd.DataFrame({"a": ["2023-10-01 12:00:00+0200", "2023-10-02 12:00:00+0200"]}),
        pd.DataFrame({"a": pd.to_datetime(["2023-10-01 12:00:00+0200",
                                           "2023-10-02 12:00:00+0200"]).tz_convert('UTC')
        })
    ),
])
def test_parse_date_columns_dataframe_uniform(monkeypatch, treat_timezone, input_df, expected):
    monkeypatch.setattr(Configuration, "treat_timezone", treat_timezone)
    result = parse_date_columns(treat_timezone, input_df)
    pd.testing.assert_frame_equal(result, expected)


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
    config.csv_extension = ".csv"  # Adjust based on your file extension

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

    @pytest.fixture
    def mock_device_status_file():
        # Mock CSV content
        csv_content = """created_at,pump/clock
    2023-10-01 12:00:00+0200,2023-10-01 12:00:00+0200
    2023-10-01T12:00:00Z,2023-10-01T12:00:00Z
    """
        return StringIO(csv_content)

    @pytest.fixture
    def mock_config():
        return TestConfiguration()

    def test_read_device_status_file_into_df_treat_timezone(
            mock_device_status_file, mock_config):
        # Test with treat_timezone='localise'
        mock_config.treat_timezone = 'localise'
        df = read_device_status_file_into_df(mock_device_status_file,
                                             mock_config)
        assert df["created_at"].iloc[0] == datetime(2023, 10, 1, 12, 0, 0)
        assert df["pump/clock"].iloc[1] == datetime(2023, 10, 1, 12, 0, 0)

        # Test with treat_timezone='utc'
        mock_device_status_file.seek(0)  # Reset file pointer
        mock_config.treat_timezone = 'utc'
        df = read_device_status_file_into_df(mock_device_status_file,
                                             mock_config)
        assert df["created_at"].iloc[0] == datetime(2023, 10, 1, 10, 0, 0)
        assert df["pump/clock"].iloc[1] == datetime(2023, 10, 1, 12, 0, 0)

    def test_parse_date_columns_treat_timezone(mock_config):
        # Mock DataFrame with datetime columns
        data = {
            "created_at": ["2023-10-01 12:00:00+0200", "2023-10-01T12:00:00Z"],
            "pump/clock": ["2023-10-01 12:00:00+0200", "2023-10-01T12:00:00Z"]
        }
        df = pd.DataFrame(data)

        # Test with treat_timezone='localise'
        mock_config.treat_timezone = 'localise'
        parsed_df = parse_date_columns(mock_config.treat_timezone, df)
        assert parsed_df["created_at"].iloc[0] == datetime(2023, 10, 1, 12, 0,
                                                           0)
        assert parsed_df["pump/clock"].iloc[1] == datetime(2023, 10, 1, 12, 0,
                                                           0)

        # Test with treat_timezone='utc'
        mock_config.treat_timezone = 'utc'
        parsed_df = parse_date_columns(mock_config.treat_timezone, df)
        assert parsed_df["created_at"].iloc[0] == datetime(2023, 10, 1, 10, 0,
                                                           0)
        assert parsed_df["pump/clock"].iloc[1] == datetime(2023, 10, 1, 12, 0,
                                                           0)

        @pytest.mark.parametrize(
            "treat_timezone, input_date, expected_output", [
                # Test cases for treat_timezone='localise' (timezones removed)
                ('localise', "2023-10-01 12:00:00+0200",
                 datetime(2023, 10, 1, 12, 0, 0)),
                ('localise', "2023-10-01T12:00:00Z", datetime(2023, 10, 1, 12, 0, 0)),
                ('localise', "2023-10-01T12:00:00+0200",
                 datetime(2023, 10, 1, 12, 0, 0)),

                # Test cases for treat_timezone='utc' (timezones converted to UTC)
                ('utc', "2023-10-01 12:00:00+0200",
                 datetime(2023, 10, 1, 10, 0, 0)),
                (
                'utc', "2023-10-01T12:00:00Z", datetime(2023, 10, 1, 12, 0, 0)),
                ('utc', "2023-10-01T12:00:00+0200",
                 datetime(2023, 10, 1, 10, 0, 0)),
            ])
        def test_treat_timezone_flag(treat_timezone, input_date,
                                        expected_output):
            # Setup configuration
            config = Configuration()
            config.treat_timezone = treat_timezone

            # Mock the Configuration class to return the desired flag
            def mock_treat_timezone():
                return config.treat_timezone

            # Replace the Configuration method with the mock
            Configuration.treat_timezone = property(
                mock_treat_timezone)

            # Call the function
            result = parse_standard_date(config.treat_timezone, input_date)

            # Assert the result matches the expected output
            assert result == expected_output
