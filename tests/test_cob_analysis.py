import pytest
import pandas as pd
from src.cob_analysis import Cob

@pytest.fixture
def cob_instance():
    return Cob()

@pytest.fixture
def mock_data_dir(tmp_path):
    # Create a temporary directory for mock data
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir

def test_check_consecutive_intervals(cob_instance):

    # Scenario 1: Valid intervals (15 minutes)
    datetimes = pd.Series(pd.date_range("2023-01-01 00:00",
                                        periods=5,
                                        freq="15min"))
    assert cob_instance._check_consecutive_intervals(datetimes, 15)

    # Scenario 2: Invalid intervals (not consistent)
    datetimes = (
        pd.Series(["2023-01-01 00:00", "2023-01-01 00:15", "2023-01-01 00:45"]))
    datetimes = pd.to_datetime(datetimes)
    assert not cob_instance._check_consecutive_intervals(datetimes, 15)

    # Scenario 3: Empty series
    datetimes = pd.Series([], dtype="datetime64[ns]")
    assert cob_instance._check_consecutive_intervals(datetimes, 15)

    # Scenario 4: Single datetime (no intervals to check)
    datetimes = pd.Series(["2023-01-01 00:00"], dtype="datetime64[ns]")
    assert cob_instance._check_consecutive_intervals(datetimes, 15)

    # Scenario 5: Multiple people with overlapping datetime ranges
    df = pd.DataFrame({
        "id": [1, 1, 1, 2, 2, 2],
        "datetime": [
            "2023-01-01 00:00", "2023-01-01 00:15", "2023-01-01 00:30",
            "2023-01-01 00:00", "2023-01-01 00:20", "2023-01-01 00:40"
        ]
    })
    df["datetime"] = pd.to_datetime(df["datetime"])

    # Check intervals for each person
    for person_id, group in df.groupby("id"):
        if person_id == 1:
            assert cob_instance._check_consecutive_intervals(group["datetime"], 15)
        elif person_id == 2:
            assert not cob_instance._check_consecutive_intervals(group["datetime"], 15)


def test_validate_sampling_rate(cob_instance):
    # Scenario 1: Valid sampling rate (15 minutes)
    data = {
        "id": [1, 1, 1],
        "datetime": ["2023-01-01 00:00", "2023-01-01 00:15", "2023-01-01 00:30"]
    }
    df = pd.DataFrame(data)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index(["id", "datetime"], inplace=True)
    cob_instance.dataset = df
    cob_instance.sampling_rate = 15
    assert cob_instance._validate_sampling_rate()

    # Scenario 2: Invalid sampling rate (minute values not divisible by 15)
    data = {
        "id": [1, 1, 1],
        "datetime": ["2023-01-01 00:01", "2023-01-01 00:16", "2023-01-01 00:31"]
    }
    df = pd.DataFrame(data)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index(["id", "datetime"], inplace=True)
    cob_instance.dataset = df
    cob_instance.sampling_rate = 15
    assert not cob_instance._validate_sampling_rate()

    # Scenario 3: Sampling rate not set
    cob_instance.sampling_rate = None
    assert not cob_instance._validate_sampling_rate()

    # Scenario 4: Sampling rate is not an integer
    cob_instance.sampling_rate = "15"
    assert not cob_instance._validate_sampling_rate()

    # Scenario 5: Multiple IDs with valid and invalid minute factors
    data = {
        "id": [1, 1, 1, 2, 2, 2],
        "datetime": [
            "2023-01-01 00:00", "2023-01-01 00:15", "2023-01-01 00:30",
            "2023-01-01 00:01", "2023-01-01 00:16", "2023-01-01 00:31"
        ]
    }
    df = pd.DataFrame(data)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index(["id", "datetime"], inplace=True)
    cob_instance.dataset = df
    cob_instance.sampling_rate = 15
    assert not cob_instance._validate_sampling_rate()

def test_read_interim_data_valid_csv(cob_instance, mock_data_dir):
    # Create a valid CSV file with consistent intervals
    file_path = mock_data_dir / "valid_data.csv"
    data = {
        "id": [1, 1, 1],
        "datetime": ["2023-01-01 00:00", "2023-01-01 00:15", "2023-01-01 00:30"],
        "system": ["A", "A", "A"]
    }
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)

    # Set the data file path
    cob_instance.data_file_path = mock_data_dir

    # Call the method
    cob_instance.read_interim_data(file_name="valid_data", file_type="csv", sampling_rate=15)

    # Assertions
    assert cob_instance.dataset is not None
    assert not cob_instance.dataset.empty
    assert cob_instance.sampling_rate == 15

    # Verify that the sampling rate matches the datetime intervals
    for p_id in cob_instance.dataset.index.get_level_values('id').drop_duplicates():
        datetimes = cob_instance.dataset.loc[p_id].reset_index()['datetime'].sort_values()
        assert Cob._check_consecutive_intervals(datetimes, 15)
        assert Cob._check_minute_factor(datetimes, 15)

def test_read_interim_data_invalid_sampling_rate(cob_instance, mock_data_dir):
    # Create a CSV file with inconsistent minute factors
    file_path = mock_data_dir / "invalid_sampling.csv"
    data = {
        "id": [1, 1, 1],
        "datetime": ["2023-01-01 00:01", "2023-01-01 00:16", "2023-01-01 00:31"],
        "system": ["A", "A", "A"]
    }
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)

    # Set the data file path
    cob_instance.data_file_path = mock_data_dir

    # Call the method and expect a ValueError
    with pytest.raises(ValueError, match="Sampling rate 15 does not match the minute values for ID 1."):
        cob_instance.read_interim_data(file_name="invalid_sampling",
                                       file_type="csv",
                                       sampling_rate=15)

def test_read_interim_data_file_not_found(cob_instance, mock_data_dir):
    # Set a non-existent file path
    non_existent_file = mock_data_dir / "non_existent_file.csv"
    cob_instance.data_file_path = mock_data_dir

    # Ensure the file does not exist
    assert not non_existent_file.exists()

    # Call the method and expect a FileNotFoundError
    with pytest.raises(FileNotFoundError, match="File not found"):
        cob_instance.read_interim_data(file_name="non_existent_file",
                                       file_type="csv",
                                       sampling_rate=15)

def test_read_interim_data_valid_parquet(cob_instance, mock_data_dir):
    # Create a valid Parquet file
    file_path = mock_data_dir / "valid_data.parquet"
    data = {
        "id": [1, 1, 1],
        "datetime": ["2023-01-01 00:00", "2023-01-01 00:15", "2023-01-01 00:30"],
        "system": ["A", "A", "A"]
    }
    df = pd.DataFrame(data)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index(["id", "datetime"], inplace=True)  # Set multi-level index
    df.to_parquet(file_path)

    # Ensure the file exists
    assert file_path.exists()

    # Set the data file path
    cob_instance.data_file_path = mock_data_dir

    # Call the method
    cob_instance.read_interim_data(file_name="valid_data", file_type="parquet", sampling_rate=15)

    # Assertions
    assert cob_instance.dataset is not None
    assert not cob_instance.dataset.empty
    assert cob_instance.sampling_rate == 15