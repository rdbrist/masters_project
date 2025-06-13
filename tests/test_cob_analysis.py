import pytest
import pandas as pd
from src.cob_analysis import Cob

@pytest.fixture
def mock_data_dir(tmp_path):
    # Create a temporary directory for mock data
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir

@pytest.fixture
def mock_dataset():
    # Create a mock dataset with a multi-level index
    data = {
        'id': [1, 1, 2, 2, 3, 3],
        'datetime': pd.date_range(start='2023-01-01', periods=6, freq='15min'),
        'system': ['A','A','A','A','A','A'],
        'cob max': [10, 20, 30, 40, 50, 60],
    }
    df = pd.DataFrame(data)
    df.set_index(['id', 'datetime'], inplace=True)
    return df

@pytest.fixture
def cob_instance(mock_dataset):
    cob = Cob(mock_dataset)
    return cob

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
    with pytest.raises(ValueError,
                       match='Sampling rate 15 does not match the minute values '
                             'for ID 1.'):
        cob_instance._validate_sampling_rate()

    # Scenario 3: Sampling rate not set
    cob_instance.sampling_rate = None
    assert not cob_instance._validate_sampling_rate()

    # Scenario 4: Sampling rate is not an integer
    cob_instance.sampling_rate = "15"
    assert not cob_instance._validate_sampling_rate()

@pytest.mark.parametrize("id, datetimes, expected_error", [
    (1, ["2023-01-01 00:01", "2023-01-01 00:16", "2023-01-01 00:31"],
     "Sampling rate 15 does not match the minute values for ID 1."),
    (2, ["2023-01-01 00:02", "2023-01-01 00:17", "2023-01-01 00:32"],
     "Sampling rate 15 does not match the minute values for ID 2."),
])
def test_validate_sampling_rate_with_dynamic_id(cob_instance, id, datetimes,
                                                expected_error):
    # Prepare the dataset
    data = {"id": [id] * len(datetimes), "datetime": datetimes}
    df = pd.DataFrame(data)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index(["id", "datetime"], inplace=True)
    cob_instance.dataset = df
    cob_instance.sampling_rate = 15

    # Assert the ValueError with the dynamic ID in the message
    with pytest.raises(ValueError, match=expected_error):
        cob_instance._validate_sampling_rate()

def test_read_interim_data_valid_csv(cob_instance, mock_data_dir):
    # Create a valid CSV file with consistent intervals
    file_path = mock_data_dir / "valid_data.csv"
    data = {
        "id": [1, 1, 1],
        "datetime": ["2023-01-01 00:00",
                     "2023-01-01 00:15",
                     "2023-01-01 00:30"],
        "system": ["A", "A", "A"]
    }
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)

    # Set the data file path
    cob_instance.data_file_path = mock_data_dir

    # Call the method
    cob_instance.read_interim_data(file_name="valid_data",
                                   file_type="csv",
                                   sampling_rate=15)

    # Assertions
    assert cob_instance.dataset is not None
    assert not cob_instance.dataset.empty
    assert cob_instance.sampling_rate == 15

    # Verify that the sampling rate matches the datetime intervals
    for p_id in (cob_instance.
            dataset.
            index.
            get_level_values('id').
            drop_duplicates()):
        datetimes = (cob_instance.dataset.loc[p_id].
                     reset_index()['datetime'].
                     sort_values())
        assert Cob._check_consecutive_intervals(datetimes, 15)
        assert Cob._check_minute_factor(datetimes, 15)

def test_read_interim_data_invalid_sampling_rate(cob_instance, mock_data_dir):
    # Create a CSV file with inconsistent minute factors
    file_path = mock_data_dir / "invalid_sampling.csv"
    data = {
        "id": [1, 1, 1],
        "datetime": ["2023-01-01 00:01",
                     "2023-01-01 00:16",
                     "2023-01-01 00:31"],
        "system": ["A", "A", "A"]
    }
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)

    # Set the data file path
    cob_instance.data_file_path = mock_data_dir

    # Call the method and expect a ValueError
    with pytest.raises(ValueError, match="Sampling rate 15 does not match the "
                                         "minute values for ID 1."):
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
    # data = {
    #     "id": [1, 1, 1],
    #     "datetime": ["2023-01-01 00:00",
    #                  "2023-01-01 00:15",
    #                  "2023-01-01 00:30"],
    #     "system": ["A", "A", "A"]
    # }
    # df = pd.DataFrame(data)
    # df["datetime"] = pd.to_datetime(df["datetime"])
    # df.set_index(["id", "datetime"], inplace=True)  # Set multilevel index
    cob_instance.dataset.to_parquet(file_path)

    # Ensure the file exists
    assert file_path.exists()

    # Set the data file path
    cob_instance.data_file_path = mock_data_dir

    # Call the method
    cob_instance.read_interim_data(file_name="valid_data",
                                   file_type="parquet",
                                   sampling_rate=15)

    # Assertions
    assert cob_instance.dataset is not None
    assert not cob_instance.dataset.empty
    assert cob_instance.sampling_rate == 15

def test_get_person_data_valid_id(cob_instance):
    # Test with a valid ID
    person_data = cob_instance.get_person_data(1)
    assert not person_data.empty
    assert len(person_data) == 2
    cob_instance.dataset = cob_instance.dataset.drop(columns=['cob max'])
    with pytest.raises(ValueError, match='Column "cob max" not found in '
                                         'dataset.'):
        cob_instance.get_person_data(1)

def test_get_person_data_invalid_id(cob_instance):
    # Test with an invalid ID
    with pytest.raises(KeyError, match="Individual 999 not found in dataset"):
        cob_instance.get_person_data(999)
    with pytest.raises(KeyError, match="xxx"):
        cob_instance.get_person_data("xxx")

@pytest.fixture
def cob_with_data(tmp_path):
    # Create a minimal dataset with 15-min intervals for one id
    dt_rng = pd.date_range("2024-01-01", periods=4, freq="15min")
    df = pd.DataFrame({
        "id": [1]*4,
        "datetime": dt_rng,
        "cob max": [1, 2, 3, 4],
        "system": ["A"]*4
    }).set_index(["id", "datetime"])
    # Save to a temporary parquet file
    data_file = tmp_path / "test.parquet"
    df.to_parquet(data_file)
    cob = Cob(df)
    return cob

def test_process_one_tz_individuals_frequency(cob_with_data):
    cob = cob_with_data
    profile_offsets = pd.DataFrame({"offset": [0]}, index=[1])
    args = {"height": 1, "distance": 1}
    result = cob.process_one_tz_individuals(profile_offsets, args)
    # Check frequency for each id
    for id_val in result.index.get_level_values("id").unique():
        dt_index = result.loc[id_val].index

