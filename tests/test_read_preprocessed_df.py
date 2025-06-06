import pytest
import pandas as pd
from pathlib import Path
from src.data_processing.read_preprocessed_df import ReadPreprocessedDataFrame
from src.configurations import Resampling

@pytest.fixture
def mock_config(mocker, tmp_path):
    # Mock the Configuration class
    mock_config = mocker.patch("src.configurations.Configuration")
    mock_config.return_value.perid_data_folder = tmp_path / "per_id"
    mock_config.return_value.flat_preprocessed_file_for = lambda sampling: tmp_path / f"{sampling.file_name()}"
    return mock_config

@pytest.fixture
def mock_csv_file(tmp_path):
    # Create a mock CSV file
    file_path = tmp_path / "mock_data.csv"
    data = {
        "id": ["1", "1", "2"],
        "datetime": ["2023-01-01 00:00", "2023-01-01 00:15", "2023-01-01 00:30"],
        "system": ["A", "A", "B"]
    }
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    return file_path

def test_read_preprocessed_dataframe_with_zip_id(mock_config, mock_csv_file, tmp_path, mocker):
    # Create a per-id folder and move the mock CSV file there
    per_id_folder = tmp_path / "per_id" / "1"
    per_id_folder.mkdir(parents=True)
    mock_csv_file.rename(per_id_folder / "mock_data.csv")

    # Mock the preprocessed_file_for function
    mock_preprocessed_file_for = mocker.patch(
        "src.data_processing.read_preprocessed_df.preprocessed_file_for")
    mock_preprocessed_file_for.return_value = per_id_folder / "mock_data.csv"

    # Mock the Resampling object
    mock_sampling = Resampling()
    mock_sampling.csv_file_name = lambda: "mock_data.csv"

    # Initialize the ReadPreprocessedDataFrame with a zip_id
    reader = ReadPreprocessedDataFrame(sampling=mock_sampling, zip_id="1")

    # Assertions
    assert isinstance(reader.df, pd.DataFrame)
    assert not reader.df.empty
    assert list(reader.df.columns) == ["id", "datetime", "system"]
    assert reader.df["id"].iloc[0] == "1"

def test_read_preprocessed_dataframe_with_zip_id(mock_config, mock_csv_file, tmp_path, mocker):
    # Create a per-id folder and move the mock CSV file there
    per_id_folder = tmp_path / "per_id" / "1"
    per_id_folder.mkdir(parents=True)
    mock_csv_file.rename(per_id_folder / "mock_data.csv")

    # Mock the preprocessed_file_for function to return the correct path
    mock_preprocessed_file_for = mocker.patch(
        "src.data_processing.read_preprocessed_df.preprocessed_file_for"
    )
    mock_preprocessed_file_for.return_value = str(per_id_folder / "mock_data.csv")

    # Mock the Resampling object
    mock_sampling = Resampling()
    mock_sampling.csv_file_name = lambda: "mock_data.csv"

    # Initialize the ReadPreprocessedDataFrame with a zip_id
    reader = ReadPreprocessedDataFrame(sampling=mock_sampling, zip_id="1")

    # Assertions
    assert isinstance(reader.df, pd.DataFrame)
    assert not reader.df.empty
    assert list(reader.df.columns) == ["id", "datetime", "system"]
    assert reader.df["id"].iloc[0] == "1"