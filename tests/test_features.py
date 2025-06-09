import pandas as pd
import pytest
import re

from src.features import FeatureSet

@pytest.fixture
def simple_df():
    data = {
        "id": [1, 1, 2],
        "datetime": ["2023-01-01 00:00", "2023-01-01 00:15",
                     "2023-01-01 00:30"],
        "value": [10, 20, 30]
    }
    df = pd.DataFrame(data)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index(["id", "datetime"], inplace=True)
    return df

def test_load_preprocessed_data_valid_file_with_correct_index(tmp_path,
                                                              simple_df):
    file_path = tmp_path / "test_data.parquet"
    simple_df.to_parquet(file_path)

    feature_set = FeatureSet(input_path=file_path)
    result = feature_set.dataset

    assert not result.empty
    assert list(result.index.names) == ["id", "datetime"]

def test_load_preprocessed_data_missing_file(tmp_path):
    missing_file = tmp_path / "does_not_exist.parquet"
    with pytest.raises(FileNotFoundError,
                       match=re.escape(f"File {missing_file} not found.")):
        FeatureSet(input_path=missing_file)

def test_load_preprocessed_data_invalid_index_structure(tmp_path):
    # Create a parquet file with wrong index
    data = {"user_id": [1, 2], "datetime": ["2023-01-01", "2023-01-02"]}
    df = pd.DataFrame(data)
    file_path = tmp_path / "bad_index.parquet"
    df.to_parquet(file_path)
    with pytest.raises(ValueError,
                       match="DataFrame index must be a MultiIndex"):
        FeatureSet(input_path=file_path)

def test_load_preprocessed_data_invalid_fixable_index_structure(tmp_path,
                                                                simple_df):
    simple_df.reset_index(inplace=True)
    file_path = tmp_path / "no_index.parquet"
    simple_df.to_parquet(file_path)

    feature_set = FeatureSet(input_path=file_path)
    result = feature_set.dataset

    assert not result.empty
    assert list(result.index.names) == ["id", "datetime"]

def test_index_raises_error_for_invalid_index_names(tmp_path):
    data = {
        "user_id": [1, 1, 2],
        "timestamp": ["2023-01-01 00:00", "2023-01-01 00:15",
                      "2023-01-01 00:30"],
        "value": [10, 20, 30],
    }
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index(["user_id", "timestamp"], inplace=True)
    file_path = tmp_path / "invalid_index_names.parquet"
    df.to_parquet(file_path)
    with pytest.raises(ValueError, match="DataFrame index must be a MultiIndex "
                                         "with levels \\['id', 'datetime'\\]."):
        FeatureSet(input_path=file_path)

def test_index_raises_error_for_invalid_index_dtypes(tmp_path):
    data = {
        "id": ["a", "b", "c"],
        "datetime": ["2023-01-01 00:00", "2023-01-01 00:15",
                     "2023-01-01 00:30"],
        "value": [10, 20, 30],
    }
    df = pd.DataFrame(data)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index(["id", "datetime"], inplace=True)
    file_path = tmp_path / "invalid_index_dtypes.parquet"
    df.to_parquet(file_path)
    with pytest.raises(ValueError,
                       match="Index level 'id' must be of integer dtype."):
        FeatureSet(input_path=file_path)

def test_add_time_features_creates_valid_columns(tmp_path, simple_df):
    file_path = tmp_path / "valid_data.parquet"
    simple_df.to_parquet(file_path)

    feature_set = FeatureSet(input_path=file_path)
    feature_set.add_time_based_features()

    assert "hour_of_day" in feature_set.dataset.columns
    assert "hour_sin" in feature_set.dataset.columns
    assert "hour_cos" in feature_set.dataset.columns

def test_add_hourly_mean_creates_correct_columns(tmp_path, simple_df):
    file_path = tmp_path / "valid_data.parquet"
    simple_df.to_parquet(file_path)

    feature_set = FeatureSet(input_path=file_path)
    feature_set.add_hourly_mean(columns=["value"])

    assert "value hourly_mean" in feature_set.dataset.columns
    assert feature_set.dataset["value hourly_mean"].notnull().all()


