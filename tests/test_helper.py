import pytest
import pandas as pd
import numpy as np
from src.helper import *


@pytest.fixture
def df_base():
    data = {
        "id": [1, 2, 3],
        "datetime": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
        "value": [10, 20, 30],
    }
    return pd.DataFrame(data)

def test_check_df_index_corrects_valid_dataframe(df_base):
    result = check_df_index(df_base)
    assert isinstance(result.index, pd.MultiIndex)
    assert list(result.index.names) == ["id", "datetime"]

def test_check_df_index_raises_error_for_non_multiindex(df_base):
    df = df_base.set_index("datetime")
    with pytest.raises(ValueError, match="DataFrame index must be a MultiIndex"):
        check_df_index(df)

def test_check_df_index_raises_error_for_invalid_index_names(df_base):
    df = df_base.set_index(["id", "datetime"])
    df.index.rename(["wrong", "names"], inplace=True)
    with pytest.raises(ValueError, match="DataFrame index must be a MultiIndex with levels \['id', 'datetime'\]."):
        check_df_index(df)

def test_check_df_index_raises_error_for_non_integer_id(df_base):
    df = df_base.copy()
    df["id"] = ["a", "b", "c"]
    df = df.set_index(["id", "datetime"])
    with pytest.raises(ValueError, match="Index level 'id' must be of integer dtype."):
        check_df_index(df)

def test_check_df_index_raises_error_for_non_datetime_datetime(df_base):
    df = df_base.copy()
    df["datetime"] = ["2023-01-01", "2023-01-02", "2023-01-03"]
    df = df.set_index(["id", "datetime"])
    with pytest.raises(ValueError, match="Index level 'datetime' must be of datetime dtype."):
        check_df_index(df)

@pytest.mark.parametrize("input_type", ["DatetimeIndex", "Series", "ndarray", "list"])
def test_get_night_start_date_various_types(input_type):
    # Example timestamps: 2024-06-01 22:00, 2024-06-02 01:00, 2024-06-02 23:00
    timestamps = [
        pd.Timestamp("2024-06-01 22:00"),
        pd.Timestamp("2024-06-02 01:00"),
        pd.Timestamp("2024-06-02 23:00"),
    ]
    night_start_hour = 21  # 21:00 (9 PM)
    expected = [
        pd.Timestamp("2024-06-01").date(),  # 22:00 is after 21:00, so same day
        pd.Timestamp("2024-06-01").date(),  # 01:00 is before 21:00, so previous day
        pd.Timestamp("2024-06-02").date(),  # 23:00 is after 21:00, so same day
    ]
    if input_type == "DatetimeIndex":
        input_val = pd.DatetimeIndex(timestamps)
    elif input_type == "Series":
        input_val = pd.Series(timestamps)
    elif input_type == "ndarray":
        input_val = np.array(timestamps)
    elif input_type == "list":
        input_val = timestamps

    result = get_night_start_date(input_val, night_start_hour)
    assert list(result) == expected

def test_get_night_start_date_raises_on_none():
    with pytest.raises(ValueError):
        get_night_start_date(None, 21)
    with pytest.raises(ValueError):
        get_night_start_date(pd.Series([pd.Timestamp("2024-06-01 22:00")]), None)

def test_get_night_start_date_invalid_type():
    with pytest.raises(ValueError):
        get_night_start_date("2024-06-01 22:00", 21)

