import pytest
import pandas as pd
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

