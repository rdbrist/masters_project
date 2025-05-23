import pytest
import pandas as pd
import numpy as np
from decimal import Decimal
from src.data_processing.resampling import Resampling, GeneralisedCols, \
    ResampleDataFrame
from src.configurations import Daily, Hourly, FifteenMinute, FiveMinute

# TODO: work through tests and corredt any issues
@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "id": [1, 1, 1, 1],
        "datetime": pd.to_datetime([
            "2023-01-01 00:00", "2023-01-01 00:05",
            "2023-01-01 00:10", "2023-01-01 00:20"
        ]),
        "system": ["A", "A", "A", "A"],
        "iob": [1, 2, 3, 6],
        "cob": [1.234, 2.345, 3.456, 4.567],
        "bg": [5.678, 6.789, 7.890, 8.901]
    })

@pytest.fixture
def resampler(sample_df):
    return ResampleDataFrame(sample_df)

def test_resample_not_a_dataframe():
    with pytest.raises(TypeError):
        ResampleDataFrame("not a dataframe")

def test_resample_to_returns_empty_for_empty_df():
    df = pd.DataFrame(columns=["id", "datetime", "iob", "cob", "bg"])
    sampling = FifteenMinute()
    with pytest.raises(IndexError):
        ResampleDataFrame(df)

def test_resample_to_resamples_correctly(resampler):
    sampling = FifteenMinute()
    result = resampler.resample_to(sampling)
    assert "iob mean" in result.columns
    assert result["iob mean"].iloc[0] == 2

