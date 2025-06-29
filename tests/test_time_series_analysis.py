import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from src.time_series_analysis import (split_on_time_gaps,
                                      remove_zero_or_null_days)
from src.helper import get_night_start_date


def test_split_on_time_gaps():
    idx = pd.date_range(start='2023-01-01', periods=6, freq='D')
    df1 = pd.DataFrame({'cob max': [10, 20, 30, 40, 50, 60]}, index=idx)

    result1 = split_on_time_gaps(df1, value_col='cob max', days_threshold=3)
    assert len(result1) == 1

    df2 = pd.DataFrame({'cob max': [10, 20, 0, 0, 0, 60]}, index=idx)
    result2 = split_on_time_gaps(df2, value_col='cob max', days_threshold=3)
    assert len(result2) == 2

def test_remove_zero_or_null_days():
    df_not_dt = pd.DataFrame({'value': [0, 0, np.nan, 5, 0, 1]})
    with pytest.raises(ValueError,
                       match='DataFrame index must be a DatetimeIndex.'):
        remove_zero_or_null_days(df_not_dt, 'value')

    df = pd.DataFrame({'value': [0, 0, np.nan, 5, 0, 1]},
                      index=pd.date_range('2024-01-01', periods=6,
                                          freq='12h'))

    result = remove_zero_or_null_days(df, 'value')
    expected_dates = pd.to_datetime(['2024-01-02 00:00:00',
                                     '2024-01-02 12:00:00',
                                     '2024-01-03 00:00:00',
                                     '2024-01-03 12:00:00'])
    assert result.index.equals(expected_dates)

    df2 = pd.DataFrame({'value': [1, 2, 3, 4]},
                       index=pd.date_range('2024-01-01', periods=4,
                                           freq='D'))
    result2 = remove_zero_or_null_days(df2, 'value')
    assert len(result2) == 4

    df3 = pd.DataFrame({'value': [0, 0, np.nan, 0]},
                       index=pd.date_range('2024-01-01', periods=4,
                                           freq='D'))
    result3 = remove_zero_or_null_days(df3, 'value')
    assert result3.empty

def test_get_night_start_date_basic_usage():
    dates = pd.to_datetime([
        '2025-06-29 18:00',
        '2025-06-29 03:00',
        '2025-06-29 23:59',
        '2025-06-30 01:00',
    ])
    result = get_night_start_date(dates, night_start_hour=17)
    assert result[0] == datetime(2025, 6, 29).date()
    assert result[1] == datetime(2025, 6, 28).date()
    assert result[2] == datetime(2025, 6, 29).date()
    assert result[3] == datetime(2025, 6, 29).date()

def test_get_night_start_date_with_numpy_array():
    arr = np.array(['2025-06-29 18:00', '2025-06-29 03:00'])
    result = get_night_start_date(arr, night_start_hour=17)
    assert result.iloc[0] == datetime(2025, 6, 29).date()
    assert result.iloc[1] == datetime(2025, 6, 28).date()

def test_get_night_start_date_with_list():
    lst = ['2025-06-29 18:00', '2025-06-29 03:00']
    result = get_night_start_date(lst, night_start_hour=17)
    assert result.iloc[0] == datetime(2025, 6, 29).date()
    assert result.iloc[1] == datetime(2025, 6, 28).date()

def test_get_night_start_date_with_pandas_series():
    ser = pd.Series(pd.to_datetime(['2025-06-29 18:00', '2025-06-29 03:00']))
    result = get_night_start_date(ser, night_start_hour=17)
    assert result.iloc[0] == datetime(2025, 6, 29).date()
    assert result.iloc[1] == datetime(2025, 6, 28).date()

def test_get_night_start_date_edge_cases():
    # Midnight boundary
    dates = pd.to_datetime(['2025-06-29 00:00', '2025-06-29 16:59', '2025-06-29 17:00'])
    result = get_night_start_date(dates, night_start_hour=17)
    assert result[0] == datetime(2025, 6, 28).date()
    assert result[1] == datetime(2025, 6, 28).date()
    assert result[2] == datetime(2025, 6, 29).date()

def test_get_night_start_date_invalid_inputs():
    with pytest.raises(ValueError):
        get_night_start_date(None, night_start_hour=17)
    with pytest.raises(ValueError):
        get_night_start_date(pd.Series([1,2,3]), night_start_hour=None)
    with pytest.raises(ValueError):
        get_night_start_date(123, night_start_hour=17)

def test_get_night_start_date_different_night_start_hours():
    dates = pd.to_datetime(['2025-06-29 06:00', '2025-06-29 20:00'])
    result = get_night_start_date(dates, night_start_hour=6)
    assert result[0] == datetime(2025, 6, 29).date()
    assert result[1] == datetime(2025, 6, 29).date()
    result2 = get_night_start_date(dates, night_start_hour=20)
    assert result2[0] == datetime(2025, 6, 28).date()
    assert result2[1] == datetime(2025, 6, 29).date()
