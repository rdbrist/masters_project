import pytest
import pandas as pd
import numpy as np
from src.time_series_analysis import (split_on_time_gaps,
                                      remove_zero_or_null_days)

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
