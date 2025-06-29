import pandas as pd
import numpy as np


def resample_to_30_minute_intervals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample the dataframe to 30-minute intervals.
    :param df: DataFrame to resample
    :return: Resampled DataFrame
    """
    # Build aggregation dictionary

    agg_dict = {}
    for col in df.columns:
        if col.endswith(' mean'):
            agg_dict[col] = 'mean'
        elif col.endswith(' min'):
            agg_dict[col] = 'min'
        elif col.endswith(' max'):
            agg_dict[col] = 'max'
        elif col.endswith(' std'):
            agg_dict[col] = 'std'
        elif col.endswith(' count'):
            agg_dict[col] = 'sum'
        else:
            agg_dict[col] = 'first'  # fallback

    df_resampled = (
        df.groupby(level=0)
          .resample('30min', level=1)
          .agg(agg_dict)
    )

    # Replace zeros with NaN in count columns before dropping all-NaN rows
    count_cols = [col for col in df_resampled.columns if col.endswith(' count')]
    df_resampled[count_cols] = df_resampled[count_cols].replace(0, np.nan)

    df_resampled = df_resampled.dropna(how='all')

    return df_resampled
