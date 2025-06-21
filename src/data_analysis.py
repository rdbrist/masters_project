def nans_per_column(df):
    """
    Provides a summary of NaN counts per column in the DataFrame.
    """
    nan_counts = df.isna().sum()
    print("Total NaNs per column:")
    print(nan_counts)
    df_isna = df[df['cob mean'].isna()]
    print(
        'Count of intervals with NaNs for COB columns, i.e. missing COB data.')
    df_isna.groupby(by=['id', 'day'])['day'].count()
    print('Count number of consecutive NaNs in cob mean column:')
    df_reset = df_isna.reset_index().sort_values(['id', 'datetime'])
    df_reset['time_diff'] = df_reset.groupby('id')[
                                'datetime'].diff().dt.total_seconds() / 60
    df_reset.groupby('id')['time_diff'].apply(lambda x: (x == 30).sum())