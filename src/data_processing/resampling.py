from decimal import ROUND_HALF_UP, Decimal
import numpy as np
import pandas as pd
from loguru import logger

from src.configurations import Resampling, GeneralisedCols, Configuration


class ResampleDataFrame:
    """
    Class for resampling irregular dataframes into regular ones
    """

    def __init__(self, irregular_df: pd.DataFrame):
        """
        :param irregular_df : DataFrame: Irregular sampled df
        """
        if not isinstance(irregular_df, pd.DataFrame):
            raise TypeError('irregular_df must be a pandas DataFrame')

        self.__df = irregular_df.sort_values(by=GeneralisedCols.datetime)
        self.__config = Configuration()
        self.zip_id = self.__df['id'].iloc[0]

    def resample_to(self, sampling: Resampling):
        """
        :param sampling : Resampling: What the time series needs to be resampled
        to
        """
        gencols = GeneralisedCols

        cols_to_resample = self.__config.value_columns_to_resample()
        columns = (self.__config.info_columns() +
                   self.__config.resampled_value_columns() +
                   self.__config.resampling_count_columns())

        def round_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
            """Round specified columns to 3 decimal places."""
            for col in columns:
                if col in df:
                    try:
                        df.loc[:, col] = (df[col].apply(self.__round_numbers).
                                          astype(df[col].dtype))
                    except FutureWarning as e:
                        logger.warning(f'Error occurred for person {self.zip_id} '
                              f'in column {col}:\n{e}')
            return df

        # resample by value column to avoid resampling over missing values in
        # some of the value columns
        resulting_df = None
        for column in cols_to_resample:
            sub_columns = self.__config.info_columns() + [column]
            # df with only one value column
            sub_df = self.__df[sub_columns].copy()
            # to ensure we don't sample over missing values
            sub_df = sub_df.dropna()
            if sub_df.shape[0] == 0:
                continue
            # calculate minutes interval between non nan samples for each
            # interval (day or hour) and only keep days/hours, where the
            # interval is smaller than the max allowed gap
            if sampling.needs_max_gap_checking and sampling.sample_rule == '1D':
                # only one day of data
                if len(set(sub_df[gencols.datetime].dt.date)) == 1:
                    result = (sub_df[gencols.datetime].
                              diff().
                              astype('timedelta64[s]') /
                              pd.Timedelta(seconds=60))
                else:
                    result = (
                        sub_df.
                        groupby(
                            by=sub_df[gencols.datetime].
                            dt.date, group_keys=True).
                        apply(
                            lambda x: x[gencols.datetime].diff().
                            astype('timedelta64[s]') /
                            pd.Timedelta(seconds=60))
                    )
                sub_df['diff'] = result.reset_index(level=0, drop=True)
                # days with bigger gaps than max
                bigger_gaps_dates = set(
                    sub_df.loc[sub_df['diff'] > sampling.max_gap_in_min]
                    [gencols.datetime].dt.date)
                df_right_max_gaps = (
                    sub_df[~sub_df[gencols.datetime].
                           dt.date.isin(bigger_gaps_dates)]
                )

                # For each date left we need to calculate the gap between
                # the last/first timestamp of the day/hour and the
                # next/previous day/hour and drop that date if it is bigger
                # than 180
                last_datetime = list(
                    df_right_max_gaps.
                    groupby(df_right_max_gaps[gencols.datetime].dt.date).
                    last()[gencols.datetime])
                first_datetime = list(
                    df_right_max_gaps.
                    groupby(df_right_max_gaps[gencols.datetime].dt.date).
                    first()[gencols.datetime])
                latest_time_each_date = \
                    [t.replace(hour=23, minute=59, second=59)
                     for t in last_datetime]
                earliest_time_each_date = \
                    [t.replace(hour=0, minute=0, second=0)
                     for t in last_datetime]
                last_or_first_time_interval_too_big = []
                for idx, last_available_t in enumerate(last_datetime):
                    min_to_midnight = (
                            (latest_time_each_date[idx] -
                             last_available_t).total_seconds() / 60.0)
                    if min_to_midnight > sampling.max_gap_in_min:
                        (last_or_first_time_interval_too_big.
                         append(last_available_t.date()))

                for idx, first_available_t in enumerate(first_datetime):
                    min_to_first_timestamp = (
                            (first_available_t -
                             earliest_time_each_date[idx]).
                            total_seconds() / 60.0)
                    if min_to_first_timestamp > sampling.max_gap_in_min:
                        (last_or_first_time_interval_too_big.
                         append(first_available_t.date()))

                # only keep dates that don't have a last time stamp that's
                # more than max interval to midnight away
                df_right_max_gaps = df_right_max_gaps[
                    ~df_right_max_gaps[gencols.datetime].dt.date.
                    isin(set(last_or_first_time_interval_too_big))]

                sub_df = df_right_max_gaps.drop(['diff'], axis=1)
            elif (sampling.needs_max_gap_checking
                  and not sampling.sample_rule == '1D'):
                raise NotImplementedError

            # resample
            sub_df = sub_df.set_index([gencols.datetime])
            agg_dict = dict(sampling.general_agg_cols_dictionary)
            agg_dict[column] = sampling.agg_cols
            resampled_df = sub_df.resample(sampling.sample_rule).agg(agg_dict)

            if resampled_df.shape[0] == 0:
                continue

            if resulting_df is None:
                resulting_df = resampled_df
            else:
                resulting_df = resulting_df.combine_first(resampled_df)

        if resulting_df is None:
            return pd.DataFrame(columns=columns)

        # ensure columns are as expected
        resulting_df.columns = resulting_df.columns.to_flat_index()
        resulting_df.columns = [' '.join(col) if col[1] != 'first' else
                                col[0] for col in resulting_df.columns.values]
        resulting_df.reset_index(inplace=True)

        # add na columns for columns that don't exist
        missing_columns = list(set(self.__config.resampled_value_columns()).
                               difference(list(resulting_df.columns)))
        resulting_df[missing_columns] = np.nan

        # drop entries that are just there for the counts
        resulting_df = (
            resulting_df.
            dropna(subset=self.__config.resampled_value_columns(), how='all'))

        # round numbers to 3 decimal places
        resulting_df = round_columns(resulting_df, [
            gencols.mean_iob, gencols.mean_cob, gencols.mean_bg,
            gencols.max_iob, gencols.max_cob, gencols.max_bg,
            gencols.min_iob, gencols.min_cob, gencols.min_bg,
            gencols.std_iob, gencols.std_cob, gencols.std_bg
        ])

        # add missing columns
        missing_columns = list(set(columns) - set(resulting_df.columns))
        resulting_df.loc[:, missing_columns] = None

        # replace na with 0 for count columns
        count_columns = self.__config.resampling_count_columns()
        try:
            resulting_df.loc[:, count_columns] = \
                (resulting_df[count_columns].
                 apply(pd.to_numeric, errors='raise').
                 astype('Int64').
                 fillna(0)
                 )
        except Exception as e:
            logger.info(f'Error occurred while converting count columns to int for '
                  f'zip_id {self.zip_id}:\n{e}')
            #resulting_df.loc[:, count_columns] = 0


        # reorder columns
        return resulting_df.loc[:, columns]

    @staticmethod
    def __round_numbers(x):
        if np.isnan(x):
            return x
        return float(Decimal(str(x)).quantize(Decimal('.100'),
                                              rounding=ROUND_HALF_UP))
