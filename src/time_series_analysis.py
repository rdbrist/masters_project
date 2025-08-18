import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import seaborn as sns
import numpy as np
from loguru import logger
from datetime import datetime, time, timedelta


from sklearn.metrics import mean_absolute_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

from src.config import FIGURES_DIR
from src.dba import DBAAverager
from src.features import FeatureSet
from src.helper import normalise_overnight_time


def split_ts(y: pd.DataFrame, ratio: float) -> (
        tuple)[pd.DataFrame, pd.DataFrame]:
    """
    Split a time series into training and test sets based on a given ratio.
    :param y: (pd.DataFrame) Time series data with a DateTime index.
    :param ratio: (float) Ratio of the training set size to the total size of
        the time series.
    :return: ytrain, ytest: (tuple) Two DataFrames containing the training and
        test sets, respectively.
    """
    min_date = y.index.min()
    max_date = y.index.max()
    num_vals = len(y)
    split = round(num_vals * ratio)
    print(f'Min date: {min_date}')
    print(f'Max date: {max_date}')
    print(f'Number of values: {num_vals}')
    print(f'Train set size: {split}')
    print(f'Test set size: {num_vals - split}')
    ytrain = y.iloc[:split].copy()
    ytest = y.iloc[split:].copy()
    return ytrain, ytest


def ts_plot(y: pd.DataFrame):
    """
    Basic time series plot. Expects a DataFrame with a DateTime index and a
    single random variable.
    :param y: (pd.DataFrame) Time series data with a DateTime index.
    """
    y.plot(figsize=(18, 8))
    plt.show()


def ts_dist_plot(x: pd.Series):
    """
    Plot the distribution of a time series variable using a boxplot and a
    histogram with a kernel density estimate (KDE).
    :param x: pd.Series or pd.DataFrame column containing the time series data.
    """
    fg, ax = plt.subplots(nrows=2, figsize=(15, 8))
    sns.boxplot(x=x, ax=ax[0])
    sns.histplot(x=x, ax=ax[1], kde=True)
    plt.show()


def ts_plot_cfs(y: pd.DataFrame):
    """
    Plot the autocorrelation function (ACF) and partial autocorrelation function
    (PACF) of a time series variable.
    :param y: (pd.DataFrame) Time series data with a DateTime index.
    """
    plot_acf(y),
    plot_pacf(y),
    plt.show()


def p_q_result(ytrain: pd.DataFrame,
               pmax: int,
               qmax: int,
               pstep: int,
               qstep: int,
               ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform a grid search over ARIMA model parameters (p, q) to find the best
    combination based on Mean Absolute Error (MAE). This function trains ARIMA
    models for different combinations of p and q, calculates the MAE for each
    model, and visualizes the results in a heatmap. It also collects AIC and BIC
    values for each model in a DataFrame.
    :param ytrain: (pd.DataFrame) Training data for the time series.
    :param pmax: (int) Maximum value for the AR parameter (p).
    :param qmax: (int) Maximum value for the MA parameter (q).
    :param pstep: (int) Step size for the AR parameter (p).
    :param qstep: (int) Step size for the MA parameter (q).
    :return:
    """
    p_params = range(0, pmax, pstep)
    q_params = range(0, qmax, qstep)
    mae_grid = {}
    init_time = time.time()

    aicbic_list = []
    for p in p_params:
        mae_grid[p] = []
        for q in q_params:
            order = (p, 0, q)
            start_time = time.time()
            model = ARIMA(ytrain, order=order).fit()
            aicbic_list.append({'Order': [order],
                                'AIC': [model.aic],
                                'BIC': [model.bic]})
            elapsed_time = round(time.time() - start_time, 2)
            print(f"Trained ARIMA {order} in {elapsed_time} seconds.")
            y_pred = model.predict()
            print(y_pred.isnull().sum().sum(), "null values in predictions")
            mae = mean_absolute_error(ytrain, y_pred)
            mae_grid[p].append(mae)
    aicbic_df = pd.DataFrame(aicbic_list, columns=['Order', 'AIC', 'BIC'])
    print(f'All permutations completed in {round(time.time() - init_time, 2)} '
          f'seconds.')

    # Given you wouldn't consider an ARMA model without autoregression, also
    # makes the heatmap more meaningful by reducing significantly different MAE
    # values:
    del mae_grid[0]
    mae_df = pd.DataFrame(mae_grid)
    print(mae_df.round(4))
    sns.heatmap(mae_df, cmap="Blues")
    plt.xlabel("p values")
    plt.ylabel("q values")
    plt.title("ARIMA Model Performance")

    return mae_df, aicbic_df


def run_adf(df):
    """
    Perform the Augmented Dickey-Fuller (ADF) test to check for stationarity and
    formats the results for easy interpretation.
    :param df: (pd.DataFrame) Time series data to be tested for stationarity.
    """
    adf_test = adfuller(df, autolag='AIC', regression='ct')
    print("ADF Test Results")
    print("Null Hypothesis: The series has a unit root (non-stationary)")
    print("ADF-Statistic:", adf_test[0])
    print("P-Value:", adf_test[1])
    print("Number of lags:", adf_test[2])
    print("Number of observations:", adf_test[3])
    print("Critical Values:", adf_test[4])
    print("Note: If P-Value is smaller than 0.05, we reject the null "
          "hypothesis and the series is stationary.")
    print("A more negative test statistic indicates stronger evidence against "
          "the null hypothesis.")


def split_on_time_gaps(df: pd.DataFrame,
                       value_col: str,
                       days_threshold: int = 3) -> list:
    """
    Removes days with all zero or null values, then splits DataFrame at gaps in
    the DatetimeIndex. Expects a dataframe without the id index, only datetime.
    greater than days_threshold.
    :param df: DataFrame with a DatetimeIndex.
    :param value_col: Name of the column to check for zero or null values.
    :param days_threshold: Number of days to consider as a gap.
    :return: List of DataFrames, each representing a continuous segment of the
    time series.
    """
    # Remove days with all zero or null values
    df = remove_zero_or_null_days(df, value_col)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex.")
    df = df.sort_index()
    gaps = df.index.to_series().diff().dt.days.fillna(0)
    group = (gaps > days_threshold).cumsum()
    return [group_df for _, group_df in df.groupby(group) if not group_df.empty]


def remove_zero_or_null_days(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """
    Keep only days where all values where at least one of the timestamps is
    non-zero or non-null.
    :param df: DataFrame with DatetimeIndex.
    :param value_col: Name of the column to check.
    :return: Filtered DataFrame.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex.")

    # Find days where all values are non-zero and non-null
    mask = df.groupby(df.index.date)[value_col].apply(
        lambda x: (x.notnull() & (x != 0)).any()
    )
    logger.info(f'Masking {mask.sum()} days with non-zero or non-null values'
                f'from {len(df.groupby(df.index.date))} total days.')
    keep_dates = set(mask[mask].index)
    date_mask = np.array([d in keep_dates for d in df.index.date])
    return df[date_mask]


def _plot_time_series_profile(
        df, variables, groupby_cols, x_col, title,night_start,
        morning_end, method='mean', rolling_window=None, y_limits: tuple = None,
        global_min=None, global_max=None, prescaled=True,
        excursion_variable: str = None, excursion_plot_type='markers',
        excursion_color='red', excursion_label='Excursions'):

    if 'hour' not in df:
        dt_index = df.index.get_level_values('datetime')
        df['hour'] = dt_index.hour

    if method == 'mean':
        agg_variables = variables.copy()
        if excursion_variable and excursion_variable in df.columns:
            agg_variables.append(excursion_variable)
        stats = (df.groupby(groupby_cols)[variables].
                 agg(['mean', 'var']).reset_index())
    elif method == 'dba':
        dbaa = DBAAverager(df[variables + (
            [excursion_variable] if excursion_variable else [])],
                           night_start_hour=night_start,
                           morning_end_hour=morning_end)
        stats = (
            dbaa.get_dba_and_variance_dataframe(rolling_window=rolling_window))
        if stats is None:
            logger.warning("No DBA averaged DataFrame available.")
            return
    else:
        raise ValueError("method must be 'mean' or 'dba'")

    x = stats[x_col].apply(lambda x: normalise_overnight_time(x, morning_end))
    # x_labels = stats['time'].apply(lambda t: t.strftime('%H:%M'))

    fig, ax = plt.subplots(figsize=(7, 3))
    colors = ['tab:green', 'tab:orange', 'tab:blue', 'tab:purple', 'tab:brown']

    # Plot mean variables
    for i, var in enumerate(variables):
        avg = stats[(var, method)]
        var_col = stats.get((var, 'var'), None)

        if var_col is None or np.isscalar(var_col):
            var_col = pd.Series([np.nan] * len(stats), index=stats.index)
        else:
            var_col = pd.to_numeric(var_col, errors='coerce')

        if not prescaled:
            min_val = global_min[
                var] if global_min is not None and var in global_min else avg.min()
            max_val = global_max[
                var] if global_max is not None and var in global_max else avg.max()

            # Avoid division by zero if range is 0
            range_val = (max_val - min_val)
            if range_val == 0:
                stats[
                    (var, 'mean_scaled')] = 0.0  # or 1.0 if all values are max
                stats[(var, 'std_scaled')] = 0.0
            else:
                stats[(var, 'mean_scaled')] = (avg - min_val) / range_val
                stats[(var, 'std_scaled')] = np.sqrt(var_col) / range_val
        else:
            stats[(var, 'mean_scaled')] = avg
            stats[(var, 'std_scaled')] = np.sqrt(var_col)

        y = stats[(var, 'mean_scaled')]
        yerr = stats[(var, 'std_scaled')]

        # Plot the main variable line
        plt.plot(x, y, label=f'{var} {method}', color=colors[i % len(colors)],
                 linewidth=2)
        # Fill between for variability
        plt.fill_between(x, y - yerr, y + yerr, color=colors[i % len(colors)],
                         alpha=0.2)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator())

    plt.title(title)
    plt.ylabel('Scaled Mean')
    plt.xticks(rotation=90)
    if y_limits is not None:
        plt.ylim(y_limits)
    plt.legend()
    filename = title.replace(" ", "_").replace(":", "_").lower()
    plt.savefig(FIGURES_DIR / f'{filename}.png', bbox_inches='tight')
    plt.show()


def plot_night_time_series(df, zip_id=None, variables=None, night_start=17,
                           morning_end=11, method='mean',
                           y_limits: tuple = None, rolling_window=None,
                            global_min=None, global_max=None, prescaled=True,
                           include_excursions: bool = False,
                           excursion_plot_type='markers', cluster=None):
    """
    Plot the means of specified variables for a group of nights during night
    hours.
    :param excursion_plot_type: str, type of plot for excursions,
    :param include_excursions: bool, whether to include excursions in the plot.
    :param global_max: dict, optional global maximum values for each variable to scale
    :param global_min: dict, optional global minimum values for each variable to scale
    :param df: DataFrame containing time series data with a multi-index
    :param zip_id: int, optional identifier for the individual.
    :param variables: list of str, names of the variables to plot. If None,
        ['iob mean', 'cob mean', 'bg mean'] are used
    :param night_start: int, hour when the night starts (0-23).
    :param morning_end: int, hour when the morning ends (0-23).
    :param method: str, method to use for plotting ('mean' or 'dba').
    :param y_limits: tuple, optional limits for the y-axis (min, max).
    :param rolling_window: int, optional size of the rolling window for
        smoothing
    :param prescaled: bool, whether the data is already scaled
    :return:
    """
    if variables is None:
        variables = ['iob mean', 'cob mean', 'bg mean']

    def night_hour(hour):
        return (hour - night_start if hour >= night_start
                else 24 - night_start + hour)

    df_new = df.copy()

    night_count = len(df_new.
                      reset_index()[['id','night_start_date']].
                      drop_duplicates())
    title = f'Nights: {night_count}'
    if cluster is not None:
        title = f'Cluster {cluster} ' + title
    if zip_id is not None:
        title = f'Person {str(zip_id)} ' + title
    dt_index = df_new.index.get_level_values('datetime')
    df_new['hour'] = dt_index.hour
    df_new['night_hour'] = df_new['hour'].map(night_hour)

    df_new = df_new[(df_new['hour'] >= night_start) |
                      (df_new['hour'] < morning_end)]

    excursion_var_to_plot = None
    if include_excursions and 'excursion_amplitude' not in df_new.columns:
        raise ValueError('Excursion variable not found in variables.')
    elif include_excursions:
        excursion_var_to_plot = 'excursion_amplitude'

    _plot_time_series_profile(
        df_new, variables,
        groupby_cols=(['id', 'night_start_date', 'night_hour', 'hour', 'time']
                      if method == 'dba'
                      else ['id', 'night_hour', 'hour', 'time']),
        x_col='time',
        title=title,
        method=method,
        night_start=night_start,
        morning_end=morning_end,
        y_limits=y_limits,
        rolling_window=rolling_window,
        global_min=global_min,
        global_max=global_max,
        prescaled=prescaled,
        excursion_variable=excursion_var_to_plot,
        excursion_plot_type=excursion_plot_type
    )

#
# def plot_night_time_series(df, zip_id=None, variables=None, night_start=17,
#                            morning_end=11, method='mean',
#                            y_limits: tuple = None, rolling_window=None,
#                            global_min=None, global_max=None, prescaled=True,
#                            include_excursions: bool = False,
#                            excursion_plot_type='markers', cluster=None):
#     """
#     Plot the means of specified variables for a group of nights.
#     """
#     if variables is None:
#         variables = ['iob mean', 'cob mean', 'bg mean']
#
#     def night_hour(hour):
#         return (hour - night_start if hour >= night_start
#                 else 24 - night_start + hour)
#
#     df_new = df.copy()
#
#     night_count = len(df_new.
#                       reset_index()[['id', 'night_start_date']].
#                       drop_duplicates())
#     title = f'Nights: {night_count}'
#     if cluster is not None:
#         title = f'Cluster {cluster} ' + title
#     if zip_id is not None:
#         title = f'Person {str(zip_id)} ' + title
#
#     dt_index = df_new.index.get_level_values('datetime')
#     df_new['hour'] = dt_index.hour
#     df_new['night_hour'] = df_new['hour'].map(night_hour)
#
#     df_new = df_new[(df_new['hour'] >= night_start) |
#                     (df_new['hour'] < morning_end)]
#
#     excursion_var_to_plot = None
#     if include_excursions and 'excursion_amplitude' not in df_new.columns:
#         raise ValueError('Excursion variable not found in variables.')
#     elif include_excursions:
#         excursion_var_to_plot = 'excursion_amplitude'
#
#     groupby_cols = (['id', 'night_start_date', 'night_hour', 'hour', 'time']
#                     if method == 'dba'
#                     else ['id', 'night_hour', 'hour', 'time'])
#
#     if method == 'mean':
#         stats = (df_new.groupby(groupby_cols)[variables].
#                  agg(['mean', 'var']).reset_index())
#     elif method == 'dba':
#         dbaa = DBAAverager(df_new[variables + (
#             [excursion_var_to_plot] if excursion_var_to_plot else [])],
#                            night_start_hour=night_start,
#                            morning_end_hour=morning_end)
#         stats = (
#             dbaa.get_dba_and_variance_dataframe(rolling_window=rolling_window))
#         if stats is None:
#             logger("No DBA averaged DataFrame available.")
#             return
#     else:
#         raise ValueError("method must be 'mean' or 'dba'")
#
#     fig, ax = plt.subplots(figsize=(8, 3))
#     colors = ['tab:green', 'tab:orange', 'tab:blue', 'tab:purple', 'tab:brown']
#
#     for i, var in enumerate(variables):
#         avg = stats[(var, 'mean')]
#         var_col = stats.get((var, 'var'), None)
#
#         y = avg
#         yerr = np.sqrt(var_col) if var_col is not None else 0
#
#         plt.plot(stats['time'], y, label=f'{var} {method}',
#                  color=colors[i % len(colors)], linewidth=2)
#         plt.fill_between(stats['time'], y - yerr, y + yerr,
#                          color=colors[i % len(colors)], alpha=0.2)
#
#     ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
#     ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
#
#     plt.title(title)
#     plt.ylabel('Scaled Mean')
#     plt.xticks(rotation=90)
#     if y_limits is not None:
#         plt.ylim(y_limits)
#     plt.legend()
#     plt.tight_layout()
#     filename = title.replace(" ", "_").replace(":", "_").lower()
#     plt.savefig(FIGURES_DIR / f'{filename}.png', dpi=400, bbox_inches='tight')
#     plt.show()


def plot_hourly_means_for_individual(df, zip_id, variables=None):
    """
    Plot the hourly means of specified variables for an individual.
    :param df: DataFrame containing time series data.
    :param zip_id: int, identifier for the individual.
    :param variables: list of str, names of the variables to plot.
    """
    if variables is None:
        variables = ['iob mean', 'cob mean', 'bg mean']

    df_new = df.copy()
    if 'datetime' in df_new.index.names:
        df_new['hour'] = df_new.index.get_level_values('datetime').hour
    else:
        # Assuming 'datetime' is a column if not an index level
        df_new['hour'] = pd.to_datetime(df_new['datetime']).dt.hour

    stats = df_new.groupby('hour')[variables].agg(['mean', 'var']).reset_index()

    fig, ax = plt.subplots(figsize=(8, 3))
    colors = ['tab:green', 'tab:orange', 'tab:blue', 'tab:purple', 'tab:brown']

    for i, var in enumerate(variables):
        avg = stats[(var, 'mean')]
        var_col = stats.get((var, 'var'), None)
        yerr = np.sqrt(var_col) if var_col is not None else 0

        plt.plot(stats['hour'], avg, label=f'{var} mean',
                 color=colors[i % len(colors)], linewidth=2)
        plt.fill_between(stats['hour'], avg - yerr, avg + yerr,
                         color=colors[i % len(colors)], alpha=0.2)

    plt.title(f'Person {str(zip_id)} (hourly means across all days)')
    plt.xlabel('Hour of the day')
    plt.ylabel('Mean')
    plt.xticks(stats['hour'])
    plt.legend()
    plt.tight_layout()
    plt.show()


def return_count_intervals(start: time, end: time,
                           minute_interval: int = 30) -> int:
    today = datetime.today().date()
    dt_start = datetime.combine(today, start)
    dt_end = datetime.combine(today, end)
    if dt_end <= dt_start:
        dt_end += timedelta(days=1)
    total_minutes = int((dt_end - dt_start).total_seconds() // 60)
    return total_minutes // minute_interval
