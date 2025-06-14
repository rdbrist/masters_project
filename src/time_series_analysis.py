import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import time
import numpy as np
from loguru import logger

from sklearn.metrics import mean_absolute_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller


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

import matplotlib.pyplot as plt
import numpy as np

def plot_night_means_for_individual(df: pd.DataFrame,
                                    zip_id: int,
                                    variables: list=['iob mean', 'cob mean', 'bg mean'],
                                    night_start: int=17,
                                    morning_end: int=11):
    """
    Plots mean and variance (as error bands) of variables for a single person over the night period.
    X-axis shows hour (0-23) in correct overnight order. Assumes df has a MultiIndex ['id', 'datetime'].
    :param df: DataFrame with MultiIndex ['id', 'datetime'] and columns for variables.
    :param id_: Person ID to plot.
    :param variables: List of variable names to plot.
    :param night_start: Hour when the night period starts (default 19).
    :param morning_end: Hour when the night period ends (default 12).
    """
    def night_period_mask(dt_index):
        hours = dt_index.hour
        return (hours >= night_start) | (hours < morning_end)

    def night_hour(hour):
        if hour >= night_start:
            return hour - night_start
        else:
            return 24 - night_start + hour

    df_person = df.xs(zip_id, level='id')
    dt_index = df_person.index.get_level_values('datetime')
    mask = night_period_mask(dt_index)
    df_night = df_person[mask].copy()
    hours = dt_index[mask].hour
    df_night['hour'] = hours
    df_night['night_hour'] = hours.map(night_hour)

    stats = df_night.groupby(['night_hour', 'hour'])[variables].agg(['mean', 'var']).sort_index()
    stats = stats.reset_index()
    for var in variables:
        min_v = stats[(var, 'mean')].min()
        max_v = stats[(var, 'mean')].max()
        stats[(var, 'mean_scaled')] = (stats[(var, 'mean')] - min_v) / (max_v - min_v + 1e-9)
        stats[(var, 'std_scaled')] = np.sqrt(stats[(var, 'var')]) / (max_v - min_v + 1e-9)

    x = stats['night_hour']
    x_labels = stats['hour'].astype(str)

    plt.figure(figsize=(10, 4))
    for var, color in zip(variables, ['tab:blue', 'tab:orange', 'tab:green']):
        y = stats[(var, 'mean_scaled')]
        yerr = stats[(var, 'std_scaled')]
        plt.plot(x, y, label=f'{var} mean', color=color)
        plt.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.2)
    plt.title(f'Person {zip_id}')
    plt.ylabel('Scaled Value')
    plt.xlabel('Hour')
    plt.xticks(x, x_labels, rotation=0)
    plt.legend()
    plt.tight_layout()
    plt.show()



