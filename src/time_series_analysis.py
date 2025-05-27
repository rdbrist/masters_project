import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import time

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
    ytrain = y.iloc[:split]
    ytest = y.iloc[split:]
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


def p_q_result(pmax: int,
               qmax: int,
               pstep: int,
               qstep: int,
               ytrain: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform a grid search over ARIMA model parameters (p, q) to find the best
    combination based on Mean Absolute Error (MAE). This function trains ARIMA
    models for different combinations of p and q, calculates the MAE for each
    model, and visualizes the results in a heatmap. It also collects AIC and BIC
    values for each model in a DataFrame.
    :param pmax: (int) Maximum value for the AR parameter (p).
    :param qmax: (int) Maximum value for the MA parameter (q).
    :param pstep: (int) Step size for the AR parameter (p).
    :param qstep: (int) Step size for the MA parameter (q).
    :param ytrain: (pd.DataFrame) Training data for the time series.
    :return:
    """
    p_params = range(0, pmax, pstep)
    q_params = range(0, qmax, qstep)
    mae_grid = dict()
    aicbic_df = pd.DataFrame(columns=['Order', 'AIC', 'BIC'])
    init_time = time.time()

    for p in p_params:
        mae_grid[p] = list()
        for q in q_params:
            order = (p, 0, q)
            start_time = time.time()
            model = ARIMA(ytrain, order=order).fit()
            value_dict = {'Order': [order],
                          'AIC': [model.aic],
                          'BIC': [model.bic]}
            new_row = pd.DataFrame(value_dict, columns=['Order', 'AIC', 'BIC'])
            aicbic_df = pd.concat([aicbic_df, new_row], ignore_index=True)
            elapsed_time = round(time.time() - start_time, 2)
            print(f"Trained ARIMA {order} in {elapsed_time} seconds.")
            y_pred = model.predict()
            print(y_pred.isnull().sum().sum(), "null values in predictions")
            mae = mean_absolute_error(ytrain, y_pred)
            mae_grid[p].append(mae)

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