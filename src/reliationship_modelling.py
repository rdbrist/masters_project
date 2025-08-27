import itertools

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from statsmodels.formula.api import ols
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from src.helper import check_df_index


class AnalyseRelationships:
    """
    Class to analyse relationships between different variables in a dataset.
    """

    def __init__(self, df, train_test_split=0.8):
        """
        Initialise with the dataset, which introduces the lags needed for analysis.

        :param df: (pd.DataFrame) The dataset with the COB, IOB and Bg means.
            DataFrame should have ['id', 'datetime'] index and ['cob mean',
            'iob mean', 'bg mean'] columns, plus 'night_start_date' and
            'cluster' columns
        :param train_test_split: (float) The proportion of each night split.
        """
        if df.empty:
            raise ValueError("The provided DataFrame is empty.")
        df = check_df_index(df)

        self.df = df.sort_index()
        self.lag_range = range(0, 5)
        self.variables = ['cob mean', 'iob mean', 'bg mean']
        self.scaled_cols = ['cob_mean_scaled', 'iob_mean_scaled', 'bg_mean_scaled']

        minmax = MinMaxScaler(feature_range=(0, 1))
        self.df[self.scaled_cols] = minmax.fit_transform(self.df[self.variables])

        self.processed_df = pd.DataFrame()

        for lag in self.lag_range:
            for (id_, night_start_date, cluster), df_night in self.df.groupby(['id', 'night_start_date', 'cluster']):
                df_lagged = df_night[self.scaled_cols].reset_index(level='id', drop=True)
                df_lagged['cob_lagged'] = df_lagged['cob_mean_scaled'].shift(lag)
                df_lagged['iob_lagged'] = df_lagged['iob_mean_scaled'].shift(lag)
                df_lagged = df_lagged.dropna(subset=['cob_lagged', 'iob_lagged', 'bg_mean_scaled'])
                df_lagged[['id', 'night_start_date', 'cluster', 'lag']] = (id_, night_start_date, cluster, lag)
                self.processed_df = pd.concat([self.processed_df, df_lagged])

        self.train_test_sets = self._train_test_split(train_test_split)

    def _train_test_split(self, train_test_split=0.8):
        """
        Split each night into training and testing sets based on the ratio given in the parameter,
        and return a dictionary with each night's training and testing sets.

        :param train_test_split: (float) The ratio of training to testing data. Default is 0.8.
        :return: (dict) A dictionary with keys as tuples of (id, night_start_date, cluster, lag)
                 and values as tuples of (train_df, test_df).
        """
        train_test_dict = {}
        for (id_, night_start_date, cluster, lag), df_night in self.processed_df.groupby(['id', 'night_start_date', 'cluster', 'lag']):
            split_index = int(len(df_night) * train_test_split)
            train_df = df_night.iloc[:split_index]
            test_df = df_night.iloc[split_index:]
            train_test_dict[(id_, night_start_date, cluster, lag)] = (train_df, test_df)
        return train_test_dict

    def _get_data_iter(self, split):
        """
        Internal helper to yield (key, (train_df, test_df)) pairs for either split or full data.

        :param split: (bool) If True, use train/test splits; if False, use full data for both train and test.
        :return: generator yielding (key, (train_df, test_df))
        """
        if split:
            return self.train_test_sets.items()
        else:
            return ((k, (df, df)) for k, df in self.processed_df.groupby(['id', 'night_start_date', 'cluster', 'lag']))

    def apply_ols(self, variables=['cob_lagged', 'iob_lagged'], split=False):
        """
        Apply OLS regression to the processed DataFrame.

        :param variables: (list) List of variable names to use in the regression.
        :param split: (bool) If True, fit on train and predict on test; if False, fit to whole data for each night.
        :return: (list) List of dictionaries with regression results for each night.
        """
        ols_str = " + ".join(variables)
        results = []
        for (id_, night_start_date, cluster, lag), (train_df, test_df) in (
                self._get_data_iter(split)):
            model = ols(f'bg_mean_scaled ~ {ols_str}',
                        data=train_df).fit()
            y_test = test_df['bg_mean_scaled'].values
            y_pred = model.predict(test_df)
            mse = ((y_pred - y_test) ** 2).mean()
            results.append({
                'id': id_,
                'night_start_date': night_start_date,
                'cluster': cluster,
                'lag': lag,
                'mse': mse,
                'r_squared': model.rsquared,
                **{f'params_{var.split("_")[0]}': model.params[var] for var in variables},
                **{f'pvalue_{var.split("_")[0]}': model.pvalues[var] for var in variables}
            })
        return results

    def apply_cv_regression(self, model: str = None,
                            variables=['cob_lagged', 'iob_lagged'], n_splits=5,
                            **model_kwargs):
        """
        Apply K-fold cross-validation regression to each night series.

        :param model: (str) The regression model (e.g., SVR,
            RandomForestRegressor).
        :param variables: (list) List of variable names to use in the regression.
        :param n_splits: (int) Number of cross-validation folds.
        :param model_kwargs: Additional keyword arguments for the model.
        :return: (list) List of dictionaries with mean metrics for each night.
        """

        if model == 'SVR':
            model_class = SVR
        elif model == 'RandomForestRegressor':
            model_class = RandomForestRegressor
        elif model == 'DecisionTreeRegressor':
            model_class = DecisionTreeRegressor
        elif model == 'LinearRegression':
            model_class = LinearRegression
        else:
            raise ValueError("Model class must be provided and a valid option"
                             " for cross-validation.")

        results = []
        for (id_, night_start_date, cluster, lag), df_night in (
                self.processed_df.groupby(['id', 'night_start_date',
                                           'cluster', 'lag'])):
            X = df_night[variables].values
            y = df_night['bg_mean_scaled'].values
            if len(df_night) < n_splits:
                continue  # Not enough samples for CV
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            mses, r2s = [], []
            for train_idx, test_idx in kf.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                model = model_class(**model_kwargs)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                mses.append(((y_pred - y_test) ** 2).mean())
                r2s.append(model.score(X_test, y_test))
            results.append({
                'id': id_,
                'night_start_date': night_start_date,
                'cluster': cluster,
                'lag': lag,
                'mse': np.mean(mses),
                'r_squared': np.mean(r2s)
            })
        return results

    def get_processed_df(self):
        """
        Get the lagged sets of the dataset.
        :return: (pd.DataFrame) The processed DataFrame with lagged variables.
        """
        return self.processed_df

    def calculate_correlation(self, corr_type='spearman'):
        variables = ['bg_mean_scaled', 'cob_lagged', 'iob_lagged']
        df_corr = pd.DataFrame(
            columns=['lag', 'cluster', 'variables', 'time', 'correlation'])
        df_corr = df_corr.astype(
            {'lag': 'int', 'cluster': 'int', 'variables': 'str',
             # 'time': 'str',
             'correlation': 'float'})
        for lag, df in self.processed_df.groupby('lag'):
            for v1, v2 in itertools.combinations(variables, 2):
                df_corr_temp =(
                    AnalyseRelationships._correlation_by_time(df, v1, v2,
                                                              corr_type))
                df_corr_temp['lag'] = lag
                df_corr_temp['variables'] = f'{v1} vs {v2}'
                df_corr = pd.concat([df_corr, df_corr_temp])
        return df_corr

    @staticmethod
    def _correlation_by_time(df, col1, col2, corr_type):
        results = []
        for cluster, group in df.reset_index().groupby('cluster'):
            group['time'] = group['datetime'].dt.time
            for time, time_group in group.groupby('time'):
                correlation = time_group[col1].corr(time_group[col2],
                                                    method=corr_type)
                results.append({'cluster': cluster, 'time': time,
                                'correlation': correlation})
        return pd.DataFrame(results)


