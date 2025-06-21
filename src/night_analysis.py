import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from tsfresh import extract_features
from tsfresh.feature_extraction import ComprehensiveFCParameters, \
    EfficientFCParameters, MinimalFCParameters
from tsfresh.utilities.dataframe_functions import impute

from src.helper import check_df_index
from src.config import FIGURES_DIR


class NightAnalyser:
    def __init__(self, df, feature_settings='custom'):
        """
        Initialises the NightAnalyser with the preprocessed time series data. It is assumed the DataFrame has a MultiIndex with 'id' and 'datetime', of night periods with consistent and complete intervals between a consistent start and end time.
        :param df: Pandas DataFrame containing the time series data.
        :param feature_settings: str, 'comprehensive', 'efficient', 'minimal', or 'custom'. Defines tsfresh feature extraction settings.
        """
        df = check_df_index(df)  # Ensure the DataFrame has a MultiIndex with 'id' and 'datetime'

        self.df = df.copy()
        self.variable_cols = [col for col in df.columns if col not in ['id', 'datetime']]
        print(self.variable_cols)
        self.customised_feature_dict = None
        self.feature_settings = self._get_feature_settings(feature_settings)
        self.night_start_hour = 17  # Default night start hour
        self.night_features_df = None
        self.scaled_night_features = None
        self.night_pca_components = None
        self.night_clusters = None
        self.rolling_features_df = None
        self.silhouette_score = None


        # Store scalers and PCA models
        self.scaler = None
        self.pca_model = None

    def _get_feature_settings(self, setting_name):
        """Helper to get tsfresh feature extraction settings."""
        if setting_name == 'comprehensive':
            return ComprehensiveFCParameters()
        elif setting_name == 'efficient':
            return EfficientFCParameters()
        elif setting_name == 'minimal':
            return MinimalFCParameters()
        elif setting_name == 'custom':
            self.customised_feature_dict = {
                'iob mean': {
                    'mean': None,
                    'variance': None,
                    'maximum': None,
                    'minimum': None,
                    'median': None,
                    'standard_deviation': None,
                    'root_mean_square': None,
                    'first_location_of_maximum': None,
                    'last_location_of_maximum': None,
                    'first_location_of_minimum': None,
                    'last_location_of_minimum': None,
                    'sample_entropy': None,
                    'longest_strike_below_mean': None,
                    'longest_strike_above_mean': None
                },
                'cob mean': {
                    'mean': None,
                    'variance': None,
                    'maximum': None,
                    'minimum': None,
                    'median': None,
                    'standard_deviation': None,
                    'root_mean_square': None,
                    'first_location_of_maximum': None,
                    'last_location_of_maximum': None,
                    'first_location_of_minimum': None,
                    'last_location_of_minimum': None,
                    'sample_entropy': None,
                    'longest_strike_below_mean': None,
                    'longest_strike_above_mean': None
                },
                'bg mean': {
                    'mean': None,
                    'variance': None,
                    'maximum': None,
                    'minimum': None,
                    'median': None,
                    'standard_deviation': None,
                    'root_mean_square': None,
                    'count_above_mean': None,
                    'count_below_mean': None,
                    'count_above': [{'t': 100}],
                    'count_below': [{'t': 50}],
                    'change_quantiles': [{'ql': 0.2, 'qh': 0.8, 'isabs': True, 'f_agg': 'mean'}],
                    'first_location_of_maximum': None,
                    'last_location_of_maximum': None,
                    'first_location_of_minimum': None,
                    'last_location_of_minimum': None,
                    'sample_entropy': None,
                    'longest_strike_below_mean': None,
                    'longest_strike_above_mean': None,
                    'mean_abs_change': None,
                },
                'cob max': {
                    'count_above': [{'t': 100}],
                    'maximum': None,
                    'minimum': None,
                    'median': None,
                    'standard_deviation': None,
                    'root_mean_square': None,
                    'first_location_of_maximum': None,
                    'last_location_of_maximum': None,
                    'first_location_of_minimum': None,
                    'last_location_of_minimum': None,
                },
                'iob max': {
                    'count_above': [{'t': 100}],
                    'maximum': None,
                    'minimum': None,
                    'median': None,
                    'standard_deviation': None,
                    'root_mean_square': None,
                    'first_location_of_maximum': None,
                    'last_location_of_maximum': None,
                    'first_location_of_minimum': None,
                    'last_location_of_minimum': None,
                }
            }
            return None
        else:
            raise ValueError("Invalid feature_settings. Choose 'comprehensive', 'efficient', 'minimal', or 'custom'.")

    def extract_night_level_features(self, night_start_hour=19):
        """
        Extracts aggregated tsfresh features for each complete night period.
        The MultiIndex needs a unique 'night_id' for each night (e.g., id + night start date).
        :param night_start_hour: (int), hour at which the night period starts (e.g., 19 for 19:00).
        """
        print(f"Extracting night-level features using {self.feature_settings.__class__.__name__} settings...")
        temp_df = self.df.reset_index()
        hour = temp_df['datetime'].dt.hour
        # Assign night to previous date if before night_start_hour
        night_start_date = temp_df['datetime'].dt.date - pd.to_timedelta((hour < night_start_hour).astype(int), unit='D')
        temp_df['night_id'] = temp_df['id'].astype(str) + '_' + night_start_date.astype(str)
        column_kind, value_name = None, None
        if self.feature_settings is None:  # And therefore using custom features
            column_kind = 'kind'
            value_name = 'value'
            temp_df = temp_df.melt(id_vars=['night_id', 'id', 'datetime'], value_vars=self.variable_cols, var_name=column_kind, value_name=value_name)
        self.night_features_df = extract_features(
            temp_df.drop(columns='id'),
            column_id='night_id',
            column_sort='datetime',
            column_kind=column_kind,
            column_value=value_name,
            kind_to_fc_parameters=self.customised_feature_dict,
            default_fc_parameters=self.feature_settings,
            impute_function=impute,
            show_warnings=True
            )

        print(f"Extracted {self.night_features_df.shape[1]} features for {self.night_features_df.shape[0]} nights.")
        return self.night_features_df

    def preprocess_night_features(self, n_components=0.95):
        """
        Scales features and applies PCA for dimensionality reduction.
        :param n_components: (float or int), number of PCA components or variance explained (0-1.0).
        """
        if self.night_features_df is None:
            raise ValueError("Night features not extracted yet. Run extract_night_level_features first.")

        print("Preprocessing night-level features (scaling and PCA)...")

        # Handle NaNs from tsfresh. Use impute again to catch any new NaNs from feature extraction.
        X_imputed = impute(self.night_features_df.copy())

        # Drop columns with zero variance after imputation (can cause issues with StandardScaler)
        X_imputed = X_imputed.loc[:, X_imputed.var() != 0]

        self.scaler = StandardScaler()
        self.scaled_night_features = self.scaler.fit_transform(X_imputed)
        self.scaled_night_features = pd.DataFrame(
            self.scaled_night_features,
            columns=X_imputed.columns,
            index=X_imputed.index
        )

        if n_components is not None:
            self.pca_model = PCA(n_components=n_components)
            self.night_pca_components = self.pca_model.fit_transform(self.scaled_night_features)
            print(f"PCA reduced dimensions from {self.scaled_night_features.shape[1]} to {self.night_pca_components.shape[1]}.")
            return self.night_pca_components
        else:
            return self.scaled_night_features

    def plot_pca_cumulative_variance(self):
        """ Plots the cumulative explained variance ratio from PCA. """
        plt.plot(np.cumsum(self.pca_model.explained_variance_ratio_))
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance')
        plt.show()

    def cluster_nights(self, n_clusters: int=3, print_clusters: bool=True,
                       plot_2d: bool=True):
        """
        Clusters the nights using K-Means.
        :param print_clusters: bool, Whether to print cluster distribution.
        :param n_clusters: (int), Number of clusters for K-Means.
        :param plot_2d: (bool), Whether to plot 2D PCA for clusters.
        """
        if (self.night_pca_components is None and
                self.scaled_night_features is None):
            raise ValueError("Features not preprocessed yet. Run p"
                             "reprocess_night_features first.")

        data_for_clustering = self.night_pca_components if self.night_pca_components is not None else self.scaled_night_features.values
        if data_for_clustering.shape[0] < n_clusters:
             raise ValueError(
                 f"Number of nights ({data_for_clustering.shape[0]}) is less "
                 f"than n_clusters ({n_clusters}). Cannot cluster.")

        print(f"Clustering nights into {n_clusters} clusters...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10) # n_init for robustness
        self.night_clusters = kmeans.fit_predict(data_for_clustering)

        self.night_features_df['cluster_label'] = self.night_clusters
        if print_clusters:
            print("Night cluster distribution:")
            print(self.night_features_df['cluster_label'].value_counts())

        self.silhouette_score = silhouette_score(data_for_clustering,
                                                 self.night_clusters)

        if (plot_2d and self.night_pca_components is not None
                and self.night_pca_components.shape[1] >= 2):
            id_list = (pd.Categorical(self.reindex_night_features()['id']).
                       codes.astype(int) + 1000)

            # Build DataFrame for plotting
            plot_df = pd.DataFrame({
                'PC1': self.night_pca_components[:, 0],
                'PC2': self.night_pca_components[:, 1],
                'cluster_label': self.night_clusters,
                'id': id_list
            })

            plt.figure(figsize=(8, 6))
            ax = sns.scatterplot(
                data=plot_df,
                x='PC1',
                y='PC2',
                hue='cluster_label',
                style='id',
                palette='viridis',
                alpha=0.7
            )

            # Remove the default legend
            ax.legend_.remove()

            # Cluster legend
            handles1, labels1 = ax.get_legend_handles_labels()
            unique_clusters = plot_df['cluster_label'].unique()
            cluster_handles = []
            cluster_labels = []
            for c in sorted(unique_clusters):
                idx = labels1.index(str(c)) if str(
                    c) in labels1 else labels1.index(int(c))
                cluster_handles.append(handles1[idx])
                cluster_labels.append(f'Cluster {c}')
            legend1 = ax.legend(cluster_handles, cluster_labels,
                                title='Cluster', loc='upper right')
            ax.add_artist(legend1)

            # ID legend
            unique_ids = plot_df['id'].unique()
            id_handles = []
            id_labels = []
            for i in unique_ids:
                idx = labels1.index(str(i)) if str(
                    i) in labels1 else labels1.index(int(i))
                id_handles.append(handles1[idx])
                id_labels.append(f'ID {i}')
            legend2 = ax.legend(id_handles, id_labels, title='ID',
                                loc='lower right')

            plt.title('Nights Clustered (KMeans)')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.savefig(FIGURES_DIR / 'clustered_nights_pca.png', dpi=400,
                        bbox_inches='tight')
            plt.show()
        elif plot_2d and (self.night_pca_components is None or self.night_pca_components.shape[1] < 2):
            print("Cannot plot 2D PCA: PCA not performed or less than 2 components.")

        return self.night_clusters

    def get_cluster_centroids(self):
        """Returns the mean feature values for each cluster (in original feature space)."""
        if self.night_clusters is None:
            raise ValueError("Nights not clustered yet. Run cluster_nights first.")

        # Inverse transform scaled features before averaging for interpretability
        original_features_df = pd.DataFrame(
            self.scaler.inverse_transform(self.scaled_night_features),
            columns=self.scaled_night_features.columns,
            index=self.scaled_night_features.index
        )
        original_features_df['cluster_label'] = self.night_clusters

        return original_features_df.groupby('cluster_label').mean()


    def silhouette_analysis(self, cluster_range: range) -> list:
        """
        Perform silhouette analysis for a range of cluster numbers.
        :param cluster_range: Range of cluster numbers to evaluate.
        :return: List of silhouette scores for each cluster number.
        """
        silhouette_scores = []
        for n_clusters in cluster_range:
            self.cluster_nights(n_clusters=n_clusters,
                                print_clusters=False,
                                plot_2d=False)
            silhouette_scores.append(self.silhouette_score)

        plt.plot(cluster_range, silhouette_scores)
        plt.title('Silhouette Analysis')
        plt.xlabel('Number of clusters')
        plt.ylabel('Silhouette Score')
        plt.xticks(cluster_range)
        plt.show()

        return silhouette_scores

    def reindex_night_features(self):
        """
        Reindexes the night features DataFrame to ensure it has a MultiIndex
        with 'id' and 'date'.
        :return: DataFrame with MultiIndex ['id', 'date'].
        """
        if self.night_features_df is None:
            raise ValueError("Night features not extracted yet. "
                             "Run extract_night_level_features first.")

        temp_df = self.night_features_df.reset_index().copy()
        temp_df[['id', 'date']] = temp_df['index'].str.split('_', expand=True)
        temp_df['date'] = pd.to_datetime(temp_df['date'])
        temp_df['id'] = temp_df['id'].astype(int)
        temp_df.drop('index', axis=1).set_index(['id', 'date'], inplace=True)

        return temp_df

    def return_dataset_with_clusters(self, df: pd.DataFrame = None):
        """
        Returns the original dataset with the cluster labels added.
        :param df: (pd.DataFrame), Optional DataFrame to use instead of self.df.
                   If None, uses the original DataFrame passed during
                   initialisation. DataFrame should have a MultiIndex ['id',
                   'datetime'].
        :return: DataFrame with original features and cluster labels, plus
            'night_start_date' indicating the start of the night period.
        """
        if df is None:
            df = self.df.copy()

        if self.night_features_df is None:
            raise ValueError("Night features not extracted yet. Run "
                             "extract_night_level_features first.")

        temp = self.reindex_night_features()
        temp = (temp[['id', 'date', 'cluster_label']].
                rename(columns={'date': 'night_start_date'}).
                set_index(['id', 'night_start_date'])
                )

        df_clustered = df.reset_index().copy()
        hour = df_clustered['datetime'].dt.hour
        df_clustered['night_start_date'] = (
                    df_clustered['datetime'] - pd.to_timedelta(
                (hour < self.night_start_hour).astype(int), unit='D')).dt.floor(
            'D')  # to enable alignment of dtypes as datetime64[ns] as a date
        # print(df_clustered.dtypes)
        df_clustered = df_clustered.reset_index().set_index(
            ['id', 'night_start_date'])

        df_clustered = df_clustered.join(temp)

        return df_clustered.reset_index().set_index(['id', 'datetime'])

    def visualise_night_features(self, feature_name, cluster_label=None):
        """
        Visualises the distribution of a specific night feature.
        :param feature_name: (str), Name of the feature to visualize.
        :param cluster_label: (int or None), If provided, filters by cluster label.
        """
        if self.night_features_df is None:
            raise ValueError("Night features not extracted yet. Run "
                             "extract_night_level_features first.")

        if feature_name not in self.night_features_df.columns:
            raise ValueError(f"Feature '{feature_name}' not found in night features.")

        data = self.night_features_df.copy()
        if cluster_label is not None:
            data = data[data['cluster_label'] == cluster_label]

        plt.figure(figsize=(10, 6))
        sns.boxplot(x='cluster_label', y=feature_name, data=data)
        plt.title(f'Distribution of {feature_name} by Cluster')
        plt.xlabel('Cluster Label')
        plt.ylabel(feature_name)
        plt.savefig(FIGURES_DIR / f'night_feature_{feature_name}.png', dpi=400, bbox_inches='tight')
        plt.show()

    def heatmap_cluster_features(self):
        """
        Visualizes the mean features for each cluster as a heatmap.
        :param cluster_label: (int or None), If provided, filters by cluster label.
        """
        if self.night_features_df is None:
            raise ValueError("Night features not extracted yet. Run "
                             "extract_night_level_features first.")

        heatmap_data = self.scaled_night_features.copy()
        heatmap_data['cluster_label'] = self.night_clusters
        heatmap_data = heatmap_data.groupby('cluster_label').mean().T
        plt.figure(figsize=(6, 14))
        sns.heatmap(heatmap_data, annot=True, cmap='viridis')
        plt.title('Cluster Centroids: Feature Means')
        plt.xlabel('Cluster', verticalalignment='top')
        plt.ylabel('Feature')
        plt.show()