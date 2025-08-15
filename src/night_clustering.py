import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from tsfresh import extract_features
from tsfresh.feature_extraction import ComprehensiveFCParameters, \
    EfficientFCParameters, MinimalFCParameters
from tsfresh.utilities.dataframe_functions import impute
from umap import UMAP

from src.helper import check_df_index, get_night_start_date, cluster_colours
from src.config import FIGURES_DIR


class NightClustering:
    """
    Class uses feature-based clustering to identify patterns in night periods,
    using the tsfresh library for feature extraction.
    """
    def __init__(self, df, feature_settings='custom', night_start_hour=None,
                 feature_variables=None):
        """
        Initialises the NightClustering with the preprocessed time series data.
        It is assumed the DataFrame has a MultiIndex with 'id' and 'datetime',
        of night periods with consistent and complete intervals between a
        consistent start and end time.
        :param df: Pandas DataFrame containing the time series data.
        :param feature_settings: str, 'comprehensive', 'efficient', 'minimal',
            or 'custom'. Defines tsfresh feature extraction settings.
        """
        df = check_df_index(df)

        self.df = df.sort_index().copy()
        if feature_variables:
            self.variable_cols = feature_variables
        else:
            self.variable_cols = [col for col in df.columns if col not in
                                  ['id', 'datetime']]
        if df[self.variable_cols].isna().sum().sum() != 0:
            raise ValueError("Input DataFrame contains NaN values. "
                             "Please handle missing data before clustering.")
        self.customised_feature_dict = None
        self.feature_settings = self._get_feature_settings(feature_settings)
        self.night_start_hour = night_start_hour  # Default night start hour
        self.night_features_df = None
        self.scaled_night_features = None
        self.night_pca_components = None
        self.night_clusters = None
        self.tsne_clusters = None
        self.tsne_results = None
        self.umap_clusters = None
        self.umap_results = None
        self.rolling_features_df = None
        self.silhouette_score = None

        # Store scalers and PCA models
        self.scaler = None
        self.pca_model = None

        self.cluster_color_map = cluster_colours()

    def get_unique_ids(self):
        """
        Returns a list of unique IDs from the DataFrame.
        :return: List of unique IDs.
        """
        return self.df.index.get_level_values('id').unique().tolist()

    def get_summary_statistics(self):
        """
        Returns summary statistics for the DataFrame.
        :return: DataFrame with summary statistics.
        """
        df = self.df.copy().reset_index()
        df['night_start_date'] = (
            get_night_start_date(df['datetime'], self.night_start_hour))
        return (df[['id', 'night_start_date']].
                drop_duplicates().groupby('id').size().
                reset_index(name='nights')).set_index('id')

    def _get_feature_settings(self, setting_name):
        """Helper to get tsfresh feature extraction settings."""
        cob_iob_features = {
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
            }
        }
        bg_features = {
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
                'change_quantiles': [{'ql': 0.2, 'qh': 0.8, 'isabs': True,
                                      'f_agg': 'mean'}],
                'first_location_of_maximum': None,
                'last_location_of_maximum': None,
                'first_location_of_minimum': None,
                'last_location_of_minimum': None,
                'sample_entropy': None,
                'longest_strike_below_mean': None,
                'longest_strike_above_mean': None,
                'mean_abs_change': None,
            },
            'bg max': {
                'maximum': None,
                'minimum': None,
                'median': None,
                'standard_deviation': None,
                'root_mean_square': None,
            },
            'bg min': {
                'maximum': None,
                'minimum': None,
                'median': None,
                'standard_deviation': None,
                'root_mean_square': None,
            },
        }
        if setting_name == 'comprehensive':
            return ComprehensiveFCParameters()
        elif setting_name == 'efficient':
            return EfficientFCParameters()
        elif setting_name == 'minimal':
            return MinimalFCParameters()
        elif setting_name == 'custom':
            self.customised_feature_dict = cob_iob_features | bg_features
        elif setting_name == 'custom_no_bg':
            self.customised_feature_dict = cob_iob_features
            return None
        else:
            raise ValueError("Invalid feature_settings. Choose "
                             "'comprehensive', 'efficient', 'minimal', "
                             "'custom' or 'custom_no_bg.")

    def extract_night_level_features(self, multi_threaded=True):
        """
        Extracts aggregated tsfresh features for each complete night period.
        The MultiIndex needs a unique 'night_id' for each night (e.g., id +
        night start date).
        :param multi_threaded: (bool), If True, uses multi-threading for feature

        """
        temp_df = self.df[self.variable_cols].reset_index()
        night_start_date = (
            get_night_start_date(temp_df['datetime'], self.night_start_hour))
        temp_df['night_id'] = (
                temp_df['id'].astype(str) + '_' + night_start_date.astype(str))
        column_kind, value_name = None, None
        if self.feature_settings is None:  # And therefore using custom features
            column_kind = 'kind'
            value_name = 'value'
            temp_df = temp_df.melt(id_vars=['night_id', 'id', 'datetime'],
                                   value_vars=self.variable_cols,
                                   var_name=column_kind,
                                   value_name=value_name)
        temp_df_sorted = (
            temp_df.sort_values(['night_id', 'datetime']).drop(columns='id'))
        n_jobs = 1 if multi_threaded else 0
        self.night_features_df = extract_features(
            temp_df_sorted,
            column_id='night_id',
            column_sort='datetime',
            column_kind=column_kind,
            column_value=value_name,
            kind_to_fc_parameters=self.customised_feature_dict,
            default_fc_parameters=self.feature_settings,
            impute_function=impute,
            show_warnings=False,
            n_jobs=n_jobs
            )

        print(f"Extracted {self.night_features_df.shape[1]} features for "
              f"{self.night_features_df.shape[0]} nights.")
        return self.night_features_df

    def preprocess_night_features(self, n_components=0.95):
        """
        Scales features and applies PCA for dimensionality reduction.
        :param n_components: (float or int), number of PCA components or
        variance explained (0-1.0).
        """
        if self.night_features_df is None:
            raise ValueError("Night features not extracted yet. Run "
                             "extract_night_level_features first.")

        print("Preprocessing night-level features (scaling and PCA)...")

        # Handle NaNs from tsfresh features
        X_imputed = impute(self.night_features_df.copy())

        # Drop columns with zero variance to avoid scaling issues
        X_imputed = X_imputed.loc[:, X_imputed.var() != 0]

        self.scaler = StandardScaler()
        unscaled_cols = set(self.night_features_df.columns)
        self.scaled_night_features = self.scaler.fit_transform(X_imputed)
        self.scaled_night_features = pd.DataFrame(
            self.scaled_night_features,
            columns=X_imputed.columns,
            index=X_imputed.index
        )
        scaled_cols = set(self.scaled_night_features.columns)
        if len(unscaled_cols.difference(scaled_cols)) > 10:
            print(f"Dropped {len(unscaled_cols.difference(scaled_cols))} "
                  f"features.")
        else:
            print(f"Dropped features from scaling: "
                  f"{unscaled_cols.difference(scaled_cols)}")

        if n_components is not None:
            self.pca_model = PCA(n_components=n_components)
            self.night_pca_components = (
                self.pca_model.fit_transform(self.scaled_night_features))
            print(f"PCA reduced dimensions from "
                  f"{self.scaled_night_features.shape[1]} to "
                  f"{self.night_pca_components.shape[1]}.")
            return self.night_pca_components
        else:
            return self.scaled_night_features

    def plot_pca_cumulative_variance(self):
        """ Plots the cumulative explained variance ratio from PCA. """
        fig = plt.figure(figsize=(5, 4))
        plt.plot(np.cumsum(self.pca_model.explained_variance_ratio_))
        plt.title('Cumulative Explained Variance by PCA Components')
        plt.xlabel('Number of components')
        plt.ylabel('Cumulative explained variance')
        plt.show()

    def cluster_nights(self, n_clusters: int = 3, print_clusters: bool = True,
                       plot_2d: bool = True):
        """
        Clusters the nights using K-Means.
        :param print_clusters: (bool), Whether to print cluster distribution.
        :param n_clusters: (int), Number of clusters for K-Means.
        :param plot_2d: (bool), Whether to plot 2D PCA for clusters.
        """
        if (self.night_pca_components is None and
                self.scaled_night_features is None):
            raise ValueError("Features not preprocessed yet. Run "
                             "preprocess_night_features first.")

        data_for_clustering = (
            self.night_pca_components if self.night_pca_components is not None
            else self.scaled_night_features.values)
        if data_for_clustering.shape[0] < n_clusters:
            raise ValueError(
                f"Number of nights ({data_for_clustering.shape[0]}) is less "
                f"than n_clusters ({n_clusters}). Cannot cluster.")

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.night_clusters = kmeans.fit_predict(data_for_clustering)

        if print_clusters:
            print("Night cluster distribution:")
            print(np.unique(self.night_clusters, return_counts=True))

        self.silhouette_score = silhouette_score(data_for_clustering,
                                                 self.night_clusters)

        if (plot_2d and self.night_pca_components is not None
                and self.night_pca_components.shape[1] >= 2):
            id_list = self.reindex_night_features().reset_index()['id']
            # Build DataFrame for plotting
            plot_df = pd.DataFrame({
                'PC1': self.night_pca_components[:, 0],
                'PC2': self.night_pca_components[:, 1],
                'cluster_label': self.night_clusters,
                'id': id_list
            })

            plt.figure(figsize=(6, 5))
            sns.scatterplot(data=plot_df, x='PC1', y='PC2', hue='cluster_label',
                            palette=self.cluster_color_map, alpha=0.7)

            plt.title('Nights Clustered (KMeans)')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.legend(title='Cluster')
            plt.savefig(FIGURES_DIR /
                f'clustered_nights_pca_{n_clusters}clusters.png',
                dpi=400, bbox_inches='tight')
            plt.show()
        elif (plot_2d and
              (self.night_pca_components is None
               or self.night_pca_components.shape[1] < 2)):
            print("Cannot plot 2D PCA: PCA not performed or less than 2 "
                  "components.")

        return self.night_clusters

    def get_cluster_centroids(self):
        """
        Returns the mean feature values for each cluster (in original feature
        space).
        """
        if self.night_clusters is None:
            raise ValueError("Nights not clustered yet. Run cluster_nights "
                             "first.")

        # Inverse scaled features before averaging for interpretability
        original_features_df = pd.DataFrame(
            self.scaler.inverse_transform(self.scaled_night_features),
            columns=self.scaled_night_features.columns,
            index=self.scaled_night_features.index
        )
        original_features_df['cluster_label'] = self.night_clusters

        return original_features_df.groupby('cluster_label').mean()

    def silhouette_analysis(self, cluster_range: range,
                            cluster_type: str = 'kmeans',
                            plot_results: bool = True,
                            **kwargs) -> list:
        """
        Perform silhouette analysis for a range of cluster numbers.
        :param cluster_range: (range) Range of cluster numbers to evaluate
        :param cluster_type: (str), Type of clustering to use for silhouette
            analysis, currently only 'kmeans' is supported
        :param plot_results: (bool) Whether to plot the silhouette scores
        :return: List of silhouette scores for each cluster number
        """
        silhouette_scores = []
        for n_clusters in cluster_range:
            if cluster_type == 'kmeans':
                self.cluster_nights(n_clusters=n_clusters,
                                    print_clusters=False,
                                    plot_2d=False)
            elif cluster_type == 'tsne':
                self.fit_tsne(**kwargs)
                self.clustering_tsne(n_clusters=n_clusters)
            elif cluster_type == 'umap':
                self.fit_umap(**kwargs)
                self.clustering_umap(n_clusters=n_clusters)
            silhouette_scores.append(self.silhouette_score)

        if plot_results:
            fig = plt.figure(figsize=(5, 4))
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

        df = self.night_features_df.reset_index().copy()
        df[['id', 'date']] = df['index'].str.split('_', expand=True)
        df['date'] = pd.to_datetime(df['date'])
        df['id'] = df['id'].astype(int)
        df = df.drop('index', axis=1).set_index(['id', 'date'])
        return df

    def return_dataset_with_clusters(self, df: pd.DataFrame = None,
                                     scaled: bool = False):
        """
        Returns the original dataset with the cluster labels added. Adds
        clusters from both the original KMeans clustering and the t-SNE if
        this has been performed, and also the start date of the night.
        :param df: (pd.DataFrame), Optional DataFrame to use instead of self.df.
                   If None, uses the original DataFrame passed during
                   initialisation. DataFrame should have a MultiIndex ['id',
                   'datetime'].
        :param scaled: (bool), If True, returns the scaled night features
        :return: DataFrame with original features and cluster labels, plus
            'night_start_date' indicating the start of the night period.
        """
        if df is None:
            df = self.df.copy()

        if self.night_features_df is None:
            raise ValueError("Night features not extracted yet. Run "
                             "extract_night_level_features first.")

        temp = self.reindex_night_features().reset_index()
        cols = ['id', 'date', 'cluster_label']
        temp['cluster_label'] = self.night_clusters

        if self.tsne_clusters is not None:
            temp['tsne_cluster_label'] = self.tsne_clusters
            cols.append('tsne_cluster_label')
        # The date in the feature ID is the night start date
        temp = (temp[cols].rename(columns={'date': 'night_start_date'}).
                set_index(['id', 'night_start_date']))

        new_df = df.reset_index().copy()
        new_df['night_start_date'] = pd.to_datetime(
            get_night_start_date(new_df['datetime'], self.night_start_hour))
        new_df = new_df.reset_index().set_index(['id', 'night_start_date'])
        if scaled:
            float_cols = [col for col in new_df.columns if
                          pd.api.types.is_float_dtype(new_df[col])]
            new_df[float_cols] = StandardScaler().fit_transform(
                new_df[float_cols].values)
        new_df = new_df.join(temp)
        return new_df.reset_index().set_index(['id', 'datetime'])

    def visualise_night_features(self, feature_name, cluster_label=None):
        """
        Visualises the distribution of a specific night feature.
        :param feature_name: (str), Name of the feature to visualise.
        :param cluster_label: (int or None), If provided, filters by cluster
            label.
        """
        if self.night_features_df is None:
            raise ValueError("Night features not extracted yet. Run "
                             "extract_night_level_features first.")

        if feature_name not in self.night_features_df.columns:
            raise ValueError(f"Feature '{feature_name}' not found in night "
                             f"features.")

        data = self.night_features_df.copy()
        if cluster_label is not None:
            data = data[data['cluster_label'] == cluster_label]

        plt.figure(figsize=(10, 6))
        sns.boxplot(x='cluster_label', y=feature_name, data=data)
        plt.title(f'Distribution of {feature_name} by Cluster')
        plt.xlabel('Cluster Label')
        plt.ylabel(feature_name)
        plt.savefig(FIGURES_DIR / f'night_feature_{feature_name}.png',
                    dpi=400, bbox_inches='tight')
        plt.show()

    def heatmap_cluster_features(self, cluster_type='kmeans'):
        """
        Visualizes the mean features for each cluster as a heatmap.
        The heatmap color is based on scaled features, but the annotation
        overlays show the unscaled feature means.
        :param cluster_type: (str) Defines which clusters to use
        """
        if self.night_features_df is None:
            raise ValueError("Night features not extracted yet. Run "
                             "extract_night_level_features first.")
        if cluster_type == 'kmeans':
            clusters = self.night_clusters
        elif cluster_type == 'tsne':
            clusters = self.tsne_clusters
        elif cluster_type == 'umap':
            clusters = self.umap_clusters
        # clusters = (self.night_clusters if cluster_type == 'kmeans'
        #             elif self.tsne_clusters)
        # Scaled features for heatmap color
        scaled = self.scaled_night_features.copy()
        scaled_cols = scaled.columns
        scaled['cluster_label'] = clusters
        heatmap_data = scaled.groupby('cluster_label').mean().T

        # Unscaled features and ensure the shapes match for annotation overlay
        unscaled = self.night_features_df.copy()
        unscaled = unscaled[scaled_cols]
        unscaled['cluster_label'] = clusters
        annot_data = unscaled.groupby('cluster_label').mean().T
        # Format annotation values for better readability
        annot_fmt = annot_data.round(2).astype(str)

        plt.figure(figsize=(6, 14))
        sns.heatmap(heatmap_data, annot=annot_fmt, fmt='', cmap='coolwarm',
                    center=0)
        plt.title('Cluster Centroids: Scaled Feature Means (Unscaled Means '
                  'Overlay)')
        plt.xlabel('Cluster', verticalalignment='top')
        plt.savefig(FIGURES_DIR / f'heatmap_cluster_features_{cluster_type}_'
                                  f'{len(set(clusters))}_clusters.png', dpi=400,
                    bbox_inches='tight')
        plt.show()

    def fit_tsne(self, perplexity=30, max_iter=1000):
        """
        Fits t-SNE to the night-level features.
        :param perplexity: (int), Perplexity parameter for t-SNE.
        :param max_iter: (int), Maximum number of iterations for t-SNE.
        :param random_state: (int), Random state for reproducibility.
        :return: t-SNE results as a 2D array.
        """
        if self.night_features_df is None:
            raise ValueError("Night features not extracted yet. Run "
                             "extract_night_level_features first.")

        cluster_cols = ['cluster_label', 'tsne_cluster_label']
        features = self.night_features_df.copy()
        for col in cluster_cols:
            if col in features.columns:
                features = features.drop(columns=col)

        tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=max_iter,
                    random_state=42)
        self.tsne_results = tsne.fit_transform(self.scaled_night_features)

        return self.tsne_results

    def clustering_tsne(self, n_clusters=3):
        """
        Clusters the t-SNE results using KMeans and adds cluster labels to the
        night features DataFrame.
        """
        if self.tsne_results is None:
            raise ValueError("t-SNE results not computed yet. Run fit_tsne "
                             "first.")

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

        tsne_clusters = kmeans.fit_predict(self.tsne_results)
        self.silhouette_score = silhouette_score(self.tsne_results,
                                                 tsne_clusters)
        self.tsne_clusters = tsne_clusters

    def plot_tsne(self, cluster_type='kmeans'):
        """
        Plots a t-SNE visualisation of the night-level features, coloured by
        cluster label. Provides the option to use either KMeans clusters from
        original KMeans clustering or t-SNE clusters.
        :param cluster_type: (str), Type of clustering to plot.
        """
        if self.tsne_results is None:
            raise ValueError("t-SNE results not computed yet. Run fit_tsne "
                             "first.")

        if cluster_type == 'kmeans':
            clusters = self.night_clusters
        elif cluster_type == 'tsne':
            clusters = self.tsne_clusters
        elif cluster_type == 'umap':
            clusters = self.umap_clusters

        plt.figure(figsize=(6, 5))
        sns.scatterplot(x=self.tsne_results[:, 0], y=self.tsne_results[:, 1],
                        hue=clusters, palette=self.cluster_color_map, alpha=0.7)
        plt.title('t-SNE Visualisation of Night Features by KMeans Cluster')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.legend(title='Cluster', loc='lower right')
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f'tsne_clusters_{len(clusters)}clusters.png', dpi=400,
                    bbox_inches='tight')
        plt.show()

    def fit_umap(self, n_neighbors=5, min_dist=0.01):
        """
        Fits UMAP to the night-level features.
        :param perplexity: (int), Perplexity parameter for UMAP.
        :param max_iter: (int), Maximum number of iterations for UMAP.
        :param random_state: (int), Random state for reproducibility.
        :return: UMAP results as a 2D array.
        """
        if self.night_features_df is None:
            raise ValueError("Night features not extracted yet. Run "
                             "extract_night_level_features first.")

        cluster_cols = ['cluster_label', 'umap_cluster_label']
        features = self.night_features_df.copy()
        for col in cluster_cols:
            if col in features.columns:
                features = features.drop(columns=col)

        umap = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
        self.umap_results = umap.fit_transform(self.scaled_night_features)

        return self.umap_results

    def clustering_umap(self, n_clusters=3):
        """
        Clusters the UMAP results using KMeans and adds cluster labels to the
        night features DataFrame.
        """
        if self.umap_results is None:
            raise ValueError("UMAP results not computed yet. Run fit_umap "
                             "first.")

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

        umap_clusters = kmeans.fit_predict(self.umap_results)
        self.silhouette_score = silhouette_score(self.umap_results,
                                                 umap_clusters)
        self.umap_clusters = umap_clusters

    def plot_umap(self, cluster_type='kmeans'):
        """
        Plots a UMAP visualisation of the night-level features, coloured by
        cluster label. Provides the option to use either KMeans clusters from
        original KMeans clustering or UMAP clusters.
        :param cluster_type: (str), Type of clustering to plot.
        """
        if self.umap_results is None:
            raise ValueError("UMAP results not computed yet. Run fit_umap "
                             "first.")

        if cluster_type == 'kmeans':
            clusters = self.night_clusters
        elif cluster_type == 'tsne':
            clusters = self.tsne_clusters
        elif cluster_type == 'umap':
            clusters = self.umap_clusters

        plt.figure(figsize=(6, 5))
        sns.scatterplot(x=self.umap_results[:, 0], y=self.umap_results[:, 1],
                        hue=clusters, palette=self.cluster_color_map, alpha=0.7)
        plt.title('UMAP Visualisation of Night Features by KMeans Cluster')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.legend(title='Cluster', loc='lower right')
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'umap_clusters.png', dpi=400,
                    bbox_inches='tight')
        plt.show()

    def plot_cluster_distribution(self, pivot_counts: pd.DataFrame = None,
                                  cluster_type: str = 'kmeans'):
        """
        Plots the distribution of clusters per patient as a horizontal bar
        chart. Uses either KMeans clusters or t-SNE clusters.
        :param pivot_counts: (pd.DataFrame) Dataframe of night counts per
            cluster, with id index and columns for each cluster.
        :param cluster_type: (str), Type of clustering to use for distribution
            plot, either 'kmeans' or 'tsne'.
        :return:
        """
        from src.helper import generate_alphabetical_aliases
        if pivot_counts is None:
            pivot_counts = self.get_cluster_distributions(cluster_type)
        id_aliases = generate_alphabetical_aliases(pivot_counts.index)
        pivot_counts.index = pivot_counts.index.map(id_aliases)
        pivot_percent = pivot_counts.div(pivot_counts.sum(axis=1), axis=0) * 100
        totals = pivot_counts.sum(axis=1)
        ax = pivot_percent.plot(kind='barh',
                                stacked=True,
                                figsize=(6, 6),
                                color=[self.cluster_color_map[col] for col in
                                       pivot_percent.columns],
                                title='Distribution of Clusters per Patient',
                                xlabel='Percentage of Nights in Cluster',
                                ylabel='Patient')

        # Annotate bar segments with percentages
        for i, (idx, row) in enumerate(pivot_percent.iterrows()):
            cumulative = 0
            for cluster_val in row.values:
                if cluster_val > 5:  # Only show percentages > 5%
                    x_pos = cumulative + cluster_val / 2
                    ax.text(x_pos, i, f'{cluster_val:.0f}%',
                            ha='center', va='center', fontsize=8,
                            color='white', weight='bold')
                cumulative += cluster_val

        for i, (idx, total) in enumerate(totals.items()):
            ax.text(102, i, f'{total}', va='center', ha='left',
                    fontsize=9)

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=labels,
                  loc='upper center',
                  bbox_to_anchor=(0.5, -0.12),
                  fancybox=True,
                  shadow=True,
                  ncol=len(labels),
                  borderaxespad=0.
                  )
        plt.savefig(FIGURES_DIR / f'cluster_distribution_{cluster_type}_'
                                  f'{len(pivot_counts.columns)}_clusters.png',
                    dpi=400, bbox_inches='tight')
        plt.show()

    def get_cluster_distributions(self, cluster_type='kmeans'):
        """
        Returns the cluster distributions as a DataFrame.
        :param cluster_type: (str), Type of clustering to use for distribution
            plot, either 'kmeans' or 'tsne'.
        :return: DataFrame with cluster distributions.
        """
        if cluster_type not in ['kmeans', 'tsne']:
            raise ValueError("Invalid cluster_type. Choose 'kmeans' or 'tsne'.")
        if cluster_type == 'tsne' and self.tsne_clusters is None:
            raise ValueError("t-SNE clusters not computed yet. Run "
                             "clustering_tsne first.")
        if cluster_type == 'umap' and self.umap_clusters is None:
            raise ValueError("UMAP clusters not computed yet. Run "
                             "clustering_umap first.")

        features_df = self.return_dataset_with_clusters().reset_index()
        if cluster_type == 'kmeans':
            cluster_col = 'cluster_label'
        elif cluster_type == 'tsne':
            cluster_col = 'tsne_cluster_label'
        elif cluster_type == 'umap':
            cluster_col = 'umap_cluster_label'

        pivot_counts = (
            features_df[['id', cluster_col, 'night_start_date']].
            drop_duplicates().
            pivot_table(
                index='id',
                columns=cluster_col,
                values='night_start_date',
                aggfunc='count',
                fill_value=0
            ))
        return pivot_counts

    def calculate_distribution_metrics(self, cluster_type='kmeans'):
        """
        Calculates and prints the entropy and Gini coefficient of the cluster
        distributions per patient. Entropy measures the uncertainty in the
        distribution, while the Gini coefficient measures inequality.
        :return: DataFrame with normalised entropy per patient.
        """
        pivot_counts = self.get_cluster_distributions(cluster_type)
        entropy_per_patient = pivot_counts.apply(
            lambda row: -np.nansum(
                (row / 100) * np.log2((row / 100).replace(0, np.nan))), axis=1
        )
        # Average entropy across all patients
        mean_entropy = entropy_per_patient.mean()
        num_clusters = pivot_counts.shape[1]
        entropy_per_patient_norm = entropy_per_patient / np.log2(num_clusters)
        normalised_mean_entropy = entropy_per_patient_norm.mean()
        print(f"Mean entropy of cluster distribution per patient: "
              f"{mean_entropy:.3f}")
        print(
            f"Normalised mean entropy of cluster distribution per patient: "
            f"{normalised_mean_entropy:.3f}")

        def gini(array):
            array = np.array(array)
            array = array / array.sum()
            diffsum = np.sum(np.abs(np.subtract.outer(array, array)))
            return diffsum / (2 * len(array) * np.sum(array))

        gini_per_patient = pivot_counts.apply(gini, axis=1)
        mean_gini = gini_per_patient.mean()
        print(
            f"Mean Gini coefficient of cluster distribution per patient: "
            f"{mean_gini:.3f}")

        return entropy_per_patient_norm

    def get_cluster_nights(self, cluster_type='kmeans', cluster_label=None):
        """
        Returns the nights DataFrame filtered by cluster label.
        :param cluster_type: (str), Type of clustering to use for filtering,
            either 'kmeans' or 'tsne'.
        :param cluster_label: (int), Cluster label to filter by. If None,
            returns all nights.
        :return: DataFrame with nights for the specified cluster.
        """
        if cluster_label is None:
            raise ValueError("cluster_label must be provided to filter nights.")
        features_df = self.return_dataset_with_clusters().reset_index()
        if cluster_type == 'kmeans':
            cluster_col = 'cluster_label'
        elif cluster_type == 'tsne':
            cluster_col = 'tsne_cluster_label'
        elif cluster_type == 'umap':
            cluster_col = 'umap_cluster_label'

        return features_df[features_df[cluster_col] ==
                           cluster_label].set_index(['id', 'datetime'])
