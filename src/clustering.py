import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

from src.hmm import HMMEventDetector

class Clustering_EventDetector:
    """
    A class to detect events using K-Means clustering on time series features.

    Assumes data is in 30-minute intervals for nightly periods (17:00-11:00).
    """

    def __init__(self, df: pd.DataFrame, feature_cols: list,
                 original_var_cols: list,
                 id_col: str = 'id', datetime_col: str = 'datetime',
                 night_cluster_col: str = 'night_cluster_label'):
        """
        Initializes the Clustering_EventDetector.

        Args:
            df (pd.DataFrame): The input DataFrame containing 'id', 'datetime',
                               feature columns, original variable columns, and a night cluster label.
            feature_cols (list): List of column names to be used as features for clustering.
            original_var_cols (list): List of original variable column names for visualization
                                      (e.g., ['iob mean', 'cob mean', 'bg mean']).
            id_col (str): Column name for individual IDs.
            datetime_col (str): Column name for datetime stamps.
            night_cluster_col (str): Column name for the pre-assigned night cluster labels.
        """
        self.df = df.copy()
        self.feature_cols = feature_cols
        self.original_var_cols = original_var_cols
        self.id_col = id_col
        self.datetime_col = datetime_col
        self.night_cluster_col = night_cluster_col
        self.model = None
        self.scaler = StandardScaler()

        # Basic NaN handling for features. User should replace this with a
        # more sophisticated domain-specific imputation if necessary.
        for col in self.feature_cols:
            if self.df[col].isnull().any():
                print(
                    f"Warning: NaN values found in feature column '{col}'. Filling with 0. "
                    "Consider proper imputation based on your domain knowledge.")
                self.df[col] = self.df[col].fillna(
                    0)  # Simple fill, replace if needed

        self.df[self.datetime_col] = pd.to_datetime(self.df[self.datetime_col])
        self.df = self.df.sort_values(by=[self.id_col, self.datetime_col])

    def _prepare_data_for_clustering(self, df_subset: pd.DataFrame):
        """
        Prepares and scales the feature data for clustering.
        """
        X = df_subset[self.feature_cols].values
        # Fit scaler only on training data if you split, otherwise on full data here
        # For simplicity, fitting here, but usually, fit on training data and transform on all data
        if X.shape[0] > 0:
            return self.scaler.fit_transform(X)
        else:
            return np.array([]).reshape(0, len(self.feature_cols))

    def train_model(self, n_clusters: int, random_state: int = 42,
                    n_init: int = 10):
        """
        Trains the K-Means clustering model on the features of the entire dataset.

        Args:
            n_clusters (int): The number of clusters for K-Means.
            random_state (int): Seed for reproducibility.
            n_init (int): Number of times to run k-means with different centroid seeds.
        """
        print(f"Training K-Means with {n_clusters} clusters...")
        X_train_scaled = self._prepare_data_for_clustering(self.df)

        if X_train_scaled.shape[0] == 0:
            print(
                "No data available for K-Means training. Please check your DataFrame.")
            return

        self.model = KMeans(n_clusters=n_clusters, random_state=random_state,
                            n_init=n_init)
        self.model.fit(X_train_scaled)
        print("K-Means training complete.")
        print("Cluster Centers (scaled features):\n",
              self.model.cluster_centers_)

    def assign_clusters(self, df_to_assign: pd.DataFrame):
        """
        Assigns cluster labels to the intervals in a given DataFrame subset.

        Args:
            df_to_assign (pd.DataFrame): DataFrame subset for which to assign clusters.

        Returns:
            pd.DataFrame: Original DataFrame subset with an added 'cluster_label' column.
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train_model() first.")

        df_assigned = df_to_assign.copy()
        df_assigned = df_assigned.sort_values(
            by=[self.id_col, self.datetime_col])

        # Scale features using the *same* scaler fitted during training
        features_to_assign = df_assigned[self.feature_cols].values
        if features_to_assign.shape[0] > 0:
            scaled_features = self.scaler.transform(features_to_assign)
            df_assigned['cluster_label'] = self.model.predict(scaled_features)
        else:
            df_assigned[
                'cluster_label'] = pd.NA  # No features to assign, mark as NA

        return df_assigned

    def visualize_individual_events(self, individual_id: str):
        """
        Visualizes the assigned K-Means clusters against original variables for a specific individual.

        Args:
            individual_id (str): The ID of the individual to visualize.
        """
        if self.model is None:
            print("Model not trained. Cannot visualize.")
            return

        individual_df = self.df[self.df[self.id_col] == individual_id].copy()
        if individual_df.empty:
            print(f"Individual {individual_id} not found.")
            return

        df_with_clusters = self.assign_clusters(individual_df)

        fig, axes = plt.subplots(len(self.original_var_cols) + 1, 1, figsize=(
        15, 3 * (len(self.original_var_cols) + 1)), sharex=True)
        fig.suptitle(
            f'K-Means Clustered Events for Individual: {individual_id}',
            fontsize=16)

        # Plot original variables
        for i, var in enumerate(self.original_var_cols):
            sns.lineplot(ax=axes[i], x=self.datetime_col, y=var,
                         data=df_with_clusters, label=var,
                         color=sns.color_palette("viridis")[i])
            axes[i].set_ylabel(var)
            axes[i].legend()
            axes[i].grid(True, linestyle='--', alpha=0.6)

        # Plot assigned clusters
        sns.scatterplot(ax=axes[-1], x=self.datetime_col, y='cluster_label',
                        data=df_with_clusters,
                        hue='cluster_label', palette='tab10', legend='full',
                        s=50, marker='o')
        axes[-1].set_ylabel('Cluster Label')
        axes[-1].set_yticks(
            sorted(df_with_clusters['cluster_label'].dropna().unique()))
        axes[-1].set_title('K-Means Cluster Sequence')
        axes[-1].grid(True, linestyle='--', alpha=0.6)
        axes[-1].set_xlabel('Time')

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.show()

    def visualize_cluster_events(self, cluster_label: int):
        """
        Visualizes the average assigned K-Means clusters against average original variables
        for a specific night cluster.

        Args:
            cluster_label (int): The label of the night cluster to visualize.
        """
        if self.model is None:
            print("Model not trained. Cannot visualize.")
            return

        cluster_df = self.df[
            self.df[self.night_cluster_col] == cluster_label].copy()
        if cluster_df.empty:
            print(f"Cluster {cluster_label} not found.")
            return

        df_with_clusters = self.assign_clusters(cluster_df)

        # Calculate average values and find most common cluster label per time-of-day for the cluster
        df_with_clusters['time_index'] = df_with_clusters.groupby(
            self.id_col).cumcount()

        avg_cluster_data = df_with_clusters.groupby('time_index').agg(
            **{f'avg_{col}': (col, 'mean') for col in self.original_var_cols},
            most_common_cluster=('cluster_label', lambda x: x.mode()[
                0] if not x.mode().empty else -1)  # Use mode for state
        ).reset_index()

        # Map time_index back to representative datetime
        first_night_df = self.df[
            self.df[self.id_col] == self.df[self.id_col].unique()[
                0]].sort_values(self.datetime_col).copy()
        first_night_df['time_index'] = range(len(first_night_df))
        avg_cluster_data = pd.merge(avg_cluster_data, first_night_df[
            [self.datetime_col, 'time_index']], on='time_index', how='left')
        avg_cluster_data = avg_cluster_data.drop_duplicates(
            subset='time_index').sort_values('time_index')

        fig, axes = plt.subplots(len(self.original_var_cols) + 1, 1, figsize=(
        15, 3 * (len(self.original_var_cols) + 1)), sharex=True)
        fig.suptitle(
            f'K-Means Clustered Events for Night Cluster: {cluster_label} (Averaged)',
            fontsize=16)

        # Plot averaged original variables
        for i, var in enumerate(self.original_var_cols):
            sns.lineplot(ax=axes[i], x=self.datetime_col, y=f'avg_{var}',
                         data=avg_cluster_data, label=f'Avg {var}',
                         color=sns.color_palette("viridis")[i])
            axes[i].set_ylabel(var)
            axes[i].legend()
            axes[i].grid(True, linestyle='--', alpha=0.6)

        # Plot most common assigned clusters
        sns.scatterplot(ax=axes[-1], x=self.datetime_col,
                        y='most_common_cluster', data=avg_cluster_data,
                        hue='most_common_cluster', palette='tab10',
                        legend='full', s=50, marker='o')
        axes[-1].set_ylabel('Most Common Cluster')
        axes[-1].set_yticks(
            sorted(avg_cluster_data['most_common_cluster'].dropna().unique()))
        axes[-1].set_title('Most Common K-Means Cluster Sequence')
        axes[-1].grid(True, linestyle='--', alpha=0.6)
        axes[-1].set_xlabel('Time')

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.show()


# --- Dummy Data Generation ---
# This section creates a DataFrame that matches the specified format for demonstration.
# In a real scenario, you would load your actual prepared DataFrame here.

def generate_dummy_data(num_individuals=5, num_night_clusters=3):
    data = []
    start_time = pd.to_datetime("2023-01-01 17:00:00")
    end_time = pd.to_datetime("2023-01-02 11:00:00")
    # 18 hours * 2 intervals/hour = 36 intervals
    time_points = pd.date_range(start=start_time, end=end_time, freq='30min')[
                  :-1]  # Exclude 11:00 to keep 36 intervals

    for i in range(num_individuals):
        individual_id = f"Individual_{i + 1}"
        night_cluster_label = np.random.randint(0,
                                                num_night_clusters)  # Assign a random night cluster

        for j, dt in enumerate(time_points):
            # Simulate features (e.g., activity, heart rate, sensor X)
            feature1 = np.random.rand() * 10 + np.sin(
                j / 5) * 5  # Example: activity with some pattern
            feature2 = np.random.rand() * 5 + np.cos(
                j / 10) * 3  # Example: heart rate
            feature3 = np.random.rand() * 2  # Example: sensor reading

            # Simulate original variables ('iob mean', 'cob mean', 'bg mean')
            # Assuming these have typical patterns over the night
            iob_mean = max(0, 20 - j * 0.5 + np.random.rand() * 5)  # Declining
            cob_mean = max(0,
                           30 - j * 0.8 + np.random.rand() * 10)  # Declining faster
            bg_mean = 120 + np.sin(
                j / 4) * 20 + np.random.rand() * 10  # Fluctuating around a baseline

            data.append({
                'id': individual_id,
                'datetime': dt,
                'feature_activity': feature1,
                'feature_heart_rate': feature2,
                'feature_sensor_x': feature3,
                'iob mean': iob_mean,
                'cob mean': cob_mean,
                'bg mean': bg_mean,
                'night_cluster_label': night_cluster_label
            })
    return pd.DataFrame(data)
