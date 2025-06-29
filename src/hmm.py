import numpy as np
import pandas as pd
import seaborn as sns
from hmmlearn import hmm
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler


class HMMEventDetector:
    """
    A class to detect events using Hidden Markov Models on time series data.
    Assumes data is in 30-minute intervals for nightly periods (17:00-11:00).
    """

    def __init__(self, df: pd.DataFrame, feature_cols: list,
                 original_var_cols: list,):
        """
        Initializes the HMMEventDetector.
        :param df: (pd.DataFrame) The input DataFrame containing 'id',
            'datetime', feature columns, original variable columns, and a night
            cluster label.
        :param feature_cols: (list) List of column names to be used as features
            for the HMM.
        :param original_var_cols: (list) List of original variable column names
            for visualization (e.g., ['iob mean', 'cob mean', 'bg mean']).
        """
        self.df = df.copy()
        self.feature_cols = feature_cols
        self.original_var_cols = original_var_cols
        self.model = None
        self.scaler = StandardScaler()

        # Basic NaN handling for features. User should replace this with a
        # more sophisticated domain-specific imputation if necessary.
        for col in self.feature_cols:
            if self.df[col].isnull().any():
                print(
                    f"Warning: NaN values found in feature column '{col}'. "
                    f"Filling with 0. Consider proper imputation if required.")
                self.df[col] = self.df[col].fillna(
                    0)  # Simple fill, replace if needed

    def _prepare_data_for_hmm(self, df_subset: pd.DataFrame):
        """
        Prepares the data subset into X and lengths format required by hmmlearn.
        Assumes data is sorted by id and datetime.
        """
        X_list = []
        lengths = []

        df_subset = df_subset.reset_index().copy()

        for individual_id in df_subset['id'].unique():
            individual_df = df_subset[
                df_subset['id'] == individual_id].copy()
            individual_features = individual_df[self.feature_cols].values
            if individual_features.shape[
                0] > 0:  # Ensure there's data for the individual
                X_list.append(individual_features)
                lengths.append(individual_features.shape[0])
            else:
                print(f"Warning: No data for individual {individual_id} in "
                      f"this subset.")

        # If X_list is empty, return empty arrays to prevent concatenation
        # errors
        if not X_list:
            return np.array([]).reshape(0, len(self.feature_cols)), []

        X = np.concatenate(X_list)
        print(X)
        return self.scaler.fit_transform(X) if X.shape[0] > 0 else X, lengths

    def train_model(self, n_components: int, covariance_type: str = 'diag',
                    n_iter: int = 100, tol: float = 0.01,
                    random_state: int = 42):
        """
        Trains the Gaussian Hidden Markov Model on the full dataset.
        :param n_components: (int) The number of hidden states for the HMM.
        :param covariance_type: (str) Type of covariance matrix ('diag', 'full',
            'spherical', 'tied').
        :param n_iter: (int) Number of iterations for the EM algorithm.
        :param tol: (float) Convergence threshold.
        :param random_state: (int) Seed for reproducibility.
        """
        print(f"Training HMM with {n_components} components...")
        X_train, lengths_train = self._prepare_data_for_hmm(self.df)

        if X_train.shape[0] == 0:
            print("No data available for HMM training. Please check your "
                  "DataFrame.")
            return

        self.model = hmm.GaussianHMM(
            n_components=n_components,
            covariance_type=covariance_type,
            n_iter=n_iter,
            tol=tol,
            random_state=random_state,
            init_params="stmc"  # Initialize startprob, transmat, means, covars
        )
        self.model.fit(X_train, lengths_train)
        print("HMM training complete.")
        print("Learned Transition Matrix:\n", self.model.transmat_)
        print("\nLearned Means of States:\n", self.model.means_)
        print("\nLearned Covariances of States:\n", self.model.covars_)

    def infer_states(self, df_to_infer: pd.DataFrame):
        """
        Infers the hidden states for a given DataFrame subset.
        :param df_to_infer: (pd.DataFrame) DataFrame subset (e.g., for one
            individual or a cluster) for which to infer states.
        :return: (pd.DataFrame) Original DataFrame subset with an added
            'inferred_state' column.
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train_model() first.")

        df_inferred = df_to_infer.copy()
        df_inferred = df_inferred.sort_values(
            by=['id', 'datetime'])

        inferred_states = []
        for individual_id in df_inferred['id'].unique():
            individual_df = df_inferred[
                df_inferred['id'] == individual_id].copy()
            if individual_df.empty:
                continue
            individual_features = individual_df[self.feature_cols].values

            # Scale the individual features using the *same* scaler fitted
            # during training
            scaled_features = self.scaler.transform(individual_features)

            # Predict
            try:
                logprob, state_sequence = self.model.decode(scaled_features,
                                                            algorithm="viterbi")
                inferred_states.extend(state_sequence)
            except Exception as e:
                print(f"Could not decode for individual {individual_id}: {e}")
                inferred_states.extend(
                    [-1] * len(individual_df))  # Mark as -1 for error

        df_inferred['inferred_state'] = inferred_states
        return df_inferred

    def visualize_individual_events(self, individual_id: int):
        """
        Visualizes the inferred HMM states against original variables for a
        specific individual.
        :param individual_id: (str) The ID of the individual to visualize.
        """
        if self.model is None:
            print("Model not trained. Cannot visualize.")
            return

        individual_df = self.df[self.df['id'] == individual_id].copy()
        if individual_df.empty:
            print(f"Individual {individual_id} not found.")
            return

        df_with_states = self.infer_states(individual_df)

        fig, axes = (
            plt.subplots(len(self.original_var_cols) + 1, 1, figsize=(
            15, 3 * (len(self.original_var_cols) + 1)), sharex=True))
        fig.suptitle(f'HMM Inferred Events for Individual: {individual_id}',
                     fontsize=16)

        # Plot original variables
        for i, var in enumerate(self.original_var_cols):
            sns.lineplot(ax=axes[i], x='datetime', y=var,
                         data=df_with_states, label=var,
                         color=sns.color_palette("viridis")[i])
            axes[i].set_ylabel(var)
            axes[i].legend()
            axes[i].grid(True, linestyle='--', alpha=0.6)

        # Plot inferred states
        sns.scatterplot(ax=axes[-1], x='datetime', y='inferred_state',
                        data=df_with_states,
                        hue='inferred_state', palette='tab10', legend='full',
                        s=50, marker='o')
        axes[-1].set_ylabel('Inferred State')
        axes[-1].set_yticks(sorted(df_with_states['inferred_state'].unique()))
        axes[-1].set_title('HMM Inferred State Sequence')
        axes[-1].grid(True, linestyle='--', alpha=0.6)
        axes[-1].set_xlabel('Time')

        plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.96))
        plt.show()

    def visualize_cluster_events(self, cluster_label: int):
        """
        Visualizes the average inferred HMM states against average original
        variables for a specific night cluster.
        :param cluster_label: (int) The label of the night cluster to visualize.
        """
        if self.model is None:
            print("Model not trained. Cannot visualize.")
            return

        cluster_df = self.df[
            self.df['cluster_label'] == cluster_label].copy()
        if cluster_df.empty:
            print(f"Cluster {cluster_label} not found.")
            return

        df_with_states = self.infer_states(cluster_df)

        # Calculate average values and inferred states per time-of-day for the
        # cluster. Create a time-of-day index (e.g., 0 for 17:00, 1 for 17:30,
        # etc.). Assuming all nights have the same number of intervals and
        # start/end times
        df_with_states['time_index'] = df_with_states.groupby(
            'id').cumcount()

        # Average original variables and find most common inferred state
        avg_cluster_data = df_with_states.groupby('time_index').agg(
            **{f'avg_{col}': (col, 'mean') for col in self.original_var_cols},
            most_common_state=('inferred_state', lambda x: x.mode()[
                0] if not x.mode().empty else -1)  # Use mode for state
        ).reset_index()

        # Map time_index back to representative datetime (first night's
        # datetime values)
        first_night_df = self.df[
            self.df['id'] == self.df['id'].unique()[
                0]].sort_values('datetime').copy()
        first_night_df['time_index'] = range(len(first_night_df))
        avg_cluster_data = pd.merge(avg_cluster_data, first_night_df[
            ['datetime', 'time_index']], on='time_index', how='left')
        avg_cluster_data = avg_cluster_data.drop_duplicates(
            subset='time_index').sort_values('time_index')

        fig, axes = plt.subplots(len(self.original_var_cols) + 1, 1, figsize=(
            15, 3 * (len(self.original_var_cols) + 1)), sharex=True)
        fig.suptitle(
            f'HMM Inferred Events for Night Cluster: '
            f'{cluster_label} (Averaged)', fontsize=16)

        # Plot averaged original variables
        for i, var in enumerate(self.original_var_cols):
            sns.lineplot(ax=axes[i], x='datetime', y=f'avg_{var}',
                         data=avg_cluster_data, label=f'Avg {var}',
                         color=sns.color_palette("viridis")[i])
            axes[i].set_ylabel(var)
            axes[i].legend()
            axes[i].grid(True, linestyle='--', alpha=0.6)

        # Plot most common inferred states
        sns.scatterplot(ax=axes[-1], x='datetime', y='most_common_state',
                        data=avg_cluster_data,
                        hue='most_common_state', palette='tab10', legend='full',
                        s=50, marker='o')
        axes[-1].set_ylabel('Most Common State')
        axes[-1].set_yticks(
            sorted(avg_cluster_data['most_common_state'].unique()))
        axes[-1].set_title('Most Common HMM Inferred State Sequence')
        axes[-1].grid(True, linestyle='--', alpha=0.6)
        axes[-1].set_xlabel('Time')

        plt.tight_layout(rect=(0, 0.03, 1, 0.96))
        plt.show()
