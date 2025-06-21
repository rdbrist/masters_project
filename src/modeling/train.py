from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import pandas as pd

from src.config import MODELS_DIR, PROCESSED_DATA_DIR
from src.configurations import Configuration
from src.hmm import HMMEventDetector
from src.clustering import Clustering_EventDetector

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    # -----------------------------------------
):
    logger.info("Training some model...")
    for i in tqdm(range(10), total=10):
        if __name__ == "__main__":
            config = Configuration()
            df = pd.read_csv(config.final_filtered_csv)
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.set_index(['id', 'datetime'])

            print(
                f"Total rows: {len(df)}, Columns: {df.columns.tolist()}")

            original_var_cols = ['iob mean', 'cob mean', 'bg mean']

            # --- HMM Event Detection ---
            print("\n--- HMM Event Detection Example ---")
            hmm_detector = HMMEventDetector(df, feature_cols,
                                             original_var_cols)
            hmm_detector.train_model(n_components=3)

            # Visualize for a specific individual
            some_individual_id = df['id'].unique()[0]
            hmm_detector.visualize_individual_events(some_individual_id)

            # Visualize for a specific night cluster (e.g., cluster 0)
            some_cluster_label = df['night_cluster_label'].unique()[0]
            hmm_detector.visualize_cluster_events(some_cluster_label)

            # # --- Clustering Event Detection (K-Means) ---
            # print("\n--- K-Means Clustering Event Detection Example ---")
            # clustering_detector = Clustering_EventDetector(df,
            #                                                feature_cols,
            #                                                original_var_cols)
            # clustering_detector.train_model(n_clusters=3)
            #
            # # Visualize for a specific individual
            # clustering_detector.visualize_individual_events(some_individual_id)
            #
            # # Visualize for a specific night cluster (e.g., cluster 0)
            # clustering_detector.visualize_cluster_events(some_cluster_label)


if __name__ == "__main__":
    app()
