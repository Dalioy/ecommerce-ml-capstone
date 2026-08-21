"""
train.py
========
Preprocessing + model fitting for customer segmentation.

Mirrors sections 8.1 - 8.3 of the original notebook:
  - scale / log-transform the RFM+ features
  - pick the optimal K for KMeans (elbow + silhouette)
  - fit KMeans, DBSCAN and Agglomerative clustering
  - persist the final model to models/clustering_model.pkl
"""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

SKEWED_COLS = ["monetary", "avg_order_value", "frequency", "tenure_days"]
FILLNA_MEDIAN_COLS = ["delivery_delay", "avg_delivery_time", "avg_freight_ratio"]

FEATURES_FOR_SCALING = [
    "recency",
    "monetary_log",
    "frequency_log",
    "avg_order_value_log",
    "return_rate",
    "avg_installments",
    "avg_review_score",
    "delivery_delay",
    "avg_delivery_time",
    "avg_freight_ratio",
    "category_diversity",
]


# ---------------------------------------------------------------------------
# 8.1  Preprocessing
# ---------------------------------------------------------------------------
def preprocess_features(rfm: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, StandardScaler]:
    """
    Log-transform skewed columns, impute delivery/freight NaNs with the
    median, then standard-scale the modeling feature set.

    Returns
    -------
    cluster_df : pd.DataFrame
        rfm copy with the extra *_log columns added.
    X_scaled : np.ndarray
        Scaled feature matrix ready for clustering.
    scaler : StandardScaler
        Fitted scaler (needed to transform new data consistently).
    """
    cluster_df = rfm.copy()

    for col in SKEWED_COLS:
        cluster_df[f"{col}_log"] = np.log1p(cluster_df[col].clip(lower=0))

    for col in FILLNA_MEDIAN_COLS:
        cluster_df[col] = cluster_df[col].fillna(cluster_df[col].median())

    X = cluster_df[FEATURES_FOR_SCALING].fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return cluster_df, X_scaled, scaler


# ---------------------------------------------------------------------------
# 8.2  Determine optimal K (elbow + silhouette)
# ---------------------------------------------------------------------------
def determine_optimal_k(
    X_scaled: np.ndarray,
    k_range: range = range(2, 9),
    save_plots: bool = True,
) -> tuple[int, dict, dict]:
    """
    Fit KMeans for every K in k_range, plot the elbow / silhouette curves,
    and return the K that maximizes silhouette score.
    """
    inertias = []
    silhouette_scores = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouette_scores.append(
            silhouette_score(X_scaled, labels, sample_size=5000, random_state=42)
        )

    if save_plots:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(8, 5))
        plt.plot(list(k_range), inertias, marker="o")
        plt.title("Elbow Method — Inertia vs K")
        plt.xlabel("Number of clusters (K)")
        plt.ylabel("Inertia")
        plt.grid(alpha=0.3)
        plt.savefig(FIGURES_DIR / "elbow_plot.png", dpi=150, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(list(k_range), silhouette_scores, marker="o", color="darkorange")
        plt.title("Silhouette Score vs K")
        plt.xlabel("Number of clusters (K)")
        plt.ylabel("Silhouette Score")
        plt.grid(alpha=0.3)
        plt.savefig(FIGURES_DIR / "silhouette_plot.png", dpi=150, bbox_inches="tight")
        plt.close()

    best_k = list(k_range)[int(np.argmax(silhouette_scores))]
    inertias_dict = dict(zip(k_range, inertias))
    silhouette_dict = dict(zip(k_range, silhouette_scores))

    print(f"Best K by silhouette score: {best_k}")
    print("Inertias:", inertias_dict)
    print("Silhouette scores:", silhouette_dict)

    return best_k, inertias_dict, silhouette_dict


# ---------------------------------------------------------------------------
# 8.3  Fit KMeans / DBSCAN / Agglomerative
# ---------------------------------------------------------------------------
def fit_models(X_scaled: np.ndarray, best_k: int = 5, sample_size: int = 5000, random_state: int = 42):
    """
    Fit the three clustering models used for comparison.

    Returns a dict with fitted estimators + label arrays:
        {
          "kmeans": {"model": ..., "labels": ...},
          "dbscan": {"model": ..., "labels": ...},
          "hierarchical": {"model": ..., "labels": ..., "sample_idx": ..., "X_sample": ...},
        }
    """
    kmeans_final = KMeans(n_clusters=best_k, random_state=random_state, n_init=10)
    kmeans_labels = kmeans_final.fit_predict(X_scaled)
    print("KMeans done")

    dbscan = DBSCAN(eps=1.5, min_samples=10)
    dbscan_labels = dbscan.fit_predict(X_scaled)
    print("DBSCAN done")

    np.random.seed(random_state)
    sample_idx = np.random.choice(X_scaled.shape[0], size=min(sample_size, X_scaled.shape[0]), replace=False)
    X_sample = X_scaled[sample_idx]

    hier = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
    hier_labels_sample = hier.fit_predict(X_sample)
    print("Hierarchical done")

    return {
        "kmeans": {"model": kmeans_final, "labels": kmeans_labels},
        "dbscan": {"model": dbscan, "labels": dbscan_labels},
        "hierarchical": {
            "model": hier,
            "labels": hier_labels_sample,
            "sample_idx": sample_idx,
            "X_sample": X_sample,
        },
    }


# ---------------------------------------------------------------------------
# Persist the chosen model
# ---------------------------------------------------------------------------
def save_model(model, scaler: StandardScaler, path: Path = MODELS_DIR / "clustering_model.pkl") -> None:
    """Pickle the fitted model together with its scaler so it can be reused for scoring new data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler, "features": FEATURES_FOR_SCALING}, f)
    print("Model saved to", path)


if __name__ == "__main__":
    from src.features.feature_engineering import PROCESSED_DATA_DIR

    rfm = pd.read_csv(PROCESSED_DATA_DIR / "rfm_final.csv")
    cluster_df, X_scaled, scaler = preprocess_features(rfm)
    best_k, _, _ = determine_optimal_k(X_scaled)
    models = fit_models(X_scaled, best_k=5)
    save_model(models["kmeans"]["model"], scaler)