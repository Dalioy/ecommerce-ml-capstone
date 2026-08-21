"""
evaluate.py
===========
Evaluates the fitted clustering models, assigns final segments, profiles
them, gives each a business-friendly name, and generates plain-language
insights.

Mirrors sections 8.4 - 8.8 of the original notebook.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import davies_bouldin_score, silhouette_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
SEGMENTS_DIR = PROJECT_ROOT / "outputs" / "segments"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"


# ---------------------------------------------------------------------------
# 8.4  Evaluate a single clustering result
# ---------------------------------------------------------------------------
def evaluate_clustering(name: str, X: np.ndarray, labels: np.ndarray) -> dict | None:
    """Compute silhouette score, Davies-Bouldin index and segment size balance for one model."""
    unique_labels = set(labels)
    unique_labels.discard(-1)
    if len(unique_labels) < 2:
        print(f"{name}: not enough clusters to evaluate.")
        return None

    mask = labels != -1
    sil = silhouette_score(X[mask], labels[mask], sample_size=5000, random_state=42)
    db = davies_bouldin_score(X[mask], labels[mask])
    sizes = pd.Series(labels[mask]).value_counts(normalize=True).sort_index()

    print(f"\n--- {name} ---")
    print(f"Clusters found: {sorted(unique_labels)}")
    print(f"Silhouette Score: {sil:.3f}")
    print(f"Davies-Bouldin Index: {db:.3f} (lower is better)")
    print("Segment size balance (proportion):")
    print(sizes)

    return {"name": name, "silhouette": sil, "davies_bouldin": db, "sizes": sizes}


def evaluate_all(X_scaled: np.ndarray, models: dict) -> dict:
    """Evaluate KMeans, DBSCAN and Hierarchical results produced by train.fit_models()."""
    results = {
        "kmeans": evaluate_clustering("KMeans", X_scaled, models["kmeans"]["labels"]),
        "dbscan": evaluate_clustering("DBSCAN", X_scaled, models["dbscan"]["labels"]),
        "hierarchical": evaluate_clustering(
            "Hierarchical", models["hierarchical"]["X_sample"], models["hierarchical"]["labels"]
        ),
    }

    if METRICS_DIR:
        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        summary = pd.DataFrame([
            {"model": k, "silhouette": v["silhouette"], "davies_bouldin": v["davies_bouldin"]}
            for k, v in results.items() if v is not None
        ])
        summary.to_csv(METRICS_DIR / "clustering_model_comparison.csv", index=False)

    return results


# ---------------------------------------------------------------------------
# 8.5  Final segment assignment
# ---------------------------------------------------------------------------
def assign_segments(rfm: pd.DataFrame, kmeans_labels: np.ndarray) -> pd.DataFrame:
    rfm = rfm.copy()
    rfm["segment"] = kmeans_labels
    return rfm


# ---------------------------------------------------------------------------
# 8.6  Segment profiling
# ---------------------------------------------------------------------------
def profile_segments(rfm: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    profile = rfm.groupby("segment").agg(
        customers=("customer_unique_id", "count"),
        avg_recency=("recency", "mean"),
        avg_frequency=("frequency", "mean"),
        avg_monetary=("monetary", "mean"),
        avg_order_value=("avg_order_value", "mean"),
        avg_return_rate=("return_rate", "mean"),
        avg_review_score=("avg_review_score", "mean"),
        avg_delivery_delay=("delivery_delay", "mean"),
        avg_delivery_time=("avg_delivery_time", "mean"),
        avg_freight_ratio=("avg_freight_ratio", "mean"),
        avg_category_diversity=("category_diversity", "mean"),
        avg_tenure_days=("tenure_days", "mean"),
        total_revenue=("monetary", "sum"),
    ).reset_index()

    profile["pct_of_customers"] = 100 * profile["customers"] / profile["customers"].sum()
    profile["pct_of_revenue"] = 100 * profile["total_revenue"] / profile["total_revenue"].sum()

    top_category = rfm.groupby("segment")["preferred_category"].agg(lambda x: x.mode().iloc[0])
    top_channel = rfm.groupby("segment")["preferred_channel"].agg(lambda x: x.mode().iloc[0])
    profile["top_category"] = profile["segment"].map(top_category)
    profile["top_channel"] = profile["segment"].map(top_channel)

    print("\n=== SEGMENT PROFILE TABLE ===")
    print(profile.round(2))

    if save:
        SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)
        profile.to_csv(SEGMENTS_DIR / "segment_profile_table.csv", index=False)

        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(8, 5))
        plt.bar(profile["segment"].astype(str), profile["pct_of_customers"], color="steelblue")
        plt.title("Segment Size (% of Customers)")
        plt.xlabel("Segment")
        plt.ylabel("% of Customers")
        plt.grid(alpha=0.3, axis="y")
        plt.savefig(FIGURES_DIR / "segment_size_chart.png", dpi=150, bbox_inches="tight")
        plt.close()

    return profile


# ---------------------------------------------------------------------------
# 8.7  Name the segments (heuristic rules)
# ---------------------------------------------------------------------------
def name_segment(row: pd.Series, profile: pd.DataFrame) -> str:
    high_monetary = row["avg_monetary"] > profile["avg_monetary"].median()
    good_review = row["avg_review_score"] >= 4.0
    bad_review = row["avg_review_score"] < 3.0
    late_delivery = row["avg_delivery_delay"] > 0
    high_return = row["avg_return_rate"] > 0.10
    repeat_customer = row["avg_frequency"] > 1.5

    if repeat_customer and high_monetary and good_review:
        return "High-Value Loyalists"
    if high_monetary and good_review and not late_delivery:
        return "High-Value One-Time Buyers"
    if late_delivery and bad_review:
        return "Delivery-Frustrated"
    if high_return:
        return "Bargain Hunters / High-Return"
    if not high_monetary and good_review:
        return "Low-Value Everyday Buyers"
    return "Occasional Buyers"


def name_segments(profile: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    profile = profile.copy()
    profile["segment_name"] = profile.apply(lambda row: name_segment(row, profile), axis=1)

    print("\n=== NAMED SEGMENTS ===")
    print(
        profile[[
            "segment", "segment_name", "pct_of_customers", "pct_of_revenue",
            "avg_monetary", "avg_frequency", "avg_recency",
            "avg_review_score", "avg_return_rate", "avg_delivery_delay",
        ]]
        .sort_values("pct_of_revenue", ascending=False)
        .round(2)
    )

    if save:
        SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)
        profile.to_csv(SEGMENTS_DIR / "segment_profile_named.csv", index=False)

    return profile


# ---------------------------------------------------------------------------
# 8.8  Business insights
# ---------------------------------------------------------------------------
def generate_business_insights(profile: pd.DataFrame, save: bool = True) -> str:
    lines = ["=== BUSINESS INSIGHTS ==="]
    for _, row in profile.sort_values("pct_of_revenue", ascending=False).iterrows():
        lines.append(
            f"Segment '{row['segment_name']}' ({row['pct_of_customers']:.0f}% of customers) "
            f"generates {row['pct_of_revenue']:.0f}% of revenue, "
            f"with {row['avg_review_score']:.1f} avg rating and "
            f"{row['avg_return_rate'] * 100:.0f}% return rate."
        )

    lines.append(
        "\nRecommendation: prioritize retention (loyalty perks, early access) for "
        "high-value/low-return segments, investigate delivery/logistics for segments "
        "with high delivery_delay and low review scores, and consider reactivation "
        "campaigns (discount codes, reminder emails) for low-frequency/high-recency "
        "segments instead of loyalty investment."
    )

    text = "\n".join(lines)
    print("\n" + text)

    if save:
        SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(SEGMENTS_DIR / "business_insights.txt", "w") as f:
            f.write(text)

    return text


if __name__ == "__main__":
    from src.clustering.train import fit_models, preprocess_features
    from src.features.feature_engineering import PROCESSED_DATA_DIR

    rfm = pd.read_csv(PROCESSED_DATA_DIR / "rfm_final.csv")
    cluster_df, X_scaled, scaler = preprocess_features(rfm)
    models = fit_models(X_scaled, best_k=5)

    results = evaluate_all(X_scaled, models)
    rfm = assign_segments(rfm, models["kmeans"]["labels"])
    profile = profile_segments(rfm)
    profile = name_segments(profile)
    generate_business_insights(profile)