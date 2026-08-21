"""
pipeline.py
===========
End-to-end customer-segmentation pipeline. Run this file directly to go
from raw CSVs (in data/raw/) all the way to a trained clustering model and
named customer segments (in models/ and outputs/).

Usage
-----
    python -m src.pipeline
    # or simply: python src/pipeline.py   (run from the project root)
"""

from pathlib import Path

import pandas as pd

from src.data.load_data import load_raw_datasets, print_columns, basic_info
from src.data.clean_data import clean_datasets
from src.features.feature_engineering import build_feature_table, save_feature_table, PROCESSED_DATA_DIR
from src.clustering.train import preprocess_features, determine_optimal_k, fit_models, save_model
from src.clustering.evaluate import (
    evaluate_all,
    assign_segments,
    profile_segments,
    name_segments,
    generate_business_insights,
)
from src import visualization as viz

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_pipeline(
    run_eda_plots: bool = True,
    run_feature_plots: bool = True,
    best_k: int | None = None,
) -> dict:
    """
    Run the full pipeline end to end and return a dict with the key
    artifacts produced (datasets, rfm table, model bundle, profile).
    """

    # 1) Load ------------------------------------------------------------
    print("\n### 1. Loading raw datasets ###")
    datasets = load_raw_datasets()
    if not datasets:
        raise FileNotFoundError(
            "No raw CSVs found in data/raw/. Place the Olist dataset there before running the pipeline."
        )
    print_columns(**datasets)

    # 2) Clean -------------------------------------------------------------
    print("\n### 2. Cleaning datasets ###")
    datasets = clean_datasets(datasets)

    # 3) EDA visuals ---------------------------------------------------
    if run_eda_plots:
        print("\n### 3. Generating EDA plots ###")
        viz.run_all_eda_plots(datasets)

    # 4) Feature engineering ----------------------------------------------
    print("\n### 4. Building customer feature table (RFM+) ###")
    rfm = build_feature_table(datasets)
    save_feature_table(rfm)

    # 5) Feature-level visuals --------------------------------------------
    if run_feature_plots:
        print("\n### 5. Generating feature-level plots ###")
        viz.run_all_feature_plots(rfm)

    # 6) Preprocess + choose K ---------------------------------------------
    print("\n### 6. Preprocessing features for clustering ###")
    cluster_df, X_scaled, scaler = preprocess_features(rfm)

    if best_k is None:
        print("\n### 6b. Selecting optimal K ###")
        best_k, _, _ = determine_optimal_k(X_scaled)
        best_k = 5  # override to match validated business choice, same as source notebook

    # 7) Fit models ---------------------------------------------------------
    print(f"\n### 7. Fitting clustering models (K={best_k}) ###")
    models = fit_models(X_scaled, best_k=best_k)
    save_model(models["kmeans"]["model"], scaler)

    # 8) Evaluate + profile + name segments --------------------------------
    print("\n### 8. Evaluating models ###")
    results = evaluate_all(X_scaled, models)

    rfm = assign_segments(rfm, models["kmeans"]["labels"])
    profile = profile_segments(rfm)
    profile = name_segments(profile)
    generate_business_insights(profile)

    return {
        "datasets": datasets,
        "rfm": rfm,
        "cluster_df": cluster_df,
        "scaler": scaler,
        "models": models,
        "evaluation": results,
        "profile": profile,
    }


if __name__ == "__main__":
    run_pipeline()
