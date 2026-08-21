"""
clean_data.py
=============
Cleans the raw Olist datasets: missing-value / duplicate audits, hidden-null
handling, text normalization, dtype fixes and business-rule validation.

Mirrors the "2 Cleaning", "3 Fix Data Types" and "4 Business Rule Validation"
sections of the original exploration notebook, refactored into reusable
functions.
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"

HIDDEN_NULLS = ["NA", "N/A", "Null", "null", "-", ""]

DATE_COLUMNS = {
    "orders": [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ],
    "order_items": ["shipping_limit_date"],
    "order_reviews": ["review_creation_date", "review_answer_timestamp"],
}

NUMERIC_COLUMNS = {
    "customers": ["customer_zip_code_prefix"],
    "geolocation": [
        "geolocation_zip_code_prefix",
        "geolocation_lat",
        "geolocation_lng",
    ],
    "order_items": ["order_item_id", "price", "freight_value"],
    "order_payments": ["payment_sequential", "payment_installments", "payment_value"],
    "order_reviews": ["review_score"],
    "products": [
        "product_name_lenght",
        "product_description_lenght",
        "product_photos_qty",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
    ],
    "sellers": ["seller_zip_code_prefix"],
}

CATEGORICAL_COLUMNS = {
    "customers": ["customer_city", "customer_state"],
    "orders": ["order_status"],
    "order_payments": ["payment_type"],
    "products": ["product_category_name"],
    "sellers": ["seller_city", "seller_state"],
}

YES_NO_MAPPING = {"yes": "Yes", "y": "Yes", "no": "No", "n": "No"}
CHANNEL_MAPPING = {"online": "Online", "app": "App", "marketplace": "Marketplace"}


# ---------------------------------------------------------------------------
# 2.1 - 2.3  Audits (missing values / exact duplicates / duplicate order ids)
# ---------------------------------------------------------------------------
def audit_missing_values(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, data in datasets.items():
        for column in data.columns:
            rows.append({
                "dataset": name,
                "column": column,
                "count": data[column].isna().sum(),
                "percentage": data[column].isna().mean() * 100,
            })
    return pd.DataFrame(rows)


def audit_exact_duplicates(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = [
        {"dataset": name, "exact_duplicates": data.duplicated().sum()}
        for name, data in datasets.items()
    ]
    return pd.DataFrame(rows)


def audit_duplicate_order_ids(orders: pd.DataFrame) -> int:
    if "order_id" not in orders.columns:
        return 0
    return int(orders["order_id"].duplicated().sum())


def audit_hidden_nulls(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, data in datasets.items():
        for column in data.select_dtypes(include="object").columns:
            count = data[column].isin(HIDDEN_NULLS).sum()
            if count > 0:
                rows.append({"dataset": name, "column": column, "hidden_null_count": count})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2.5 - 2.8  Text normalization
# ---------------------------------------------------------------------------
def snake_case_columns(datasets: dict[str, pd.DataFrame]) -> None:
    """Normalize every dataset's column names to snake_case, in place."""
    for data in datasets.values():
        data.columns = (
            data.columns.str.strip()
            .str.lower()
            .str.replace(r"[^a-z0-9]+", "_", regex=True)
            .str.strip("_")
        )


def strip_whitespace_and_hidden_nulls(datasets: dict[str, pd.DataFrame]) -> None:
    """Strip whitespace on object columns and convert hidden-null tokens to NaN, in place."""
    strip_only_hidden_nulls = [v for v in HIDDEN_NULLS if v != ""]
    for data in datasets.values():
        object_columns = data.select_dtypes(include="object").columns
        for column in object_columns:
            data[column] = data[column].str.strip()
            data[column] = data[column].replace(strip_only_hidden_nulls, np.nan)


def unify_casing(datasets: dict[str, pd.DataFrame]) -> None:
    """Lowercase + strip every object column, in place."""
    for data in datasets.values():
        object_columns = data.select_dtypes(include="object").columns
        for column in object_columns:
            data[column] = data[column].str.lower().str.strip()


def standardize_text_labels(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Standardize known categorical text labels (channel, returned) using
    lookup mappings. Returns a log of every substitution rule that was applied.
    """
    mapping_log = []

    def apply_mapping(data, column, mapping, dataset_name):
        if column not in data.columns:
            return
        data[column] = data[column].replace(mapping)
        for old_value, new_value in mapping.items():
            mapping_log.append({
                "dataset": dataset_name,
                "column": column,
                "old_value": old_value,
                "new_value": new_value,
            })

    for name, data in datasets.items():
        if "channel" in data.columns:
            apply_mapping(data, "channel", CHANNEL_MAPPING, name)
        if "returned" in data.columns:
            apply_mapping(data, "returned", YES_NO_MAPPING, name)

    return pd.DataFrame(mapping_log)


# ---------------------------------------------------------------------------
# 3  Fix data types
# ---------------------------------------------------------------------------
def fix_dtypes(datasets: dict[str, pd.DataFrame]) -> None:
    """Cast date / numeric / categorical columns to the right dtype, in place."""
    for dataset_name, columns in DATE_COLUMNS.items():
        data = datasets.get(dataset_name)
        if data is None:
            continue
        for column in columns:
            if column in data.columns:
                data[column] = pd.to_datetime(data[column], errors="coerce")

    for dataset_name, columns in NUMERIC_COLUMNS.items():
        data = datasets.get(dataset_name)
        if data is None:
            continue
        for column in columns:
            if column in data.columns:
                data[column] = pd.to_numeric(data[column], errors="coerce")

    for dataset_name, columns in CATEGORICAL_COLUMNS.items():
        data = datasets.get(dataset_name)
        if data is None:
            continue
        for column in columns:
            if column in data.columns:
                data[column] = data[column].astype("category")


# ---------------------------------------------------------------------------
# 4  Business rule validation
# ---------------------------------------------------------------------------
def validate_business_rules(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Run sanity checks (price > 0, review_score in [1,5], etc.) and report violations."""
    order_items = datasets["order_items"]
    order_payments = datasets["order_payments"]
    order_reviews = datasets["order_reviews"]
    products = datasets["products"]
    orders = datasets["orders"]

    validation_results = [
        {"rule": "price > 0", "invalid_count": (order_items["price"] <= 0).sum()},
        {"rule": "freight_value >= 0", "invalid_count": (order_items["freight_value"] < 0).sum()},
        {"rule": "payment_value > 0", "invalid_count": (order_payments["payment_value"] <= 0).sum()},
        {
            "rule": "review_score between 1 and 5",
            "invalid_count": (~order_reviews["review_score"].between(1, 5)).sum(),
        },
    ]

    for column in ["product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"]:
        validation_results.append({
            "rule": f"{column} > 0",
            "invalid_count": (products[column] <= 0).sum(),
        })

    validation_results.append({
        "rule": "delivery_date >= purchase_date",
        "invalid_count": (
            orders["order_delivered_customer_date"] < orders["order_purchase_timestamp"]
        ).sum(),
    })

    return pd.DataFrame(validation_results)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def clean_datasets(
    datasets: dict[str, pd.DataFrame],
    save_metrics: bool = True,
    save_processed: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Run the full cleaning pipeline on the raw datasets dict, in place, and
    optionally persist audit reports / cleaned CSVs to disk.

    Returns the same dict (mutated in place) for convenience.
    """
    missing_audit = audit_missing_values(datasets)
    duplicate_audit = audit_exact_duplicates(datasets)
    duplicate_order_ids = audit_duplicate_order_ids(datasets["orders"])
    hidden_null_audit_before = audit_hidden_nulls(datasets)

    snake_case_columns(datasets)
    strip_whitespace_and_hidden_nulls(datasets)
    unify_casing(datasets)
    mapping_log = standardize_text_labels(datasets)

    fix_dtypes(datasets)
    validation_results = validate_business_rules(datasets)

    print("Missing values audit:\n", missing_audit)
    print("\nExact duplicates audit:\n", duplicate_audit)
    print("\nDuplicate order_ids:", duplicate_order_ids)
    print("\nHidden nulls found (pre-clean):\n", hidden_null_audit_before)
    print("\nBusiness rule validation:\n", validation_results)

    if save_metrics:
        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        missing_audit.to_csv(METRICS_DIR / "missing_values_audit.csv", index=False)
        duplicate_audit.to_csv(METRICS_DIR / "exact_duplicates_audit.csv", index=False)
        hidden_null_audit_before.to_csv(METRICS_DIR / "hidden_nulls_audit.csv", index=False)
        mapping_log.to_csv(METRICS_DIR / "text_label_mapping_log.csv", index=False)
        validation_results.to_csv(METRICS_DIR / "business_rule_validation.csv", index=False)

    if save_processed:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        for name, data in datasets.items():
            data.to_csv(PROCESSED_DATA_DIR / f"{name}_clean.csv", index=False)

    return datasets


if __name__ == "__main__":
    from load_data import load_raw_datasets

    raw = load_raw_datasets()
    clean_datasets(raw)