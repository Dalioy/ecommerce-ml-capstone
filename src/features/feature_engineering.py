"""
feature_engineering.py
=======================
Builds the customer-level modeling dataset (RFM + behavioral features) used
for clustering, starting from the cleaned Olist datasets.

Mirrors section "6 Feature Engineering" of the original notebook.
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"


# ---------------------------------------------------------------------------
# 6.1  Build the flat modeling dataset (orders + customers + items + products)
# ---------------------------------------------------------------------------
def build_modeling_dataset(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Join orders/customers/order_items/products/category_translation into one flat table."""
    df = (
        datasets["orders"]
        .merge(datasets["customers"], on="customer_id", how="left")
        .merge(datasets["order_items"], on="order_id", how="left")
        .merge(datasets["products"], on="product_id", how="left")
        .merge(datasets["category_name_translation"], on="product_category_name", how="left")
    )
    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
    return df


# ---------------------------------------------------------------------------
# 6.1 (order totals) + 6.2  Core RFM metrics
# ---------------------------------------------------------------------------
def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
    snapshot_date = df["order_purchase_timestamp"].max() + pd.Timedelta(days=1)

    order_value = (
        df.groupby(["order_id", "customer_unique_id", "order_purchase_timestamp"])
        .agg(order_total=("price", "sum"), order_freight=("freight_value", "sum"))
        .reset_index()
    )
    order_value["order_total_with_freight"] = order_value["order_total"] + order_value["order_freight"]

    rfm = order_value.groupby("customer_unique_id").agg(
        recency=("order_purchase_timestamp", lambda x: (snapshot_date - x.max()).days),
        frequency=("order_id", "nunique"),
        monetary=("order_total_with_freight", "sum"),
    ).reset_index()

    rfm["avg_order_value"] = rfm["monetary"] / rfm["frequency"]
    return rfm


# ---------------------------------------------------------------------------
# 6.3  Preferred category
# ---------------------------------------------------------------------------
def compute_preferred_category(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("customer_unique_id")["product_category_name_english"]
        .agg(lambda x: x.mode()[0] if not x.mode().empty else None)
        .reset_index()
        .rename(columns={"product_category_name_english": "preferred_category"})
    )


# ---------------------------------------------------------------------------
# 6.4  Area (state / city)
# ---------------------------------------------------------------------------
def compute_area(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("customer_unique_id")
        .agg(customer_state=("customer_state", "first"), customer_city=("customer_city", "first"))
        .reset_index()
    )


# ---------------------------------------------------------------------------
# 6.5  Return rate proxy (based on order_status)
# ---------------------------------------------------------------------------
def compute_return_rate(df: pd.DataFrame) -> pd.DataFrame:
    order_status_flags = df.groupby(["customer_unique_id", "order_id"])["order_status"].first().reset_index()

    return_rate = (
        order_status_flags.groupby("customer_unique_id")
        .agg(
            total_orders=("order_id", "nunique"),
            problematic_orders=("order_status", lambda x: x.isin(["canceled", "unavailable"]).sum()),
        )
        .reset_index()
    )
    return_rate["return_rate"] = return_rate["problematic_orders"] / return_rate["total_orders"]
    return return_rate


# ---------------------------------------------------------------------------
# 6.6  Preferred channel (payment_type) & avg installments
# ---------------------------------------------------------------------------
def compute_payment_features(df: pd.DataFrame, order_payments: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    order_customer_map = df[["order_id", "customer_unique_id"]].drop_duplicates()
    payments_with_customer = order_payments.merge(order_customer_map, on="order_id", how="left")

    preferred_channel = (
        payments_with_customer.groupby("customer_unique_id")["payment_type"]
        .agg(lambda x: x.mode()[0] if not x.mode().empty else None)
        .reset_index()
        .rename(columns={"payment_type": "preferred_channel"})
    )

    avg_installments = (
        payments_with_customer.groupby("customer_unique_id")["payment_installments"]
        .mean()
        .reset_index()
        .rename(columns={"payment_installments": "avg_installments"})
    )

    return preferred_channel, avg_installments


# ---------------------------------------------------------------------------
# 6.7  Average review score
# ---------------------------------------------------------------------------
def compute_avg_review_score(df: pd.DataFrame, order_reviews: pd.DataFrame) -> pd.DataFrame:
    order_customer_map = df[["order_id", "customer_unique_id"]].drop_duplicates()
    reviews_with_customer = order_reviews.merge(order_customer_map, on="order_id", how="left")

    return (
        reviews_with_customer.groupby("customer_unique_id")["review_score"]
        .mean()
        .reset_index()
        .rename(columns={"review_score": "avg_review_score"})
    )


# ---------------------------------------------------------------------------
# 6.8  Delivery timing features
# ---------------------------------------------------------------------------
def compute_delivery_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["order_delivered_customer_date"] = pd.to_datetime(df["order_delivered_customer_date"])
    df["order_estimated_delivery_date"] = pd.to_datetime(df["order_estimated_delivery_date"])

    df["delivery_delay_days"] = (
        df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]
    ).dt.days
    df["delivery_time_days"] = (
        df["order_delivered_customer_date"] - df["order_purchase_timestamp"]
    ).dt.days

    return df.groupby("customer_unique_id").agg(
        delivery_delay=("delivery_delay_days", "mean"),
        avg_delivery_time=("delivery_time_days", "mean"),
    ).reset_index()


# ---------------------------------------------------------------------------
# 6.9  Avg freight ratio
# ---------------------------------------------------------------------------
def compute_freight_ratio(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["freight_ratio"] = df["freight_value"] / df["price"].replace(0, pd.NA)

    return (
        df.groupby("customer_unique_id")["freight_ratio"]
        .mean()
        .reset_index()
        .rename(columns={"freight_ratio": "avg_freight_ratio"})
    )


# ---------------------------------------------------------------------------
# 6.10  Category diversity
# ---------------------------------------------------------------------------
def compute_category_diversity(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("customer_unique_id")["product_category_name_english"]
        .nunique()
        .reset_index()
        .rename(columns={"product_category_name_english": "category_diversity"})
    )


# ---------------------------------------------------------------------------
# 6.11  Tenure
# ---------------------------------------------------------------------------
def compute_tenure(df: pd.DataFrame) -> pd.DataFrame:
    tenure = (
        df.groupby("customer_unique_id")["order_purchase_timestamp"]
        .agg(first_order="min", last_order="max")
        .reset_index()
    )
    tenure["tenure_days"] = (tenure["last_order"] - tenure["first_order"]).dt.days
    return tenure[["customer_unique_id", "tenure_days"]]


# ---------------------------------------------------------------------------
# 6.12  Merge everything into the final RFM+ feature table
# ---------------------------------------------------------------------------
def build_feature_table(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Run the full feature-engineering pipeline and return the customer-level
    `rfm` DataFrame (recency, frequency, monetary + behavioral features)
    ready to feed into clustering.
    """
    df = build_modeling_dataset(datasets)

    rfm = compute_rfm(df)
    preferred_category = compute_preferred_category(df)
    area = compute_area(df)
    return_rate = compute_return_rate(df)
    preferred_channel, avg_installments = compute_payment_features(df, datasets["order_payments"])
    avg_review_score = compute_avg_review_score(df, datasets["order_reviews"])
    delivery_features = compute_delivery_features(df)
    avg_freight_ratio = compute_freight_ratio(df)
    category_diversity = compute_category_diversity(df)
    tenure = compute_tenure(df)

    rfm = (
        rfm.merge(preferred_category, on="customer_unique_id", how="left")
        .merge(area, on="customer_unique_id", how="left")
        .merge(return_rate[["customer_unique_id", "return_rate"]], on="customer_unique_id", how="left")
        .merge(preferred_channel, on="customer_unique_id", how="left")
        .merge(avg_installments, on="customer_unique_id", how="left")
        .merge(avg_review_score, on="customer_unique_id", how="left")
        .merge(delivery_features, on="customer_unique_id", how="left")
        .merge(avg_freight_ratio, on="customer_unique_id", how="left")
        .merge(category_diversity, on="customer_unique_id", how="left")
        .merge(tenure, on="customer_unique_id", how="left")
    )

    return rfm


def save_feature_table(rfm: pd.DataFrame, path: Path = PROCESSED_DATA_DIR / "rfm_final.csv") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rfm.to_csv(path, index=False)
    print("Saved:", rfm.shape, "->", path)


if __name__ == "__main__":
    from src.data.load_data import load_raw_datasets
    from src.data.clean_data import clean_datasets

    raw = load_raw_datasets()
    clean = clean_datasets(raw, save_metrics=False, save_processed=False)
    rfm_table = build_feature_table(clean)
    save_feature_table(rfm_table)