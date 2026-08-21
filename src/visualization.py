"""
visualization.py
================
All plotting functions used across the project (EDA plots from section 5 of
the original notebook + feature-level plots from section 7). Every function
saves its figure into outputs/figures/ and also calls plt.show() so it works
fine inside a notebook.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"


def _ensure_dir() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _savefig(filename: str) -> None:
    _ensure_dir()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# 5  EDA plots (raw / cleaned datasets)
# ---------------------------------------------------------------------------
def plot_order_status_distribution(orders: pd.DataFrame) -> None:
    status_counts = orders["order_status"].value_counts()
    plt.figure()
    plt.bar(status_counts.index, status_counts.values, color="steelblue")
    plt.title("Order Status Distribution")
    plt.xlabel("Order Status")
    plt.ylabel("Number of Orders")
    plt.xticks(rotation=45)
    _savefig("01_order_status_distribution.png")


def plot_review_score_distribution(order_reviews: pd.DataFrame) -> None:
    review_counts = order_reviews["review_score"].value_counts().sort_index()
    plt.figure()
    plt.bar(review_counts.index.astype(str), review_counts.values, color="darkorange")
    plt.title("Review Score Distribution")
    plt.xlabel("Review Score")
    plt.ylabel("Number of Reviews")
    _savefig("02_review_score_distribution.png")


def plot_monthly_orders_trend(orders: pd.DataFrame) -> None:
    orders_by_month = (
        orders.dropna(subset=["order_purchase_timestamp"])
        .set_index("order_purchase_timestamp")
        .resample("MS")
        .size()
    )
    plt.figure()
    plt.plot(orders_by_month.index, orders_by_month.values, marker="o", color="seagreen")
    plt.title("Monthly Orders Trend")
    plt.xlabel("Month")
    plt.ylabel("Number of Orders")
    plt.xticks(rotation=45)
    _savefig("03_monthly_orders_trend.png")


def plot_top10_states_customers(customers: pd.DataFrame) -> None:
    top_states = customers["customer_state"].value_counts().head(10)
    plt.figure()
    plt.bar(top_states.index, top_states.values, color="indianred")
    plt.title("Top 10 States by Number of Customers")
    plt.xlabel("State")
    plt.ylabel("Number of Customers")
    _savefig("04_top10_states_customers.png")


def plot_top10_categories(order_items: pd.DataFrame, products: pd.DataFrame, category_translation: pd.DataFrame) -> pd.DataFrame:
    """Returns the merged items_products table (reused by other plots) as a side effect."""
    items_products = order_items.merge(products[["product_id", "product_category_name"]], on="product_id", how="left")
    items_products = items_products.merge(category_translation, on="product_category_name", how="left")

    top_categories = items_products["product_category_name_english"].value_counts().head(10)
    plt.figure()
    plt.barh(top_categories.index[::-1], top_categories.values[::-1], color="mediumpurple")
    plt.title("Top 10 Product Categories by Items Sold")
    plt.xlabel("Number of Items Sold")
    _savefig("05_top10_categories.png")

    return items_products


def plot_payment_type_distribution(order_payments: pd.DataFrame) -> None:
    payment_type_counts = order_payments["payment_type"].value_counts()
    plt.figure()
    plt.pie(payment_type_counts.values, labels=payment_type_counts.index, autopct="%1.1f%%", startangle=90)
    plt.title("Payment Type Distribution")
    _savefig("06_payment_type_distribution.png")


def plot_installments_distribution(order_payments: pd.DataFrame) -> None:
    installments_counts = order_payments["payment_installments"].value_counts().sort_index()
    plt.figure()
    plt.bar(installments_counts.index.astype(str), installments_counts.values, color="teal")
    plt.title("Payment Installments Distribution")
    plt.xlabel("Number of Installments")
    plt.ylabel("Count")
    _savefig("07_payment_installments_distribution.png")


def plot_price_distribution(order_items: pd.DataFrame) -> None:
    price_clipped = order_items["price"].clip(upper=order_items["price"].quantile(0.99))
    plt.figure()
    plt.hist(price_clipped, bins=40, color="goldenrod", edgecolor="black")
    plt.title("Product Price Distribution (99th percentile clipped)")
    plt.xlabel("Price")
    plt.ylabel("Frequency")
    _savefig("08_price_distribution.png")


def plot_price_vs_freight(order_items: pd.DataFrame, random_state: int = 1) -> None:
    sample = order_items.sample(min(3000, len(order_items)), random_state=random_state)
    plt.figure()
    plt.scatter(sample["price"], sample["freight_value"], alpha=0.4, color="crimson", s=15)
    plt.title("Price vs Freight Value")
    plt.xlabel("Price")
    plt.ylabel("Freight Value")
    _savefig("09_price_vs_freight.png")


def plot_delivery_time_distribution(orders: pd.DataFrame) -> None:
    delivery_days = (orders["order_delivered_customer_date"] - orders["order_purchase_timestamp"]).dt.days
    delivery_days = delivery_days.dropna()
    delivery_days = delivery_days[(delivery_days >= 0) & (delivery_days <= 60)]

    plt.figure()
    plt.hist(delivery_days, bins=40, color="cornflowerblue", edgecolor="black")
    plt.title("Delivery Time Distribution (days)")
    plt.xlabel("Days from Purchase to Delivery")
    plt.ylabel("Number of Orders")
    _savefig("10_delivery_time_distribution.png")


def plot_top10_sellers(order_items: pd.DataFrame) -> None:
    top_sellers = order_items["seller_id"].value_counts().head(10)
    plt.figure()
    plt.bar(range(len(top_sellers)), top_sellers.values, color="slategray")
    plt.title("Top 10 Sellers by Number of Order Items")
    plt.xlabel("Seller (ranked)")
    plt.ylabel("Number of Items Sold")
    plt.xticks(range(len(top_sellers)), [f"S{i + 1}" for i in range(len(top_sellers))])
    _savefig("11_top10_sellers.png")


def plot_avg_review_score_top10_categories(
    order_reviews: pd.DataFrame,
    orders: pd.DataFrame,
    order_items: pd.DataFrame,
    products: pd.DataFrame,
    category_translation: pd.DataFrame,
    items_products: pd.DataFrame,
) -> None:
    reviews_orders = order_reviews.merge(orders[["order_id"]], on="order_id", how="inner")
    reviews_items = reviews_orders.merge(order_items[["order_id", "product_id"]], on="order_id", how="left")
    reviews_items = reviews_items.merge(products[["product_id", "product_category_name"]], on="product_id", how="left")
    reviews_items = reviews_items.merge(category_translation, on="product_category_name", how="left")

    top10_cats = items_products["product_category_name_english"].value_counts().head(10).index
    avg_score = (
        reviews_items[reviews_items["product_category_name_english"].isin(top10_cats)]
        .groupby("product_category_name_english")["review_score"]
        .mean()
        .sort_values()
    )

    plt.figure()
    plt.barh(avg_score.index, avg_score.values, color="mediumseagreen")
    plt.title("Average Review Score - Top 10 Categories")
    plt.xlabel("Average Review Score")
    plt.xlim(0, 5)
    _savefig("12_avg_review_score_top10_categories.png")


def run_all_eda_plots(datasets: dict[str, pd.DataFrame]) -> None:
    """Convenience wrapper: run every EDA plot in section-5 order."""
    plot_order_status_distribution(datasets["orders"])
    plot_review_score_distribution(datasets["order_reviews"])
    plot_monthly_orders_trend(datasets["orders"])
    plot_top10_states_customers(datasets["customers"])
    items_products = plot_top10_categories(
        datasets["order_items"], datasets["products"], datasets["category_name_translation"]
    )
    plot_payment_type_distribution(datasets["order_payments"])
    plot_installments_distribution(datasets["order_payments"])
    plot_price_distribution(datasets["order_items"])
    plot_price_vs_freight(datasets["order_items"])
    plot_delivery_time_distribution(datasets["orders"])
    plot_top10_sellers(datasets["order_items"])
    plot_avg_review_score_top10_categories(
        datasets["order_reviews"],
        datasets["orders"],
        datasets["order_items"],
        datasets["products"],
        datasets["category_name_translation"],
        items_products,
    )


# ---------------------------------------------------------------------------
# 7  Feature-level plots (post feature engineering, on the rfm table)
# ---------------------------------------------------------------------------
def plot_correlation_heatmap(rfm: pd.DataFrame) -> None:
    numeric_cols = [
        "recency", "frequency", "monetary", "avg_order_value", "return_rate",
        "avg_installments", "avg_review_score", "delivery_delay",
        "avg_delivery_time", "avg_freight_ratio", "category_diversity", "tenure_days",
    ]
    corr = rfm[numeric_cols].corr()

    plt.figure(figsize=(9, 7))
    im = plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=90)
    plt.yticks(range(len(numeric_cols)), numeric_cols)
    plt.title("Correlation Heatmap - Customer Features")
    _savefig("13_correlation_heatmap.png")


def plot_monetary_distribution(rfm: pd.DataFrame) -> None:
    monetary_clipped = rfm["monetary"].clip(upper=rfm["monetary"].quantile(0.99))
    plt.figure()
    plt.hist(monetary_clipped, bins=40, color="darkgreen", edgecolor="black")
    plt.title("Monetary Value Distribution (99th percentile clipped)")
    plt.xlabel("Monetary Value")
    plt.ylabel("Number of Customers")
    _savefig("14_monetary_distribution.png")


def plot_recency_vs_frequency(rfm: pd.DataFrame, random_state: int = 1) -> None:
    sample = rfm.sample(min(3000, len(rfm)), random_state=random_state)
    plt.figure()
    sc = plt.scatter(
        sample["recency"], sample["frequency"], c=sample["monetary"],
        cmap="viridis", alpha=0.6, s=20,
    )
    plt.colorbar(sc, label="Monetary Value")
    plt.title("Recency vs Frequency (colored by Monetary)")
    plt.xlabel("Recency (days)")
    plt.ylabel("Frequency")
    _savefig("15_recency_vs_frequency.png")


def plot_top10_states_features(rfm: pd.DataFrame) -> None:
    top_states = rfm["customer_state"].value_counts().head(10)
    plt.figure()
    plt.bar(top_states.index, top_states.values, color="steelblue")
    plt.title("Top 10 States by Number of Customers")
    plt.xlabel("State")
    plt.ylabel("Number of Customers")
    _savefig("16_top10_states_customer_features.png")


def plot_preferred_category_distribution(rfm: pd.DataFrame) -> None:
    top_pref_categories = rfm["preferred_category"].value_counts().head(10)
    plt.figure()
    plt.barh(top_pref_categories.index[::-1], top_pref_categories.values[::-1], color="mediumpurple")
    plt.title("Top 10 Preferred Categories Among Customers")
    plt.xlabel("Number of Customers")
    _savefig("17_preferred_category_distribution.png")


def run_all_feature_plots(rfm: pd.DataFrame) -> None:
    """Convenience wrapper: run every feature-level plot in section-7 order."""
    plot_correlation_heatmap(rfm)
    plot_monetary_distribution(rfm)
    plot_recency_vs_frequency(rfm)
    plot_top10_states_features(rfm)
    plot_preferred_category_distribution(rfm)

    