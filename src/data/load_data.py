"""
load_data.py
============
Loads the raw Olist e-commerce CSV files from `data/raw/` and provides
helper functions to inspect them (columns, shapes, info, describe).

The raw CSVs are NOT shipped with this repo - place them manually in
`data/raw/` before running this module:

    data/raw/olist_customers_dataset.csv
    data/raw/olist_geolocation_dataset.csv
    data/raw/olist_order_items_dataset.csv
    data/raw/olist_order_payments_dataset.csv
    data/raw/olist_order_reviews_dataset.csv
    data/raw/olist_orders_dataset.csv
    data/raw/olist_products_dataset.csv
    data/raw/olist_sellers_dataset.csv
    data/raw/product_category_name_translation.csv
"""

from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

RAW_FILES = {
    "customers": "olist_customers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "order_payments": "olist_order_payments_dataset.csv",
    "order_reviews": "olist_order_reviews_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "category_name_translation": "product_category_name_translation.csv",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_raw_datasets(raw_dir: Path = RAW_DATA_DIR) -> dict[str, pd.DataFrame]:
    """
    Load all raw Olist CSVs into a dict of DataFrames.

    Parameters
    ----------
    raw_dir : Path
        Directory containing the raw CSV files.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys match RAW_FILES keys (customers, geolocation, order_items, ...).
    """
    datasets = {}
    missing = []

    for name, filename in RAW_FILES.items():
        filepath = raw_dir / filename
        if not filepath.exists():
            missing.append(str(filepath))
            continue
        datasets[name] = pd.read_csv(filepath)

    if missing:
        print("WARNING: the following raw files were not found:")
        for path in missing:
            print(f"  - {path}")
        print("Place the raw Olist CSVs in data/raw/ before running the pipeline.\n")

    return datasets


# ---------------------------------------------------------------------------
# Inspection helpers
# ---------------------------------------------------------------------------
def print_columns(**dataframes: pd.DataFrame) -> None:
    """Print the column list for each provided dataframe."""
    for name, df in dataframes.items():
        print(f"{name}:")
        print(df.columns.tolist())
        print()


def basic_info(datasets: dict[str, pd.DataFrame]) -> None:
    """Print shape, info() and describe() for every dataset in the dict."""
    for name, data in datasets.items():
        print("\n" + "=" * 70)
        print(f"{name.upper()}")
        print("=" * 70)

        print("Shape:", data.shape)

        print("\nInfo:")
        data.info()

        print("\nDescribe:")
        print(data.describe(include="all").T)


if __name__ == "__main__":
    dfs = load_raw_datasets()
    print_columns(**dfs)
    basic_info(dfs)