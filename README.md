# Customer Segmentation — Olist E-commerce Dataset

An end-to-end **RFM-based customer segmentation pipeline** built on the
Olist Brazilian e-commerce dataset. Raw transactional data is cleaned,
enriched into a customer-level behavioral feature table (RFM+), clustered
with three different algorithms, and translated into named, business-ready
customer segments with actionable recommendations.

```
Raw CSVs  →  Cleaning  →  Feature Engineering (RFM+)  →  Clustering  →  Segment Profiles & Insights
```

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Outputs](#outputs)
- [Requirements](#requirements)

---

## Overview

The goal of this project is to turn raw order/customer/payment/review data
into a small number of **actionable customer segments** — e.g. loyal
high-value customers, delivery-frustrated customers, bargain hunters — that
a business team can act on directly (retention offers, logistics fixes,
reactivation campaigns).

The codebase separates **reusable logic** (`src/`) from **exploration and
storytelling** (`notebooks/`), so the same functions power both the
interactive notebooks and a single reproducible pipeline script.

**Key techniques used:**
- Data auditing: missing values, exact duplicates, hidden nulls, dtype validation, business-rule checks
- Feature engineering: Recency, Frequency, Monetary (RFM) plus behavioral signals (return rate, delivery delay, freight ratio, category diversity, tenure, preferred channel/category)
- Preprocessing: log-transforms for skewed features, median imputation, standard scaling
- Clustering: **KMeans**, **DBSCAN**, and **Agglomerative (hierarchical)** clustering, compared via Silhouette Score and Davies-Bouldin Index
- Segment profiling: automatic segment naming and business-insight generation

---

## Project Structure

```
data/
├── raw/                         # place the raw Olist CSVs here manually (not tracked)
└── processed/                   # cleaned datasets + rfm_final.csv (generated)

src/
├── data/
│   ├── load_data.py             # load raw CSVs, basic inspection (shape/info/describe)
│   └── clean_data.py            # missing/duplicate audits, dtype fixes, business-rule validation
├── features/
│   └── feature_engineering.py   # builds the customer-level RFM+ feature table
├── clustering/
│   ├── train.py                 # preprocessing, optimal-K selection, model fitting
│   └── evaluate.py              # metrics, segment profiling, naming, business insights
├── visualization.py             # all EDA + feature-level plots (saved to outputs/figures)
└── pipeline.py                  # orchestrates the full pipeline end-to-end

notebooks/
├── 01_exploration.ipynb         # load, inspect, clean, EDA plots
├── 02_feature_engineering.ipynb # build the RFM+ feature table
└── 03_clustering_analysis.ipynb # cluster, evaluate, profile & name segments

outputs/
├── figures/                     # all saved plots (.png)
├── metrics/                     # audit reports & model comparison metrics (.csv)
└── segments/                    # segment profile tables & business insights (.csv/.txt)

models/
└── clustering_model.pkl         # final fitted model + scaler (generated)
```

Every notebook simply calls into `src/` — the logic lives in one place
whether you run it interactively or through the pipeline script.

---

## Methodology

| Stage | Module | What happens |
|---|---|---|
| **1. Load** | `src/data/load_data.py` | Reads the 9 raw Olist CSVs, prints shapes/columns/dtypes |
| **2. Clean** | `src/data/clean_data.py` | Missing-value & duplicate audits → snake_case columns → strip/normalize text → fix dtypes (dates/numeric/categorical) → validate business rules (price > 0, review_score ∈ [1,5], etc.) |
| **3. EDA** | `src/visualization.py` | Order status, review scores, monthly trend, top states/categories/sellers, price & delivery distributions |
| **4. Feature Engineering** | `src/features/feature_engineering.py` | Builds the flat orders×customers×items×products table, then aggregates to one row per `customer_unique_id`: recency, frequency, monetary, avg order value, preferred category/channel, return rate, avg installments/review score, delivery delay & time, freight ratio, category diversity, tenure |
| **5. Preprocessing** | `src/clustering/train.py` | Log-transforms skewed columns, imputes missing delivery/freight values with the median, standard-scales the final feature set |
| **6. Model Selection** | `src/clustering/train.py` | Elbow method + Silhouette score across K=2..8 to pick the optimal number of clusters |
| **7. Clustering** | `src/clustering/train.py` | Fits **KMeans**, **DBSCAN**, and **Agglomerative** clustering for comparison |
| **8. Evaluation** | `src/clustering/evaluate.py` | Silhouette Score & Davies-Bouldin Index per model, segment size balance |
| **9. Profiling & Naming** | `src/clustering/evaluate.py` | Aggregates each segment's stats, assigns a heuristic business name (e.g. *High-Value Loyalists*, *Delivery-Frustrated*, *Bargain Hunters*), generates plain-language recommendations |

---

## Getting Started

### 1. Clone / unzip the project and install dependencies

```bash
pip install -r requirements.txt
```

### 2. Add the raw data

This repo does **not** ship the dataset. Download the
[Olist Brazilian E-commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
and place the following files manually in `data/raw/`:

```
data/raw/olist_customers_dataset.csv
data/raw/olist_geolocation_dataset.csv
data/raw/olist_order_items_dataset.csv
data/raw/olist_order_payments_dataset.csv
data/raw/olist_order_reviews_dataset.csv
data/raw/olist_orders_dataset.csv
data/raw/olist_products_dataset.csv
data/raw/olist_sellers_dataset.csv
data/raw/product_category_name_translation.csv
```

---

## Usage

### Option A — Run the full pipeline (recommended)

```bash
python -m src.pipeline
```

This single command loads the raw data, cleans it, builds the RFM+ feature
table, generates all EDA/feature plots, fits and evaluates the clustering
models, profiles and names the segments, and saves every artifact to
`data/processed/`, `outputs/`, and `models/`.

### Option B — Walk through the notebook

Open **`notebooks/customer_segmentation.ipynb`** — it walks through the full
analysis in order: data loading & inspection, cleaning, EDA, feature
engineering (RFM+), and clustering (model selection, fitting, evaluation,
segment profiling & naming).

---

## Outputs

After a full run you'll find:

- **`data/processed/`** — cleaned datasets + `rfm_final.csv` (the customer feature table)
- **`outputs/figures/`** — ~17 PNG charts (EDA + feature + clustering diagnostics)
- **`outputs/metrics/`** — audit CSVs (missing values, duplicates, business-rule violations) + model comparison table
- **`outputs/segments/`** — `segment_profile_table.csv`, `segment_profile_named.csv`, `business_insights.txt`
- **`models/clustering_model.pkl`** — the final fitted KMeans model + its `StandardScaler`, pickled together for reuse on new data

---

## Requirements

See [`requirements.txt`](./requirements.txt):

| Package | Purpose |
|---|---|
| `numpy` | Array math, log-transforms, clipping |
| `pandas` | Data loading, joins, aggregation (the backbone of every module) |
| `matplotlib` | All EDA and feature-level visualizations |
| `scipy` | Hierarchical clustering utilities (`linkage`, `dendrogram`) |
| `scikit-learn` | `KMeans`, `DBSCAN`, `AgglomerativeClustering`, `StandardScaler`, evaluation metrics |
| `jupyter` | Running the notebooks |
| `nbformat` | Notebook file I/O (used internally by Jupyter) |

```bash
pip install -r requirements.txt
```