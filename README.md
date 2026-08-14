# Customer Churn Prediction & Retention Analytics

A Streamlit dashboard for exploring customer churn: upload any customer usage/transaction
CSV (or use the built-in sample data), map your columns, and get RFM features,
cohort/retention analysis, and churn risk scoring — no code required.

## Features

- **Bring your own data** — upload an event-level log (one row per usage/transaction event)
  or a snapshot table (one row per customer). Map your column names to what the app needs;
  nothing needs to be renamed ahead of time.
- **Built-in sample data** — try the app instantly with either a generated sample dataset or
  IBM's published [Telco Customer Churn](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)
  benchmark (7,043 customers).
- **RFM + rolling-engagement features** — recency, frequency, monetary value, tenure, and
  30/60/90-day activity windows, computed automatically from your mapped columns.
- **Cohort & retention analysis** — DuckDB-powered monthly cohort retention heatmap and
  average retention curve (event-level data only).
- **Churn risk scoring** — logistic regression and random forest models trained on your data,
  with per-customer risk scores, risk tiers, and a downloadable CSV.
- **Model insights** — ROC curves, accuracy/precision/recall/AUC, feature importances /
  coefficients, and a confusion matrix on a held-out test set.

## Getting started

### Requirements

- Python 3.10+

### Install

```bash
pip install -r requirements.txt
```

### Run the app

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

### (Optional) Pre-fetch the Telco benchmark dataset

The app downloads and caches this automatically the first time you select it, but you can
also fetch it ahead of time:

```bash
python scripts/download_telco.py
```

## Using the app

1. **Choose a data source** in the sidebar — the Telco benchmark dataset, or upload your own
   CSV (a template CSV is available to download for reference).
2. **Map your columns** — customer ID, date, and optionally amount/revenue, frequency,
   plan/segment, and an existing churn label. If you don't have a churn label but have an
   event-level log, define churn as "no activity in the last N days" instead.
3. **Set a snapshot date** — the "as of" date used to compute recency, tenure, and rolling
   activity windows.
4. Explore the **Overview**, **Cohort & Retention**, **Churn Risk Scores**, and
   **Model Insights** tabs (tabs appear based on what your data supports).

## Project structure

```
app.py                    Streamlit app / UI
src/
  telco.py                Fetch + prepare the IBM Telco benchmark dataset
  sample_data.py          Generate the built-in sample dataset
  features.py             RFM + rolling-engagement feature engineering
  cohort.py                DuckDB cohort/retention SQL
  modeling.py              Train/score logistic regression + random forest models
  theme.py                 Shared Plotly/Streamlit styling
scripts/
  download_telco.py        CLI to pre-fetch the Telco dataset
data/                       Cached datasets (gitignored)
```
