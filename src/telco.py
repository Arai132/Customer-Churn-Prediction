"""Fetch + prepare the IBM Telco Customer Churn dataset - the standard churn
benchmark (7,043 customers, one row per customer) with widely-published
accuracy/AUC numbers to compare a model against.

Shared by scripts/download_telco.py (manual CLI fetch) and app.py (one-click
sample data option).
"""

import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

SOURCE_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_PATH = DATA_DIR / "telco_customer_churn_raw.csv"
PREPARED_PATH = DATA_DIR / "telco_customer_churn.csv"

COLUMN_MAPPING_GUIDE = {
    "Customer ID column": "customerID",
    "Date column": "signup_date",
    "Amount / revenue column": "MonthlyCharges (or TotalCharges)",
    "Plan / segment column": "Contract (or InternetService)",
    "Existing churn label": "Churn (churned value: 'Yes')",
}


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)


def prepare(raw_path: Path, as_of: pd.Timestamp) -> pd.DataFrame:
    df = pd.read_csv(raw_path)
    # TotalCharges ships as a string with 11 blank entries (brand-new customers, tenure=0).
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    df["signup_date"] = as_of - pd.to_timedelta(df["tenure"] * 30, unit="D")
    return df


def fetch_or_load(as_of: pd.Timestamp | None = None, force_refresh: bool = False) -> pd.DataFrame:
    """Return the prepared Telco dataset, using the cached CSV in data/ if present."""
    as_of = as_of or pd.Timestamp.today().normalize()

    if PREPARED_PATH.exists() and not force_refresh:
        return pd.read_csv(PREPARED_PATH, parse_dates=["signup_date"])

    if not RAW_PATH.exists() or force_refresh:
        download(SOURCE_URL, RAW_PATH)

    df = prepare(RAW_PATH, as_of)
    df.to_csv(PREPARED_PATH, index=False)
    return df
