"""Download and prepare the IBM Telco Customer Churn dataset for use with this app.

This is the standard churn benchmark dataset (7,043 customers, one row per
customer) with widely-published accuracy/AUC numbers to compare your model
against. It has no real date column, so this script derives a `signup_date`
from `tenure` (months) since the app requires a date column to map.

Note: the app's "Telco benchmark dataset" sample-data option does this
automatically - this script is only needed if you want the CSV on disk
ahead of time, or want to inspect/edit it before uploading.

Usage:
    ./.venv/bin/python scripts/download_telco.py [--as-of YYYY-MM-DD]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import telco


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--as-of", type=str, default=None,
        help="Reference date for deriving signup_date from tenure (default: today)",
    )
    args = parser.parse_args()
    as_of = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp.today().normalize()

    print(f"Downloading Telco Customer Churn dataset from {telco.SOURCE_URL} ...")
    try:
        df = telco.fetch_or_load(as_of=as_of, force_refresh=True)
    except (OSError, ValueError) as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        print(
            "Download it manually instead: search Kaggle for 'Telco Customer Churn' "
            f"(uploader: blastchar) and save the CSV to {telco.PREPARED_PATH}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Saved {len(df):,} rows x {df.shape[1]} columns to {telco.PREPARED_PATH}")
    print("\nWhen you upload this CSV in the app sidebar, map columns as:")
    for field, value in telco.COLUMN_MAPPING_GUIDE.items():
        print(f"  {field:<26} -> {value}")
    print("This is a snapshot table (one row per customer), so the Cohort & Retention tab won't apply.")


if __name__ == "__main__":
    main()
