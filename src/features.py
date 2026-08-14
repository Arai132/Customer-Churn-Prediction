"""RFM + rolling-engagement feature engineering, built to work on either
an event-level log (many rows per customer) or a snapshot table (one row per customer)."""

import numpy as np
import pandas as pd

ROLLING_WINDOWS = (30, 60, 90)


def is_event_level(df: pd.DataFrame, id_col: str) -> bool:
    return df[id_col].duplicated().any()


def build_customer_features(
    df: pd.DataFrame,
    id_col: str,
    date_col: str,
    snapshot_date: pd.Timestamp,
    event_level: bool,
    amount_col: str | None = None,
    frequency_col: str | None = None,
    plan_col: str | None = None,
    rolling_windows: tuple[int, ...] = ROLLING_WINDOWS,
) -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    snapshot_date = pd.Timestamp(snapshot_date)

    if event_level:
        grouped = df.groupby(id_col)
        features = grouped[date_col].agg(first_date="min", last_date="max")
        features["frequency"] = grouped.size()
        features["monetary"] = grouped[amount_col].sum() if amount_col else features["frequency"].astype(float)
        if plan_col:
            features["plan_tier"] = grouped[plan_col].agg(lambda s: s.mode().iat[0] if not s.mode().empty else np.nan)
        for window in rolling_windows:
            cutoff = snapshot_date - pd.Timedelta(days=window)
            recent_counts = df.loc[df[date_col] > cutoff].groupby(id_col).size()
            features[f"activity_last_{window}d"] = recent_counts.reindex(features.index).fillna(0)
    else:
        features = df.set_index(id_col).rename(columns={date_col: "first_date"})
        features["last_date"] = features["first_date"]
        features["frequency"] = features[frequency_col] if frequency_col else 1.0
        features["monetary"] = features[amount_col] if amount_col else 0.0
        if plan_col:
            features["plan_tier"] = features[plan_col]
        for window in rolling_windows:
            features[f"activity_last_{window}d"] = np.nan

    features["recency_days"] = (snapshot_date - features["last_date"]).dt.days
    features["tenure_days"] = (snapshot_date - features["first_date"]).dt.days
    features["cohort_month"] = features["first_date"].values.astype("datetime64[M]")

    return features.reset_index().rename(columns={id_col: "customer_id"})
