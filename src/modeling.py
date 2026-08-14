"""Train logistic regression + random forest churn models and score every customer."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

MIN_ROWS_TO_TRAIN = 20


def prepare_model_matrix(features: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    X = features[feature_cols].copy()
    categorical_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    if categorical_cols:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    X = X.apply(pd.to_numeric, errors="coerce")
    return X.fillna(X.median(numeric_only=True))


def train_models(
    features: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    test_size: float = 0.25,
    random_state: int = 42,
) -> dict:
    X = prepare_model_matrix(features, feature_cols)
    y = features[label_col].astype(int)

    stratify = y if y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logreg = LogisticRegression(max_iter=1000, random_state=random_state).fit(X_train_scaled, y_train)
    rf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=random_state, n_jobs=-1).fit(
        X_train, y_train
    )

    metrics = {}
    for name, model, X_te in [("Logistic Regression", logreg, X_test_scaled), ("Random Forest", rf, X_test)]:
        proba = model.predict_proba(X_te)[:, 1]
        preds = (proba >= 0.5).astype(int)
        fpr, tpr, _ = roc_curve(y_test, proba)
        metrics[name] = {
            "accuracy": accuracy_score(y_test, preds),
            "auc": roc_auc_score(y_test, proba) if y_test.nunique() > 1 else np.nan,
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
            "fpr": fpr,
            "tpr": tpr,
            "confusion_matrix": confusion_matrix(y_test, preds),
        }

    # Refit on the full dataset so every customer gets an operational risk score.
    full_scaler = StandardScaler().fit(X)
    logreg_full = LogisticRegression(max_iter=1000, random_state=random_state).fit(full_scaler.transform(X), y)
    rf_full = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=random_state, n_jobs=-1).fit(X, y)

    metrics["Logistic Regression"]["coefficients"] = pd.Series(logreg_full.coef_[0], index=X.columns).sort_values()
    metrics["Random Forest"]["feature_importances"] = pd.Series(
        rf_full.feature_importances_, index=X.columns
    ).sort_values()

    return {
        "metrics": metrics,
        "full_models": {
            "Logistic Regression": (full_scaler, logreg_full),
            "Random Forest": (None, rf_full),
        },
        "feature_columns": X.columns.tolist(),
        "X_full": X,
    }


def score_customers(full_models: dict, model_name: str, X_full: pd.DataFrame) -> np.ndarray:
    scaler, model = full_models[model_name]
    X_input = scaler.transform(X_full) if scaler is not None else X_full
    return model.predict_proba(X_input)[:, 1]
