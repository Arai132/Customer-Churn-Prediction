"""Customer Churn Prediction & Retention Analytics Dashboard.

Streamlit app: upload any customer usage/transaction CSV (or try the built-in
sample data), map your columns, and get RFM + rolling-engagement features,
DuckDB-powered cohort/retention analysis, and logistic regression + random
forest churn risk scoring - no code required.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import telco, theme
from src.cohort import build_cohort_retention_table
from src.features import ROLLING_WINDOWS, build_customer_features, is_event_level
from src.modeling import MIN_ROWS_TO_TRAIN, score_customers, train_models
from src.sample_data import generate_sample_data

POSITIVE_CHURN_VALUES = {"yes", "true", "1", "churned", "cancelled", "canceled"}

st.set_page_config(page_title="Customer Churn & Retention Dashboard", layout="wide")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

@st.cache_data
def load_sample_data() -> pd.DataFrame:
    return generate_sample_data()


@st.cache_data(show_spinner="Downloading the Telco benchmark dataset...")
def load_telco_sample() -> pd.DataFrame:
    return telco.fetch_or_load()


def guess_churned_value(values: list) -> object:
    for v in values:
        if str(v).strip().lower() in POSITIVE_CHURN_VALUES:
            return v
    return values[0]


@st.cache_data
def read_csv(file) -> pd.DataFrame:
    try:
        return pd.read_csv(file)
    except UnicodeDecodeError:
        file.seek(0)
        return pd.read_csv(file, encoding="latin-1")


def guess_index(columns: list[str], keywords: list[str], has_none_option: bool = False) -> int:
    offset = 1 if has_none_option else 0
    cols_lower = [c.lower() for c in columns]
    for kw in keywords:
        for i, c in enumerate(cols_lower):
            if kw in c:
                return i + offset
    return 0


def sequential_colorscale(hex_steps: list[str]) -> list[list]:
    n = len(hex_steps)
    return [[i / (n - 1), color] for i, color in enumerate(hex_steps)]


def style_fig(fig: go.Figure) -> go.Figure:
    fig.update_layout(**theme.plotly_layout_defaults())
    return fig


@st.cache_data(show_spinner="Training churn models...")
def cached_train_models(features_key: pd.DataFrame, feature_cols: list[str], label_col: str):
    return train_models(features_key, feature_cols, label_col)


# --------------------------------------------------------------------------
# Sidebar: data source
# --------------------------------------------------------------------------

st.sidebar.header("1. Data source")
source = st.sidebar.radio(
    "Choose a data source",
    ["Telco benchmark dataset", "Upload your own CSV"],
    label_visibility="collapsed",
)

raw_df = None
if source == "Telco benchmark dataset":
    try:
        raw_df = load_telco_sample()
    except OSError as exc:
        st.sidebar.error(f"Couldn't download the dataset: {exc}")
        st.stop()
    st.sidebar.caption(
        "IBM's real Telco Customer Churn dataset (7,043 customers) - a published "
        "benchmark, useful for sanity-checking model performance against known results."
    )
else:
    st.sidebar.download_button(
        "Download a template CSV",
        data=load_sample_data().to_csv(index=False),
        file_name="churn_data_template.csv",
        mime="text/csv",
        help="An event-level log (one row per usage/transaction event) shaped the way this app expects.",
    )
    uploaded = st.sidebar.file_uploader("CSV file", type="csv")
    if uploaded is not None:
        raw_df = read_csv(uploaded)

if raw_df is None:
    st.title("Customer Churn Prediction & Retention Analytics")
    st.info("Upload a CSV in the sidebar, or switch to sample data, to get started.")
    st.markdown(
        "**What kind of file works best?** Either an event-level log (one row per login, "
        "usage event, or transaction, with a customer ID and a date) or a snapshot table "
        "(one row per customer). Either way, you'll map your own column names to what the "
        "app needs in the next step - nothing needs to be renamed ahead of time."
    )
    st.stop()

st.title("Customer Churn Prediction & Retention Analytics")

with st.expander("Preview of your data", expanded=False):
    st.dataframe(raw_df.head(20), width='stretch')
    st.caption(f"{len(raw_df):,} rows, {raw_df.shape[1]} columns")

columns = raw_df.columns.tolist()

# --------------------------------------------------------------------------
# Sidebar: column mapping
# --------------------------------------------------------------------------

st.sidebar.header("2. Map your columns")

id_col = st.sidebar.selectbox(
    "Customer ID column", columns, index=guess_index(columns, ["customer_id", "customer", "user_id", "id"])
)
date_col = st.sidebar.selectbox(
    "Date column (event date, or signup date if one row per customer)",
    columns,
    index=guess_index(columns, ["event_date", "date", "signup"]),
)

try:
    raw_df[date_col] = pd.to_datetime(raw_df[date_col])
except Exception as exc:
    st.error(f"Couldn't parse '{date_col}' as a date: {exc}")
    st.stop()

event_level = is_event_level(raw_df, id_col)
st.sidebar.caption(
    "Detected **event-level log** (multiple rows per customer)." if event_level
    else "Detected **one row per customer** (snapshot table)."
)

none_option = "None"
amount_col = st.sidebar.selectbox(
    "Amount / revenue column (optional)", [none_option] + columns,
    index=guess_index(columns, ["amount", "revenue", "price", "spend", "charges"], has_none_option=True),
)
amount_col = None if amount_col == none_option else amount_col

frequency_col = None
if not event_level:
    frequency_col = st.sidebar.selectbox(
        "Frequency column (optional - only used for snapshot tables)", [none_option] + columns,
        index=guess_index(columns, ["frequency", "visits", "logins", "sessions"], has_none_option=True),
    )
    frequency_col = None if frequency_col == none_option else frequency_col

plan_col = st.sidebar.selectbox(
    "Plan / segment column (optional)", [none_option] + columns,
    index=guess_index(columns, ["plan", "tier", "segment", "contract"], has_none_option=True),
)
plan_col = None if plan_col == none_option else plan_col

churn_col = st.sidebar.selectbox(
    "Existing churn label column (optional)", [none_option] + columns,
    index=guess_index(columns, ["churn", "cancelled", "canceled"], has_none_option=True),
)
churn_col = None if churn_col == none_option else churn_col

churned_value = None
inactivity_threshold = None
if churn_col:
    unique_vals = raw_df[churn_col].dropna().unique().tolist()
    default_churned = guess_churned_value(unique_vals)
    churned_value = st.sidebar.selectbox(
        "Which value in that column means 'churned'?", unique_vals,
        index=unique_vals.index(default_churned),
    )
elif event_level:
    inactivity_threshold = st.sidebar.slider(
        "No churn column? Define churn as: no activity in the last N days", 30, 365, 90, step=15
    )
else:
    st.sidebar.warning("No churn column and no repeated activity to infer inactivity from - map a churn column to enable modeling.")

st.sidebar.header("3. Snapshot date")
default_snapshot = raw_df[date_col].max()
snapshot_date = st.sidebar.date_input("Analyze data as of", value=default_snapshot)
snapshot_date = pd.Timestamp(snapshot_date)

# --------------------------------------------------------------------------
# Feature engineering
# --------------------------------------------------------------------------

features_df = build_customer_features(
    raw_df,
    id_col=id_col,
    date_col=date_col,
    snapshot_date=snapshot_date,
    event_level=event_level,
    amount_col=amount_col,
    frequency_col=frequency_col,
    plan_col=plan_col,
    rolling_windows=ROLLING_WINDOWS,
)

# Attach churn label
if churn_col:
    label_by_customer = (
        raw_df.groupby(id_col)[churn_col]
        .agg(lambda s: s.dropna().iloc[-1] if s.dropna().size else np.nan)
        .rename("churned_raw")
    )
    features_df = features_df.merge(label_by_customer, left_on="customer_id", right_index=True, how="left")
    features_df["churned"] = (features_df["churned_raw"] == churned_value).astype(int)
    features_df = features_df.drop(columns=["churned_raw"])
    can_model = True
elif event_level:
    features_df["churned"] = (features_df["recency_days"] > inactivity_threshold).astype(int)
    can_model = True
else:
    can_model = False

feature_cols = ["recency_days", "frequency", "monetary", "tenure_days"]
if event_level:
    feature_cols += [f"activity_last_{w}d" for w in ROLLING_WINDOWS]
if "plan_tier" in features_df.columns:
    feature_cols.append("plan_tier")

can_model = can_model and len(features_df) >= MIN_ROWS_TO_TRAIN and features_df["churned"].nunique() > 1

# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------

tab_names = ["Overview"]
if event_level:
    tab_names.append("Cohort & Retention")
if can_model:
    tab_names += ["Churn Risk Scores", "Model Insights"]

tabs = st.tabs(tab_names)
tab_map = dict(zip(tab_names, tabs))

# ---- Overview ----
with tab_map["Overview"]:
    n_customers = len(features_df)
    churn_rate = features_df["churned"].mean() if "churned" in features_df else np.nan
    avg_tenure = features_df["tenure_days"].mean()
    avg_monetary = features_df["monetary"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{n_customers:,}")
    c2.metric("Churn rate", f"{churn_rate:.1%}" if pd.notna(churn_rate) else "n/a")
    c3.metric("Avg tenure", f"{avg_tenure:,.0f} days")
    c4.metric("Avg monetary value", f"${avg_monetary:,.0f}")

    if "plan_tier" in features_df.columns and "churned" in features_df:
        by_plan = features_df.groupby("plan_tier")["churned"].mean().reset_index()
        fig = px.bar(
            by_plan, x="plan_tier", y="churned", title="Churn rate by plan / segment",
            color="plan_tier", color_discrete_sequence=theme.CATEGORICAL,
        )
        fig.update_layout(yaxis_tickformat=".0%", showlegend=False)
        st.plotly_chart(style_fig(fig), width='stretch')

    st.subheader("RFM distributions")
    r1, r2, r3 = st.columns(3)
    for col, label, container in [
        ("recency_days", "Recency (days since last activity)", r1),
        ("frequency", "Frequency (activity count)", r2),
        ("monetary", "Monetary (total value)", r3),
    ]:
        fig = px.histogram(features_df, x=col, title=label, color_discrete_sequence=[theme.CATEGORICAL[0]])
        container.plotly_chart(style_fig(fig), width='stretch')

# ---- Cohort & Retention ----
if "Cohort & Retention" in tab_map:
    with tab_map["Cohort & Retention"]:
        cohort_raw = build_cohort_retention_table(raw_df, id_col, date_col)
        pivot = cohort_raw.pivot(index="cohort_month", columns="period_number", values="retention_rate")
        pivot.index = pivot.index.strftime("%Y-%m")

        st.subheader("Retention heatmap (% of cohort still active, by month since signup)")
        fig = go.Figure(
            data=go.Heatmap(
                z=pivot.values,
                x=[f"Month {p}" for p in pivot.columns],
                y=pivot.index,
                colorscale=sequential_colorscale(theme.SEQUENTIAL_BLUE),
                zmin=0,
                zmax=1,
                hovertemplate="Cohort %{y}<br>%{x}<br>Retention: %{z:.0%}<extra></extra>",
                colorbar=dict(tickformat=".0%"),
            )
        )
        st.plotly_chart(style_fig(fig), width='stretch')

        st.subheader("Average retention curve")
        avg_curve = cohort_raw.groupby("period_number").apply(
            lambda g: np.average(g["retention_rate"], weights=g["cohort_size"])
        ).reset_index(name="retention_rate")
        fig = px.line(
            avg_curve, x="period_number", y="retention_rate", markers=True,
            title="Retention across all cohorts, by month since signup",
            color_discrete_sequence=[theme.CATEGORICAL[0]],
        )
        fig.update_layout(yaxis_tickformat=".0%", xaxis_title="Months since signup", yaxis_title="Retention")
        st.plotly_chart(style_fig(fig), width='stretch')

# ---- Modeling tabs ----
if can_model:
    with st.sidebar:
        st.header("4. Scoring model")
        model_choice = st.selectbox("Model used for risk scores", ["Random Forest", "Logistic Regression"])

    train_result = cached_train_models(features_df, feature_cols, "churned")
    scores = score_customers(train_result["full_models"], model_choice, train_result["X_full"])
    features_df = features_df.copy()
    features_df["risk_score"] = scores
    features_df["risk_tier"] = features_df["risk_score"].apply(theme.risk_bucket)

    with tab_map["Churn Risk Scores"]:
        st.subheader(f"Risk scores - {model_choice}")
        d1, d2 = st.columns([2, 1])
        with d1:
            fig = px.histogram(
                features_df, x="risk_score", color="risk_tier", nbins=20,
                title="Distribution of churn risk scores",
                category_orders={"risk_tier": ["Low", "Medium", "High", "Critical"]},
                color_discrete_map=theme.RISK_COLOR_MAP,
            )
            st.plotly_chart(style_fig(fig), width='stretch')
        with d2:
            tier_counts = features_df["risk_tier"].value_counts().reindex(["Low", "Medium", "High", "Critical"]).fillna(0)
            for tier in ["Critical", "High", "Medium", "Low"]:
                st.metric(f"{tier} risk", f"{int(tier_counts[tier]):,}")

        st.subheader("At-risk customers")
        display_cols = ["customer_id", "risk_score", "risk_tier", "recency_days", "frequency", "monetary"]
        if "plan_tier" in features_df.columns:
            display_cols.append("plan_tier")
        table = features_df[display_cols].sort_values("risk_score", ascending=False)
        st.dataframe(
            table,
            width='stretch',
            column_config={
                "risk_score": st.column_config.ProgressColumn("Risk score", min_value=0, max_value=1, format="%.2f"),
            },
            hide_index=True,
        )
        st.download_button(
            "Download risk scores as CSV", data=table.to_csv(index=False),
            file_name="churn_risk_scores.csv", mime="text/csv",
        )

    with tab_map["Model Insights"]:
        st.subheader("Held-out test performance")
        metric_cols = st.columns(len(train_result["metrics"]))
        for col, (name, m) in zip(metric_cols, train_result["metrics"].items()):
            with col:
                st.markdown(f"**{name}**")
                st.metric("AUC", f"{m['auc']:.3f}" if pd.notna(m["auc"]) else "n/a")
                st.metric("Accuracy", f"{m['accuracy']:.1%}")
                st.metric("Precision", f"{m['precision']:.1%}")
                st.metric("Recall", f"{m['recall']:.1%}")

        st.subheader("ROC curve")
        fig = go.Figure()
        for i, (name, m) in enumerate(train_result["metrics"].items()):
            fig.add_trace(go.Scatter(x=m["fpr"], y=m["tpr"], mode="lines", name=name, line=dict(color=theme.CATEGORICAL[i])))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(color=theme.STATUS["warning"], dash="dash")))
        fig.update_layout(xaxis_title="False positive rate", yaxis_title="True positive rate")
        st.plotly_chart(style_fig(fig), width='stretch')

        st.subheader(f"What drives churn risk - {model_choice}")
        if model_choice == "Random Forest":
            importances = train_result["metrics"]["Random Forest"]["feature_importances"].tail(15)
            fig = px.bar(
                x=importances.values, y=importances.index, orientation="h",
                title="Feature importance", color_discrete_sequence=[theme.CATEGORICAL[0]],
            )
        else:
            coefs = train_result["metrics"]["Logistic Regression"]["coefficients"].tail(15)
            colors = [theme.STATUS["critical"] if v > 0 else theme.CATEGORICAL[0] for v in coefs.values]
            fig = go.Figure(go.Bar(x=coefs.values, y=coefs.index, orientation="h", marker_color=colors))
            fig.update_layout(title="Coefficients (red = increases risk, blue = decreases risk)")
        fig.update_layout(showlegend=False, xaxis_title="Impact", yaxis_title="")
        st.plotly_chart(style_fig(fig), width='stretch')

        st.subheader("Confusion matrix (test set)")
        cm = train_result["metrics"][model_choice]["confusion_matrix"]
        fig = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=["Predicted: stayed", "Predicted: churned"],
                y=["Actual: stayed", "Actual: churned"],
                colorscale=sequential_colorscale(theme.SEQUENTIAL_BLUE),
                text=cm,
                texttemplate="%{text}",
                showscale=False,
            )
        )
        st.plotly_chart(style_fig(fig), width='stretch')
elif event_level:
    st.info("Map a churn column, or lower the inactivity threshold, to unlock churn risk scoring.")
