"""Validated color palette and Plotly template shared by every chart in the dashboard."""

import streamlit as st

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]

STATUS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}

RISK_COLOR_MAP = {
    "Low": STATUS["good"],
    "Medium": STATUS["warning"],
    "High": STATUS["serious"],
    "Critical": STATUS["critical"],
}

_SURFACES = {
    "light": {"surface": "#fcfcfb", "grid": "#e1e0d9", "axis": "#c3c2b7", "ink": "#0b0b0b", "muted": "#898781"},
    "dark": {"surface": "#1a1a19", "grid": "#2c2c2a", "axis": "#383835", "ink": "#ffffff", "muted": "#898781"},
}


def current_mode() -> str:
    base = st.get_option("theme.base")
    return "dark" if base == "dark" else "light"


def plotly_layout_defaults() -> dict:
    """Common layout kwargs to spread into every fig.update_layout() call."""
    colors = _SURFACES[current_mode()]
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color=colors["ink"],
        colorway=CATEGORICAL,
        legend_title_text="",
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(gridcolor=colors["grid"], linecolor=colors["axis"], zerolinecolor=colors["axis"]),
        yaxis=dict(gridcolor=colors["grid"], linecolor=colors["axis"], zerolinecolor=colors["axis"]),
    )


def risk_bucket(score: float) -> str:
    if score >= 0.75:
        return "Critical"
    if score >= 0.5:
        return "High"
    if score >= 0.25:
        return "Medium"
    return "Low"
