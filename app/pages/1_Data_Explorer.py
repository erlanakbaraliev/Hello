"""Exploratory data analysis for uploaded air-quality CSV panels."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_here = Path(__file__).resolve()
_bootstrap = (
    _here.parent / "invoke_bootstrap.py"
    if (_here.parent / "invoke_bootstrap.py").is_file()
    else _here.parent.parent / "invoke_bootstrap.py"
)
_bs_spec = importlib.util.spec_from_file_location("_streamlit_invoke_bootstrap", _bootstrap)
if _bs_spec is None or _bs_spec.loader is None:
    raise RuntimeError("invoke_bootstrap: could not load spec")
_bs_mod = importlib.util.module_from_spec(_bs_spec)
_bs_spec.loader.exec_module(_bs_mod)

import io

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from app import charts as ch
from app.ui import configure_authenticated_workspace_page, hero_title

configure_authenticated_workspace_page(page_title="Data Explorer · Urban air quality")

_plot_font = ch.PLOT_FONT

hero_title(
    "Data explorer",
    "Upload a CSV panel to inspect scale, dispersion, and linear association among numeric variables.",
)

uploaded = st.file_uploader(
    "Historical air quality CSV",
    type=["csv"],
    help="Any table with at least one numeric column; time-like columns improve the date-range summary.",
)

if uploaded is None:
    st.info("Upload a CSV file to populate the dashboard.")
    st.stop()

try:
    raw = pd.read_csv(io.BytesIO(uploaded.getvalue()))
except Exception as exc:
    st.error(f"Could not parse CSV: {exc}")
    st.stop()

if raw.empty:
    st.error("The uploaded file contains no rows.")
    st.stop()

numeric = raw.select_dtypes(include=[np.number])
if numeric.shape[1] == 0:
    st.error("No numeric columns found for correlation and distribution plots.")
    st.stop()

st.markdown("### Dataset overview")
c1, c2, c3 = st.columns(3)
c1.metric("Total rows", f"{len(raw):,}")
time_col = None
for name in ("time", "timestamp", "datetime", "date"):
    if name in raw.columns:
        time_col = name
        break
if time_col is not None:
    tser = pd.to_datetime(raw[time_col], utc=True, errors="coerce")
    valid = tser.dropna()
    if len(valid) > 0:
        dr = f"{valid.min().date()} — {valid.max().date()}"
    else:
        dr = "Could not parse dates"
else:
    dr = "No time column detected"
c2.metric("Date range (UTC)", dr)
c3.metric("Numeric features", numeric.shape[1])

st.markdown("")

with st.expander("View statistical summary", expanded=False):
    st.caption(
        "Descriptive statistics (count, central tendency, dispersion, quantiles) for numeric columns."
    )
    st.dataframe(
        numeric.describe().T,
        use_container_width=True,
        height=min(480, 60 + 28 * len(numeric.columns)),
    )

st.markdown("### Correlation structure")
st.caption("Pearson correlation matrix among numeric variables.")
corr = numeric.corr()
fig_hm = px.imshow(
    corr,
    text_auto=".2f",
    aspect="auto",
    color_continuous_scale="RdBu_r",
    zmin=-1,
    zmax=1,
    template="plotly_white",
)
fig_hm.update_layout(
    margin=dict(l=48, r=48, t=56, b=48),
    height=max(440, 44 * len(corr.columns)),
    title=dict(text="Correlation heatmap", font=dict(size=17, family=_plot_font)),
    font=dict(family=_plot_font, size=12, color="#334155"),
    paper_bgcolor="rgba(255,255,255,0.92)",
    plot_bgcolor="#fafbfc",
)
st.plotly_chart(fig_hm, use_container_width=True)

st.markdown("### Feature distributions")
col_choice = st.selectbox("Numeric column", list(numeric.columns))
vals = numeric[col_choice].dropna()
if vals.empty:
    st.warning("No non-missing values for the selected column.")
else:
    hist_df = pd.DataFrame({col_choice: vals})
    fig_hist = px.histogram(
        hist_df,
        x=col_choice,
        nbins=min(60, max(10, int(np.sqrt(len(vals))))),
        template="plotly_white",
    )
    fig_hist.update_layout(
        title=dict(text=f"Histogram: {col_choice}", font=dict(size=17, family=_plot_font)),
        xaxis_title=col_choice,
        yaxis_title="Count",
        bargap=0.1,
        margin=dict(l=48, r=48, t=56, b=48),
        height=460,
        font=dict(family=_plot_font, size=12, color="#334155"),
        paper_bgcolor="rgba(255,255,255,0.92)",
        plot_bgcolor="#fafbfc",
    )
    st.plotly_chart(fig_hist, use_container_width=True)
