"""Exploratory data analysis for uploaded air-quality CSV panels."""

from __future__ import annotations

import io

import invoke_bootstrap  # noqa: F401 — ensures project root on sys.path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from back_end.data_processing import drop_geo_columns
from front_end import charts as ch
from front_end.ui import configure_authenticated_workspace_page, empty_state, hero_title

configure_authenticated_workspace_page(page_title="Data Explorer · Air quality")

hero_title(
    "Data Explorer",
    "Upload a CSV panel to inspect distributions, correlations, and summary statistics.",
)

uploaded = st.file_uploader(
    "Historical air quality CSV",
    type=["csv"],
    help="Any table with at least one numeric column.",
)

if uploaded is None:
    empty_state("📊", "No data loaded", "Upload a CSV file above to start exploring.")
    st.stop()

try:
    raw = pd.read_csv(io.BytesIO(uploaded.getvalue()))
except Exception as exc:
    st.error(f"Could not parse CSV: {exc}")
    st.stop()

if raw.empty:
    st.error("The uploaded file contains no rows.")
    st.stop()

raw = drop_geo_columns(raw)
numeric = raw.select_dtypes(include=[np.number])
if numeric.shape[1] == 0:
    st.error("No numeric columns found.")
    st.stop()

# ── Overview metrics ──────────────────────────────────────────────────────────
# Extra width on the center column so long date ranges are not clipped.
c1, c2, c3 = st.columns([1.0, 2.4, 1.0])
c1.metric("Rows", f"{len(raw):,}")
time_col = next((n for n in ("time", "timestamp", "datetime", "date") if n in raw.columns), None)
if time_col:
    tser = pd.to_datetime(raw[time_col], utc=True, errors="coerce").dropna()
    dr = f"{tser.min().date()} — {tser.max().date()}" if len(tser) > 0 else "—"
else:
    dr = "No time column"
c2.metric("Date range", dr)
c3.metric("Numeric features", numeric.shape[1])

# ── Statistical summary ───────────────────────────────────────────────────────
with st.expander("Statistical summary", expanded=False):
    st.dataframe(
        numeric.describe().T,
        use_container_width=True,
        height=min(420, 56 + 28 * len(numeric.columns)),
    )

# ── Correlation ───────────────────────────────────────────────────────────────
st.markdown("### Correlation")
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
    margin=dict(l=48, r=48, t=48, b=48),
    height=max(400, 40 * len(corr.columns)),
    font=dict(family=ch.PLOT_FONT, size=12, color=ch.plotly_label_color()),
    paper_bgcolor=ch.plotly_paper_bg(),
    plot_bgcolor=ch.plotly_plot_bg(),
)
st.plotly_chart(fig_hm, use_container_width=True)

# ── Distribution ──────────────────────────────────────────────────────────────
st.markdown("### Distribution")
col_choice = st.selectbox("Column", list(numeric.columns))
vals = numeric[col_choice].dropna()
if vals.empty:
    st.warning("No non-missing values for the selected column.")
else:
    fig_hist = px.histogram(
        pd.DataFrame({col_choice: vals}),
        x=col_choice,
        nbins=min(60, max(10, int(np.sqrt(len(vals))))),
        template="plotly_white",
        color_discrete_sequence=["#0d9488"],
    )
    fig_hist.update_layout(
        xaxis_title=col_choice,
        yaxis_title="Count",
        bargap=0.08,
        margin=dict(l=48, r=48, t=40, b=48),
        height=400,
        font=dict(family=ch.PLOT_FONT, size=12, color=ch.plotly_label_color()),
        paper_bgcolor=ch.plotly_paper_bg(),
        plot_bgcolor=ch.plotly_plot_bg(),
    )
    st.plotly_chart(fig_hist, use_container_width=True)
