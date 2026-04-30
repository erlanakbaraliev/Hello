"""Dashboard page helpers: default series load, AQI banding, and plot styling."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from back_end.data_processing import load_dataset_from_data_folder, validate_and_prepare


def aqi_category(pm25: float) -> tuple[int, str]:
    if pm25 <= 12.0:
        return 50, "Low"
    if pm25 <= 35.4:
        return 100, "Moderate"
    return 150, "High"


_POLLUTANT_LABELS: dict[str, str] = {
    "pm2_5": "PM2.5",
    "pm10": "PM10",
    "nitrogen_dioxide": "NO₂",
    "carbon_monoxide": "CO",
    "ozone": "O₃",
    "sulphur_dioxide": "SO₂",
}


def pollutant_label(key: str) -> str:
    return _POLLUTANT_LABELS.get(key, key.replace("_", " ").title())


@st.cache_data
def load_default_prepared() -> pd.DataFrame:
    raw = load_dataset_from_data_folder("london_2024.csv")
    return validate_and_prepare(raw)


def render_dashboard_charts(prepared_df: pd.DataFrame, ch: object) -> None:
    """Metrics row and trend / bar charts for the dashboard body."""
    latest_pm25 = float(prepared_df["pm2_5"].iloc[-1])
    aqi_level, risk = aqi_category(latest_pm25)

    m1, m2, m3 = st.columns(3)
    m1.metric("Latest PM2.5 (µg/m³)", f"{latest_pm25:.1f}")
    m2.metric("Estimated AQI", f"{aqi_level}")
    m3.metric("Risk level", risk)

    trend_col, cta_col = st.columns([1, 1])
    with trend_col:
        trend_days = st.selectbox(
            "Trend window", ["Last 30 days", "Last 90 days", "Full period"], index=0
        )
    with cta_col:
        st.markdown("<div style='height:1.6rem'></div>", unsafe_allow_html=True)
        st.page_link("pages/2_Predict_future_quality.py", label="Run a forecast →", icon="🔮")
    if trend_days == "Last 30 days":
        trend_df = prepared_df.tail(720).reset_index()
    elif trend_days == "Last 90 days":
        trend_df = prepared_df.tail(2160).reset_index()
    else:
        trend_df = prepared_df.reset_index()

    c1, c2 = st.columns([1.3, 1.0])
    trend_cols = ["pm2_5", "pm10", "nitrogen_dioxide", "ozone"]
    trend_rename = {c: pollutant_label(c) for c in trend_cols}
    trend_plot = trend_df.rename(columns=trend_rename)
    fig_trend = px.line(
        trend_plot,
        x="time",
        y=list(trend_rename.values()),
        title="Air quality trend",
        template=ch.plotly_template(),
    )
    fig_trend.update_layout(
        font=dict(family=ch.PLOT_FONT, size=12, color=ch.plotly_label_color()),
        title=dict(font=dict(color=ch.plotly_title_color())),
        paper_bgcolor=ch.plotly_paper_bg(),
        plot_bgcolor=ch.plotly_plot_bg(),
        margin=dict(l=48, r=24, t=56, b=48),
        height=420,
    )
    fig_trend.update_xaxes(gridcolor=ch.plotly_grid_color(), zeroline=False)
    fig_trend.update_yaxes(gridcolor=ch.plotly_grid_color(), zeroline=False)
    c1.plotly_chart(fig_trend, use_container_width=True)

    latest_row = prepared_df.iloc[-1]
    pollutant_names = ["pm2_5", "pm10", "nitrogen_dioxide", "carbon_monoxide", "ozone"]
    bar_df = pd.DataFrame(
        {
            "Pollutant": [pollutant_label(n) for n in pollutant_names],
            "µg/m³": [float(latest_row.get(name, 0.0)) for name in pollutant_names],
        }
    )
    fig_bar = px.bar(
        bar_df,
        x="Pollutant",
        y="µg/m³",
        title="Current snapshot",
        template=ch.plotly_template(),
        color_discrete_sequence=["#0d9488"],
    )
    fig_bar.update_layout(
        font=dict(family=ch.PLOT_FONT, size=12, color=ch.plotly_label_color()),
        title=dict(font=dict(color=ch.plotly_title_color())),
        paper_bgcolor=ch.plotly_paper_bg(),
        plot_bgcolor=ch.plotly_plot_bg(),
        margin=dict(l=48, r=24, t=56, b=48),
        height=420,
    )
    fig_bar.update_xaxes(gridcolor=ch.plotly_grid_color(), zeroline=False)
    fig_bar.update_yaxes(gridcolor=ch.plotly_grid_color(), zeroline=False)
    c2.plotly_chart(fig_bar, use_container_width=True)
