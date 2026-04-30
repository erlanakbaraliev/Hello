"""Predict-future-quality page: hourly forecast dispatch and Plotly figure."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def dispatch_hourly_forecast(
    df: pd.DataFrame, model_choice: str, fc: Any
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Run the selected hourly model; returns ``(forecast_index, forecast_values)``."""
    if model_choice == "ARIMA":
        arima = fc.load_arima_model()
        return fc.forecast_arima(df["pm2_5"], arima)
    if model_choice == "LSTM":
        lstm = fc.load_lstm_model()
        scaler = fc.load_lstm_scaler()
        return fc.forecast_lstm(df, lstm, scaler)
    xgb = fc.load_xgboost_model()
    return fc.forecast_xgboost(df, xgb)


def build_pm25_forecast_figure(
    hist_idx: Any,
    y_hist: np.ndarray,
    fc_idx: Any,
    fc_vals: np.ndarray,
    ch: Any,
) -> go.Figure:
    """History + 168h forecast line chart (Plotly)."""
    plot_font = ch.PLOT_FONT
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=hist_idx,
            y=y_hist,
            mode="lines",
            name="Observed PM2.5",
            line=dict(color="#0e7490", width=2.2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[hist_idx[-1], fc_idx[0]],
            y=[y_hist[-1], fc_vals[0]],
            mode="lines",
            line=dict(color="#b45309", width=2, dash="dot"),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fc_idx,
            y=fc_vals,
            mode="lines",
            name="Forecast PM2.5",
            line=dict(color="#c2410c", width=2.2, dash="dash"),
        )
    )
    fig.update_layout(
        title=dict(
            text="PM2.5 (µg/m³): history and 168-hour forecast",
            font=dict(size=18, family=plot_font, color=ch.plotly_title_color()),
        ),
        xaxis_title="Time (UTC)",
        yaxis_title="PM2.5 (µg/m³)",
        hovermode="x unified",
        template=ch.plotly_template(),
        font=dict(family=plot_font, size=12, color=ch.plotly_label_color()),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.06,
            xanchor="right",
            x=1,
            bgcolor=ch.plotly_legend_bg(),
            bordercolor=ch.plotly_legend_border(),
            borderwidth=1,
        ),
        height=540,
        margin=dict(l=56, r=40, t=72, b=56),
        paper_bgcolor=ch.plotly_paper_bg(),
        plot_bgcolor=ch.plotly_plot_bg(),
        xaxis=dict(showgrid=True, gridcolor=ch.plotly_grid_color(), zeroline=False),
        yaxis=dict(showgrid=True, gridcolor=ch.plotly_grid_color(), zeroline=False),
    )
    return fig
