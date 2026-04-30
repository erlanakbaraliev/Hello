"""Forecast PM2.5 using trained models and log results to SQLite."""

from __future__ import annotations

import invoke_bootstrap  # noqa: F401 — ensures project root on sys.path

import pandas as pd
import streamlit as st

from front_end import charts as ch
from back_end import forecasting as fc
from back_end import page_predict as pp
from front_end.ui import configure_authenticated_workspace_page, current_user_id, hero_title
from db import get_user_settings, save_prediction

configure_authenticated_workspace_page(page_title="Predict · Urban air quality")

uid = current_user_id()
if uid is None:
    st.stop()
_settings = get_user_settings(uid)
_model_options = ("LSTM", "XGBoost")
_default_model = _settings.get("default_model") or "LSTM"
_model_index = _model_options.index(_default_model) if _default_model in _model_options else 0

hero_title(
    "Predict future quality",
    "Upload recent hourly observations and generate a 168-hour PM₂.₅ forecast.",
)

ctrl1, ctrl2, ctrl3 = st.columns([1.2, 0.5, 0.3], gap="medium")
with ctrl1:
    uploaded = st.file_uploader(
        "Hourly air quality CSV",
        type=["csv"],
        help="Columns: time, pm10, pm2_5, carbon_monoxide, nitrogen_dioxide, sulphur_dioxide, ozone, temperature, wind_speed.",
    )
with ctrl2:
    model_choice = st.selectbox(
        "Model",
        _model_options,
        index=_model_index,
        help="Default from Settings. Both use recursive 168-hour forecasting.",
    )
with ctrl3:
    st.markdown("<div style='height:1.6rem'></div>", unsafe_allow_html=True)
    run_clicked = st.button("Run forecast", type="primary", use_container_width=True)

if run_clicked:
    if uploaded is None:
        st.warning("Upload a CSV file first.")
    else:
        try:
            df = fc.preprocess_upload_bytes(uploaded.getvalue())
        except (ValueError, pd.errors.ParserError) as exc:
            st.error(f"Could not read or validate the CSV: {exc}")
        except Exception as exc:
            st.error(f"Unexpected error while loading data: {exc}")
        else:
            hist_df = df[fc.FORECAST_OUTPUT_COLUMNS].astype(float).copy()
            y_hist = hist_df["pm2_5"].to_numpy()
            hist_idx = df.index

            try:
                fc_idx, fc_df = pp.dispatch_hourly_forecast(df, model_choice, fc)
            except FileNotFoundError as exc:
                st.error(str(exc))
            except ImportError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Forecast failed: {exc}")
            else:
                fc_vals = fc_df["pm2_5"].astype(float).to_numpy()
                fig = pp.build_pm25_forecast_figure(hist_idx, y_hist, fc_idx, fc_vals, ch)
                st.plotly_chart(fig, use_container_width=True)

                try:
                    payload = fc.prediction_payload(model_choice, hist_idx, hist_df, fc_idx, fc_df)
                    save_prediction(uid, model_choice, payload)
                    st.success("Prediction saved to your history.")
                except KeyError:
                    st.error("Session expired; please log in again.")
                except (TypeError, ValueError, RuntimeError) as exc:
                    st.warning(f"Forecast succeeded but logging failed: {exc}")
