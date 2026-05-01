"""Upload + horizon-selectable forecasting page."""

from __future__ import annotations

import io

import invoke_bootstrap  # noqa: F401
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from front_end.risk_bands import add_pm25_safety_column
from front_end.ui import configure_authenticated_workspace_page, current_user_id, empty_state, hero_title
from back_end.constants import (
    SUPPORTED_HORIZONS,
    ArtifactMissingError,
    HorizonUnavailableError,
    UploadValidationError,
)
from back_end.inference import prediction_payload
from back_end.lstm_pred import forecast_lstm_full
from back_end.upload_preprocess import PreprocessedUpload, preprocess_uploaded_dataset
from back_end.xgboost_pred import forecast_xgboost_full
from db import save_prediction_artifacts

configure_authenticated_workspace_page(page_title="Forecast · Air quality")

uid = current_user_id()
if uid is None:
    st.stop()

hero_title("Forecast", "Upload your CSV, click Preprocess, then run a 24h, 72h, or 168h forecast.")

with st.expander("CSV format (short)", expanded=False):
    st.markdown(
        """
- **One CSV**, hourly **UTC** times, no gaps or duplicate times.
- **Past:** at least **48** rows with all pollutant columns **and** temperature & wind_speed filled.
- **Future:** each hour needs **time**, **temperature**, and **wind_speed** filled. Pollutants can be empty.
- **How long:** you need **24**, **72**, or **168** future hours for that horizon.
"""
    )

_SESSION_KEY = "forecast_preprocessed"
_LAST_FILE_KEY = "forecast_last_file_signature"


def _file_signature(file_bytes: bytes, name: str) -> str:
    return f"{name}:{len(file_bytes)}:{hash(file_bytes)}"


def _clear_preprocessed() -> None:
    st.session_state.pop(_SESSION_KEY, None)
    st.session_state.pop(_LAST_FILE_KEY, None)


def _summary(pre: PreprocessedUpload) -> None:
    n_past = len(pre.past_df_aligned)
    n_future = len(pre.future_weather_df)
    cols = st.columns(3)
    cols[0].metric("Past rows", f"{n_past:,}")
    cols[1].metric("Future rows", f"{n_future:,}")
    cols[2].metric("Cutoff (UTC)", pd.Timestamp(pre.cutoff_ts).strftime("%Y-%m-%d %H:%M"))
    if pre.available_horizons:
        st.caption("Available horizons: " + ", ".join(f"{h}h" for h in pre.available_horizons))
    else:
        st.warning("No supported horizon available from this upload.")


def _build_figure(history_pm25: pd.Series, forecast_df: pd.DataFrame, model_name: str, horizon: int) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=history_pm25.index, y=history_pm25.to_numpy(), mode="lines", name="Observed PM2.5")
    )
    fig.add_trace(
        go.Scatter(
            x=pd.to_datetime(forecast_df["time"]),
            y=forecast_df["pm2_5_pred"].to_numpy(),
            mode="lines",
            name=f"Forecast PM2.5 ({model_name}, {horizon}h)",
            line=dict(dash="dash"),
        )
    )
    fig.update_layout(
        xaxis_title="Time (UTC)",
        yaxis_title="PM2.5 (ug/m3)",
        hovermode="x unified",
        height=480,
        margin=dict(l=40, r=30, t=30, b=40),
    )
    return fig


st.markdown("### 1. Upload dataset")
uploaded = st.file_uploader("Hourly air quality CSV", type=["csv"], key="forecast_file")

if uploaded is None:
    _clear_preprocessed()
    empty_state("📥", "No data loaded", "Upload a CSV file above and click Preprocess to validate it.")
    st.stop()

file_bytes = uploaded.getvalue()
sig = _file_signature(file_bytes, uploaded.name)
if st.session_state.get(_LAST_FILE_KEY) != sig:
    _clear_preprocessed()

if st.button("Preprocess", type="primary", key="forecast_preprocess_btn"):
    try:
        raw_df = pd.read_csv(io.BytesIO(file_bytes))
        pre = preprocess_uploaded_dataset(raw_df)
    except (pd.errors.ParserError, ValueError, UploadValidationError) as exc:
        st.error(str(exc))
        _clear_preprocessed()
        st.stop()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Unexpected preprocessing error: {exc}")
        _clear_preprocessed()
        st.stop()
    st.session_state[_SESSION_KEY] = pre
    st.session_state[_LAST_FILE_KEY] = sig
    st.success("Dataset validated and preprocessed.")

pre_obj: PreprocessedUpload | None = st.session_state.get(_SESSION_KEY)
if pre_obj is None:
    st.info("Click **Preprocess** to validate the upload before forecasting.")
    st.stop()

st.markdown("### 2. Preprocessing summary")
_summary(pre_obj)

st.markdown("### 3. Run forecast")
valid_horizons = pre_obj.available_horizons
if not valid_horizons:
    st.stop()

ctrl_h, ctrl_m, ctrl_btn = st.columns([0.6, 0.6, 0.4], gap="medium")
with ctrl_h:
    horizon = st.radio(
        "Forecast horizon",
        options=list(valid_horizons),
        index=0,
        format_func=lambda h: f"{h}h",
        horizontal=True,
    )
with ctrl_m:
    model_choice = st.selectbox("Model", options=("LSTM", "XGBoost"), index=0)
with ctrl_btn:
    st.markdown("<div style='height:1.6rem'></div>", unsafe_allow_html=True)
    run_clicked = st.button("Run forecast", type="primary", use_container_width=True)

if run_clicked:
    with st.spinner(f"Running {model_choice} {horizon}h forecast..."):
        try:
            if model_choice == "LSTM":
                forecast_df = forecast_lstm_full(pre_obj, int(horizon))
            else:
                forecast_df = forecast_xgboost_full(pre_obj, int(horizon))
        except (HorizonUnavailableError, ArtifactMissingError, UploadValidationError, ImportError) as exc:
            st.error(str(exc))
            st.stop()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Forecast failed: {exc}")
            st.stop()

    forecast_df = add_pm25_safety_column(forecast_df)
    st.markdown("### 4. Forecast")
    history_pm25 = pre_obj.past_df_aligned["pm2_5"].astype(float).tail(7 * 24)
    st.plotly_chart(
        _build_figure(history_pm25, forecast_df, model_choice, int(horizon)),
        use_container_width=True,
    )

    st.dataframe(forecast_df, use_container_width=True, height=min(420, 56 + 28 * len(forecast_df)))
    pred_csv_bytes = forecast_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download forecast CSV",
        data=pred_csv_bytes,
        file_name=f"forecast_{model_choice.lower()}_{int(horizon)}h.csv",
        mime="text/csv",
    )

    try:
        payload = prediction_payload(pre_obj, forecast_df, model_name=model_choice, horizon=int(horizon))
        mean_pm25 = float(forecast_df["pm2_5_pred"].astype(float).mean())
        save_prediction_artifacts(
            uid,
            model_choice,
            payload,
            dataset_name=str(uploaded.name),
            avg_aqi=mean_pm25,
            risk_level="-",
            dataset_csv=file_bytes,
            prediction_csv=pred_csv_bytes,
            details={"source": "upload_full_models", "forecast_horizon": int(horizon)},
        )
        st.success("Forecast saved to your history.")
    except KeyError:
        st.error("Session expired; please log in again.")
    except (TypeError, ValueError, RuntimeError) as exc:
        st.warning(f"Forecast succeeded but logging failed: {exc}")
