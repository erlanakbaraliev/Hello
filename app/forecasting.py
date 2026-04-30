"""Shared forecasting utilities and cached model loaders for Streamlit pages."""

from __future__ import annotations

import hashlib
import io
import json
import math
from datetime import timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from statsmodels.tsa.statespace.sarimax import SARIMAX

_ROOT = Path(__file__).resolve().parent.parent
_MODELS_DIR = _ROOT / "models"
_HASHES_PATH = _MODELS_DIR / "hashes.json"

TRAINING_COLUMNS = [
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
]
WEATHER_COLUMNS = ["temperature", "wind_speed"]
FORECAST_OUTPUT_COLUMNS = TRAINING_COLUMNS + WEATHER_COLUMNS
OPTIONAL_DROP = {"city", "latitude", "longitude"}
FORECAST_HOURS = 168


def models_dir() -> Path:
    return _MODELS_DIR


def verify_artifact_hash(path: Path) -> None:
    if not _HASHES_PATH.is_file():
        return
    try:
        hashes = json.loads(_HASHES_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return
    expected = hashes.get(path.name)
    if not expected:
        return
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected:
        raise ValueError(f"Artifact integrity check failed for {path.name}")


@st.cache_resource
def load_arima_model():
    path = models_dir() / "arima_model.pkl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing model file: {path}")
    verify_artifact_hash(path)
    return joblib.load(path)


@st.cache_resource
def load_lstm_model():
    path = models_dir() / "lstm_multivariate_model.keras"
    if not path.is_file():
        raise FileNotFoundError(f"Missing model file: {path}")
    verify_artifact_hash(path)
    try:
        from tensorflow import keras
    except ImportError as exc:
        raise ImportError("TensorFlow/Keras is required for the LSTM model.") from exc
    return keras.models.load_model(path)


@st.cache_resource
def load_lstm_artifacts() -> dict[str, Any]:
    path = models_dir() / "lstm_multivariate_artifacts.joblib"
    if not path.is_file():
        raise FileNotFoundError(f"Missing artifacts file: {path}")
    verify_artifact_hash(path)
    return joblib.load(path)


@st.cache_resource
def load_xgboost_model():
    path = models_dir() / "xgboost_multivariate_artifacts.joblib"
    if not path.is_file():
        raise FileNotFoundError(f"Missing model file: {path}")
    verify_artifact_hash(path)
    return joblib.load(path)


def preprocess_upload(raw: pd.DataFrame) -> pd.DataFrame:
    if "time" not in raw.columns:
        raise ValueError("CSV must include a 'time' column.")
    df = raw.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    if df["time"].isna().all():
        raise ValueError("Could not parse any valid timestamps in 'time'.")
    df = df.dropna(subset=["time"])
    for c in list(df.columns):
        if c in OPTIONAL_DROP:
            df = df.drop(columns=[c])
    required_cols = TRAINING_COLUMNS + WEATHER_COLUMNS
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    df = df[["time"] + required_cols].sort_values("time")
    df = df.set_index("time").asfreq("h")
    df = df.ffill()
    if len(df) < 48:
        raise ValueError("Need at least 48 hourly rows after cleaning for reliable forecasting.")
    return df


def preprocess_upload_bytes(file_bytes: bytes) -> pd.DataFrame:
    raw = pd.read_csv(io.BytesIO(file_bytes))
    return preprocess_upload(raw)


def _future_time_features(ts: pd.Timestamp) -> dict[str, float]:
    hour = float(ts.hour)
    dow = float(ts.dayofweek)
    month = float(ts.month)
    return {
        "h_sin": math.sin(2.0 * math.pi * hour / 24.0),
        "h_cos": math.cos(2.0 * math.pi * hour / 24.0),
        "h2_sin": math.sin(4.0 * math.pi * hour / 24.0),
        "h2_cos": math.cos(4.0 * math.pi * hour / 24.0),
        "dow_sin": math.sin(2.0 * math.pi * dow / 7.0),
        "dow_cos": math.cos(2.0 * math.pi * dow / 7.0),
        "mon_sin": math.sin(2.0 * math.pi * month / 12.0),
        "mon_cos": math.cos(2.0 * math.pi * month / 12.0),
    }


def _append_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["h_sin"] = np.sin(2.0 * np.pi * out.index.hour / 24.0)
    out["h_cos"] = np.cos(2.0 * np.pi * out.index.hour / 24.0)
    out["h2_sin"] = np.sin(4.0 * np.pi * out.index.hour / 24.0)
    out["h2_cos"] = np.cos(4.0 * np.pi * out.index.hour / 24.0)
    out["dow_sin"] = np.sin(2.0 * np.pi * out.index.dayofweek / 7.0)
    out["dow_cos"] = np.cos(2.0 * np.pi * out.index.dayofweek / 7.0)
    out["mon_sin"] = np.sin(2.0 * np.pi * out.index.month / 12.0)
    out["mon_cos"] = np.cos(2.0 * np.pi * out.index.month / 12.0)
    return out


def _validate_multivariate_artifacts(artifacts: dict[str, Any], model_name: str) -> None:
    needed = {"window", "target_cols", "feat_cols"}
    if model_name == "LSTM":
        needed.add("scaler")
    if model_name == "XGBoost":
        needed.update({"mu_all", "sigma_all", "boosters"})
    missing = sorted(needed.difference(artifacts.keys()))
    if missing:
        raise ValueError(f"{model_name} artifacts are missing keys: {', '.join(missing)}")
    targets = list(artifacts["target_cols"])
    for col in TRAINING_COLUMNS:
        if col not in targets:
            raise ValueError(f"{model_name} artifacts do not include required target column '{col}'.")
    feat_cols = list(artifacts["feat_cols"])
    for col in WEATHER_COLUMNS:
        if col not in feat_cols:
            raise ValueError(f"{model_name} artifacts do not include required feature column '{col}'.")


def _forecast_weather_recursive(seq: pd.DataFrame, column: str, window: int) -> float:
    recent = seq[column].astype(float).tail(window)
    if recent.empty:
        return 0.0
    if len(recent) >= 24:
        return float(recent.iloc[-24])
    return float(recent.iloc[-1])


def forecast_arima(y: pd.Series, model: Any) -> tuple[pd.DatetimeIndex, np.ndarray]:
    order = tuple(model.order)
    seasonal_order = tuple(model.seasonal_order)
    y_f = y.astype(float).asfreq("h")
    res = SARIMAX(
        y_f,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)
    fc = res.get_forecast(steps=FORECAST_HOURS)
    mean = fc.predicted_mean.to_numpy(dtype=float)
    start = y_f.index[-1] + timedelta(hours=1)
    idx = pd.date_range(start=start, periods=FORECAST_HOURS, freq="h")
    return idx, mean


def forecast_lstm(
    df: pd.DataFrame, keras_model: Any, artifacts: dict[str, Any]
) -> tuple[pd.DatetimeIndex, pd.DataFrame]:
    _validate_multivariate_artifacts(artifacts, "LSTM")
    window = int(artifacts["window"])
    feat_cols = list(artifacts["feat_cols"])
    target_cols = list(artifacts["target_cols"])
    scaler = artifacts["scaler"]
    df_feat = _append_time_features(df)
    if len(df) < window:
        raise ValueError(f"Need at least {window} rows for LSTM forecasting.")
    data = scaler.transform(df_feat[feat_cols]).astype(np.float32)
    seq = df_feat[feat_cols].astype(float).copy()
    win = data[-window:].copy()
    preds: list[dict[str, float]] = []
    for _ in range(FORECAST_HOURS):
        x = win.reshape(1, window, len(feat_cols))
        nxt_scaled = keras_model.predict(x, verbose=0)[0]
        if nxt_scaled.shape[0] != len(target_cols):
            raise ValueError("LSTM output dimension does not match target columns.")
        unscaled = np.zeros((1, scaler.n_features_in_), dtype=np.float64)
        for i, col in enumerate(target_cols):
            col_idx = feat_cols.index(col)
            unscaled[0, col_idx] = float(nxt_scaled[i])
        nxt_unscaled = scaler.inverse_transform(unscaled)[0]
        next_ts = df.index[-1] + timedelta(hours=len(preds) + 1)
        next_row_raw: dict[str, float] = {}
        for col in target_cols:
            col_idx = feat_cols.index(col)
            next_row_raw[col] = float(nxt_unscaled[col_idx])
        for col in WEATHER_COLUMNS:
            if col not in next_row_raw:
                next_row_raw[col] = _forecast_weather_recursive(seq, col, window)
        next_row_raw.update(_future_time_features(next_ts))
        preds.append({k: next_row_raw[k] for k in FORECAST_OUTPUT_COLUMNS})

        nxt_input = np.array([next_row_raw[col] for col in feat_cols], dtype=np.float64).reshape(1, -1)
        nxt_scaled_row = scaler.transform(nxt_input).reshape(-1)
        win = np.vstack([win[1:], nxt_scaled_row.reshape(1, -1)])
        seq = pd.concat([seq, pd.DataFrame([next_row_raw], index=[next_ts])], axis=0)

    start = df.index[-1] + timedelta(hours=1)
    idx = pd.date_range(start=start, periods=FORECAST_HOURS, freq="h")
    out = pd.DataFrame(preds, index=idx, columns=FORECAST_OUTPUT_COLUMNS)
    if not np.isfinite(out.to_numpy(dtype=float)).all():
        raise ValueError("LSTM forecast produced non-finite values.")
    return idx, out


def forecast_xgboost(df: pd.DataFrame, model: Any) -> tuple[pd.DatetimeIndex, pd.DataFrame]:
    _validate_multivariate_artifacts(model, "XGBoost")
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise ImportError("xgboost is required for XGBoost inference.") from exc

    window = int(model["window"])
    feat_cols = list(model["feat_cols"])
    target_cols = list(model["target_cols"])
    mu_all = np.asarray(model["mu_all"], dtype=np.float64)
    sigma_all = np.asarray(model["sigma_all"], dtype=np.float64)
    sigma_all = np.where(sigma_all == 0.0, 1.0, sigma_all)
    boosters = list(model["boosters"])
    if len(boosters) != len(target_cols):
        raise ValueError("XGBoost artifacts have mismatched boosters and target columns.")
    if len(df) < window:
        raise ValueError(f"Need at least {window} rows for XGBoost forecasting.")

    df_feat = _append_time_features(df)
    seq = df_feat[feat_cols].astype(float).copy()
    preds: list[dict[str, float]] = []
    for _ in range(FORECAST_HOURS):
        window_np = seq.tail(window).to_numpy(dtype=np.float64)
        x_scaled = ((window_np - mu_all) / sigma_all).reshape(1, -1)
        dmx = xgb.DMatrix(x_scaled)
        preds_step = [float(booster.predict(dmx)[0]) for booster in boosters]
        next_ts = df.index[-1] + timedelta(hours=len(preds) + 1)
        next_row: dict[str, float] = {col: preds_step[i] for i, col in enumerate(target_cols)}
        for col in WEATHER_COLUMNS:
            if col not in next_row:
                next_row[col] = _forecast_weather_recursive(seq, col, window)
        next_row.update(_future_time_features(next_ts))
        preds.append({k: next_row[k] for k in FORECAST_OUTPUT_COLUMNS})
        seq = pd.concat([seq, pd.DataFrame([next_row], index=[next_ts])], axis=0)

    start = df.index[-1] + timedelta(hours=1)
    idx = pd.date_range(start=start, periods=FORECAST_HOURS, freq="h")
    out = pd.DataFrame(preds, index=idx, columns=FORECAST_OUTPUT_COLUMNS)
    if not np.isfinite(out.to_numpy(dtype=float)).all():
        raise ValueError("XGBoost forecast produced non-finite values.")
    return idx, out


def prediction_payload(
    model_label: str,
    hist_index: pd.DatetimeIndex,
    hist_df: pd.DataFrame,
    fc_index: pd.DatetimeIndex,
    fc_df: pd.DataFrame,
) -> dict[str, Any]:
    hist_records = []
    for t, (_, row) in zip(hist_index, hist_df.iterrows()):
        rec = {"time": t.isoformat()}
        for col in FORECAST_OUTPUT_COLUMNS:
            rec[col] = float(row[col])
        hist_records.append(rec)
    fc_records = []
    for t, (_, row) in zip(fc_index, fc_df.iterrows()):
        rec = {"time": t.isoformat()}
        for col in FORECAST_OUTPUT_COLUMNS:
            rec[col] = float(row[col])
        fc_records.append(rec)
    return {
        "model": model_label,
        "forecast_hours": FORECAST_HOURS,
        "columns": list(FORECAST_OUTPUT_COLUMNS),
        "history": hist_records,
        "forecast": fc_records,
    }
