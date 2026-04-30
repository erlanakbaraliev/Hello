"""Shared forecasting utilities and cached model loaders for Streamlit pages."""

from __future__ import annotations

import hashlib
import io
import json
from datetime import timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
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
OPTIONAL_DROP = {"city", "latitude", "longitude"}
FORECAST_HOURS = 168
LOOKBACK_LSTM = 24


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
    path = models_dir() / "lstm_model.keras"
    if not path.is_file():
        raise FileNotFoundError(f"Missing model file: {path}")
    verify_artifact_hash(path)
    try:
        from tensorflow import keras
    except ImportError as exc:
        raise ImportError("TensorFlow/Keras is required for the LSTM model.") from exc
    return keras.models.load_model(path)


@st.cache_resource
def load_lstm_scaler() -> MinMaxScaler:
    path = models_dir() / "lstm_scaler.pkl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing scaler file: {path}")
    verify_artifact_hash(path)
    return joblib.load(path)


@st.cache_resource
def load_xgboost_model():
    path = models_dir() / "xgboost_model.joblib"
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
    missing = [c for c in TRAINING_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    df = df[["time"] + TRAINING_COLUMNS].sort_values("time")
    df = df.set_index("time").asfreq("h")
    df = df.ffill()
    if len(df) < 48:
        raise ValueError("Need at least 48 hourly rows after cleaning for reliable forecasting.")
    return df


def preprocess_upload_bytes(file_bytes: bytes) -> pd.DataFrame:
    raw = pd.read_csv(io.BytesIO(file_bytes))
    return preprocess_upload(raw)


def inverse_pm25_column(scaled_1d: np.ndarray, scaler: MinMaxScaler, col_index: int) -> np.ndarray:
    n = len(scaled_1d)
    pad = np.zeros((n, scaler.n_features_in_), dtype=np.float64)
    pad[:, col_index] = np.asarray(scaled_1d).reshape(-1)
    return scaler.inverse_transform(pad)[:, col_index]


def feature_columns_for_scaler(scaler: MinMaxScaler) -> list[str]:
    names = getattr(scaler, "feature_names_in_", None)
    if names is not None:
        return list(names)
    return list(TRAINING_COLUMNS)


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
    df: pd.DataFrame, keras_model: Any, scaler: MinMaxScaler
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    cols = feature_columns_for_scaler(scaler)
    pm25_idx = cols.index("pm2_5")
    data = scaler.transform(df[cols]).astype(np.float32)
    window = data[-LOOKBACK_LSTM:].copy()
    preds_scaled: list[float] = []
    last_row = window[-1].copy()
    for _ in range(FORECAST_HOURS):
        x = window.reshape(1, LOOKBACK_LSTM, len(cols))
        nxt = float(keras_model.predict(x, verbose=0)[0, 0])
        preds_scaled.append(nxt)
        new_row = last_row.copy()
        new_row[pm25_idx] = nxt
        window = np.vstack([window[1:], new_row.reshape(1, -1)])
        last_row = new_row
    preds = inverse_pm25_column(np.array(preds_scaled), scaler, pm25_idx)
    start = df.index[-1] + timedelta(hours=1)
    idx = pd.date_range(start=start, periods=FORECAST_HOURS, freq="h")
    return idx, preds


def build_xgb_row(
    ts: pd.Timestamp,
    hist_pm25: list[float],
    base_exog: pd.Series,
) -> pd.Series:
    hour = int(ts.hour)
    dow = int(ts.dayofweek)
    month = int(ts.month)
    lag1 = hist_pm25[-1]
    lag2 = hist_pm25[-2]
    lag24 = hist_pm25[-24] if len(hist_pm25) >= 24 else hist_pm25[0]
    row = base_exog.copy()
    row["hour"] = hour
    row["day_of_week"] = dow
    row["month"] = month
    row["pm2_5_lag1"] = lag1
    row["pm2_5_lag2"] = lag2
    row["pm2_5_lag24"] = lag24
    return row


def forecast_xgboost(df: pd.DataFrame, model: Any) -> tuple[pd.DatetimeIndex, np.ndarray]:
    feat_order = list(getattr(model, "feature_names_in_", []))
    if not feat_order:
        raise ValueError(
            "XGBoost model has no feature_names_in_; retrain with a recent scikit-learn API."
        )
    base = df.iloc[-1][TRAINING_COLUMNS].astype(float)
    hist = [float(x) for x in df["pm2_5"].tolist()]
    if len(hist) < 25:
        raise ValueError("Uploaded series too short for XGBoost lag-24 features.")
    preds: list[float] = []
    cur_ts = df.index[-1]
    for _ in range(FORECAST_HOURS):
        cur_ts = cur_ts + timedelta(hours=1)
        row = build_xgb_row(cur_ts, hist, base)
        x = row[feat_order].values.reshape(1, -1)
        p = float(model.predict(x)[0])
        preds.append(p)
        hist.append(p)
    start = df.index[-1] + timedelta(hours=1)
    idx = pd.date_range(start=start, periods=FORECAST_HOURS, freq="h")
    return idx, np.array(preds, dtype=float)


def prediction_payload(
    model_label: str,
    hist_index: pd.DatetimeIndex,
    hist_pm25: np.ndarray,
    fc_index: pd.DatetimeIndex,
    fc_pm25: np.ndarray,
) -> dict[str, Any]:
    return {
        "model": model_label,
        "forecast_hours": FORECAST_HOURS,
        "history": [
            {"time": t.isoformat(), "pm2_5": float(v)} for t, v in zip(hist_index, hist_pm25)
        ],
        "forecast": [{"time": t.isoformat(), "pm2_5": float(v)} for t, v in zip(fc_index, fc_pm25)],
    }
