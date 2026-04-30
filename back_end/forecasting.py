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
    """168-hour recursive LSTM forecast. Logic matches models/lstm_forecast.py."""
    _validate_multivariate_artifacts(artifacts, "LSTM")
    window = int(artifacts["window"])
    feat_cols = list(artifacts["feat_cols"])
    target_cols = list(artifacts["target_cols"])
    tgt_idxs = list(artifacts["tgt_idxs"])
    scaler = artifacts["scaler"]

    if len(df) < window:
        raise ValueError(f"Need at least {window} rows for LSTM forecasting.")

    df_feat = _append_time_features(df)
    df_feat = df_feat[feat_cols].interpolate(limit_direction="both").ffill().bfill()

    mu_all = scaler.mean_.astype(np.float32)
    sigma_all = scaler.scale_.astype(np.float32)
    sigma_safe = np.where(sigma_all < 1e-8, 1.0, sigma_all)

    frame_scaled = scaler.transform(df_feat.values).astype(np.float32)
    history = frame_scaled[-window:].copy()
    history_orig = scaler.inverse_transform(history).astype(np.float32)

    idx_map = {c: i for i, c in enumerate(feat_cols)}
    preds: list[dict[str, float]] = []

    for step in range(FORECAST_HOURS):
        x = history.reshape(1, window, len(feat_cols)).astype(np.float32)
        pred_targets_scaled = keras_model.predict(x, verbose=0)[0].astype(np.float32)

        next_row_scaled = np.zeros(len(feat_cols), dtype=np.float32)
        next_row_scaled[tgt_idxs] = pred_targets_scaled
        next_row_orig = np.zeros(len(feat_cols), dtype=np.float32)
        next_row_orig[tgt_idxs] = pred_targets_scaled * sigma_safe[tgt_idxs] + mu_all[tgt_idxs]

        # Weather: seasonal naive (24h lag)
        for col in WEATHER_COLUMNS:
            cidx = idx_map[col]
            w_val = float(history_orig[-24, cidx]) if len(history_orig) >= 24 else float(history_orig[-1, cidx])
            next_row_orig[cidx] = w_val
            next_row_scaled[cidx] = (w_val - mu_all[cidx]) / sigma_safe[cidx]

        # Time features from future timestamp
        next_ts = df.index[-1] + timedelta(hours=step + 1)
        time_feats = _future_time_features(next_ts)
        for col, val in time_feats.items():
            if col in idx_map:
                cidx = idx_map[col]
                next_row_orig[cidx] = float(val)
                next_row_scaled[cidx] = (float(val) - mu_all[cidx]) / sigma_safe[cidx]

        # Store prediction (unscaled target values)
        row_dict: dict[str, float] = {}
        for col in target_cols:
            row_dict[col] = float(next_row_orig[idx_map[col]])
        for col in WEATHER_COLUMNS:
            row_dict[col] = float(next_row_orig[idx_map[col]])
        preds.append(row_dict)

        # Slide window
        history = np.vstack([history[1:], next_row_scaled.reshape(1, -1)])
        history_orig = np.vstack([history_orig[1:], next_row_orig.reshape(1, -1)])

    start = df.index[-1] + timedelta(hours=1)
    idx = pd.date_range(start=start, periods=FORECAST_HOURS, freq="h")
    out = pd.DataFrame(preds, index=idx, columns=FORECAST_OUTPUT_COLUMNS)
    return idx, out


def forecast_xgboost(df: pd.DataFrame, model: Any) -> tuple[pd.DatetimeIndex, pd.DataFrame]:
    """168-hour recursive XGBoost forecast. Logic matches models/xgboost_forecast.py."""
    _validate_multivariate_artifacts(model, "XGBoost")
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise ImportError("xgboost is required for XGBoost inference.") from exc

    window = int(model["window"])
    feat_cols = list(model["feat_cols"])
    target_cols = list(model["target_cols"])
    tgt_idxs = list(model["tgt_idxs"])
    mu_all = np.asarray(model["mu_all"], dtype=np.float32)
    sigma_all = np.asarray(model["sigma_all"], dtype=np.float32)
    sigma_safe = np.where(sigma_all < 1e-8, 1.0, sigma_all)
    boosters = list(model["boosters"])

    if len(boosters) != len(target_cols):
        raise ValueError("XGBoost artifacts have mismatched boosters and target columns.")
    if len(df) < window:
        raise ValueError(f"Need at least {window} rows for XGBoost forecasting.")

    df_feat = _append_time_features(df)
    df_feat = df_feat[feat_cols].interpolate(limit_direction="both").ffill().bfill()

    all_scaled = ((df_feat.values - mu_all) / sigma_safe).astype(np.float32)
    history = all_scaled[-window:].copy()
    history_orig = (history * sigma_safe + mu_all).astype(np.float32)

    idx_map = {c: i for i, c in enumerate(feat_cols)}
    preds: list[dict[str, float]] = []

    for step in range(FORECAST_HOURS):
        x2d = history.reshape(1, -1)
        dmat = xgb.DMatrix(x2d)
        pred_targets_scaled = np.array([b.predict(dmat)[0] for b in boosters], dtype=np.float32)

        next_row_scaled = np.zeros(len(feat_cols), dtype=np.float32)
        next_row_scaled[tgt_idxs] = pred_targets_scaled
        next_row_orig = np.zeros(len(feat_cols), dtype=np.float32)
        next_row_orig[tgt_idxs] = pred_targets_scaled * sigma_safe[tgt_idxs] + mu_all[tgt_idxs]

        # Weather: seasonal naive (24h lag)
        for col in WEATHER_COLUMNS:
            cidx = idx_map[col]
            w_val = float(history_orig[-24, cidx]) if len(history_orig) >= 24 else float(history_orig[-1, cidx])
            next_row_orig[cidx] = w_val
            next_row_scaled[cidx] = (w_val - mu_all[cidx]) / sigma_safe[cidx]

        # Time features from future timestamp
        next_ts = df.index[-1] + timedelta(hours=step + 1)
        time_feats = _future_time_features(next_ts)
        for col, val in time_feats.items():
            if col in idx_map:
                cidx = idx_map[col]
                next_row_orig[cidx] = float(val)
                next_row_scaled[cidx] = (float(val) - mu_all[cidx]) / sigma_safe[cidx]

        # Store prediction (unscaled target values)
        row_dict: dict[str, float] = {}
        for col in target_cols:
            row_dict[col] = float(next_row_orig[idx_map[col]])
        for col in WEATHER_COLUMNS:
            row_dict[col] = float(next_row_orig[idx_map[col]])
        preds.append(row_dict)

        # Slide window
        history = np.vstack([history[1:], next_row_scaled.reshape(1, -1)])
        history_orig = np.vstack([history_orig[1:], next_row_orig.reshape(1, -1)])

    start = df.index[-1] + timedelta(hours=1)
    idx = pd.date_range(start=start, periods=FORECAST_HOURS, freq="h")
    out = pd.DataFrame(preds, index=idx, columns=FORECAST_OUTPUT_COLUMNS)
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
