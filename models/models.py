"""Forecasting models for daily PM2.5 prediction."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

MODELS_DIR = Path(__file__).resolve().parent / "models"
HASHES_PATH = MODELS_DIR / "hashes.json"
FORECAST_DAYS = 7
LSTM_LOOKBACK = 24

# ---------------------------------------------------------------------------
# Model metadata helpers
# ---------------------------------------------------------------------------


def get_model_metadata(model_name: str) -> dict[str, Any]:
    """Load metadata for a saved model artifact.

    Each trained model can have a companion ``<name>_metadata.json`` file
    alongside the artifact.  If the file is absent the function returns a
    minimal dict so callers never have to guard against ``None``.

    Expected metadata keys (all optional):
        trained_at   – ISO-8601 timestamp of the training run
        data_version – identifier / hash of the training dataset
        metrics      – dict of evaluation metrics (rmse, mae, …)
        notes        – free-form string
    """
    meta_path = MODELS_DIR / f"{model_name.lower()}_metadata.json"
    if meta_path.is_file():
        try:
            with meta_path.open() as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError):
            pass
    return {"model": model_name, "note": "No metadata file found."}


def save_model_metadata(model_name: str, metadata: dict[str, Any]) -> None:
    """Persist metadata for a model artifact.

    Args:
        model_name: Base name used to derive the JSON filename
                    (e.g. ``"arima"`` → ``models/arima_metadata.json``).
        metadata:   JSON-serialisable dict.  A ``saved_at`` timestamp is
                    injected automatically if not already present.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = MODELS_DIR / f"{model_name.lower()}_metadata.json"
    payload = dict(metadata)
    payload.setdefault("saved_at", datetime.now(timezone.utc).isoformat())
    with meta_path.open("w") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


def verify_artifact_hash(path: Path) -> None:
    if not HASHES_PATH.is_file():
        return
    try:
        with HASHES_PATH.open() as fh:
            hashes = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return
    expected = hashes.get(path.name)
    if not expected:
        return
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected:
        raise ValueError(f"Artifact integrity check failed for {path.name}")


# ---------------------------------------------------------------------------
# Artifact loaders
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def load_lstm_artifacts() -> tuple[Any, Any]:
    """Load the pre-trained LSTM model and its MinMaxScaler.

    Returns:
        ``(keras_model, scaler)`` tuple.

    Raises:
        FileNotFoundError: If either artifact is missing from ``models/``.
        ImportError: If TensorFlow is not installed.
    """
    model_path = MODELS_DIR / "lstm_model.keras"
    scaler_path = MODELS_DIR / "lstm_scaler.pkl"
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing LSTM model artifact: {model_path}")
    if not scaler_path.is_file():
        raise FileNotFoundError(f"Missing LSTM scaler artifact: {scaler_path}")
    verify_artifact_hash(model_path)
    verify_artifact_hash(scaler_path)
    try:
        from tensorflow import keras
    except ImportError as exc:
        raise ImportError("TensorFlow is required to run LSTM predictions.") from exc

    model = keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


@lru_cache(maxsize=1)
def load_xgboost_artifact() -> Any:
    """Load the pre-trained XGBoost model.

    Returns:
        Fitted ``XGBRegressor`` (or compatible) instance.

    Raises:
        FileNotFoundError: If the artifact is missing from ``models/``.
    """
    model_path = MODELS_DIR / "xgboost_model.joblib"
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing XGBoost model artifact: {model_path}")
    verify_artifact_hash(model_path)
    return joblib.load(model_path)


# ---------------------------------------------------------------------------
# ARIMA
# ---------------------------------------------------------------------------


def train_arima_and_forecast(
    pm25_series: pd.Series, forecast_days: int = FORECAST_DAYS
) -> pd.DataFrame:
    """Fit ARIMA(2,1,2) on *pm25_series* and return a point-forecast DataFrame."""
    y = pd.Series(pm25_series).dropna().astype(float)
    if len(y) < 30:
        raise ValueError("ARIMA requires at least 30 observations.")

    model = ARIMA(y, order=(2, 1, 2))
    fitted = model.fit()
    preds = fitted.forecast(steps=forecast_days)

    if isinstance(y.index, pd.DatetimeIndex) and len(y.index) > 0:
        start = y.index[-1] + timedelta(days=1)
        pred_index = pd.date_range(start=start, periods=forecast_days, freq="D")
    else:
        pred_index = pd.RangeIndex(start=1, stop=forecast_days + 1)

    return pd.DataFrame(
        {
            "date": pred_index,
            "model": "ARIMA",
            "predicted_pm2_5": np.asarray(preds, dtype=float),
        }
    )


def train_arima_with_intervals(
    pm25_series: pd.Series,
    forecast_days: int = FORECAST_DAYS,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Fit ARIMA(2,1,2) and return forecast with confidence intervals.

    Args:
        pm25_series: Historical PM2.5 values.
        forecast_days: Number of days to forecast.
        alpha: Significance level for confidence intervals (default 0.05 → 95 %).

    Returns:
        DataFrame with columns: date, model, predicted_pm2_5, lower_ci, upper_ci.
    """
    y = pd.Series(pm25_series).dropna().astype(float)
    if len(y) < 30:
        raise ValueError("ARIMA requires at least 30 observations.")
    model = ARIMA(y, order=(2, 1, 2))
    fitted = model.fit()
    fc = fitted.get_forecast(steps=forecast_days)
    pred = fc.predicted_mean.to_numpy(dtype=float)
    conf = fc.conf_int(alpha=alpha)
    lower = conf.iloc[:, 0].to_numpy(dtype=float)
    upper = conf.iloc[:, 1].to_numpy(dtype=float)
    if isinstance(y.index, pd.DatetimeIndex) and len(y.index) > 0:
        start = y.index[-1] + timedelta(days=1)
        pred_index = pd.date_range(start=start, periods=forecast_days, freq="D")
    else:
        pred_index = pd.RangeIndex(start=1, stop=forecast_days + 1)
    return pd.DataFrame(
        {
            "date": pred_index,
            "model": "ARIMA",
            "predicted_pm2_5": pred,
            "lower_ci": lower,
            "upper_ci": upper,
        }
    )


# ---------------------------------------------------------------------------
# LSTM
# ---------------------------------------------------------------------------


def _resolve_feature_columns(df: pd.DataFrame, scaler: Any) -> list[str]:
    feature_names = getattr(scaler, "feature_names_in_", None)
    if feature_names is not None:
        cols = [c for c in feature_names if c in df.columns]
        if len(cols) == len(feature_names):
            return cols
    fallback = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return fallback


def forecast_lstm_pretrained(df: pd.DataFrame, forecast_days: int = FORECAST_DAYS) -> pd.DataFrame:
    """Generate a *forecast_days*-step PM2.5 forecast using the saved LSTM.

    Args:
        df: Preprocessed DataFrame with a DatetimeIndex and at least a
            ``pm2_5`` column (plus whatever features the scaler expects).
        forecast_days: Number of days to forecast ahead.

    Returns:
        DataFrame with columns: date, model, predicted_pm2_5.
    """
    model, scaler = load_lstm_artifacts()
    feature_cols = _resolve_feature_columns(df, scaler)
    if "pm2_5" not in feature_cols:
        raise ValueError("LSTM input must contain a 'pm2_5' feature.")
    if len(df) < LSTM_LOOKBACK:
        raise ValueError(f"LSTM needs at least {LSTM_LOOKBACK} rows.")

    transformed = scaler.transform(df[feature_cols]).astype(np.float32)
    window = transformed[-LSTM_LOOKBACK:].copy()
    pm25_idx = feature_cols.index("pm2_5")

    predictions_scaled: list[float] = []
    last_row = window[-1].copy()
    for _ in range(forecast_days):
        x = window.reshape(1, LSTM_LOOKBACK, len(feature_cols))
        next_value = float(model.predict(x, verbose=0)[0, 0])
        predictions_scaled.append(next_value)
        updated_row = last_row.copy()
        updated_row[pm25_idx] = next_value
        window = np.vstack([window[1:], updated_row.reshape(1, -1)])
        last_row = updated_row

    pad = np.zeros((forecast_days, len(feature_cols)), dtype=np.float32)
    pad[:, pm25_idx] = np.array(predictions_scaled, dtype=np.float32)
    predictions = scaler.inverse_transform(pad)[:, pm25_idx]

    if isinstance(df.index, pd.DatetimeIndex) and len(df.index) > 0:
        start = df.index[-1] + timedelta(days=1)
        pred_index = pd.date_range(start=start, periods=forecast_days, freq="D")
    else:
        pred_index = pd.RangeIndex(start=1, stop=forecast_days + 1)

    return pd.DataFrame(
        {
            "date": pred_index,
            "model": "LSTM",
            "predicted_pm2_5": predictions.astype(float),
        }
    )


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------


def _build_xgb_row(
    ts: pd.Timestamp,
    hist_pm25: list[float],
    base_exog: pd.Series,
) -> pd.Series:
    """Construct a single feature row for the XGBoost model."""
    row = base_exog.copy()
    row["hour"] = int(ts.hour)
    row["day_of_week"] = int(ts.dayofweek)
    row["month"] = int(ts.month)
    row["pm2_5_lag1"] = hist_pm25[-1]
    row["pm2_5_lag2"] = hist_pm25[-2] if len(hist_pm25) >= 2 else hist_pm25[-1]
    row["pm2_5_lag24"] = hist_pm25[-24] if len(hist_pm25) >= 24 else hist_pm25[0]
    return row


def forecast_xgboost_pretrained(
    df: pd.DataFrame, forecast_days: int = FORECAST_DAYS
) -> pd.DataFrame:
    """Generate a *forecast_days*-step PM2.5 forecast using the saved XGBoost model.

    The model is expected to have been trained with features:
    ``hour``, ``day_of_week``, ``month``, ``pm2_5_lag1``, ``pm2_5_lag2``,
    ``pm2_5_lag24``, plus the other pollutant columns.

    Args:
        df: Preprocessed DataFrame with a DatetimeIndex.
        forecast_days: Number of days to forecast ahead.

    Returns:
        DataFrame with columns: date, model, predicted_pm2_5.
    """
    xgb_model = load_xgboost_artifact()
    feat_order: list[str] = list(getattr(xgb_model, "feature_names_in_", []))
    if not feat_order:
        raise ValueError(
            "XGBoost model has no feature_names_in_; retrain with a recent scikit-learn API."
        )

    pollutant_cols = [
        c
        for c in ("pm10", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone")
        if c in df.columns
    ]
    base_exog = df.iloc[-1][pollutant_cols].astype(float)
    hist_pm25 = [float(x) for x in df["pm2_5"].tolist()]
    if len(hist_pm25) < 25:
        raise ValueError("Uploaded series too short for XGBoost lag-24 features (need ≥ 25 rows).")

    preds: list[float] = []
    cur_ts = df.index[-1]
    for _ in range(forecast_days):
        cur_ts = cur_ts + timedelta(days=1)
        row = _build_xgb_row(cur_ts, hist_pm25, base_exog)
        # Align to the exact feature order the model was trained with
        x = row.reindex(feat_order, fill_value=0.0).values.reshape(1, -1)
        p = float(xgb_model.predict(x)[0])
        preds.append(p)
        hist_pm25.append(p)

    if isinstance(df.index, pd.DatetimeIndex) and len(df.index) > 0:
        start = df.index[-1] + timedelta(days=1)
        pred_index = pd.date_range(start=start, periods=forecast_days, freq="D")
    else:
        pred_index = pd.RangeIndex(start=1, stop=forecast_days + 1)

    return pd.DataFrame(
        {
            "date": pred_index,
            "model": "XGBoost",
            "predicted_pm2_5": np.array(preds, dtype=float),
        }
    )


# ---------------------------------------------------------------------------
# Unified interface
# ---------------------------------------------------------------------------


def predict_next_7_days(model_name: str, prepared_df: pd.DataFrame) -> pd.DataFrame:
    """Dispatch to the correct model and return a 7-day PM2.5 forecast.

    Args:
        model_name: One of ``"ARIMA"``, ``"LSTM"``, or ``"XGBoost"``
                    (case-insensitive).
        prepared_df: Preprocessed DataFrame produced by
                     ``data_processing.preprocess_pipeline``.

    Returns:
        DataFrame with columns: date, model, predicted_pm2_5.

    Raises:
        ValueError: If *model_name* is not recognised.
    """
    choice = (model_name or "").strip().upper()
    if choice == "ARIMA":
        return train_arima_and_forecast(prepared_df["pm2_5"], forecast_days=FORECAST_DAYS)
    if choice == "LSTM":
        return forecast_lstm_pretrained(prepared_df, forecast_days=FORECAST_DAYS)
    if choice == "XGBOOST":
        return forecast_xgboost_pretrained(prepared_df, forecast_days=FORECAST_DAYS)
    raise ValueError(f"Unknown model '{model_name}'. Choose ARIMA, LSTM, or XGBoost.")


def compare_models(prepared_df: pd.DataFrame) -> pd.DataFrame:
    """Run ARIMA and LSTM forecasts and return a combined DataFrame.

    XGBoost is excluded from the comparison view because it requires
    different feature engineering (lag columns) that may not be present
    in all prepared DataFrames.

    Returns:
        DataFrame with columns: date, model, predicted_pm2_5.
    """
    arima = train_arima_and_forecast(prepared_df["pm2_5"], forecast_days=FORECAST_DAYS)
    lstm = forecast_lstm_pretrained(prepared_df, forecast_days=FORECAST_DAYS)
    return pd.concat([arima, lstm], ignore_index=True)
