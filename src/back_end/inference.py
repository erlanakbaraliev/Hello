"""Shared recursive single-step inference engine for the upload forecast flow.

The two model-specific entry points (:mod:`back_end.lstm_pred`,
:mod:`back_end.xgboost_pred`) feed this module a ``predict_one_step``
callable that maps a 48-row scaled history to a 1-D vector of scaled target
predictions. This module then:

* validates the requested horizon against the preprocessed upload;
* drives the recursive loop;
* injects the user-provided future weather and the per-step calendar features
  into the next history row;
* unscales the predictions and assembles the public output frame; and
* builds the JSON-safe payload persisted to the prediction history table.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from back_end.constants import (
    POLLUTANT_COLUMNS,
    SUPPORTED_HORIZONS,
    TIME_FEATURE_COLUMNS,
    WEATHER_COLUMNS,
    WINDOW,
    HorizonUnavailableError,
    UploadValidationError,
)
from back_end.upload_preprocess import PreprocessedUpload

# ─── Horizon validation ──────────────────────────────────────────────────────


def validate_horizon(preprocessed: PreprocessedUpload, horizon: int) -> None:
    """Raise :class:`HorizonUnavailableError` if H is unsupported or unavailable."""
    if horizon not in SUPPORTED_HORIZONS:
        raise HorizonUnavailableError(
            f"Unsupported forecast horizon {horizon}. "
            f"Choose from {', '.join(str(h) for h in SUPPORTED_HORIZONS)}."
        )
    n_future = len(preprocessed.future_weather_df)
    if n_future < horizon:
        raise HorizonUnavailableError(
            f"Selected horizon {horizon}h requires {horizon} future hourly rows "
            f"of temperature/wind_speed; only {n_future} available."
        )


# ─── Scaling helpers ─────────────────────────────────────────────────────────


def _safe_sigma(sigma: np.ndarray) -> np.ndarray:
    return np.where(sigma < 1e-8, 1.0, sigma).astype(np.float64)


def _scale_row(values: np.ndarray, mu: np.ndarray, sigma_safe: np.ndarray) -> np.ndarray:
    return ((values - mu) / sigma_safe).astype(np.float32)


def _time_features_for_timestamp(ts: pd.Timestamp) -> dict[str, float]:
    """Single-row time features for a future timestamp ``ts``."""
    h = float(ts.hour)
    dow = float(ts.dayofweek)
    m = float(ts.month)
    return {
        "h_sin": float(np.sin(2 * np.pi * h / 24)),
        "h_cos": float(np.cos(2 * np.pi * h / 24)),
        "h2_sin": float(np.sin(4 * np.pi * h / 24)),
        "h2_cos": float(np.cos(4 * np.pi * h / 24)),
        "dow_sin": float(np.sin(2 * np.pi * dow / 7)),
        "dow_cos": float(np.cos(2 * np.pi * dow / 7)),
        "mon_sin": float(np.sin(2 * np.pi * m / 12)),
        "mon_cos": float(np.cos(2 * np.pi * m / 12)),
    }


# ─── Recursive single-step engine ────────────────────────────────────────────


def _run_recursive_single_step(
    *,
    preprocessed: PreprocessedUpload,
    horizon: int,
    feat_cols: list[str],
    target_cols: list[str],
    tgt_idxs: list[int],
    mu: np.ndarray,
    sigma: np.ndarray,
    predict_one_step: Any,
) -> pd.DataFrame:
    """Drive H recursive single-step predictions and return ``(H x 6)`` originals.

    ``predict_one_step`` takes ``history`` of shape ``(WINDOW, n_features)``
    and returns a 1-D array of length ``len(target_cols)`` in *scaled* space.
    """
    validate_horizon(preprocessed, horizon)
    sigma_safe = _safe_sigma(sigma.astype(np.float64))
    mu64 = mu.astype(np.float64)

    past_aligned = preprocessed.past_df_aligned[feat_cols]
    future_aligned = preprocessed.future_weather_df[feat_cols].head(horizon)

    if past_aligned.shape[1] != len(feat_cols):
        raise ValueError(
            f"Past block has {past_aligned.shape[1]} features; "
            f"expected {len(feat_cols)} matching artifact feat_cols."
        )

    n_features = len(feat_cols)
    idx_map: dict[str, int] = {c: i for i, c in enumerate(feat_cols)}
    weather_indices = [idx_map[c] for c in WEATHER_COLUMNS]
    time_feature_indices = [idx_map[c] for c in TIME_FEATURE_COLUMNS]
    target_indices = list(tgt_idxs)

    past_raw = past_aligned.tail(WINDOW).to_numpy(dtype=np.float64)
    history = ((past_raw - mu64) / sigma_safe).astype(np.float32)

    one_hour = pd.Timedelta(hours=1)
    future_index = future_aligned.index
    if len(future_index) < horizon:
        raise HorizonUnavailableError(
            f"Internal: aligned future block has {len(future_index)} rows "
            f"but horizon is {horizon}."
        )
    expected_first = preprocessed.cutoff_ts + one_hour
    if future_index[0] != expected_first:
        raise UploadValidationError(
            f"Future block must start exactly 1 hour after the cutoff "
            f"({expected_first}); first future row is {future_index[0]}."
        )

    pred_records: list[dict[str, float]] = []
    for step in range(horizon):
        next_ts = future_index[step]

        pred_targets_scaled = np.asarray(predict_one_step(history), dtype=np.float32)
        if pred_targets_scaled.shape != (len(target_cols),):
            raise ValueError(
                f"Model returned shape {pred_targets_scaled.shape}; "
                f"expected ({len(target_cols)},)."
            )

        next_row_scaled = np.zeros(n_features, dtype=np.float32)
        for j, ti in enumerate(target_indices):
            next_row_scaled[ti] = pred_targets_scaled[j]

        # Future weather from the upload — never seasonal-naive.
        for col, ci in zip(WEATHER_COLUMNS, weather_indices):
            v = float(future_aligned.iloc[step][col])
            next_row_scaled[ci] = (v - mu64[ci]) / sigma_safe[ci]

        # Time features derived from next_ts.
        time_vals = _time_features_for_timestamp(next_ts)
        for col, ci in zip(TIME_FEATURE_COLUMNS, time_feature_indices):
            v = float(time_vals[col])
            next_row_scaled[ci] = (v - mu64[ci]) / sigma_safe[ci]

        # Record originals.
        rec: dict[str, float] = {"time": next_ts}  # type: ignore[dict-item]
        for j, col in enumerate(target_cols):
            ti = target_indices[j]
            rec[f"{col}_pred"] = float(
                pred_targets_scaled[j] * sigma_safe[ti] + mu64[ti]
            )
        rec["temperature_used"] = float(future_aligned.iloc[step]["temperature"])
        rec["wind_speed_used"] = float(future_aligned.iloc[step]["wind_speed"])
        pred_records.append(rec)

        history = np.vstack([history[1:], next_row_scaled.reshape(1, -1)])

    out = pd.DataFrame(pred_records).set_index("time")
    if not np.isfinite(out.select_dtypes(include=[np.number]).to_numpy()).all():
        raise ValueError("Forecast produced non-finite values.")
    return out


# ─── Output assembly ─────────────────────────────────────────────────────────


def build_forecast_output(
    raw_preds: pd.DataFrame,
    future_actuals_df: pd.DataFrame | None,
    *,
    target_cols: list[str],
    model_name: str,
    horizon: int,
) -> pd.DataFrame:
    """Assemble the final per-step forecast DataFrame in the documented order."""
    out = raw_preds.reset_index().rename(columns={"index": "time"})
    if "time" not in out.columns:
        out = raw_preds.reset_index()

    out["model_name"] = model_name
    out["forecast_horizon"] = int(horizon)

    pred_cols = [f"{c}_pred" for c in target_cols]
    ordered = [
        "time",
        *pred_cols,
        "temperature_used",
        "wind_speed_used",
        "model_name",
        "forecast_horizon",
    ]

    if future_actuals_df is not None:
        actuals = future_actuals_df.head(len(out)).reset_index(drop=True)
        actuals.columns = [f"{c}_actual" for c in actuals.columns]
        out = out.reset_index(drop=True)
        for col in actuals.columns:
            out[col] = actuals[col].to_numpy()
        ordered = ordered + [f"{c}_actual" for c in target_cols]

    return out[ordered]


# ─── JSON-safe prediction payload (for database.save_prediction) ─────────────


def prediction_payload(
    preprocessed: PreprocessedUpload,
    forecast_df: pd.DataFrame,
    *,
    model_name: str,
    horizon: int,
) -> dict[str, Any]:
    """Build a JSON-serialisable payload for ``database.save_prediction``."""
    target_cols = list(POLLUTANT_COLUMNS)
    history_records: list[dict[str, Any]] = []
    past = preprocessed.past_df_aligned
    for ts, row in past[target_cols + WEATHER_COLUMNS].tail(WINDOW).iterrows():
        rec: dict[str, Any] = {"time": pd.Timestamp(ts).isoformat()}
        for col in target_cols + WEATHER_COLUMNS:
            rec[col] = float(row[col])
        history_records.append(rec)

    forecast_records: list[dict[str, Any]] = []
    for _, row in forecast_df.iterrows():
        rec = {"time": pd.Timestamp(row["time"]).isoformat()}
        for col in target_cols:
            rec[col] = float(row[f"{col}_pred"])
        rec["temperature_used"] = float(row["temperature_used"])
        rec["wind_speed_used"] = float(row["wind_speed_used"])
        if "pm25_safety_level" in forecast_df.columns:
            rec["pm25_safety_level"] = str(row["pm25_safety_level"])
        if f"{target_cols[0]}_actual" in forecast_df.columns:
            for col in target_cols:
                rec[f"{col}_actual"] = float(row[f"{col}_actual"])
        forecast_records.append(rec)

    return {
        "model": model_name,
        "forecast_horizon": int(horizon),
        "source": "upload_full_models",
        "cutoff_ts": pd.Timestamp(preprocessed.cutoff_ts).isoformat(),
        "columns": target_cols + WEATHER_COLUMNS,
        "history": history_records,
        "forecast": forecast_records,
    }
