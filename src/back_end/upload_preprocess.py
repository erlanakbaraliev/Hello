"""Upload-side validation and preprocessing for the forecasting pipeline.

This module is **UI-free**. It owns:

* :class:`PreprocessedUpload` — the output container consumed by the
  recursive single-step inference engine in :mod:`back_end.inference`.
* :func:`preprocess_uploaded_dataset` — the public entry point used by
  the Streamlit forecast page after the user uploads a CSV.
* All schema/structure checks: hourly cadence, no duplicate or unparseable
  timestamps, required column presence, and the strict no-interpolation
  policy on future weather drivers.

Future pollutant values, if present in the upload, are surfaced as
``*_actual`` for evaluation but are **never** fed back as model inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import joblib
import numpy as np
import pandas as pd

from back_end.constants import (
    INTERPOLATE_LIMIT,
    LSTM_ARTIFACT_PATH,
    OPTIONAL_DROP,
    POLLUTANT_COLUMNS,
    REQUIRED_FUTURE_COLUMNS,
    REQUIRED_PAST_COLUMNS,
    SUPPORTED_HORIZONS,
    TIME_FEATURE_COLUMNS,
    WEATHER_COLUMNS,
    WINDOW,
    XGB_ARTIFACT_PATH,
    UploadValidationError,
)

# ─── Result container ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PreprocessedUpload:
    """Output of :func:`preprocess_uploaded_dataset`.

    All frames use a tz-aware ``DatetimeIndex`` named ``time``.

    Attributes
    ----------
    past_df_aligned :
        Past block aligned exactly to the model artifact's ``feat_cols``
        (16 columns: 6 pollutants + 2 weather + 8 sin/cos calendar).
        Length >= 48; only the last 48 rows are used as inference history.
    future_weather_df :
        Future block aligned to ``feat_cols`` for convenience. Pollutant
        columns are present but **must not** be read as inputs; only
        ``temperature``, ``wind_speed`` and the 8 calendar columns are used.
    cutoff_ts :
        Last timestamp at which all six pollutants AND both weather columns
        were non-null. Past block ends here (inclusive); future block starts
        at ``cutoff_ts + 1h``.
    future_actuals_df :
        If the future block contains valid pollutant values (no NaNs in any
        of the six target columns for the available future rows), this holds
        them as a 6-column frame for optional ``*_actual`` reporting.
        Otherwise ``None``.
    available_horizons :
        The subset of ``SUPPORTED_HORIZONS`` for which enough future weather
        rows exist (i.e. ``len(future_weather_df) >= H``). Useful for
        disabling unsupported choices in the UI.
    """

    past_df_aligned: pd.DataFrame
    future_weather_df: pd.DataFrame
    cutoff_ts: pd.Timestamp
    future_actuals_df: pd.DataFrame | None
    available_horizons: tuple[int, ...] = field(default_factory=tuple)


# ─── Calendar feature engineering (used by both past and future blocks) ──────


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append the eight sin/cos calendar features used in training.

    Formulae are byte-identical to ``lstm_full_training.py`` and
    ``xgboost_full_training.py``. The frame must have a ``DatetimeIndex``.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("add_time_features requires a DatetimeIndex.")
    out = df.copy()
    h = df.index.hour
    dow = df.index.dayofweek
    m = df.index.month
    out["h_sin"] = np.sin(2 * np.pi * h / 24)
    out["h_cos"] = np.cos(2 * np.pi * h / 24)
    out["h2_sin"] = np.sin(4 * np.pi * h / 24)
    out["h2_cos"] = np.cos(4 * np.pi * h / 24)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    out["mon_sin"] = np.sin(2 * np.pi * m / 12)
    out["mon_cos"] = np.cos(2 * np.pi * m / 12)
    return out


# ─── Validation primitives ───────────────────────────────────────────────────


def _normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def _drop_optional_columns(df: pd.DataFrame) -> pd.DataFrame:
    drop = [c for c in df.columns if c in OPTIONAL_DROP]
    return df.drop(columns=drop, errors="ignore") if drop else df


def _check_columns_present(df: pd.DataFrame, required: list[str], where: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise UploadValidationError(
            f"Upload is missing required {where} column(s): {', '.join(missing)}."
        )


def _parse_time_column(df: pd.DataFrame) -> pd.DataFrame:
    if "time" not in df.columns:
        raise UploadValidationError("Upload must contain a 'time' column.")
    out = df.copy()
    out["time"] = pd.to_datetime(out["time"], utc=True, errors="coerce")
    if out["time"].isna().all():
        raise UploadValidationError(
            "Could not parse any valid timestamps in 'time'. "
            "Use ISO 8601 format such as '2024-01-01 00:00:00+00:00'."
        )
    out = out.dropna(subset=["time"])
    return out


def _check_unique_and_hourly(idx: pd.DatetimeIndex) -> None:
    if not idx.is_unique:
        dups = idx[idx.duplicated()].unique().tolist()
        examples = ", ".join(str(t) for t in dups[:3])
        raise UploadValidationError(
            f"Duplicate timestamps detected: {examples}. "
            "Each timestamp must be unique."
        )
    if len(idx) < 2:
        return
    diffs = idx.to_series().diff().dropna()
    one_hour = pd.Timedelta(hours=1)
    bad_mask = diffs != one_hour
    if bad_mask.any():
        first_bad_pos = int(np.where(bad_mask.to_numpy())[0][0])
        # diffs index is aligned with idx[1:]; the gap is between idx[first_bad_pos]
        # and idx[first_bad_pos + 1].
        ts_a = idx[first_bad_pos]
        ts_b = idx[first_bad_pos + 1]
        delta_h = (ts_b - ts_a).total_seconds() / 3600.0
        raise UploadValidationError(
            "Timestamps are not strictly hourly. "
            f"First non-hourly gap detected at {ts_a} -> {ts_b} "
            f"(delta = {delta_h:g}h)."
        )


def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _detect_cutoff(df_indexed: pd.DataFrame) -> pd.Timestamp:
    """Auto-detect cutoff = last row where all pollutants AND both weather cols
    are non-null AND at least one row follows it.

    The "next-row-exists" guard handles the backtest case where pollutants are
    present throughout the dataset: the auto-detector then picks the second-to-
    last fully-observed row so that at least one future row remains. Callers
    that need a specific boundary should pass ``cutoff_ts`` to
    :func:`preprocess_uploaded_dataset` explicitly.
    """
    needed = POLLUTANT_COLUMNS + WEATHER_COLUMNS
    fully_observed = df_indexed[needed].notna().all(axis=1)
    if not fully_observed.any():
        raise UploadValidationError(
            "No row has all six pollutant columns AND both weather columns "
            "non-null at the same time. Cannot determine the past/future cutoff."
        )
    candidates = df_indexed.index[fully_observed]
    last_ts = df_indexed.index[-1]
    for ts in reversed(candidates):
        if ts != last_ts:
            return cast(pd.Timestamp, ts)
    # Only the very last row is fully observed → cannot leave any future rows.
    raise UploadValidationError(
        "Auto-detected cutoff would leave no future rows. The dataset has "
        "fully-observed pollutant values only at the very last timestamp. "
        "Either remove future-row pollutants or pass an explicit cutoff_ts."
    )


def _assert_future_weather_user_filled(weather_df: pd.DataFrame) -> pd.DataFrame:
    """Require complete future ``temperature`` and ``wind_speed`` — no interpolation.

    Unlike the past block, missing future weather is an upload error so users
    cannot accidentally run forecasts on imputed future drivers.
    """
    if weather_df.empty:
        raise UploadValidationError(
            "No future rows after the cutoff timestamp. Upload at least one "
            "future hourly row containing temperature and wind_speed for the "
            "selected horizon."
        )
    work = weather_df[WEATHER_COLUMNS].copy()
    vals = work.to_numpy(dtype=np.float64, copy=False)
    bad = np.isnan(vals) | ~np.isfinite(vals)
    if not bad.any():
        return work.astype(np.float64)

    bad_mask = bad.any(axis=1)
    bad_times = work.index[bad_mask]
    examples = ", ".join(str(t) for t in bad_times[:5])
    more = f" (+{len(bad_times) - 5} more)" if len(bad_times) > 5 else ""
    raise UploadValidationError(
        "Future weather must have valid numeric temperature and wind_speed for "
        "every future hour (no missing or non-finite values; interpolation is "
        f"not applied to future weather). Problem timestamp(s): {examples}{more}."
    )


def _interpolate_block(df: pd.DataFrame, cols: list[str], where: str) -> pd.DataFrame:
    out = df.copy()
    if not cols:
        return out
    for c in cols:
        if c not in out.columns:
            continue
        out[c] = out[c].interpolate(
            method="time", limit=INTERPOLATE_LIMIT, limit_direction="both"
        )
    # Safe ffill/bfill only for residual edges within the block.
    out[cols] = out[cols].ffill().bfill()
    remaining = out[cols].isna().sum()
    bad = remaining[remaining > 0]
    if not bad.empty:
        details = ", ".join(f"{name} ({int(n)} rows)" for name, n in bad.items())
        raise UploadValidationError(
            f"Preprocessing left missing values in {where}: {details}. "
            "Please clean these or extend interpolation."
        )
    return out


def _align_to_feat_cols(
    df_with_time_feats: pd.DataFrame, feat_cols: list[str]
) -> pd.DataFrame:
    """Reorder + restrict columns to match the artifact's ``feat_cols`` exactly."""
    missing = [c for c in feat_cols if c not in df_with_time_feats.columns]
    if missing:
        # Fill missing pollutant columns in the future block with 0.0; they are
        # never read as inputs but must exist so column order matches.
        out = df_with_time_feats.copy()
        for c in missing:
            out[c] = 0.0
        return out[feat_cols]
    return df_with_time_feats[feat_cols].copy()


def _load_feat_cols_for_validation() -> list[str]:
    """Try to load ``feat_cols`` from the LSTM artifact for column alignment.

    Falls back to the static list in this module if the artifact is missing,
    so preprocessing still works in unit tests that don't ship the model.
    """
    try:
        if LSTM_ARTIFACT_PATH.is_file():
            artifacts = joblib.load(LSTM_ARTIFACT_PATH)
            cols = list(artifacts.get("feat_cols", []))
            if cols:
                return cols
    except Exception:  # noqa: BLE001 — fallback path
        pass
    if XGB_ARTIFACT_PATH.is_file():
        try:
            artifacts = joblib.load(XGB_ARTIFACT_PATH)
            cols = list(artifacts.get("feat_cols", []))
            if cols:
                return cols
        except Exception:  # noqa: BLE001 — fallback path
            pass
    return [*POLLUTANT_COLUMNS, *WEATHER_COLUMNS, *TIME_FEATURE_COLUMNS]


# ─── Public entry point ──────────────────────────────────────────────────────


def preprocess_uploaded_dataset(
    raw_df: pd.DataFrame,
    *,
    cutoff_ts: pd.Timestamp | None = None,
) -> PreprocessedUpload:
    """Validate, clean, and align an uploaded dataset for inference.

    Performs (in order):
      1. Lowercase column names; drop ``OPTIONAL_DROP`` (city/lat/lon).
      2. Parse ``time`` (UTC); drop rows whose timestamp could not be parsed.
      3. Convert past-required numeric columns with ``errors="coerce"``.
      4. Sort by time; reject duplicates and non-hourly gaps.
      5. Determine ``cutoff_ts`` (caller-supplied or auto-detected).
      6. Slice into past/future blocks.
      7. Interpolate (time, limit=6) + safe ffill/bfill on the **past** block only;
         future ``temperature`` / ``wind_speed`` must be user-filled (strict check).
      8. Add the 8 sin/cos calendar features to both blocks.
      9. Align both blocks to the artifact's ``feat_cols`` order.
     10. Validate len(past) >= 48.
     11. Build optional ``future_actuals_df`` from any clean future pollutants.

    Parameters
    ----------
    raw_df :
        The user's upload as a pandas DataFrame.
    cutoff_ts :
        Optional explicit boundary between past and future. When ``None``,
        the cutoff is auto-detected as the last timestamp at which all six
        pollutant columns AND both weather columns are non-null AND at least
        one row follows. Pass an explicit timestamp for backtests where
        pollutants are present throughout.
    """
    df = _normalize_column_names(raw_df)
    df = _drop_optional_columns(df)

    df = _parse_time_column(df)

    _check_columns_present(df, REQUIRED_PAST_COLUMNS, where="past")

    df = _coerce_numeric(df, POLLUTANT_COLUMNS + WEATHER_COLUMNS)

    df = df.sort_values("time").reset_index(drop=True)
    df_indexed = df.set_index("time")

    _check_unique_and_hourly(cast(pd.DatetimeIndex, df_indexed.index))

    if cutoff_ts is None:
        cutoff_ts = _detect_cutoff(df_indexed)
    else:
        cutoff_ts = pd.Timestamp(cutoff_ts)
        if cutoff_ts.tzinfo is None:
            cutoff_ts = cutoff_ts.tz_localize("UTC")
        if cutoff_ts not in df_indexed.index:
            raise UploadValidationError(
                f"Explicit cutoff_ts={cutoff_ts} is not present in the upload's "
                "timestamps."
            )
    one_hour = pd.Timedelta(hours=1)

    past_raw = df_indexed.loc[:cutoff_ts]
    future_raw = df_indexed.loc[cutoff_ts + one_hour :]

    if len(past_raw) < WINDOW:
        raise UploadValidationError(
            f"Past block has only {len(past_raw)} valid hourly rows; "
            f"at least {WINDOW} are required "
            f"(last fully-observed timestamp: {cutoff_ts})."
        )

    # Interpolate past block on pollutants + weather (all required as inputs).
    past_clean = _interpolate_block(
        past_raw[POLLUTANT_COLUMNS + WEATHER_COLUMNS],
        POLLUTANT_COLUMNS + WEATHER_COLUMNS,
        where="past block",
    )

    # Future block: pollutants are NOT required as model inputs. Only require
    # temperature + wind_speed for the rows we will actually consume.
    _check_columns_present(future_raw.reset_index(), REQUIRED_FUTURE_COLUMNS, where="future")

    future_weather_clean = _assert_future_weather_user_filled(
        future_raw[WEATHER_COLUMNS],
    )

    past_with_time = add_time_features(past_clean)
    future_with_time = add_time_features(future_weather_clean)

    feat_cols = _load_feat_cols_for_validation()
    past_aligned = _align_to_feat_cols(past_with_time, feat_cols)
    future_aligned = _align_to_feat_cols(future_with_time, feat_cols)

    # Optional: surface future actuals if all pollutant columns are clean for
    # at least the same number of rows as the future weather block.
    future_actuals_df: pd.DataFrame | None = None
    future_pollutants = future_raw.reindex(columns=POLLUTANT_COLUMNS)
    if not future_pollutants.empty and not future_pollutants.isna().any().any():
        future_actuals_df = future_pollutants.astype(float).copy()

    n_future = len(future_aligned)
    available = tuple(h for h in SUPPORTED_HORIZONS if n_future >= h)

    return PreprocessedUpload(
        past_df_aligned=past_aligned,
        future_weather_df=future_aligned,
        cutoff_ts=cutoff_ts,
        future_actuals_df=future_actuals_df,
        available_horizons=available,
    )
