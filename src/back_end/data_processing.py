"""Generic dataset loading, schema preparation, EDA, and training helpers.

This module is the *batch* counterpart to :mod:`back_end.upload_preprocess`:
it covers the raw-CSV → cleaned-frame stage and the lightweight feature
engineering used by the offline training pipeline. The user-upload forecast
flow is handled separately under :mod:`back_end.upload_preprocess`,
:mod:`back_end.inference`, and the per-model entry points.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from back_end.constants import (
    EXPECTED_COLUMNS,
    GEO_COLUMN_NAMES,
    POLLUTANT_COLUMNS,
)


# ─── Schema helpers ──────────────────────────────────────────────────────────


def drop_geo_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop geographic coordinate columns if present (case-insensitive names)."""
    to_drop = [c for c in df.columns if str(c).strip().lower() in GEO_COLUMN_NAMES]
    return df.drop(columns=to_drop, errors="ignore")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def load_dataset_from_data_folder(filename: str) -> pd.DataFrame:
    """Load a CSV from ``src/data/`` by filename. Raises if not found."""
    data_path = Path(__file__).resolve().parent.parent / "data" / filename
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    return pd.read_csv(data_path)


def ensure_expected_schema(df: pd.DataFrame, source: str = "uploaded") -> pd.DataFrame:
    """Coerce arbitrary input into the expected modeling schema."""
    data = _normalize_columns(df)
    for col in EXPECTED_COLUMNS:
        if col not in data.columns:
            if col == "city":
                data[col] = "unknown"
            elif col == "time":
                data[col] = pd.NaT
            else:
                data[col] = 0.0
    data = data[EXPECTED_COLUMNS].copy()
    data["source"] = source
    return data


def validate_and_prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce → DatetimeIndex → daily resample with simple gap filling."""
    data = ensure_expected_schema(df)

    data["time"] = pd.to_datetime(data["time"], utc=True, errors="coerce")
    data = data.dropna(subset=["time"]).sort_values("time").set_index("time")

    numeric_columns = list(POLLUTANT_COLUMNS)
    for col in numeric_columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    daily = data.resample("D").mean(numeric_only=True)
    daily = daily.interpolate(method="time").ffill().bfill()
    return daily


def combine_sources(
    uploaded_df: pd.DataFrame | None, live_df: pd.DataFrame | None
) -> pd.DataFrame:
    """Concatenate uploaded + live API frames, deduplicate on timestamp."""
    chunks: list[pd.DataFrame] = []
    if uploaded_df is not None and not uploaded_df.empty:
        chunks.append(ensure_expected_schema(uploaded_df, source="uploaded"))
    if live_df is not None and not live_df.empty:
        chunks.append(ensure_expected_schema(live_df, source="live_api"))
    if not chunks:
        raise ValueError("No source data available to combine.")
    out = pd.concat(chunks, ignore_index=True)
    out["time"] = pd.to_datetime(out["time"], utc=True, errors="coerce")
    out = out.dropna(subset=["time"]).sort_values("time")
    out = out.drop_duplicates(subset=["time"], keep="last")
    return out


# ─── Lightweight training-pipeline helpers ───────────────────────────────────


def create_lag_features(df: pd.DataFrame, target_col: str = "pm2_5") -> pd.DataFrame:
    """Add lag-1, lag-2 of ``target_col`` and lag-1 of selected drivers."""
    if target_col not in df.columns:
        raise ValueError(f"Target column not found: {target_col}")
    out = df.copy()
    out[f"{target_col}_lag1"] = out[target_col].shift(1)
    out[f"{target_col}_lag2"] = out[target_col].shift(2)
    for feature in ("pm10", "nitrogen_dioxide", "ozone"):
        if feature in out.columns:
            out[f"{feature}_lag1"] = out[feature].shift(1)
    return out.dropna()


def split_train_test(
    df: pd.DataFrame, test_days: int = 7
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reserve the last ``test_days`` rows as the holdout test set."""
    if len(df) <= test_days:
        raise ValueError("Not enough rows to reserve last 7 days for test.")
    train = df.iloc[:-test_days].copy()
    test = df.iloc[-test_days:].copy()
    return train, test


def fit_lstm_scaler(train_df: pd.DataFrame) -> MinMaxScaler:
    """Fit a MinMax scaler on the training frame."""
    scaler = MinMaxScaler()
    scaler.fit(train_df.values)
    return scaler


def transform_with_scaler(
    scaler: MinMaxScaler, train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply a fitted scaler to both train and test frames, preserving index/cols."""
    train_scaled = pd.DataFrame(
        scaler.transform(train_df.values), index=train_df.index, columns=train_df.columns
    )
    test_scaled = pd.DataFrame(
        scaler.transform(test_df.values), index=test_df.index, columns=test_df.columns
    )
    return train_scaled, test_scaled


def preprocess_pipeline(
    raw_df: pd.DataFrame, target_col: str = "pm2_5", test_days: int = 7
) -> dict[str, pd.DataFrame | MinMaxScaler]:
    """Run the offline batch pipeline: clean → lag → split → scale."""
    prepared = validate_and_prepare(raw_df)
    with_lags = create_lag_features(prepared, target_col=target_col)
    train_df, test_df = split_train_test(with_lags, test_days=test_days)
    scaler = fit_lstm_scaler(train_df)
    train_scaled, test_scaled = transform_with_scaler(scaler, train_df, test_df)
    return {
        "prepared": prepared,
        "with_lags": with_lags,
        "train": train_df,
        "test": test_df,
        "train_scaled": train_scaled,
        "test_scaled": test_scaled,
        "scaler": scaler,
    }


# ─── EDA helpers ─────────────────────────────────────────────────────────────


def statistical_summary(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Per-column mean / median / min / max / std summary."""
    stats = df[columns].agg(["mean", "median", "min", "max", "std"]).T
    return stats.reset_index().rename(columns={"index": "pollutant"})


def missing_values_report(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Per-column NaN count."""
    missing = df[columns].isna().sum().reset_index()
    missing.columns = ["column", "missing_count"]
    return missing


def detect_outliers_iqr(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Per-column IQR-based outlier counts and percentages."""
    rows: list[dict[str, float | str | int]] = []
    for col in columns:
        series = df[col].dropna()
        if series.empty:
            rows.append({"column": col, "outlier_count": 0, "outlier_pct": 0.0})
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (series < lo) | (series > hi)
        rows.append(
            {
                "column": col,
                "outlier_count": int(mask.sum()),
                "outlier_pct": float(mask.mean() * 100.0),
            }
        )
    return pd.DataFrame(rows)


def seasonal_aggregates(
    df: pd.DataFrame, columns: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-month and per-hour means; requires a DatetimeIndex."""
    work = df.copy()
    if not isinstance(work.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be DatetimeIndex for seasonal aggregates.")
    monthly = work[columns].groupby(work.index.month).mean(numeric_only=True).reset_index()
    monthly = monthly.rename(columns={monthly.columns[0]: "month"})
    hourly = work[columns].groupby(work.index.hour).mean(numeric_only=True).reset_index()
    hourly = hourly.rename(columns={hourly.columns[0]: "hour"})
    return monthly, hourly
