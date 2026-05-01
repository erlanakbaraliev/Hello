"""Tests for ``src/back_end/data_processing.py``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import MinMaxScaler

from back_end import data_processing as dp
from back_end.constants import EXPECTED_COLUMNS, POLLUTANT_COLUMNS
from back_end.data_processing import (
    combine_sources,
    create_lag_features,
    detect_outliers_iqr,
    drop_geo_columns,
    ensure_expected_schema,
    fit_lstm_scaler,
    load_dataset_from_data_folder,
    missing_values_report,
    preprocess_pipeline,
    seasonal_aggregates,
    split_train_test,
    statistical_summary,
    transform_with_scaler,
    validate_and_prepare,
)


def _make_hourly_frame(n: int = 96, start: str = "2026-01-01 00:00:00") -> pd.DataFrame:
    times = pd.date_range(start=start, periods=n, freq="h", tz="UTC")
    rng = np.random.default_rng(42)
    df = pd.DataFrame({"time": times})
    for col in POLLUTANT_COLUMNS:
        df[col] = rng.normal(loc=20.0, scale=4.0, size=n)
    df["city"] = "london"
    return df


# ─── drop_geo_columns ────────────────────────────────────────────────────────


def test_drop_geo_columns_removes_known_aliases() -> None:
    df = pd.DataFrame(
        {"a": [1], "Latitude": [51.0], "longitude": [0.1], "lat": [1], "Lng": [2], "lon": [3]}
    )
    out = drop_geo_columns(df)
    assert list(out.columns) == ["a"]


def test_drop_geo_columns_no_op_when_absent() -> None:
    df = pd.DataFrame({"x": [1, 2]})
    out = drop_geo_columns(df)
    assert list(out.columns) == ["x"]


# ─── ensure_expected_schema ──────────────────────────────────────────────────


def test_ensure_expected_schema_adds_all_required_columns() -> None:
    src = pd.DataFrame({"time": ["2026-01-01"], "pm2_5": [12.5]})
    out = ensure_expected_schema(src, source="uploaded")
    for col in EXPECTED_COLUMNS:
        assert col in out.columns
    assert out.loc[0, "city"] == "unknown"
    assert out.loc[0, "source"] == "uploaded"


def test_ensure_expected_schema_normalizes_column_case() -> None:
    src = pd.DataFrame({"Time": ["2026-01-01"], "PM2_5": [10], "City": ["x"]})
    out = ensure_expected_schema(src)
    assert "time" in out.columns
    assert "pm2_5" in out.columns
    assert out.loc[0, "city"] == "x"


def test_ensure_expected_schema_default_source_is_uploaded() -> None:
    src = pd.DataFrame({"time": ["2026-01-01"]})
    out = ensure_expected_schema(src)
    assert out["source"].iloc[0] == "uploaded"


# ─── validate_and_prepare ────────────────────────────────────────────────────


def test_validate_and_prepare_returns_daily_index() -> None:
    df = _make_hourly_frame(n=72)
    out = validate_and_prepare(df)
    assert isinstance(out.index, pd.DatetimeIndex)
    # 72 hourly rows starting at midnight => 3 days of daily resample (D[0], D[1], D[2])
    assert len(out) == 3
    for col in POLLUTANT_COLUMNS:
        assert col in out.columns
        assert not out[col].isna().any()


def test_validate_and_prepare_drops_unparseable_times() -> None:
    df = _make_hourly_frame(n=24)
    # Cast to object before injecting an invalid timestamp literal to avoid
    # pandas dtype-assignment FutureWarning in test setup.
    df["time"] = df["time"].astype("object")
    df.loc[0, "time"] = "not-a-date"
    out = validate_and_prepare(df)
    # Bad row dropped silently.
    assert not out.empty


# ─── combine_sources ─────────────────────────────────────────────────────────


def test_combine_sources_merges_uploaded_and_live() -> None:
    a = pd.DataFrame(
        {
            "time": ["2026-01-01 00:00:00", "2026-01-01 01:00:00"],
            "pm2_5": [10.0, 11.0],
        }
    )
    b = pd.DataFrame(
        {
            "time": ["2026-01-01 01:00:00", "2026-01-01 02:00:00"],
            "pm2_5": [99.0, 12.0],
        }
    )
    out = combine_sources(a, b)
    assert len(out) == 3
    # Duplicate time keeps the last (i.e. the live API row).
    one_am = out[out["time"] == pd.Timestamp("2026-01-01 01:00:00", tz="UTC")]
    assert one_am["pm2_5"].iloc[0] == 99.0
    assert one_am["source"].iloc[0] == "live_api"


def test_combine_sources_only_uploaded() -> None:
    a = pd.DataFrame({"time": ["2026-01-01 00:00:00"], "pm2_5": [10.0]})
    out = combine_sources(a, None)
    assert len(out) == 1
    assert out["source"].iloc[0] == "uploaded"


def test_combine_sources_only_live() -> None:
    b = pd.DataFrame({"time": ["2026-01-01 00:00:00"], "pm2_5": [10.0]})
    out = combine_sources(None, b)
    assert len(out) == 1
    assert out["source"].iloc[0] == "live_api"


def test_combine_sources_treats_empty_as_missing() -> None:
    empty = pd.DataFrame()
    with pytest.raises(ValueError, match="No source data"):
        combine_sources(empty, empty)


def test_combine_sources_raises_when_nothing_provided() -> None:
    with pytest.raises(ValueError, match="No source data"):
        combine_sources(None, None)


# ─── load_dataset_from_data_folder ───────────────────────────────────────────


def test_load_dataset_from_data_folder_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        load_dataset_from_data_folder("definitely-not-here.csv")


def test_load_dataset_from_data_folder_reads_existing_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Redirect data folder to a temporary location.
    fake_pkg_root = tmp_path / "back_end"
    fake_pkg_root.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "tiny.csv").write_text("time,pm2_5\n2026-01-01,12.0\n")

    fake_module_file = fake_pkg_root / "data_processing.py"
    fake_module_file.write_text("# placeholder\n")

    monkeypatch.setattr(dp, "__file__", str(fake_module_file))
    out = load_dataset_from_data_folder("tiny.csv")
    assert list(out.columns) == ["time", "pm2_5"]
    assert out.iloc[0]["pm2_5"] == 12.0


# ─── create_lag_features / split_train_test ──────────────────────────────────


def test_create_lag_features_adds_lag_columns_and_drops_nans() -> None:
    df = pd.DataFrame(
        {
            "pm2_5": list(range(10)),
            "pm10": list(range(10)),
            "nitrogen_dioxide": list(range(10)),
            "ozone": list(range(10)),
        }
    )
    out = create_lag_features(df)
    assert "pm2_5_lag1" in out.columns
    assert "pm2_5_lag2" in out.columns
    assert "pm10_lag1" in out.columns
    assert "nitrogen_dioxide_lag1" in out.columns
    assert "ozone_lag1" in out.columns
    # Two leading rows lost to lag2 NaNs.
    assert len(out) == 8


def test_create_lag_features_missing_target_raises() -> None:
    df = pd.DataFrame({"pm10": [1, 2, 3]})
    with pytest.raises(ValueError, match="Target column not found"):
        create_lag_features(df, target_col="pm2_5")


def test_split_train_test_default_window() -> None:
    df = pd.DataFrame({"x": list(range(20))})
    train, test = split_train_test(df, test_days=7)
    assert len(train) == 13
    assert len(test) == 7


def test_split_train_test_too_few_rows_raises() -> None:
    df = pd.DataFrame({"x": list(range(5))})
    with pytest.raises(ValueError, match="Not enough rows"):
        split_train_test(df, test_days=7)


# ─── Scaler helpers ──────────────────────────────────────────────────────────


def test_fit_and_transform_scaler_round_trip() -> None:
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [10.0, 20.0, 30.0, 40.0]})
    scaler = fit_lstm_scaler(df.iloc[:3])
    assert isinstance(scaler, MinMaxScaler)
    train_scaled, test_scaled = transform_with_scaler(scaler, df.iloc[:3], df.iloc[3:])
    assert train_scaled.shape == (3, 2)
    assert test_scaled.shape == (1, 2)
    # min/max of train must lie in [0, 1].
    assert train_scaled.values.min() == pytest.approx(0.0)
    assert train_scaled.values.max() == pytest.approx(1.0)


# ─── preprocess_pipeline (integration) ───────────────────────────────────────


def test_preprocess_pipeline_returns_full_bundle() -> None:
    df = _make_hourly_frame(n=24 * 30)  # 30 days
    bundle = preprocess_pipeline(df, test_days=5)
    assert set(bundle.keys()) == {
        "prepared",
        "with_lags",
        "train",
        "test",
        "train_scaled",
        "test_scaled",
        "scaler",
    }
    train_df = bundle["train"]
    test_df = bundle["test"]
    train_scaled = bundle["train_scaled"]
    test_scaled = bundle["test_scaled"]
    assert isinstance(train_df, pd.DataFrame)
    assert len(test_df) == 5
    assert train_scaled.shape == train_df.shape
    assert test_scaled.shape == test_df.shape


# ─── statistical_summary / missing_values_report / outliers / seasonal ───────


def test_statistical_summary_columns() -> None:
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]})
    out = statistical_summary(df, ["a", "b"])
    assert set(out.columns) == {"pollutant", "mean", "median", "min", "max", "std"}
    assert set(out["pollutant"]) == {"a", "b"}


def test_missing_values_report_counts_nan_per_column() -> None:
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [np.nan, np.nan, 0.0]})
    out = missing_values_report(df, ["a", "b"])
    counts = dict(zip(out["column"], out["missing_count"]))
    assert counts == {"a": 1, "b": 2}


def test_detect_outliers_iqr_finds_extreme_values() -> None:
    df = pd.DataFrame({"a": [1, 1, 1, 1, 1, 100], "b": [0, 0, 0, 0, 0, 0]})
    out = detect_outliers_iqr(df, ["a", "b"])
    rows = {row["column"]: row for _, row in out.iterrows()}
    assert rows["a"]["outlier_count"] >= 1
    assert rows["b"]["outlier_count"] == 0


def test_detect_outliers_iqr_handles_empty_series() -> None:
    df = pd.DataFrame({"a": [np.nan, np.nan], "b": [1.0, 2.0]})
    out = detect_outliers_iqr(df, ["a", "b"])
    rows = {row["column"]: row for _, row in out.iterrows()}
    assert rows["a"]["outlier_count"] == 0
    assert rows["a"]["outlier_pct"] == 0.0


def test_seasonal_aggregates_monthly_and_hourly() -> None:
    idx = pd.date_range("2026-01-01", periods=48, freq="h", tz="UTC")
    df = pd.DataFrame({"a": np.arange(48, dtype=float), "b": np.arange(48, dtype=float)}, index=idx)
    monthly, hourly = seasonal_aggregates(df, ["a", "b"])
    assert "month" in monthly.columns
    assert "hour" in hourly.columns
    assert len(hourly) == 24


def test_seasonal_aggregates_requires_datetime_index() -> None:
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValueError, match="DatetimeIndex"):
        seasonal_aggregates(df, ["a"])
