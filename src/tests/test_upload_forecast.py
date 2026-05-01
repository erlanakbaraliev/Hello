"""Tests for the user-upload forecasting pipeline.

Heavy model dependencies (TensorFlow / XGBoost) and the trained artifacts in
``experiments/models/full_models/`` are optional: tests that need them are
skipped when missing.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from back_end.constants import (
    LSTM_ARTIFACT_PATH,
    LSTM_KERAS_PATH,
    POLLUTANT_COLUMNS,
    SUPPORTED_HORIZONS,
    TIME_FEATURE_COLUMNS,
    WEATHER_COLUMNS,
    XGB_ARTIFACT_PATH,
    HorizonUnavailableError,
    UploadValidationError,
)
from back_end.inference import _run_recursive_single_step, validate_horizon
from back_end.lstm_pred import forecast_lstm_full, load_full_lstm
from back_end.upload_preprocess import (
    PreprocessedUpload,
    add_time_features,
    preprocess_uploaded_dataset,
)
from back_end.xgboost_pred import forecast_xgboost_full, load_full_xgboost

# ─── Synthetic upload helpers ────────────────────────────────────────────────


def _make_upload(
    *,
    n_past: int = 60,
    n_future: int = 24,
    start: str = "2024-01-01 00:00:00+00:00",
    drop_pollutant_in_future: bool = True,
) -> pd.DataFrame:
    """Build a synthetic, well-formed hourly upload."""
    n_total = n_past + n_future
    times = pd.date_range(start=start, periods=n_total, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    data: dict[str, np.ndarray | pd.DatetimeIndex] = {"time": times}
    for col in POLLUTANT_COLUMNS:
        vals = rng.normal(loc=20.0, scale=4.0, size=n_total).astype(float)
        if drop_pollutant_in_future:
            vals[n_past:] = np.nan
        data[col] = vals
    data["temperature"] = rng.normal(loc=10.0, scale=2.0, size=n_total).astype(float)
    data["wind_speed"] = rng.normal(loc=3.5, scale=0.5, size=n_total).astype(float)
    return pd.DataFrame(data)


# ─── Validation tests (no model required) ────────────────────────────────────


def test_validate_missing_time_column_raises() -> None:
    df = _make_upload().drop(columns=["time"])
    with pytest.raises(UploadValidationError, match="'time' column"):
        preprocess_uploaded_dataset(df)


def test_validate_missing_pollutant_column_raises_with_column_list() -> None:
    df = _make_upload().drop(columns=["pm10", "ozone"])
    with pytest.raises(UploadValidationError) as exc:
        preprocess_uploaded_dataset(df)
    msg = str(exc.value)
    assert "pm10" in msg and "ozone" in msg


def test_validate_non_hourly_gap_raises_with_gap_location() -> None:
    df = _make_upload(n_past=60, n_future=24)
    # Drop one row in the middle to create a 2h gap.
    df = df.drop(index=20).reset_index(drop=True)
    with pytest.raises(UploadValidationError, match="not strictly hourly"):
        preprocess_uploaded_dataset(df)


def test_validate_duplicate_timestamp_raises() -> None:
    df = _make_upload(n_past=60, n_future=24)
    # Duplicate row 5.
    dup = df.iloc[[5]].copy()
    df = pd.concat([df, dup], ignore_index=True)
    with pytest.raises(UploadValidationError, match="Duplicate timestamps"):
        preprocess_uploaded_dataset(df)


def test_validate_fewer_than_48_past_rows_raises() -> None:
    df = _make_upload(n_past=10, n_future=24)
    with pytest.raises(UploadValidationError, match="at least 48"):
        preprocess_uploaded_dataset(df)


def test_validate_unparseable_timestamps_raises() -> None:
    df = _make_upload()
    df["time"] = "not-a-date"
    with pytest.raises(UploadValidationError, match="parse any valid timestamps"):
        preprocess_uploaded_dataset(df)


def test_validate_no_future_rows_raises() -> None:
    """Auto-detector cannot pick the very last row as cutoff; passing it
    explicitly via cutoff_ts triggers the empty-future-block error."""
    df = _make_upload(n_past=60, n_future=0, drop_pollutant_in_future=False)
    last_ts = pd.Timestamp(df["time"].iloc[-1])
    with pytest.raises(UploadValidationError, match="No future rows"):
        preprocess_uploaded_dataset(df, cutoff_ts=last_ts)


def test_validate_auto_cutoff_at_only_last_row_raises() -> None:
    """When only the last row is fully observed, auto-detection must error."""
    df = _make_upload(n_past=60, n_future=24, drop_pollutant_in_future=True)
    # Wipe all past pollutants except the very last past row so that the only
    # fully-observed row is the very last past row at index 59. Then drop
    # all rows after that so it's also the last row of the dataset.
    for col in POLLUTANT_COLUMNS:
        df.loc[: 58, col] = np.nan
    df = df.iloc[: 60].reset_index(drop=True)
    with pytest.raises(UploadValidationError, match="leave no future rows"):
        preprocess_uploaded_dataset(df)


def test_validate_missing_future_weather_raises() -> None:
    df = _make_upload(n_past=60, n_future=24)
    df.loc[60:, "temperature"] = np.nan
    df.loc[60:, "wind_speed"] = np.nan
    with pytest.raises(UploadValidationError, match="interpolation is not applied"):
        preprocess_uploaded_dataset(df)


def test_future_weather_single_missing_cell_not_interpolated_raises() -> None:
    """One missing future temperature must error — we do not fill future weather."""
    df = _make_upload(n_past=60, n_future=24)
    df.loc[70, "temperature"] = np.nan
    with pytest.raises(UploadValidationError, match="interpolation is not applied"):
        preprocess_uploaded_dataset(df)


# ─── Preprocessing tests ─────────────────────────────────────────────────────


def test_preprocess_aligns_to_feat_cols_and_adds_time_features() -> None:
    df = _make_upload(n_past=60, n_future=24)
    pre = preprocess_uploaded_dataset(df)
    expected_cols = (
        POLLUTANT_COLUMNS + WEATHER_COLUMNS + TIME_FEATURE_COLUMNS
    )
    assert list(pre.past_df_aligned.columns) == expected_cols
    assert list(pre.future_weather_df.columns) == expected_cols
    assert len(pre.past_df_aligned) == 60
    assert len(pre.future_weather_df) == 24
    one_h = pd.Timedelta(hours=1)
    assert pre.future_weather_df.index[0] == pre.cutoff_ts + one_h
    # Calendar features must be exactly in [-1, 1].
    arr = pre.past_df_aligned[TIME_FEATURE_COLUMNS].to_numpy()
    assert np.all(np.abs(arr) <= 1.0 + 1e-9)


def test_preprocess_interpolates_short_gap_in_past() -> None:
    df = _make_upload(n_past=60, n_future=24)
    df.loc[10, "pm2_5"] = np.nan  # single missing past pollutant value
    pre = preprocess_uploaded_dataset(df)
    assert not pre.past_df_aligned["pm2_5"].isna().any()


def test_preprocess_surfaces_future_actuals_when_complete() -> None:
    df = _make_upload(n_past=60, n_future=24, drop_pollutant_in_future=False)
    cutoff = pd.Timestamp(df["time"].iloc[59])
    pre = preprocess_uploaded_dataset(df, cutoff_ts=cutoff)
    assert pre.future_actuals_df is not None
    assert list(pre.future_actuals_df.columns) == POLLUTANT_COLUMNS
    assert len(pre.future_actuals_df) == 24


def test_preprocess_no_actuals_when_pollutants_missing_in_future() -> None:
    df = _make_upload(n_past=60, n_future=24, drop_pollutant_in_future=True)
    pre = preprocess_uploaded_dataset(df)
    assert pre.future_actuals_df is None


def test_preprocess_available_horizons_reports_only_supported() -> None:
    df = _make_upload(n_past=60, n_future=80)
    pre = preprocess_uploaded_dataset(df)
    assert pre.available_horizons == (24, 72)


def test_add_time_features_columns_and_range() -> None:
    idx = pd.date_range("2024-06-01", periods=24, freq="h", tz="UTC")
    df = pd.DataFrame({"x": np.arange(24)}, index=idx)
    out = add_time_features(df)
    for c in TIME_FEATURE_COLUMNS:
        assert c in out.columns
        assert np.all(np.abs(out[c].to_numpy()) <= 1.0 + 1e-9)


# ─── Horizon validation ──────────────────────────────────────────────────────


@pytest.mark.parametrize("h", SUPPORTED_HORIZONS)
def test_horizon_longer_than_future_weather_raises(h: int) -> None:
    df = _make_upload(n_past=60, n_future=h - 1)
    pre = preprocess_uploaded_dataset(df)
    with pytest.raises(HorizonUnavailableError, match="future hourly rows"):
        validate_horizon(pre, h)


def test_validate_horizon_unsupported_value_raises() -> None:
    df = _make_upload(n_past=60, n_future=24)
    pre = preprocess_uploaded_dataset(df)
    with pytest.raises(HorizonUnavailableError, match="Unsupported forecast horizon"):
        validate_horizon(pre, 48)


# ─── Recursive inference with a stub model (no heavy deps) ──────────────────


def _make_pre_for_horizon(n_future: int) -> PreprocessedUpload:
    df = _make_upload(n_past=60, n_future=n_future, drop_pollutant_in_future=True)
    return preprocess_uploaded_dataset(df)


def _stub_predict_zero(history: np.ndarray) -> np.ndarray:
    return np.zeros(len(POLLUTANT_COLUMNS), dtype=np.float32)


@pytest.mark.parametrize("horizon", SUPPORTED_HORIZONS)
def test_recursive_loop_runs_for_each_horizon_with_stub(horizon: int) -> None:
    pre = _make_pre_for_horizon(horizon)
    feat_cols = (
        POLLUTANT_COLUMNS + WEATHER_COLUMNS + TIME_FEATURE_COLUMNS
    )
    n_features = len(feat_cols)
    mu = np.zeros(n_features)
    sigma = np.ones(n_features)
    out = _run_recursive_single_step(
        preprocessed=pre,
        horizon=horizon,
        feat_cols=feat_cols,
        target_cols=list(POLLUTANT_COLUMNS),
        tgt_idxs=list(range(len(POLLUTANT_COLUMNS))),
        mu=mu,
        sigma=sigma,
        predict_one_step=_stub_predict_zero,
    )
    assert len(out) == horizon
    pred_cols = [f"{c}_pred" for c in POLLUTANT_COLUMNS]
    for c in pred_cols + ["temperature_used", "wind_speed_used"]:
        assert c in out.columns


def test_recursive_uses_uploaded_weather_via_history() -> None:
    """When future temperature is perturbed, the scaled history fed to the model
    in subsequent steps differs accordingly."""
    pre = _make_pre_for_horizon(24)

    seen_histories: list[np.ndarray] = []

    def capture_predict(history: np.ndarray) -> np.ndarray:
        seen_histories.append(history.copy())
        return np.zeros(len(POLLUTANT_COLUMNS), dtype=np.float32)

    feat_cols = (
        POLLUTANT_COLUMNS + WEATHER_COLUMNS + TIME_FEATURE_COLUMNS
    )
    n_features = len(feat_cols)
    mu = np.zeros(n_features)
    sigma = np.ones(n_features)
    _run_recursive_single_step(
        preprocessed=pre,
        horizon=24,
        feat_cols=feat_cols,
        target_cols=list(POLLUTANT_COLUMNS),
        tgt_idxs=list(range(len(POLLUTANT_COLUMNS))),
        mu=mu,
        sigma=sigma,
        predict_one_step=capture_predict,
    )

    pre2 = deepcopy(pre)
    pre2.future_weather_df.iloc[:, pre2.future_weather_df.columns.get_loc("temperature")] += 50.0
    seen_histories2: list[np.ndarray] = []

    def capture_predict2(history: np.ndarray) -> np.ndarray:
        seen_histories2.append(history.copy())
        return np.zeros(len(POLLUTANT_COLUMNS), dtype=np.float32)

    _run_recursive_single_step(
        preprocessed=pre2,
        horizon=24,
        feat_cols=feat_cols,
        target_cols=list(POLLUTANT_COLUMNS),
        tgt_idxs=list(range(len(POLLUTANT_COLUMNS))),
        mu=mu,
        sigma=sigma,
        predict_one_step=capture_predict2,
    )

    # First history (step 0) is identical (depends only on past). Step 1 onward
    # must differ because future temperature is now baked into the appended row.
    np.testing.assert_allclose(seen_histories[0], seen_histories2[0])
    assert not np.allclose(seen_histories[1], seen_histories2[1])


def test_no_future_pollutants_used_as_inputs() -> None:
    """Corrupting future pollutant values must not change the histories the
    model sees, because those values are never read into ``next_row_scaled``."""
    base_df = _make_upload(n_past=60, n_future=24, drop_pollutant_in_future=False)
    cutoff = pd.Timestamp(base_df["time"].iloc[59])
    pre_clean = preprocess_uploaded_dataset(base_df, cutoff_ts=cutoff)

    corrupted_df = base_df.copy()
    for col in POLLUTANT_COLUMNS:
        corrupted_df.loc[60:, col] = 1e6  # extreme garbage
    pre_corrupted = preprocess_uploaded_dataset(corrupted_df, cutoff_ts=cutoff)

    feat_cols = (
        POLLUTANT_COLUMNS + WEATHER_COLUMNS + TIME_FEATURE_COLUMNS
    )
    n_features = len(feat_cols)
    mu = np.zeros(n_features)
    sigma = np.ones(n_features)

    histories_clean: list[np.ndarray] = []
    histories_corrupt: list[np.ndarray] = []

    def _capture(target: list[np.ndarray]):
        def _predict(history: np.ndarray) -> np.ndarray:
            target.append(history.copy())
            return np.zeros(len(POLLUTANT_COLUMNS), dtype=np.float32)

        return _predict

    _run_recursive_single_step(
        preprocessed=pre_clean,
        horizon=24,
        feat_cols=feat_cols,
        target_cols=list(POLLUTANT_COLUMNS),
        tgt_idxs=list(range(len(POLLUTANT_COLUMNS))),
        mu=mu,
        sigma=sigma,
        predict_one_step=_capture(histories_clean),
    )
    _run_recursive_single_step(
        preprocessed=pre_corrupted,
        horizon=24,
        feat_cols=feat_cols,
        target_cols=list(POLLUTANT_COLUMNS),
        tgt_idxs=list(range(len(POLLUTANT_COLUMNS))),
        mu=mu,
        sigma=sigma,
        predict_one_step=_capture(histories_corrupt),
    )

    # All histories must match exactly across both runs — proves that future
    # pollutant values are NOT used as inputs.
    assert len(histories_clean) == len(histories_corrupt) == 24
    for a, b in zip(histories_clean, histories_corrupt):
        np.testing.assert_allclose(a, b)


# ─── Artifact-gated end-to-end tests ─────────────────────────────────────────


def _artifacts_present(*paths: Path) -> bool:
    return all(p.is_file() for p in paths)


@pytest.mark.skipif(
    not _artifacts_present(LSTM_KERAS_PATH, LSTM_ARTIFACT_PATH),
    reason="LSTM full-model artifacts not present in experiments/models/full_models/",
)
@pytest.mark.parametrize("horizon", SUPPORTED_HORIZONS)
def test_lstm_full_end_to_end(horizon: int) -> None:
    pytest.importorskip("tensorflow")
    n_past = 72
    df = _make_upload(n_past=n_past, n_future=horizon, drop_pollutant_in_future=False)
    cutoff = pd.Timestamp(df["time"].iloc[n_past - 1])
    pre = preprocess_uploaded_dataset(df, cutoff_ts=cutoff)
    keras_model, artifacts = load_full_lstm()
    out = forecast_lstm_full(pre, horizon, keras_model=keras_model, artifacts=artifacts)
    pred_cols = [f"{c}_pred" for c in POLLUTANT_COLUMNS]
    actual_cols = [f"{c}_actual" for c in POLLUTANT_COLUMNS]
    expected = [
        "time", *pred_cols, "temperature_used", "wind_speed_used",
        "model_name", "forecast_horizon", *actual_cols,
    ]
    assert list(out.columns) == expected
    assert len(out) == horizon
    assert (out["model_name"] == "LSTM").all()
    assert (out["forecast_horizon"] == horizon).all()
    assert np.isfinite(out[pred_cols].to_numpy()).all()


@pytest.mark.skipif(
    not _artifacts_present(XGB_ARTIFACT_PATH),
    reason="XGBoost full-model artifacts not present in experiments/models/full_models/",
)
@pytest.mark.parametrize("horizon", SUPPORTED_HORIZONS)
def test_xgboost_full_end_to_end(horizon: int) -> None:
    pytest.importorskip("xgboost")
    n_past = 72
    df = _make_upload(n_past=n_past, n_future=horizon, drop_pollutant_in_future=False)
    cutoff = pd.Timestamp(df["time"].iloc[n_past - 1])
    pre = preprocess_uploaded_dataset(df, cutoff_ts=cutoff)
    artifacts = load_full_xgboost()
    out = forecast_xgboost_full(pre, horizon, artifacts=artifacts)
    pred_cols = [f"{c}_pred" for c in POLLUTANT_COLUMNS]
    actual_cols = [f"{c}_actual" for c in POLLUTANT_COLUMNS]
    expected = [
        "time", *pred_cols, "temperature_used", "wind_speed_used",
        "model_name", "forecast_horizon", *actual_cols,
    ]
    assert list(out.columns) == expected
    assert len(out) == horizon
    assert (out["model_name"] == "XGBoost").all()
    assert (out["forecast_horizon"] == horizon).all()
    assert np.isfinite(out[pred_cols].to_numpy()).all()


@pytest.mark.skipif(
    not _artifacts_present(XGB_ARTIFACT_PATH),
    reason="XGBoost full-model artifacts not present in experiments/models/full_models/",
)
def test_xgboost_full_no_future_pollutants_used_end_to_end() -> None:
    """End-to-end regression: extreme garbage in future pollutants must not
    change the predictions, because they are never fed into the model."""
    pytest.importorskip("xgboost")
    n_past = 72
    base = _make_upload(n_past=n_past, n_future=24, drop_pollutant_in_future=False)
    cutoff = pd.Timestamp(base["time"].iloc[n_past - 1])
    pre_clean = preprocess_uploaded_dataset(base, cutoff_ts=cutoff)

    corrupted = base.copy()
    for col in POLLUTANT_COLUMNS:
        corrupted.loc[72:, col] = 1e6
    pre_corrupt = preprocess_uploaded_dataset(corrupted, cutoff_ts=cutoff)

    artifacts = load_full_xgboost()
    out_clean = forecast_xgboost_full(pre_clean, 24, artifacts=artifacts)
    out_corrupt = forecast_xgboost_full(pre_corrupt, 24, artifacts=artifacts)

    pred_cols = [f"{c}_pred" for c in POLLUTANT_COLUMNS]
    np.testing.assert_allclose(
        out_clean[pred_cols].to_numpy(),
        out_corrupt[pred_cols].to_numpy(),
        atol=1e-9,
    )
