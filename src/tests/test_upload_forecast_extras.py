"""Additional coverage for ``src/back_end/upload_forecast.py``.

These tests target the helper functions, validation branches, output assembly,
and full-model entry points. Heavy ML dependencies are stubbed so no real
model artifacts or TensorFlow runtime is required.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest

from back_end import lstm_pred as lstm_mod
from back_end import upload_preprocess as up
from back_end import xgboost_pred as xgb_mod
from back_end.constants import (
    POLLUTANT_COLUMNS,
    TIME_FEATURE_COLUMNS,
    WEATHER_COLUMNS,
    WINDOW,
    ArtifactMissingError,
    HorizonUnavailableError,
    UploadValidationError,
)
from back_end.inference import (
    _run_recursive_single_step,
    _safe_sigma,
    _scale_row,
    build_forecast_output,
    prediction_payload,
)
from back_end.lstm_pred import forecast_lstm_full, load_full_lstm
from back_end.upload_preprocess import (
    PreprocessedUpload,
    _interpolate_block,
    _load_feat_cols_for_validation,
    add_time_features,
    preprocess_uploaded_dataset,
)
from back_end.xgboost_pred import forecast_xgboost_full, load_full_xgboost


FEAT_COLS = POLLUTANT_COLUMNS + WEATHER_COLUMNS + TIME_FEATURE_COLUMNS
N_FEATURES = len(FEAT_COLS)
N_TARGETS = len(POLLUTANT_COLUMNS)


# ─── Synthetic helpers ───────────────────────────────────────────────────────


def _make_upload_df(
    *,
    n_past: int = 60,
    n_future: int = 24,
    drop_pollutant_in_future: bool = True,
    start: str = "2024-01-01 00:00:00+00:00",
) -> pd.DataFrame:
    n = n_past + n_future
    times = pd.date_range(start=start, periods=n, freq="h", tz="UTC")
    rng = np.random.default_rng(7)
    data: dict[str, Any] = {"time": times}
    for col in POLLUTANT_COLUMNS:
        vals = rng.normal(loc=20.0, scale=4.0, size=n).astype(float)
        if drop_pollutant_in_future:
            vals[n_past:] = np.nan
        data[col] = vals
    data["temperature"] = rng.normal(loc=10.0, scale=2.0, size=n).astype(float)
    data["wind_speed"] = rng.normal(loc=3.5, scale=0.5, size=n).astype(float)
    return pd.DataFrame(data)


def _make_pre(n_future: int = 24) -> PreprocessedUpload:
    df = _make_upload_df(n_past=60, n_future=n_future, drop_pollutant_in_future=True)
    return preprocess_uploaded_dataset(df)


# ─── add_time_features error path ────────────────────────────────────────────


def test_add_time_features_requires_datetime_index() -> None:
    df = pd.DataFrame({"x": [1, 2, 3]})  # default RangeIndex
    with pytest.raises(TypeError, match="DatetimeIndex"):
        add_time_features(df)


# ─── _interpolate_block branches ─────────────────────────────────────────────


def test_interpolate_block_empty_cols_is_no_op() -> None:
    idx = pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC")
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]}, index=idx)
    out = _interpolate_block(df, [], where="past block")
    pd.testing.assert_frame_equal(out, df)


def test_interpolate_block_raises_when_residual_nans_remain() -> None:
    """A column that is entirely NaN cannot be interpolated → raises."""
    idx = pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC")
    df = pd.DataFrame({"a": [np.nan, np.nan, np.nan]}, index=idx)
    with pytest.raises(UploadValidationError, match="missing values"):
        _interpolate_block(df, ["a"], where="past block")


# ─── _load_feat_cols_for_validation fallbacks ────────────────────────────────


def test_load_feat_cols_returns_static_fallback_when_no_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(up, "LSTM_ARTIFACT_PATH", tmp_path / "missing_lstm.joblib")
    monkeypatch.setattr(up, "XGB_ARTIFACT_PATH", tmp_path / "missing_xgb.joblib")
    cols = _load_feat_cols_for_validation()
    assert cols == FEAT_COLS


def test_load_feat_cols_uses_lstm_artifact_when_available(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    artifact = {"feat_cols": ["custom_a", "custom_b"]}
    artifact_path = tmp_path / "lstm.joblib"
    joblib.dump(artifact, artifact_path)
    monkeypatch.setattr(up, "LSTM_ARTIFACT_PATH", artifact_path)
    monkeypatch.setattr(up, "XGB_ARTIFACT_PATH", tmp_path / "missing_xgb.joblib")
    assert _load_feat_cols_for_validation() == ["custom_a", "custom_b"]


def test_load_feat_cols_falls_back_to_xgb_when_lstm_corrupt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bad = tmp_path / "bad_lstm.joblib"
    bad.write_bytes(b"not-a-joblib-file")
    xgb_path = tmp_path / "xgb.joblib"
    joblib.dump({"feat_cols": ["xgb_only"]}, xgb_path)
    monkeypatch.setattr(up, "LSTM_ARTIFACT_PATH", bad)
    monkeypatch.setattr(up, "XGB_ARTIFACT_PATH", xgb_path)
    assert _load_feat_cols_for_validation() == ["xgb_only"]


def test_load_feat_cols_falls_back_to_static_when_both_corrupt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bad1 = tmp_path / "bad1.joblib"
    bad1.write_bytes(b"junk")
    bad2 = tmp_path / "bad2.joblib"
    bad2.write_bytes(b"junk")
    monkeypatch.setattr(up, "LSTM_ARTIFACT_PATH", bad1)
    monkeypatch.setattr(up, "XGB_ARTIFACT_PATH", bad2)
    assert _load_feat_cols_for_validation() == FEAT_COLS


# ─── Explicit cutoff_ts paths ────────────────────────────────────────────────


def test_preprocess_explicit_cutoff_with_naive_timestamp_localized() -> None:
    df = _make_upload_df(n_past=60, n_future=24, drop_pollutant_in_future=False)
    naive = pd.Timestamp(df["time"].iloc[59]).tz_convert(None)
    pre = preprocess_uploaded_dataset(df, cutoff_ts=naive)
    assert pre.cutoff_ts == pd.Timestamp(df["time"].iloc[59])


def test_preprocess_explicit_cutoff_not_in_index_raises() -> None:
    df = _make_upload_df(n_past=60, n_future=24, drop_pollutant_in_future=False)
    bogus = pd.Timestamp("1999-01-01 00:00:00+00:00")
    with pytest.raises(UploadValidationError, match="not present"):
        preprocess_uploaded_dataset(df, cutoff_ts=bogus)


# ─── Model loaders: missing-artifact errors ──────────────────────────────────


def test_load_full_lstm_missing_keras_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(lstm_mod, "LSTM_KERAS_PATH", tmp_path / "missing.keras")
    monkeypatch.setattr(lstm_mod, "LSTM_ARTIFACT_PATH", tmp_path / "missing.joblib")
    with pytest.raises(ArtifactMissingError, match="missing.keras"):
        load_full_lstm()


def test_load_full_lstm_missing_artifacts_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    keras_path = tmp_path / "model.keras"
    keras_path.write_bytes(b"placeholder")  # exists but not a real model
    monkeypatch.setattr(lstm_mod, "LSTM_KERAS_PATH", keras_path)
    monkeypatch.setattr(lstm_mod, "LSTM_ARTIFACT_PATH", tmp_path / "missing.joblib")
    with pytest.raises(ArtifactMissingError, match="missing.joblib"):
        load_full_lstm()


def test_load_full_xgboost_missing_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(xgb_mod, "XGB_ARTIFACT_PATH", tmp_path / "absent.joblib")
    with pytest.raises(ArtifactMissingError, match="absent.joblib"):
        load_full_xgboost()


def test_load_full_xgboost_returns_artifact_dict_when_present(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    artifact_path = tmp_path / "xgb.joblib"
    payload = {"feat_cols": ["a"], "target_cols": ["t"], "boosters": []}
    joblib.dump(payload, artifact_path)
    monkeypatch.setattr(xgb_mod, "XGB_ARTIFACT_PATH", artifact_path)
    out = load_full_xgboost()
    assert out == payload


# ─── Tiny scaling helpers ────────────────────────────────────────────────────


def test_safe_sigma_replaces_tiny_values_with_one() -> None:
    sigma = np.array([0.0, 1e-12, 1.0, 5.0])
    out = _safe_sigma(sigma)
    assert out[0] == 1.0
    assert out[1] == 1.0
    assert out[2] == 1.0
    assert out[3] == 5.0
    assert out.dtype == np.float64


def test_scale_row_subtracts_mean_and_divides() -> None:
    values = np.array([10.0, 20.0])
    mu = np.array([0.0, 10.0])
    sigma = np.array([2.0, 5.0])
    out = _scale_row(values, mu, sigma)
    np.testing.assert_allclose(out, np.array([5.0, 2.0], dtype=np.float32))
    assert out.dtype == np.float32


# ─── _run_recursive_single_step error branches ───────────────────────────────


def test_recursive_predict_wrong_shape_raises() -> None:
    pre = _make_pre(24)
    feat_cols = list(FEAT_COLS)
    mu = np.zeros(N_FEATURES)
    sigma = np.ones(N_FEATURES)

    def bad_predict(_history: np.ndarray) -> np.ndarray:
        return np.zeros(N_TARGETS - 1, dtype=np.float32)  # wrong length

    with pytest.raises(ValueError, match="Model returned shape"):
        _run_recursive_single_step(
            preprocessed=pre,
            horizon=24,
            feat_cols=feat_cols,
            target_cols=list(POLLUTANT_COLUMNS),
            tgt_idxs=list(range(N_TARGETS)),
            mu=mu,
            sigma=sigma,
            predict_one_step=bad_predict,
        )


def test_recursive_non_finite_prediction_raises() -> None:
    pre = _make_pre(24)
    feat_cols = list(FEAT_COLS)
    mu = np.zeros(N_FEATURES)
    sigma = np.ones(N_FEATURES)

    def nan_predict(_history: np.ndarray) -> np.ndarray:
        out = np.zeros(N_TARGETS, dtype=np.float32)
        out[0] = np.nan
        return out

    with pytest.raises(ValueError, match="non-finite"):
        _run_recursive_single_step(
            preprocessed=pre,
            horizon=24,
            feat_cols=feat_cols,
            target_cols=list(POLLUTANT_COLUMNS),
            tgt_idxs=list(range(N_TARGETS)),
            mu=mu,
            sigma=sigma,
            predict_one_step=nan_predict,
        )


# ─── build_forecast_output ───────────────────────────────────────────────────


def _build_raw_preds(n: int = 3) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
    rec = {f"{c}_pred": np.linspace(10.0, 20.0, n) for c in POLLUTANT_COLUMNS}
    rec["temperature_used"] = np.linspace(8.0, 12.0, n)
    rec["wind_speed_used"] = np.linspace(2.0, 5.0, n)
    return pd.DataFrame(rec, index=idx).rename_axis("time")


def test_build_forecast_output_no_actuals() -> None:
    raw = _build_raw_preds(3)
    out = build_forecast_output(
        raw, None, target_cols=list(POLLUTANT_COLUMNS), model_name="LSTM", horizon=3
    )
    pred_cols = [f"{c}_pred" for c in POLLUTANT_COLUMNS]
    expected = ["time", *pred_cols, "temperature_used", "wind_speed_used", "model_name", "forecast_horizon"]
    assert list(out.columns) == expected
    assert (out["model_name"] == "LSTM").all()
    assert (out["forecast_horizon"] == 3).all()


def test_build_forecast_output_with_actuals() -> None:
    raw = _build_raw_preds(3)
    actuals = pd.DataFrame(
        {col: np.linspace(11.0, 19.0, 3) for col in POLLUTANT_COLUMNS},
        index=pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC"),
    )
    out = build_forecast_output(
        raw, actuals, target_cols=list(POLLUTANT_COLUMNS), model_name="XGBoost", horizon=3
    )
    for col in POLLUTANT_COLUMNS:
        assert f"{col}_actual" in out.columns
    assert (out["model_name"] == "XGBoost").all()


# ─── prediction_payload ─────────────────────────────────────────────────────


def test_prediction_payload_serialises_full_round_trip() -> None:
    pre = _make_pre(24)
    raw = _build_raw_preds(24)
    forecast = build_forecast_output(
        raw, None, target_cols=list(POLLUTANT_COLUMNS), model_name="LSTM", horizon=24
    )
    payload = prediction_payload(pre, forecast, model_name="LSTM", horizon=24)
    assert payload["model"] == "LSTM"
    assert payload["forecast_horizon"] == 24
    assert payload["source"] == "upload_full_models"
    assert "history" in payload and len(payload["history"]) == WINDOW
    assert "forecast" in payload and len(payload["forecast"]) == 24
    # Must round-trip through json without errors.
    json.dumps(payload)


def test_prediction_payload_includes_safety_level_and_actuals_when_present() -> None:
    pre = _make_pre(24)
    raw = _build_raw_preds(24)
    forecast = build_forecast_output(
        raw,
        pd.DataFrame(
            {col: np.linspace(11.0, 19.0, 24) for col in POLLUTANT_COLUMNS},
            index=pd.date_range("2026-01-01", periods=24, freq="h", tz="UTC"),
        ),
        target_cols=list(POLLUTANT_COLUMNS),
        model_name="LSTM",
        horizon=24,
    )
    forecast["pm25_safety_level"] = ["Low"] * 24
    payload = prediction_payload(pre, forecast, model_name="LSTM", horizon=24)
    first = payload["forecast"][0]
    assert first["pm25_safety_level"] == "Low"
    for col in POLLUTANT_COLUMNS:
        assert f"{col}_actual" in first


# ─── forecast_lstm_full / forecast_xgboost_full with stubs ──────────────────


class _FakeKerasModel:
    """Minimal stand-in for a Keras LSTM model."""

    def predict(self, x: np.ndarray, *, verbose: int = 0) -> np.ndarray:  # noqa: ARG002
        # Return zeros of shape (batch, n_targets)
        return np.zeros((x.shape[0], N_TARGETS), dtype=np.float32)


class _FakeScaler:
    def __init__(self) -> None:
        self.mean_ = np.zeros(N_FEATURES, dtype=np.float64)
        self.scale_ = np.ones(N_FEATURES, dtype=np.float64)


def _fake_lstm_artifacts() -> dict[str, Any]:
    return {
        "feat_cols": list(FEAT_COLS),
        "target_cols": list(POLLUTANT_COLUMNS),
        "tgt_idxs": list(range(N_TARGETS)),
        "scaler": _FakeScaler(),
    }


def test_forecast_lstm_full_with_stubs_returns_full_frame() -> None:
    pre = _make_pre(24)
    out = forecast_lstm_full(
        pre,
        24,
        keras_model=_FakeKerasModel(),
        artifacts=_fake_lstm_artifacts(),
    )
    pred_cols = [f"{c}_pred" for c in POLLUTANT_COLUMNS]
    for col in pred_cols + ["temperature_used", "wind_speed_used"]:
        assert col in out.columns
    assert (out["model_name"] == "LSTM").all()
    assert (out["forecast_horizon"] == 24).all()
    assert len(out) == 24


class _FakeBooster:
    """Stand-in XGBoost booster that returns a constant scaled value."""

    def __init__(self, value: float = 0.0) -> None:
        self._value = value

    def predict(self, _dmat: Any) -> np.ndarray:
        return np.array([self._value], dtype=np.float32)


def _fake_xgb_artifacts() -> dict[str, Any]:
    return {
        "feat_cols": list(FEAT_COLS),
        "target_cols": list(POLLUTANT_COLUMNS),
        "tgt_idxs": list(range(N_TARGETS)),
        "mu_all": np.zeros(N_FEATURES, dtype=np.float64),
        "sigma_all": np.ones(N_FEATURES, dtype=np.float64),
        "boosters": [_FakeBooster(0.0) for _ in POLLUTANT_COLUMNS],
    }


@pytest.fixture()
def fake_xgboost_module() -> Any:
    """Install a stub ``xgboost`` module so ``forecast_xgboost_full`` runs without
    the real dependency. Restored after the test."""
    original = sys.modules.get("xgboost")
    fake = types.ModuleType("xgboost")

    class DMatrix:
        def __init__(self, data: Any) -> None:
            self.data = data

    fake.DMatrix = DMatrix  # type: ignore[attr-defined]
    sys.modules["xgboost"] = fake
    try:
        yield fake
    finally:
        if original is None:
            sys.modules.pop("xgboost", None)
        else:
            sys.modules["xgboost"] = original


def test_forecast_xgboost_full_with_stubs_returns_full_frame(
    fake_xgboost_module: Any,
) -> None:
    _ = fake_xgboost_module
    pre = _make_pre(24)
    out = forecast_xgboost_full(pre, 24, artifacts=_fake_xgb_artifacts())
    pred_cols = [f"{c}_pred" for c in POLLUTANT_COLUMNS]
    for col in pred_cols + ["temperature_used", "wind_speed_used"]:
        assert col in out.columns
    assert (out["model_name"] == "XGBoost").all()
    assert (out["forecast_horizon"] == 24).all()
    assert len(out) == 24


def test_forecast_xgboost_full_mismatched_boosters_raises(
    fake_xgboost_module: Any,
) -> None:
    _ = fake_xgboost_module
    pre = _make_pre(24)
    artifacts = _fake_xgb_artifacts()
    artifacts["boosters"] = artifacts["boosters"][:-1]  # one short
    with pytest.raises(ValueError, match="boosters but"):
        forecast_xgboost_full(pre, 24, artifacts=artifacts)


def test_forecast_lstm_full_loads_artifacts_when_not_passed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When no model/artifacts are supplied, the function calls ``load_full_lstm``."""
    pre = _make_pre(24)
    monkeypatch.setattr(
        lstm_mod, "load_full_lstm", lambda: (_FakeKerasModel(), _fake_lstm_artifacts())
    )
    out = forecast_lstm_full(pre, 24)
    assert len(out) == 24


def test_forecast_xgboost_full_loads_artifacts_when_not_passed(
    monkeypatch: pytest.MonkeyPatch,
    fake_xgboost_module: Any,
) -> None:
    _ = fake_xgboost_module
    pre = _make_pre(24)
    monkeypatch.setattr(xgb_mod, "load_full_xgboost", _fake_xgb_artifacts)
    out = forecast_xgboost_full(pre, 24)
    assert len(out) == 24


# ─── HorizonUnavailableError end-to-end via stubs ────────────────────────────


def test_forecast_lstm_full_horizon_too_long_raises() -> None:
    pre = _make_pre(24)  # only 24 future hours
    with pytest.raises(HorizonUnavailableError):
        forecast_lstm_full(
            pre,
            72,
            keras_model=_FakeKerasModel(),
            artifacts=_fake_lstm_artifacts(),
        )
