"""LSTM full-model loader and forecast entry point.

Loads the artifacts produced by
``experiments/forecasting/notebooks/lstm_full_training.py`` and runs a
recursive single-step forecast for ``horizon`` ∈ ``SUPPORTED_HORIZONS`` via
the shared engine in :mod:`back_end.inference`.
"""

from __future__ import annotations

from typing import Any

import joblib
import numpy as np
import pandas as pd

from back_end.constants import (
    LSTM_ARTIFACT_PATH,
    LSTM_KERAS_PATH,
    WINDOW,
    ArtifactMissingError,
)
from back_end.inference import _run_recursive_single_step, build_forecast_output
from back_end.upload_preprocess import PreprocessedUpload


def load_full_lstm() -> tuple[Any, dict[str, Any]]:
    """Return ``(keras_model, artifacts_dict)`` from the full-data LSTM."""
    if not LSTM_KERAS_PATH.is_file():
        raise ArtifactMissingError(
            f"Required model artifact not found at {LSTM_KERAS_PATH}."
        )
    if not LSTM_ARTIFACT_PATH.is_file():
        raise ArtifactMissingError(
            f"Required model artifact not found at {LSTM_ARTIFACT_PATH}."
        )
    try:
        from tensorflow import keras
    except ImportError as exc:  # pragma: no cover — env-dependent
        raise ImportError(
            "TensorFlow/Keras is required to load the LSTM full model."
        ) from exc
    model = keras.models.load_model(LSTM_KERAS_PATH)
    artifacts = joblib.load(LSTM_ARTIFACT_PATH)
    return model, artifacts


def forecast_lstm_full(
    preprocessed: PreprocessedUpload,
    horizon: int,
    *,
    keras_model: Any | None = None,
    artifacts: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Recursive single-step LSTM forecast for ``horizon`` ∈ {24, 72, 168}."""
    if keras_model is None or artifacts is None:
        keras_model, artifacts = load_full_lstm()

    feat_cols = list(artifacts["feat_cols"])
    target_cols = list(artifacts["target_cols"])
    tgt_idxs = list(artifacts["tgt_idxs"])
    scaler = artifacts["scaler"]
    mu = np.asarray(scaler.mean_, dtype=np.float64)
    sigma = np.asarray(scaler.scale_, dtype=np.float64)

    def predict_one_step(history: np.ndarray) -> np.ndarray:
        x = history.reshape(1, WINDOW, len(feat_cols)).astype(np.float32)
        return keras_model.predict(x, verbose=0)[0]

    raw_preds = _run_recursive_single_step(
        preprocessed=preprocessed,
        horizon=horizon,
        feat_cols=feat_cols,
        target_cols=target_cols,
        tgt_idxs=tgt_idxs,
        mu=mu,
        sigma=sigma,
        predict_one_step=predict_one_step,
    )
    return build_forecast_output(
        raw_preds,
        preprocessed.future_actuals_df,
        target_cols=target_cols,
        model_name="LSTM",
        horizon=horizon,
    )
