"""XGBoost full-model loader and forecast entry point.

Loads the artifact produced by
``experiments/forecasting/notebooks/xgboost_full_training.py`` and runs a
recursive single-step forecast for ``horizon`` ∈ ``SUPPORTED_HORIZONS`` via
the shared engine in :mod:`back_end.inference`.
"""

from __future__ import annotations

from typing import Any

import joblib
import numpy as np
import pandas as pd

from back_end.constants import XGB_ARTIFACT_PATH, ArtifactMissingError
from back_end.inference import _run_recursive_single_step, build_forecast_output
from back_end.upload_preprocess import PreprocessedUpload


def load_full_xgboost() -> dict[str, Any]:
    """Return the XGBoost full-data artifact dict (contains ``boosters`` list)."""
    if not XGB_ARTIFACT_PATH.is_file():
        raise ArtifactMissingError(
            f"Required model artifact not found at {XGB_ARTIFACT_PATH}."
        )
    return joblib.load(XGB_ARTIFACT_PATH)


def forecast_xgboost_full(
    preprocessed: PreprocessedUpload,
    horizon: int,
    *,
    artifacts: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Recursive single-step XGBoost forecast for ``horizon`` ∈ {24, 72, 168}."""
    if artifacts is None:
        artifacts = load_full_xgboost()
    try:
        import xgboost as xgb
    except ImportError as exc:  # pragma: no cover — env-dependent
        raise ImportError(
            "xgboost is required to run XGBoost full-model inference."
        ) from exc

    feat_cols = list(artifacts["feat_cols"])
    target_cols = list(artifacts["target_cols"])
    tgt_idxs = list(artifacts["tgt_idxs"])
    mu = np.asarray(artifacts["mu_all"], dtype=np.float64)
    sigma = np.asarray(artifacts["sigma_all"], dtype=np.float64)
    boosters = list(artifacts["boosters"])
    if len(boosters) != len(target_cols):
        raise ValueError(
            f"XGBoost artifacts have {len(boosters)} boosters but "
            f"{len(target_cols)} target columns."
        )

    def predict_one_step(history: np.ndarray) -> np.ndarray:
        flat = history.reshape(1, -1).astype(np.float32)
        dmat = xgb.DMatrix(flat)
        return np.asarray(
            [float(b.predict(dmat)[0]) for b in boosters], dtype=np.float32
        )

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
        model_name="XGBoost",
        horizon=horizon,
    )
