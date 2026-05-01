"""Shared constants, paths, and domain exceptions for the back-end pipeline.

Everything in this module is intentionally dependency-free so it can be
imported by any other ``back_end`` module without risking circular imports.
"""

from __future__ import annotations

from pathlib import Path

# ─── Column lists ────────────────────────────────────────────────────────────

EXPECTED_COLUMNS: list[str] = [
    "time",
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "city",
]

POLLUTANT_COLUMNS: list[str] = [
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
]

WEATHER_COLUMNS: list[str] = ["temperature", "wind_speed"]

TIME_FEATURE_COLUMNS: list[str] = [
    "h_sin", "h_cos", "h2_sin", "h2_cos",
    "dow_sin", "dow_cos", "mon_sin", "mon_cos",
]

REQUIRED_PAST_COLUMNS: list[str] = ["time", *POLLUTANT_COLUMNS, *WEATHER_COLUMNS]
REQUIRED_FUTURE_COLUMNS: list[str] = ["time", *WEATHER_COLUMNS]

GEO_COLUMN_NAMES: frozenset[str] = frozenset(
    {"latitude", "longitude", "lat", "lon", "lng"}
)
OPTIONAL_DROP: frozenset[str] = frozenset({"city", *GEO_COLUMN_NAMES})


# ─── Forecast configuration ──────────────────────────────────────────────────

SUPPORTED_HORIZONS: tuple[int, ...] = (24, 72, 168)
WINDOW: int = 48

# Maximum consecutive-hour gap that interpolation will fill before erroring.
INTERPOLATE_LIMIT: int = 6


# ─── Model artifact paths ────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_FULL_MODELS_DIR = _PROJECT_ROOT / "experiments" / "models" / "full_models"

LSTM_KERAS_PATH: Path = _FULL_MODELS_DIR / "lstm_multivariate_full_model.keras"
LSTM_ARTIFACT_PATH: Path = _FULL_MODELS_DIR / "lstm_multivariate_full_artifacts.joblib"
XGB_ARTIFACT_PATH: Path = _FULL_MODELS_DIR / "xgboost_multivariate_full_artifacts.joblib"


# ─── Domain exceptions (caught by the Streamlit forecast page) ───────────────


class UploadValidationError(ValueError):
    """Raised when the uploaded dataset fails any schema/structure check."""


class HorizonUnavailableError(ValueError):
    """Raised when the user picks H but fewer than H future weather rows exist."""


class ArtifactMissingError(FileNotFoundError):
    """Raised when a required model artifact file is missing on disk."""
