"""
Multivariate XGBoost — train on 100% of the dataset (no train/test split, no plots).

Saves separate artifacts from xgboost_training.py:
  models/full_models/xgboost_multivariate_full_artifacts.joblib
  models/full_models/xgboost_multivariate_full_forecast_bundle.zip
"""
from pathlib import Path
import importlib
import joblib
import sys
import sysconfig
import zipfile

_script_dir = Path(__file__).resolve().parent
_here = str(_script_dir)
sys.path[:] = [p for p in sys.path if not (p and str(Path(p).resolve()) == _here)] + [_here]

if (_script_dir / "xgboost.py").is_file():
    raise ImportError(
        f"Delete or rename {_script_dir / 'xgboost.py'} — that filename blocks the real XGBoost package."
    )
if (_script_dir / "xgboost").is_dir() and not (_script_dir / "xgboost" / "core.py").is_file():
    raise ImportError(
        f"Delete or complete the local folder {_script_dir / 'xgboost'} "
        "(it has no core.py). It is treated as the `xgboost` package when your env install is missing."
    )

import numpy as np
import pandas as pd


def _load_xgb_train_api():
    pure = Path(sysconfig.get_paths()["purelib"])
    xgb_pkg = pure / "xgboost"
    if not xgb_pkg.is_dir():
        raise ImportError(
            "No Python package `xgboost` in site-packages for this interpreter.\n"
            f"  python: {sys.executable}\n"
            f"  missing: {xgb_pkg}\n"
            "  pip install xgboost"
        )
    try:
        core = importlib.import_module("xgboost.core")
        training = importlib.import_module("xgboost.training")
        return core.DMatrix, training.train
    except ImportError as e:
        raise ImportError(f"Cannot import xgboost.core: {e!r}") from e


DMatrix, xgb_train = _load_xgb_train_api()


def _standardize_all(arr: np.ndarray):
    mu = arr.mean(axis=0)
    sigma = arr.std(axis=0)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    return ((arr - mu) / sigma).astype(np.float32), mu, sigma


WINDOW = 48
HORIZON = 1
TREES = 200
LR = 0.05
DEPTH = 6

print("=" * 60)
print(" MULTIVARIATE XGBOOST | FULL DATA (100% TRAIN) | NO PLOTS")
print("=" * 60)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = _PROJECT_ROOT / "data" / "london_2024.csv"
TARGET_COLS = [
    "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
    "sulphur_dioxide", "ozone",
]
WEATHER_COLS = ["temperature", "wind_speed"]

df = pd.read_csv(DATA_PATH, parse_dates=["time"], index_col="time")
missing = [c for c in TARGET_COLS + WEATHER_COLS if c not in df.columns]
if missing:
    raise ValueError(f"CSV missing columns: {missing}. Path: {DATA_PATH}")
df = df[TARGET_COLS + WEATHER_COLS].apply(pd.to_numeric, errors="coerce").dropna()
if len(df) < WINDOW + HORIZON + 100:
    raise ValueError(f"Not enough rows after dropna: {len(df)}")
print(f"[DATA] Loaded {len(df)} hourly rows (all used for training) from {DATA_PATH}")

h = df.index.hour
dow = df.index.dayofweek
m = df.index.month
df["h_sin"] = np.sin(2 * np.pi * h / 24)
df["h_cos"] = np.cos(2 * np.pi * h / 24)
df["h2_sin"] = np.sin(4 * np.pi * h / 24)
df["h2_cos"] = np.cos(4 * np.pi * h / 24)
df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
df["mon_sin"] = np.sin(2 * np.pi * m / 12)
df["mon_cos"] = np.cos(2 * np.pi * m / 12)

FEAT_COLS = list(df.columns)
TGT_IDXS = [FEAT_COLS.index(c) for c in TARGET_COLS]

all_raw = df.values.astype(np.float64)
all_scaled, mu_all, sigma_all = _standardize_all(all_raw)


def make_windows(arr, window, horizon):
    n_samples = len(arr) - window - horizon + 1
    X = np.stack([arr[i : i + window] for i in range(n_samples)], axis=0).astype(np.float32)
    y = arr[window + horizon - 1 : window + horizon - 1 + n_samples, TGT_IDXS].astype(np.float32)
    return X, y


X_train_3d, y_train = make_windows(all_scaled, WINDOW, HORIZON)
X_train_2d = X_train_3d.reshape(X_train_3d.shape[0], -1)
print(f"[WINDOWS] X_train flattened: {X_train_2d.shape} -> y_train: {y_train.shape}")

xgb_params = {
    "objective": "reg:pseudohubererror",
    "max_depth": DEPTH,
    "eta": LR,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": 42,
}

print("\n[TRAIN] xgboost.train on 100% of windows (one booster per target)...")
n_targets = y_train.shape[1]
boosters = []
for k in range(n_targets):
    dtrain = DMatrix(X_train_2d, label=y_train[:, k])
    booster = xgb_train(xgb_params, dtrain, num_boost_round=TREES)
    boosters.append(booster)
print("[TRAIN] Complete.")

MODELS_DIR = _PROJECT_ROOT / "models"
FULL_MODELS_DIR = MODELS_DIR / "full_models"
FULL_MODELS_DIR.mkdir(parents=True, exist_ok=True)
_art = FULL_MODELS_DIR / "xgboost_multivariate_full_artifacts.joblib"
_zip = FULL_MODELS_DIR / "xgboost_multivariate_full_forecast_bundle.zip"

joblib.dump(
    {
        "version": "multivariate_xgb_full_v1",
        "window": WINDOW,
        "horizon": HORIZON,
        "target_cols": TARGET_COLS,
        "weather_cols": WEATHER_COLS,
        "feat_cols": FEAT_COLS,
        "tgt_idxs": TGT_IDXS,
        "mu_all": mu_all,
        "sigma_all": sigma_all,
        "xgb_params": xgb_params,
        "n_trees": TREES,
        "n_features_per_step": len(FEAT_COLS),
        "boosters": boosters,
        "trained_on": "100pct_no_split",
    },
    _art,
)
with zipfile.ZipFile(_zip, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write(_art, arcname=_art.name)

print(
    "\n[SAVED] Full-data XGBoost (no plots):\n"
    f"  • {_art}\n"
    f"  • {_zip}\n"
    "\nNote: xgboost_forecast.py still points at models/xgboost_multivariate_artifacts.joblib — "
    "use models/full_models/xgboost_multivariate_full_artifacts.joblib if you want this model in inference."
)
print("\nDone.")
