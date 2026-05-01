"""
Multivariate LSTM — train on 100% of the dataset (no train/test split, no plots).

Saves separate artifacts from lstm_training.py:
  models/full_models/lstm_multivariate_full_model.keras
  models/full_models/lstm_multivariate_full_artifacts.joblib
  models/full_models/lstm_multivariate_full_forecast_bundle.zip
"""
from pathlib import Path
import zipfile

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# 1. HYPERPARAMETERS (same architecture as lstm_training.py)
WINDOW = 48
HORIZON = 1
HIDDEN = 64
DROPOUT = 0.2
BATCH = 256
EPOCHS = 32
PATIENCE = 6

print("=" * 60)
print(" MULTIVARIATE LSTM | FULL DATA (100% TRAIN) | NO PLOTS")
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
print(f"Loaded {len(df)} hourly rows (all used for training) from {DATA_PATH}")

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
scaler = StandardScaler()
all_scaled = scaler.fit_transform(all_raw)


def make_windows(arr, window, horizon):
    n_samples = len(arr) - window - horizon + 1
    X = np.stack([arr[i : i + window] for i in range(n_samples)], axis=0).astype(np.float32)
    y = arr[window + horizon - 1 : window + horizon - 1 + n_samples, TGT_IDXS].astype(np.float32)
    return X, y


X_train, y_train = make_windows(all_scaled, WINDOW, HORIZON)
n_features = X_train.shape[2]
n_targets = len(TARGET_COLS)
print(f"Windows: X={X_train.shape}, y={y_train.shape}")

model = Sequential([
    LSTM(HIDDEN, return_sequences=True, input_shape=(WINDOW, n_features)),
    Dropout(DROPOUT),
    LSTM(HIDDEN, return_sequences=False),
    Dropout(DROPOUT),
    Dense(n_targets),
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.Huber(delta=1.0),
)
# No validation set — monitor training loss only
early_stop = EarlyStopping(monitor="loss", patience=PATIENCE, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="loss", factor=0.5, patience=3, min_lr=1e-6)

print("Training on 100% of samples (no held-out split)...")
model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH,
    callbacks=[early_stop, reduce_lr],
    verbose=1,
)

MODELS_DIR = _PROJECT_ROOT / "models"
FULL_MODELS_DIR = MODELS_DIR / "full_models"
FULL_MODELS_DIR.mkdir(parents=True, exist_ok=True)
_keras = FULL_MODELS_DIR / "lstm_multivariate_full_model.keras"
_art = FULL_MODELS_DIR / "lstm_multivariate_full_artifacts.joblib"
_zip = FULL_MODELS_DIR / "lstm_multivariate_full_forecast_bundle.zip"

model.save(_keras)
joblib.dump(
    {
        "version": "multivariate_lstm_full_v1",
        "window": WINDOW,
        "horizon": HORIZON,
        "target_cols": TARGET_COLS,
        "weather_cols": WEATHER_COLS,
        "feat_cols": FEAT_COLS,
        "tgt_idxs": TGT_IDXS,
        "scaler": scaler,
        "trained_on": "100pct_no_split",
    },
    _art,
)
with zipfile.ZipFile(_zip, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write(_keras, arcname=_keras.name)
    zf.write(_art, arcname=_art.name)

print(
    "\n[SAVED] Full-data LSTM (no plots):\n"
    f"  • {_keras}\n"
    f"  • {_art}\n"
    f"  • {_zip}\n"
    "\nNote: lstm_forecast.py still points at models/lstm_multivariate_model.keras — "
    "use models/full_models/lstm_multivariate_full_* if you want this model in inference."
)
print("\nDone.")
