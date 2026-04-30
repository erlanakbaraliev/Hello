from pathlib import Path
import zipfile

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# 1. HYPERPARAMETERS
WINDOW   = 48    # Look-back window (48 hours)
HORIZON  = 1     # Predict t+1
HIDDEN   = 64
DROPOUT  = 0.2
BATCH    = 256
EPOCHS   = 32
PATIENCE = 6

print("=" * 60)
print(" MULTIVARIATE LSTM PIPELINE | TENSORFLOW / KERAS")
print("=" * 60)

# 2. LOAD DATA — London 2024 hourly (pollutants + weather)
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "london_2024.csv"
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
print(f"Loaded {len(df)} hourly rows from {DATA_PATH}")

# 3. FOURIER HARMONICS (The genius time-handling trick)
h = df.index.hour
dow = df.index.dayofweek
m = df.index.month

df["h_sin"]   = np.sin(2 * np.pi * h   / 24)
df["h_cos"]   = np.cos(2 * np.pi * h   / 24)
df["h2_sin"]  = np.sin(4 * np.pi * h   / 24)
df["h2_cos"]  = np.cos(4 * np.pi * h   / 24)
df["dow_sin"] = np.sin(2 * np.pi * dow  / 7)
df["dow_cos"] = np.cos(2 * np.pi * dow  / 7)
df["mon_sin"] = np.sin(2 * np.pi * m   / 12)
df["mon_cos"] = np.cos(2 * np.pi * m   / 12)

# 4. FEATURE COLUMN ORDER (targets + weather + harmonics)
FEAT_COLS = list(df.columns)
TGT_IDXS = [FEAT_COLS.index(c) for c in TARGET_COLS]

# 5. CHRONOLOGICAL SPLIT & SCALING
split_idx = int(len(df) * 0.80)
tr_raw = df.values[:split_idx]
te_raw = df.values[split_idx:]

scaler = StandardScaler()
tr_scaled = scaler.fit_transform(tr_raw) # Fit ONLY on train (Zero Leakage)
te_scaled = scaler.transform(te_raw)

# 6. SLIDING WINDOW FUNCTION (Multivariate)
def make_windows(arr, window, horizon):
    n_samples = len(arr) - window - horizon + 1
    X = np.stack([arr[i : i + window] for i in range(n_samples)], axis=0).astype(np.float32)
    # y grabs ALL 6 pollutant columns for the future time step
    y = arr[window + horizon - 1 : window + horizon - 1 + n_samples, TGT_IDXS].astype(np.float32)
    return X, y

X_train, y_train = make_windows(tr_scaled, WINDOW, HORIZON)
X_test,  y_test  = make_windows(te_scaled, WINDOW, HORIZON)

n_features = X_train.shape[2]
n_targets = len(TARGET_COLS)

print(f"Train Shape: X={X_train.shape}, y={y_train.shape}")
print(f"Test Shape : X={X_test.shape}, y={y_test.shape}")

# 7. BUILD KERAS LSTM ARCHITECTURE
model = Sequential([
    LSTM(HIDDEN, return_sequences=True, input_shape=(WINDOW, n_features)),
    Dropout(DROPOUT),
    LSTM(HIDDEN, return_sequences=False),
    Dropout(DROPOUT),
    Dense(n_targets) # CRITICAL: Output layer predicts 6 numbers, not 1.
])

# Huber Loss is built directly into TensorFlow
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.Huber(delta=1.0) 
)

# 8. KERAS CALLBACKS (Replaces PyTorch's manual loops)
# Stops early if no improvement, and automatically restores the best weights!
early_stop = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
# Cuts learning rate in half if it gets stuck
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

print("Training Model...")

# 9. TRAIN THE MODEL
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH,
    callbacks=[early_stop, reduce_lr],
    verbose=1 # Shows a nice progress bar
)

# 9b. SAVE MODEL + SCALER (for forecasting / backup — under project models/)
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
_lstm_keras = MODELS_DIR / "lstm_multivariate_model.keras"
_lstm_art = MODELS_DIR / "lstm_multivariate_artifacts.joblib"
_lstm_zip = MODELS_DIR / "lstm_multivariate_forecast_bundle.zip"
model.save(_lstm_keras)
joblib.dump(
    {
        "version": "multivariate_lstm_v1",
        "window": WINDOW,
        "horizon": HORIZON,
        "target_cols": TARGET_COLS,
        "weather_cols": WEATHER_COLS,
        "feat_cols": FEAT_COLS,
        "tgt_idxs": TGT_IDXS,
        "scaler": scaler,
    },
    _lstm_art,
)
with zipfile.ZipFile(_lstm_zip, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write(_lstm_keras, arcname=_lstm_keras.name)
    zf.write(_lstm_art, arcname=_lstm_art.name)
print(
    "\n[SAVED] Trained LSTM for forecasting:\n"
    f"  • {_lstm_keras}\n"
    f"  • {_lstm_art}  (StandardScaler + column metadata)\n"
    f"  • {_lstm_zip}  (single file to copy / download)"
)

# 10. INFERENCE & INVERSE SCALING
print("Predicting on Test Set...")
preds_scaled = model.predict(X_test)

# Inverse transform only the 6 target columns using the saved Scaler properties
mu_targets = scaler.mean_[TGT_IDXS]
sigma_targets = scaler.scale_[TGT_IDXS]

y_pred_orig = preds_scaled * sigma_targets + mu_targets
y_true_orig = y_test * sigma_targets + mu_targets

# Extract training history for plotting
train_losses = history.history['loss']
val_losses = history.history['val_loss']

# 11. PLOTTING FUNCTION
def plot_multivariate_results(df, y_true_multi, y_pred_multi, train_losses, val_losses, split_idx):
    
    # We only want to plot PM2.5 on the graph
    PM25_IDX = TARGET_COLS.index('pm2_5')
    y_true = y_true_multi[:, PM25_IDX]
    y_pred = y_pred_multi[:, PM25_IDX]

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    ss_r = np.sum((y_true - y_pred) ** 2)
    ss_t = np.sum((y_true - y_true.mean()) ** 2)
    r2   = 1 - ss_r / ss_t if ss_t > 0 else float("nan")

    BG, PAN = "#0d1117", "#161b22"
    C2, CT  = "#ef9a9a", "#b0bec5"

    def sax(ax, title, fs=10):
        ax.set_facecolor(PAN)
        ax.set_title(title, color="white", fontsize=fs, pad=8)
        ax.tick_params(colors="#8b949e", labelsize=8)
        ax.spines[:].set_color("#30363d")
        for it in [ax.xaxis.label, ax.yaxis.label]: it.set_color("#8b949e")
        return ax

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.patch.set_facecolor(BG)

    ax = sax(axes[0, 0], "Full Year — PM2.5 Observed", 10)
    ax.plot(df.index, df["pm2_5"], color=CT, lw=0.5, alpha=0.9, label="Observed")
    ax.axvline(df.index[split_idx], color="#ffa726", lw=1.2, ls="--", label="Train | Test split")
    ax.set_ylabel("PM2.5  µg/m³")
    ax.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="white", fontsize=8)

    nb = min(600, len(y_true))
    ax = sax(axes[0, 1], f"TF/Keras Predictions — last {nb} h (PM2.5 Only)", 10)
    ax.plot(y_true[-nb:], color=CT,  lw=0.9, label="Actual")
    ax.plot(y_pred[-nb:], color=C2,  lw=0.9, alpha=0.85, label="LSTM pred")
    ax.text(0.02, 0.93, f"MAE={mae:.3f} µg/m³\nR²={r2:.3f}",
            transform=ax.transAxes, color="white", fontsize=8, va="top",
            bbox=dict(facecolor="#21262d", edgecolor="none", pad=2))
    ax.set_ylabel("PM2.5  µg/m³")
    ax.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="white", fontsize=8)

    ax = sax(axes[1, 0], "Training Curves (Huber loss, scaled space)", 10)
    epochs_range = range(1, len(train_losses) + 1)
    ax.plot(epochs_range, train_losses, color=C2, lw=1.8, label="Train Huber")
    ax.plot(epochs_range, val_losses, color="#4fc3f7", lw=1.8, label="Val Huber")
    best_ep = int(np.argmin(val_losses)) + 1
    ax.scatter([best_ep], [min(val_losses)], color="#a5d6a7", s=80, zorder=5, label=f"Best epoch {best_ep}")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Huber Loss")
    ax.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="white", fontsize=8)

    ax = sax(axes[1, 1], "Actual vs Predicted (PM2.5 Test Set)", 10)
    ax.scatter(y_true, y_pred, s=2, alpha=0.2, color=C2, rasterized=True)
    lo = min(y_true.min(), y_pred.min()) - 0.3
    hi = max(y_true.max(), y_pred.max()) + 0.3
    ax.plot([lo, hi], [lo, hi], "w--", lw=0.8, label="Perfect fit")
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    ax.text(0.05, 0.93, f"Pearson r={corr:.3f}\npred std={y_pred.std():.2f}  true std={y_true.std():.2f}",
            transform=ax.transAxes, color="white", fontsize=8, va="top",
            bbox=dict(facecolor="#21262d", edgecolor="none", pad=2))
    ax.set_xlabel("Actual PM2.5  µg/m³"); ax.set_ylabel("Predicted  µg/m³")
    ax.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="white", fontsize=8)

    fig.suptitle(
        "TF/Keras Multivariate LSTM (48-h window) — PM2.5 Evaluation\n"
        f"Test MAE={mae:.4f} µg/m³   RMSE={rmse:.4f} µg/m³   R²={r2:.4f}",
        color="white", fontsize=11, y=0.995, fontweight="bold",
    )
    plt.tight_layout(pad=1.5)
    
    out_path = "keras_multivariate_lstm_report.png"
    plt.savefig(out_path, dpi=140, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"\n[SUCCESS] Plot saved to {out_path}")

plot_multivariate_results(df, y_true_orig, y_pred_orig, train_losses, val_losses, split_idx)