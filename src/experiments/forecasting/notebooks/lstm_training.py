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
# Repo layout: train_folder/data/ … (this file lives in train_folder/forecasting/notebooks/)
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
MODELS_DIR = _PROJECT_ROOT / "models"
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

# 11. PLOTTING — one PNG per figure (A4-friendly: ~160 mm text width @ 200 dpi)
_DOC_DPI = 200
# ~6.7" × 3.9" ≈ 170 × 99 mm — embed at ~0.95×A4 text width without downscaling
_DOC_FIGSIZE = (6.7, 3.9)


def plot_multivariate_results(df, y_true_multi, y_pred_multi, train_losses, val_losses, split_idx):
    out_dir = Path(__file__).resolve().parent
    PM25_IDX = TARGET_COLS.index("pm2_5")
    y_true = y_true_multi[:, PM25_IDX]
    y_pred = y_pred_multi[:, PM25_IDX]

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    ss_r = np.sum((y_true - y_pred) ** 2)
    ss_t = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_r / ss_t if ss_t > 0 else float("nan")

    BG, PAN = "#0d1117", "#161b22"
    C2, CT = "#ef9a9a", "#b0bec5"

    def sax(ax, title, fs=11):
        ax.set_facecolor(PAN)
        ax.set_title(title, color="white", fontsize=fs, pad=10)
        ax.tick_params(colors="#8b949e", labelsize=9)
        ax.spines[:].set_color("#30363d")
        for it in [ax.xaxis.label, ax.yaxis.label]:
            it.set_color("#8b949e")
        return ax

    def new_fig():
        fig, ax = plt.subplots(figsize=_DOC_FIGSIZE)
        fig.patch.set_facecolor(BG)
        return fig, ax

    def save_doc(fig, filename):
        path = out_dir / filename
        fig.tight_layout()
        fig.savefig(path, dpi=_DOC_DPI, bbox_inches="tight", facecolor=BG)
        plt.close(fig)
        return path

    saved = []

    # 1 — Full year PM2.5
    fig, ax = new_fig()
    sax(ax, "Full year — PM2.5 observed (hourly)")
    ax.plot(df.index, df["pm2_5"], color=CT, lw=0.55, alpha=0.9, label="Observed")
    ax.axvline(
        df.index[split_idx],
        color="#ffa726",
        lw=1.2,
        ls="--",
        label="Train | test split (80% / 20%)",
    )
    ax.set_ylabel("PM2.5  µg/m³", fontsize=10)
    ax.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="white", fontsize=9)
    fig.suptitle(
        "Multivariate LSTM — study period and chronological split",
        color="white",
        fontsize=12,
        y=1.02,
        fontweight="bold",
    )
    saved.append(save_doc(fig, "keras_multivariate_lstm_doc_01_full_year_pm25.png"))

    # 2 — Last N h predictions (same window as before; metrics = full test set)
    nb = min(600, len(y_true))
    fig, ax = new_fig()
    sax(ax, f"Test set — last {nb} h: actual vs LSTM (PM2.5)")
    ax.plot(y_true[-nb:], color=CT, lw=1.0, label="Actual")
    ax.plot(y_pred[-nb:], color=C2, lw=1.0, alpha=0.88, label="LSTM prediction")
    ax.text(
        0.02,
        0.93,
        f"Metrics on full test: MAE={mae:.3f} µg/m³\n"
        f"RMSE={rmse:.3f} µg/m³\n"
        f"R²={r2:.3f}",
        transform=ax.transAxes,
        color="white",
        fontsize=9,
        va="top",
        bbox=dict(facecolor="#21262d", edgecolor="none", pad=4),
    )
    ax.set_ylabel("PM2.5  µg/m³", fontsize=10)
    ax.set_xlabel("Test hours (index within window)", fontsize=10)
    ax.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="white", fontsize=9)
    fig.suptitle(
        f"PM2.5 +1 h forecast — 48 h lookback | Test MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}",
        color="white",
        fontsize=11,
        y=1.02,
        fontweight="bold",
    )
    saved.append(save_doc(fig, "keras_multivariate_lstm_doc_02_test_predictions_pm25.png"))

    # 3 — Training curves
    fig, ax = new_fig()
    sax(ax, "Training — Huber loss (scaled target space)")
    epochs_range = range(1, len(train_losses) + 1)
    ax.plot(epochs_range, train_losses, color=C2, lw=2.0, label="Train Huber")
    ax.plot(epochs_range, val_losses, color="#4fc3f7", lw=2.0, label="Validation Huber")
    best_ep = int(np.argmin(val_losses)) + 1
    ax.scatter(
        [best_ep],
        [min(val_losses)],
        color="#a5d6a7",
        s=90,
        zorder=5,
        label=f"Best epoch {best_ep}",
    )
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Huber loss", fontsize=10)
    ax.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="white", fontsize=9)
    fig.suptitle(
        "TF/Keras LSTM — optimisation (early stopping restores best weights)",
        color="white",
        fontsize=12,
        y=1.02,
        fontweight="bold",
    )
    saved.append(save_doc(fig, "keras_multivariate_lstm_doc_03_training_loss.png"))

    # 4 — Scatter
    fig, ax = new_fig()
    sax(ax, "Test set — actual vs predicted PM2.5")
    ax.scatter(y_true, y_pred, s=3, alpha=0.22, color=C2, rasterized=True)
    lo = min(y_true.min(), y_pred.min()) - 0.3
    hi = max(y_true.max(), y_pred.max()) + 0.3
    ax.plot([lo, hi], [lo, hi], "w--", lw=0.9, label="Perfect fit (y = x)")
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    ax.text(
        0.05,
        0.93,
        f"Pearson r={corr:.3f}\npred σ={y_pred.std():.2f}   true σ={y_true.std():.2f}",
        transform=ax.transAxes,
        color="white",
        fontsize=9,
        va="top",
        bbox=dict(facecolor="#21262d", edgecolor="none", pad=4),
    )
    ax.set_xlabel("Actual PM2.5  µg/m³", fontsize=10)
    ax.set_ylabel("Predicted PM2.5  µg/m³", fontsize=10)
    ax.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="white", fontsize=9)
    fig.suptitle(
        f"Calibration on full test (n={len(y_true):,}) | MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}",
        color="white",
        fontsize=11,
        y=1.02,
        fontweight="bold",
    )
    saved.append(save_doc(fig, "keras_multivariate_lstm_doc_04_actual_vs_predicted_pm25.png"))

    print("\n[SUCCESS] Documentation figures (LSTM, 200 dpi, A4-friendly width):")
    for p in saved:
        print(f"  • {p}")


plot_multivariate_results(df, y_true_orig, y_pred_orig, train_losses, val_losses, split_idx)