from pathlib import Path
import importlib
import joblib
import sys
import sysconfig
import zipfile

_script_dir = Path(__file__).resolve().parent
_here = str(_script_dir)
# Prefer site-packages over this folder so a stray `xgboost.py` / empty `xgboost/` here does not win.
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_xgb_train_api():
    """Load DMatrix/train from xgboost submodules."""
    pure = Path(sysconfig.get_paths()["purelib"])
    xgb_pkg = pure / "xgboost"
    if not xgb_pkg.is_dir():
        raise ImportError(
            "No Python package `xgboost` in site-packages for this interpreter — "
            "`conda list` can still show `xgboost` / `libxgboost` if only the C++ library is wired.\n\n"
            f"  python: {sys.executable}\n"
            f"  missing: {xgb_pkg}\n\n"
            "Replace the defaults build with a full install (your list shows build *hecd8cb5* = defaults):\n"
            "  conda remove xgboost libxgboost _py-xgboost-mutex --force -y\n"
            "  pip install xgboost\n"
            "  # or: conda install -c conda-forge xgboost --force-reinstall --no-channel-priority\n\n"
            "Check `which python` points to .../envs/thesis_tf/bin/python"
        )

    probe = ""
    try:
        import xgboost as _xp  # noqa: F401
        probe = (
            f"\nDiagnostic: xgboost.__path__={list(getattr(_xp, '__path__', []))!r} "
            f"__file__={getattr(_xp, '__file__', None)!r}"
        )
    except Exception as ex:
        probe = f"\nDiagnostic: import xgboost failed: {ex!r}"

    try:
        core = importlib.import_module("xgboost.core")
        training = importlib.import_module("xgboost.training")
        return core.DMatrix, training.train
    except ImportError as e:
        raise ImportError(
            "Cannot import xgboost.core — incomplete install under site-packages.\n\n"
            "  pip uninstall xgboost -y\n"
            "  conda remove xgboost libxgboost _py-xgboost-mutex --force -y\n"
            "  pip install xgboost\n"
            "  python -c \"import xgboost.core; print('ok')\"\n"
            f"\nOriginal error: {e!r}{probe}"
        ) from e


DMatrix, xgb_train = _load_xgb_train_api()


def _standardize_train_test(train: np.ndarray, test: np.ndarray):
    """Per-column z-score from train only (equivalent to StandardScaler, ddof=0)."""
    mu = train.mean(axis=0)
    sigma = train.std(axis=0)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    return (train - mu) / sigma, (test - mu) / sigma, mu, sigma

# ─────────────────────────────────────────────────────────────────────────────
# 1. HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
WINDOW   = 48    # Look-back length in hours
HORIZON  = 1     # Forecast horizon: predict t+1
TREES    = 200   # Number of boosting rounds (trees)
LR       = 0.05  # Learning rate
DEPTH    = 6     # Max tree depth

print("=" * 60)
print(" PIPELINE C : MULTIVARIATE XGBOOST (48-h window, +1h forecast)")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────────────────
# 2. LOAD DATA — London 2024 hourly (pollutants + weather)
# ─────────────────────────────────────────────────────────────────────────────
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
print(f"[DATA] Loaded {len(df)} hourly rows from {DATA_PATH}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. FOURIER HARMONICS (Time engineering)
# ─────────────────────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────────────────────
# 4. FEATURE ORDER & CHRONOLOGICAL SPLIT
# ─────────────────────────────────────────────────────────────────────────────
FEAT_COLS = list(df.columns)
TGT_IDXS = [FEAT_COLS.index(c) for c in TARGET_COLS]
PM25_IDX_IN_TARGETS = TARGET_COLS.index('pm2_5') # To pull PM2.5 for the plot later

split_idx = int(len(df) * 0.80)
tr_raw = df.values[:split_idx]
te_raw = df.values[split_idx:]

print(f"[SPLIT] Train: {len(tr_raw)} rows | Test: {len(te_raw)} rows")

# ─────────────────────────────────────────────────────────────────────────────
# 5. SCALING & WINDOWING (Identical to LSTM to ensure fairness)
# ─────────────────────────────────────────────────────────────────────────────
tr_scaled, te_scaled, mu_all, sigma_all = _standardize_train_test(tr_raw, te_raw)

def make_windows(arr, window, horizon):
    n_samples = len(arr) - window - horizon + 1
    X = np.stack([arr[i : i + window] for i in range(n_samples)], axis=0).astype(np.float32)
    y = arr[window + horizon - 1 : window + horizon - 1 + n_samples, TGT_IDXS].astype(np.float32)
    return X, y

X_train_3d, y_train = make_windows(tr_scaled, WINDOW, HORIZON)
X_test_3d,  y_test  = make_windows(te_scaled, WINDOW, HORIZON)

# ⚡ THE XGBOOST TRICK: Flatten the 3D sliding windows into 2D tables!
# (Samples, Time Steps, Features) -> (Samples, Time Steps * Features)
X_train_2d = X_train_3d.reshape(X_train_3d.shape[0], -1)
X_test_2d  = X_test_3d.reshape(X_test_3d.shape[0], -1)

print(f"[WINDOWS] X_train flattened: {X_train_2d.shape} -> y_train: {y_train.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. BUILD & TRAIN MULTIVARIATE XGBOOST
# ─────────────────────────────────────────────────────────────────────────────
print("\n[TRAIN] Native xgboost.train (DMatrix) — no sklearn XGBRegressor...")

n_targets = y_train.shape[1]
xgb_params = {
    "objective": "reg:pseudohubererror",
    "max_depth": DEPTH,
    "eta": LR,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": 42,
}
dtest = DMatrix(X_test_2d)
pred_cols = []
boosters = []
for k in range(n_targets):
    dtrain = DMatrix(X_train_2d, label=y_train[:, k])
    booster = xgb_train(xgb_params, dtrain, num_boost_round=TREES)
    boosters.append(booster)
    pred_cols.append(booster.predict(dtest))
preds_scaled = np.column_stack(pred_cols).astype(np.float32)
print("[TRAIN] XGBoost training complete.")

# 6b. SAVE BOOSTERS + SCALING (for forecasting / backup — under project models/)
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
_xgb_art = MODELS_DIR / "xgboost_multivariate_artifacts.joblib"
_xgb_zip = MODELS_DIR / "xgboost_multivariate_forecast_bundle.zip"
joblib.dump(
    {
        "version": "multivariate_xgb_v1",
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
    },
    _xgb_art,
)
with zipfile.ZipFile(_xgb_zip, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write(_xgb_art, arcname=_xgb_art.name)
print(
    "\n[SAVED] Trained XGBoost for forecasting:\n"
    f"  • {_xgb_art}\n"
    f"  • {_xgb_zip}  (single file to copy / download)"
)

mu_targets = mu_all[TGT_IDXS]
sigma_targets = sigma_all[TGT_IDXS]

y_pred_orig = preds_scaled * sigma_targets + mu_targets
y_true_orig = y_test * sigma_targets + mu_targets

# ─────────────────────────────────────────────────────────────────────────────
# 8. METRICS & VISUALIZATION (Dark Mode)
# ─────────────────────────────────────────────────────────────────────────────
def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    return float(np.mean(np.abs(y_true - y_pred) / np.where(denom < 1e-8, 1e-8, denom)) * 100)

# Extract only PM2.5 for evaluation
y_true_pm25 = y_true_orig[:, PM25_IDX_IN_TARGETS]
y_pred_pm25 = y_pred_orig[:, PM25_IDX_IN_TARGETS]

mae = float(np.mean(np.abs(y_true_pm25 - y_pred_pm25)))
rmse = float(np.sqrt(np.mean((y_true_pm25 - y_pred_pm25) ** 2)))
sp   = smape(y_true_pm25, y_pred_pm25)
ss_r = np.sum((y_true_pm25 - y_pred_pm25) ** 2)
ss_t = np.sum((y_true_pm25 - y_true_pm25.mean()) ** 2)
r2   = 1 - ss_r / ss_t if ss_t > 0 else float("nan")

print(f"\n[EVALUATION] PM2.5 Test Metrics:")
print(f" MAE   : {mae:.4f} µg/m³")
print(f" RMSE  : {rmse:.4f} µg/m³")
print(f" sMAPE : {sp:.2f}%")
print(f" R²    : {r2:.4f}")

def plot_xgboost(df, y_true, y_pred, split_idx):
    BG, PAN = "#0d1117", "#161b22"
    C2, CT  = "#ef9a9a", "#b0bec5"

    def sax(ax, title, fs=10):
        ax.set_facecolor(PAN)
        ax.set_title(title, color="white", fontsize=fs, pad=8)
        ax.tick_params(colors="#8b949e", labelsize=8)
        ax.spines[:].set_color("#30363d")
        for it in [ax.xaxis.label, ax.yaxis.label]: it.set_color("#8b949e")
        return ax

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor(BG)

    # 1. Full year overview
    ax = sax(axes[0], "Full Year — PM2.5 Observed", 10)
    ax.plot(df.index, df["pm2_5"], color=CT, lw=0.5, alpha=0.9, label="Observed")
    ax.axvline(df.index[split_idx], color="#ffa726", lw=1.2, ls="--", label="Train/Test Split")
    ax.set_ylabel("PM2.5  µg/m³")
    ax.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="white", fontsize=8)

    # 2. Last 600 h predictions
    nb = min(600, len(y_true))
    ax = sax(axes[1], f"XGBoost Predictions — last {nb} h (PM2.5)", 10)
    ax.plot(y_true[-nb:], color=CT, lw=0.9, label="Actual")
    ax.plot(y_pred[-nb:], color=C2, lw=0.9, alpha=0.85, label="XGB pred")
    ax.text(0.02, 0.93, f"MAE={mae:.3f}\nR²={r2:.3f}",
            transform=ax.transAxes, color="white", fontsize=8, va="top",
            bbox=dict(facecolor="#21262d", edgecolor="none", pad=2))
    ax.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="white", fontsize=8)

    # 3. Scatter actual vs predicted
    ax = sax(axes[2], "Actual vs Predicted (PM2.5 Test Set)", 10)
    ax.scatter(y_true, y_pred, s=2, alpha=0.2, color=C2, rasterized=True)
    lo = min(y_true.min(), y_pred.min()) - 0.3
    hi = max(y_true.max(), y_pred.max()) + 0.3
    ax.plot([lo, hi], [lo, hi], "w--", lw=0.8, label="Perfect fit")
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    ax.text(0.05, 0.93, f"Pearson r={corr:.3f}\npred std={y_pred.std():.2f}",
            transform=ax.transAxes, color="white", fontsize=8, va="top",
            bbox=dict(facecolor="#21262d", edgecolor="none", pad=2))
    ax.set_xlabel("Actual PM2.5  µg/m³")
    ax.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="white", fontsize=8)

    fig.suptitle(
        f"Multivariate XGBoost (48-h window) — PM2.5 +1h Forecast\n"
        f"Test MAE={mae:.4f} µg/m³   RMSE={rmse:.4f} µg/m³   R²={r2:.4f}",
        color="white", fontsize=11, y=1.05, fontweight="bold",
    )
    plt.tight_layout()
    
    out_path = "xgboost_multivariate_report.png"
    plt.savefig(out_path, dpi=140, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"\n[PLOT] Saved → {out_path}")

plot_xgboost(df, y_true_pm25, y_pred_pm25, split_idx)
print("\nDone ✓")