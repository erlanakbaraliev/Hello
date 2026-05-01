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
MODELS_DIR = _PROJECT_ROOT / "models"
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

# A4-friendly single-column width (~160 mm @ 200 dpi when embedded)
_DOC_DPI = 200
_DOC_FIGSIZE = (6.7, 3.9)


def plot_xgboost(df, y_true, y_pred, split_idx):
    out_dir = Path(__file__).resolve().parent
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

    # 1 — Full year
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
        "Multivariate XGBoost — study period and chronological split",
        color="white",
        fontsize=12,
        y=1.02,
        fontweight="bold",
    )
    saved.append(save_doc(fig, "xgboost_multivariate_doc_01_full_year_pm25.png"))

    # 2 — Last N h
    nb = min(600, len(y_true))
    fig, ax = new_fig()
    sax(ax, f"Test set — last {nb} h: actual vs XGBoost (PM2.5)")
    ax.plot(y_true[-nb:], color=CT, lw=1.0, label="Actual")
    ax.plot(y_pred[-nb:], color=C2, lw=1.0, alpha=0.88, label="XGBoost prediction")
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
    saved.append(save_doc(fig, "xgboost_multivariate_doc_02_test_predictions_pm25.png"))

    # 3 — Scatter
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
    saved.append(save_doc(fig, "xgboost_multivariate_doc_03_actual_vs_predicted_pm25.png"))

    print("\n[PLOT] Documentation figures (XGBoost, 200 dpi, A4-friendly width):")
    for p in saved:
        print(f"  • {p}")


plot_xgboost(df, y_true_pm25, y_pred_pm25, split_idx)
print("\nDone ✓")