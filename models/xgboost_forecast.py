from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb


DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "london_2025_eval.csv"
ARTIFACT_PATH = Path(__file__).resolve().parent / "xgboost_multivariate_artifacts.joblib"
OUT_DIR = Path(__file__).resolve().parent / "forecast_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    h = df.index.hour
    dow = df.index.dayofweek
    m = df.index.month
    out = df.copy()
    out["h_sin"] = np.sin(2 * np.pi * h / 24)
    out["h_cos"] = np.cos(2 * np.pi * h / 24)
    out["h2_sin"] = np.sin(4 * np.pi * h / 24)
    out["h2_cos"] = np.cos(4 * np.pi * h / 24)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    out["mon_sin"] = np.sin(2 * np.pi * m / 12)
    out["mon_cos"] = np.cos(2 * np.pi * m / 12)
    return out


def seasonal_naive_next(history_orig: np.ndarray, col_idx: int, daily_lag: int = 24) -> float:
    if len(history_orig) >= daily_lag:
        return float(history_orig[-daily_lag, col_idx])
    return float(history_orig[-1, col_idx])


def main():
    artifacts = joblib.load(ARTIFACT_PATH)
    window = int(artifacts["window"])
    target_cols = artifacts["target_cols"]
    weather_cols = artifacts["weather_cols"]
    feat_cols = artifacts["feat_cols"]
    tgt_idxs = artifacts["tgt_idxs"]
    mu_all = np.asarray(artifacts["mu_all"], dtype=np.float32)
    sigma_all = np.asarray(artifacts["sigma_all"], dtype=np.float32)
    boosters = artifacts["boosters"]

    df = pd.read_csv(DATA_PATH, parse_dates=["time"])
    df = df.set_index("time").sort_index()
    base_cols = target_cols + weather_cols
    df = df[base_cols].apply(pd.to_numeric, errors="coerce")
    df = add_time_features(df)
    df = df[feat_cols].interpolate(limit_direction="both").ffill().bfill()

    jan_mask = df.index.month == 1
    feb_mask = df.index.month == 2
    jan_df = df.loc[jan_mask]
    feb_df = df.loc[feb_mask]
    if len(jan_df) < window or len(feb_df) == 0:
        raise ValueError("Need January history and February rows in london_2025_eval.csv")

    sigma_safe = np.where(sigma_all < 1e-8, 1.0, sigma_all)
    all_scaled = ((df.values - mu_all) / sigma_safe).astype(np.float32)
    frame_scaled = pd.DataFrame(all_scaled, index=df.index, columns=feat_cols)

    history = frame_scaled.loc[jan_mask].tail(window).to_numpy(dtype=np.float32)
    history_orig = (history * sigma_safe + mu_all).astype(np.float32)
    pred_rows_scaled = []
    pred_weather_orig = []
    idx_map = {c: i for i, c in enumerate(feat_cols)}

    for ts in feb_df.index:
        x2d = history.reshape(1, -1)
        dmat = xgb.DMatrix(x2d)
        pred_targets_scaled = np.array([b.predict(dmat)[0] for b in boosters], dtype=np.float32)

        next_row_scaled = np.zeros(len(feat_cols), dtype=np.float32)
        next_row_scaled[tgt_idxs] = pred_targets_scaled
        next_row_orig = np.zeros(len(feat_cols), dtype=np.float32)
        next_row_orig[tgt_idxs] = pred_targets_scaled * sigma_safe[tgt_idxs] + mu_all[tgt_idxs]

        # Recursively forecast weather with a 24-hour seasonal naive step.
        for col in weather_cols:
            cidx = idx_map[col]
            w_val = seasonal_naive_next(history_orig, cidx, daily_lag=24)
            next_row_orig[cidx] = w_val
            next_row_scaled[cidx] = (w_val - mu_all[cidx]) / sigma_safe[cidx]
        pred_weather_orig.append([next_row_orig[idx_map[c]] for c in weather_cols])

        # Time features are known in future from timestamp.
        h = ts.hour
        dow = ts.dayofweek
        m = ts.month
        time_vals = {
            "h_sin": np.sin(2 * np.pi * h / 24),
            "h_cos": np.cos(2 * np.pi * h / 24),
            "h2_sin": np.sin(4 * np.pi * h / 24),
            "h2_cos": np.cos(4 * np.pi * h / 24),
            "dow_sin": np.sin(2 * np.pi * dow / 7),
            "dow_cos": np.cos(2 * np.pi * dow / 7),
            "mon_sin": np.sin(2 * np.pi * m / 12),
            "mon_cos": np.cos(2 * np.pi * m / 12),
        }
        for col, val in time_vals.items():
            cidx = idx_map[col]
            next_row_orig[cidx] = float(val)
            next_row_scaled[cidx] = (float(val) - mu_all[cidx]) / sigma_safe[cidx]

        pred_rows_scaled.append(next_row_scaled.copy())
        history = np.vstack([history[1:], next_row_scaled])
        history_orig = np.vstack([history_orig[1:], next_row_orig])

    pred_rows_scaled = np.vstack(pred_rows_scaled)
    pred_targets_orig = pred_rows_scaled[:, tgt_idxs] * sigma_safe[tgt_idxs] + mu_all[tgt_idxs]
    actual_targets_orig = feb_df[target_cols].to_numpy(dtype=np.float32)
    pred_weather_orig = np.array(pred_weather_orig, dtype=np.float32)
    actual_weather_orig = feb_df[weather_cols].to_numpy(dtype=np.float32)

    pred_df = pd.DataFrame(pred_targets_orig, index=feb_df.index, columns=[f"{c}_pred" for c in target_cols])
    act_df = pd.DataFrame(actual_targets_orig, index=feb_df.index, columns=[f"{c}_actual" for c in target_cols])
    pred_w_df = pd.DataFrame(pred_weather_orig, index=feb_df.index, columns=[f"{c}_pred" for c in weather_cols])
    act_w_df = pd.DataFrame(actual_weather_orig, index=feb_df.index, columns=[f"{c}_actual" for c in weather_cols])
    out_df = pd.concat([pred_df, act_df, pred_w_df, act_w_df], axis=1).reset_index().rename(columns={"index": "time"})

    out_csv = OUT_DIR / "xgboost_february_forecast.csv"
    out_df.to_csv(out_csv, index=False)

    pm_col = "pm2_5"
    pm_actual = act_df[f"{pm_col}_actual"].values
    pm_pred = pred_df[f"{pm_col}_pred"].values
    mae = float(np.mean(np.abs(pm_actual - pm_pred)))
    rmse = float(np.sqrt(np.mean((pm_actual - pm_pred) ** 2)))

    plt.figure(figsize=(12, 5))
    plt.plot(feb_df.index, pm_actual, label="Actual PM2.5", linewidth=1.4)
    plt.plot(feb_df.index, pm_pred, label="Forecast PM2.5 (XGBoost)", linewidth=1.4, alpha=0.9)
    plt.title(f"XGBoost February Forecast (MAE={mae:.3f}, RMSE={rmse:.3f})")
    plt.xlabel("Time")
    plt.ylabel("PM2.5")
    plt.legend()
    plt.tight_layout()
    out_png = OUT_DIR / "xgboost_february_forecast_pm2_5.png"
    plt.savefig(out_png, dpi=140)
    plt.close()

    print(f"Saved predictions: {out_csv}")
    print(f"Saved plot: {out_png}")
    print(f"Forecast horizon rows: {len(feb_df)}")


if __name__ == "__main__":
    main()
