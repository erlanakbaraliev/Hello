"""PM2.5-only safety bands for forecasts and UI.

These cutpoints are used for PM2.5 forecast safety labeling.
They are a simple health-style banding of PM2.5 (µg/m³), not a full multi-pollutant
AQI index.
"""

from __future__ import annotations

import pandas as pd

# PM2.5 safety thresholds.
PM25_LOW_MAX = 12.0
PM25_MOD_MAX = 35.4


def pm25_safety_level(pm25: float) -> str:
    """Return ``Low``, ``Moderate``, or ``High`` from predicted or observed PM2.5."""
    v = float(pm25)
    if v <= PM25_LOW_MAX:
        return "Low"
    if v <= PM25_MOD_MAX:
        return "Moderate"
    return "High"


def add_pm25_safety_column(df: pd.DataFrame, pred_col: str = "pm2_5_pred") -> pd.DataFrame:
    """Append ``pm25_safety_level`` from the PM2.5 prediction column."""
    out = df.copy()
    out["pm25_safety_level"] = out[pred_col].astype(float).map(pm25_safety_level)
    return out
