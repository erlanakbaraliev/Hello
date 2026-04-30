from __future__ import annotations

import pandas as pd

from data_processing import combine_sources, ensure_expected_schema, validate_and_prepare


def test_ensure_expected_schema_fills_required_columns() -> None:
    source = pd.DataFrame({"time": ["2026-01-01 00:00:00"], "pm2_5": [12.5]})
    out = ensure_expected_schema(source, source="uploaded")
    assert "city" in out.columns
    assert "pm10" in out.columns
    assert out.loc[0, "source"] == "uploaded"


def test_validate_and_prepare_returns_daily_series() -> None:
    source = pd.DataFrame(
        {
            "time": ["2026-01-01 00:00:00", "2026-01-01 12:00:00", "2026-01-02 00:00:00"],
            "pm10": [10, 12, 14],
            "pm2_5": [5, 7, 9],
            "carbon_monoxide": [1, 1, 1],
            "nitrogen_dioxide": [3, 3, 3],
            "sulphur_dioxide": [0.1, 0.1, 0.1],
            "ozone": [2, 2, 2],
            "latitude": [51.5, 51.5, 51.5],
            "longitude": [-0.1, -0.1, -0.1],
            "city": ["london", "london", "london"],
        }
    )
    prepared = validate_and_prepare(source)
    assert isinstance(prepared.index, pd.DatetimeIndex)
    assert "pm2_5" in prepared.columns
    assert len(prepared) == 2


def test_combine_sources_rejects_empty_inputs() -> None:
    try:
        combine_sources(None, None)
    except ValueError as exc:
        assert "No source data" in str(exc)
    else:
        raise AssertionError("Expected combine_sources to raise ValueError")
