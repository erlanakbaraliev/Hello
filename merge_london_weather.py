#!/usr/bin/env python3
"""Merge temperature (LaqnData.csv) and wind_speed (LaqnData-3.csv) into london_2024.csv by hour."""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent
LONDON = ROOT / "data" / "london_2024.csv"
TEMP_SRC = ROOT / "data" / "other_datas" / "LaqnData.csv"
WIND_SRC = ROOT / "data" / "other_datas" / "LaqnData-3.csv"

CANONICAL_TAIL = ("temperature", "wind_speed")


def laqn_time_key(reading_dt: str) -> str:
    dt = datetime.strptime(reading_dt.strip(), "%d/%m/%Y %H:%M")
    return dt.strftime("%Y-%m-%d %H:%M:%S+00:00")


def load_laqn_values(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = laqn_time_key(row["ReadingDateTime"])
            raw = (row.get("Value") or "").strip()
            out[key] = str(float(raw)) if raw else ""
    return out


def _parse_float(s: str) -> Optional[float]:
    t = (s or "").strip()
    if not t:
        return None
    return float(t)


def _fmt_float(x: float) -> str:
    """Short decimal string without scientific noise for typical readings."""
    s = f"{x:.6f}".rstrip("0").rstrip(".")
    return s if s else "0"


def interpolate_hourly_gaps(values: list[str]) -> list[str]:
    """
    Fill missing values: interior gaps use linear steps between neighbours
    (one hour missing -> average of previous and next). Leading/trailing gaps
    use the nearest available value.
    """
    nums: list[Optional[float]] = [_parse_float(v) for v in values]
    n = len(nums)
    if n == 0:
        return values
    out: list[Optional[float]] = list(nums)
    i = 0
    while i < n:
        if out[i] is not None:
            i += 1
            continue
        start = i
        while i < n and out[i] is None:
            i += 1
        end = i
        left = out[start - 1] if start > 0 else None
        right = out[end] if end < n else None
        span = end - (start - 1)
        for k in range(start, end):
            if left is not None and right is not None:
                out[k] = left + (right - left) * (k - (start - 1)) / span
            elif left is not None:
                out[k] = left
            elif right is not None:
                out[k] = right
            else:
                out[k] = None
    return [_fmt_float(x) if x is not None else "" for x in out]


def read_london_deduped(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read CSV; duplicate header names keep first column only (avoids rerun doubling)."""
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        raw_header = next(reader)
        seen: set[str] = set()
        col_idx: list[int] = []
        header: list[str] = []
        for j, name in enumerate(raw_header):
            if name in seen:
                continue
            seen.add(name)
            col_idx.append(j)
            header.append(name)
        rows: list[dict[str, str]] = []
        for raw in reader:
            row = {header[i]: raw[col_idx[i]] for i in range(len(header))}
            rows.append(row)
    return header, rows


def canonical_fieldnames(header: list[str]) -> list[str]:
    """Single temperature, wind_speed at end; drop duplicates from header list."""
    h = [c for c in header if c not in CANONICAL_TAIL]
    return h + list(CANONICAL_TAIL)


def main() -> None:
    temp_map = load_laqn_values(TEMP_SRC)
    wind_map = load_laqn_values(WIND_SRC)

    header, rows = read_london_deduped(LONDON)
    fieldnames = canonical_fieldnames(header)

    empty_t = empty_w = 0
    temps: list[str] = []
    winds: list[str] = []
    for row in rows:
        t = row["time"].strip()
        row["temperature"] = temp_map.get(t, "")
        row["wind_speed"] = wind_map.get(t, "")
        if row["temperature"] == "":
            empty_t += 1
        if row["wind_speed"] == "":
            empty_w += 1
        temps.append(row["temperature"])
        winds.append(row["wind_speed"])

    temps_f = interpolate_hourly_gaps(temps)
    winds_f = interpolate_hourly_gaps(winds)
    filled_t = sum(1 for a, b in zip(temps, temps_f) if a == "" and b != "")
    filled_w = sum(1 for a, b in zip(winds, winds_f) if a == "" and b != "")

    for row, tv, wv in zip(rows, temps_f, winds_f):
        row["temperature"] = tv
        row["wind_speed"] = wv

    with LONDON.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {LONDON} ({len(rows)} rows).")
    print(
        f"Before interpolation — blank temperature: {empty_t}; wind_speed: {empty_w} "
        f"(empty Value in Laqn exports)."
    )
    print(f"Filled by interpolation — temperature: {filled_t}; wind_speed: {filled_w}.")


if __name__ == "__main__":
    main()
