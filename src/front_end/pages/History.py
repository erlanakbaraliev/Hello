"""History page for browsing saved prediction artifacts."""

from __future__ import annotations

from typing import Any

import invoke_bootstrap  # noqa: F401 — ensures project root on sys.path
import pandas as pd
import streamlit as st

from db import delete_history_entry, get_history_filtered
from front_end.ui import (
    configure_authenticated_workspace_page,
    current_user_id,
    empty_state,
    hero_title,
)

configure_authenticated_workspace_page(page_title="History · Air quality")
hero_title("History", "Review, download, and manage past prediction runs.")

uid = current_user_id()
if uid is None:
    st.stop()


def _blob_as_bytes(blob: Any) -> bytes:
    if blob is None:
        return b""
    if isinstance(blob, memoryview):
        return blob.tobytes()
    if isinstance(blob, (bytes, bytearray)):
        return bytes(blob)
    return bytes(blob)


def _dataset_csv_bytes(sel: dict[str, Any]) -> bytes:
    b = _blob_as_bytes(sel.get("dataset_csv_blob"))
    if b:
        return b
    res = sel.get("results") or {}
    hist = res.get("history")
    if hist:
        return pd.DataFrame(hist).to_csv(index=False).encode("utf-8")
    return b""


def _prediction_csv_bytes(sel: dict[str, Any]) -> bytes:
    b = _blob_as_bytes(sel.get("prediction_csv_blob"))
    if b:
        return b
    res = sel.get("results") or {}
    fc = res.get("forecast")
    if fc:
        return pd.DataFrame(fc).to_csv(index=False).encode("utf-8")
    return b""


# ── Filters ───────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
start = col1.date_input("Start date", value=None)
end = col2.date_input("End date", value=None)
model_filter = col3.selectbox("Model", ["All", "LSTM", "XGBoost"])

rows = get_history_filtered(
    uid,
    start_date=str(start) if start else None,
    end_date=str(end) if end else None,
    model_used=model_filter,
)

if not rows:
    empty_state("🕘", "No predictions yet", "Run a forecast to see your history here.")
    st.stop()

# ── Table ─────────────────────────────────────────────────────────────────────
table_rows = []
for item in rows:
    table_rows.append(
        {
            "id": item["id"],
            "Date": (item.get("upload_date") or item["timestamp"])[:16],
            "Model": item["model_used"],
        }
    )
hist_df = pd.DataFrame(table_rows).sort_values("Date", ascending=False)
st.dataframe(hist_df.drop(columns=["id"]), use_container_width=True, hide_index=True)

# ── Detail view ───────────────────────────────────────────────────────────────
_id_labels = {row["id"]: f"{row['Date']} · {row['Model']}" for _, row in hist_df.iterrows()}
selected_id = st.selectbox("Select entry", hist_df["id"].tolist(), format_func=lambda x: _id_labels.get(x, str(x)))
selected = next(x for x in rows if x["id"] == selected_id)

a_col, b_col = st.columns(2)
a_col.download_button(
    "⬇ Dataset CSV",
    data=_dataset_csv_bytes(selected),
    file_name=f"dataset_{selected_id}.csv",
    mime="text/csv",
    use_container_width=True,
)
b_col.download_button(
    "⬇ Predictions CSV",
    data=_prediction_csv_bytes(selected),
    file_name=f"predictions_{selected_id}.csv",
    mime="text/csv",
    use_container_width=True,
)

_, del_col = st.columns([3, 1])
if del_col.button("🗑 Delete entry", type="secondary", use_container_width=True):
    delete_history_entry(uid, int(selected_id))
    st.success("Entry deleted.")
    st.rerun()
