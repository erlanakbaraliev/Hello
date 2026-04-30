"""History page for browsing saved prediction artifacts."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_here = Path(__file__).resolve()
_bootstrap = (
    _here.parent / "invoke_bootstrap.py"
    if (_here.parent / "invoke_bootstrap.py").is_file()
    else _here.parent.parent / "invoke_bootstrap.py"
)
_bs_spec = importlib.util.spec_from_file_location("_streamlit_invoke_bootstrap", _bootstrap)
if _bs_spec is None or _bs_spec.loader is None:
    raise RuntimeError("invoke_bootstrap: could not load spec")
_bs_mod = importlib.util.module_from_spec(_bs_spec)
_bs_spec.loader.exec_module(_bs_mod)

import io

import pandas as pd
import plotly.express as px
import streamlit as st

from app.ui import configure_authenticated_workspace_page, current_user_id, hero_title
from database import delete_history_entry, get_history_filtered

configure_authenticated_workspace_page(page_title="History · Urban air quality")
hero_title("History", "Review, download, and manage past prediction runs.")

uid = current_user_id()
assert uid is not None

col1, col2, col3 = st.columns(3)
start = col1.date_input("Start date", value=None)
end = col2.date_input("End date", value=None)
model_filter = col3.selectbox("Model", ["All", "ARIMA", "LSTM", "XGBoost"])

rows = get_history_filtered(
    uid,
    start_date=str(start) if start else None,
    end_date=str(end) if end else None,
    model_used=model_filter,
)
if not rows:
    st.info("No prediction history yet.")
    st.stop()

table_rows = []
for item in rows:
    table_rows.append(
        {
            "id": item["id"],
            "dataset_name": item.get("dataset_name") or "n/a",
            "upload_date": item.get("upload_date") or item["timestamp"],
            "timestamp": item["timestamp"],
            "model": item["model_used"],
            "avg_aqi": item.get("avg_aqi"),
            "risk_level": item.get("risk_level"),
        }
    )
hist_df = pd.DataFrame(table_rows).sort_values("timestamp", ascending=False)
st.dataframe(hist_df, use_container_width=True)
selected_id = st.selectbox("Select history entry", hist_df["id"].tolist())
selected = next(x for x in rows if x["id"] == selected_id)

a_col, b_col, c_col = st.columns(3)
a_col.download_button(
    "Download dataset",
    data=selected.get("dataset_csv_blob") or b"",
    file_name=f"dataset_{selected_id}.csv",
    mime="text/csv",
)
b_col.download_button(
    "Download predictions",
    data=selected.get("prediction_csv_blob") or b"",
    file_name=f"predictions_{selected_id}.csv",
    mime="text/csv",
)
if c_col.button("Delete entry"):
    delete_history_entry(uid, int(selected_id))
    st.success("Entry deleted.")
    st.rerun()

with st.expander("View details"):
    st.json(selected.get("details", {}))
    pred_blob = selected.get("prediction_csv_blob")
    if pred_blob:
        pdf = pd.read_csv(io.BytesIO(pred_blob))
        if "predicted_pm2_5" in pdf.columns:
            st.plotly_chart(
                px.line(pdf, x="date", y="predicted_pm2_5", title="Saved forecast"),
                use_container_width=True,
            )
