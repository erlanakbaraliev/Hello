"""History page for browsing saved prediction artifacts."""

from __future__ import annotations

import invoke_bootstrap  # noqa: F401 — ensures project root on sys.path

import io

import pandas as pd
import plotly.express as px
import streamlit as st

from front_end import charts as ch
from front_end.ui import configure_authenticated_workspace_page, current_user_id, empty_state, hero_title
from db import delete_history_entry, get_history_filtered

configure_authenticated_workspace_page(page_title="History · Urban air quality")
hero_title("History", "Review, download, and manage past prediction runs.")

uid = current_user_id()
if uid is None:
    st.stop()

# ── Filters ───────────────────────────────────────────────────────────────────
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
    empty_state("🕘", "No predictions yet", "Run a forecast to see your history here.")
    st.stop()

# ── Table ─────────────────────────────────────────────────────────────────────
table_rows = []
for item in rows:
    table_rows.append(
        {
            "id": item["id"],
            "Dataset": item.get("dataset_name") or "—",
            "Date": (item.get("upload_date") or item["timestamp"])[:16],
            "Model": item["model_used"],
            "Avg AQI": item.get("avg_aqi"),
            "Risk": item.get("risk_level"),
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
    data=selected.get("dataset_csv_blob") or b"",
    file_name=f"dataset_{selected_id}.csv",
    mime="text/csv",
    use_container_width=True,
)
b_col.download_button(
    "⬇ Predictions CSV",
    data=selected.get("prediction_csv_blob") or b"",
    file_name=f"predictions_{selected_id}.csv",
    mime="text/csv",
    use_container_width=True,
)

with st.expander("Details & forecast chart"):
    st.json(selected.get("details", {}))
    pred_blob = selected.get("prediction_csv_blob")
    if pred_blob:
        pdf = pd.read_csv(io.BytesIO(pred_blob))
        if "predicted_pm2_5" in pdf.columns:
            fig = px.line(
                pdf, x="date", y="predicted_pm2_5",
                title="Saved forecast",
                template=ch.plotly_template(),
                color_discrete_sequence=["#0d9488"],
            )
            fig.update_layout(
                font=dict(family=ch.PLOT_FONT, size=12, color=ch.plotly_label_color()),
                paper_bgcolor=ch.plotly_paper_bg(),
                plot_bgcolor=ch.plotly_plot_bg(),
                margin=dict(l=48, r=24, t=48, b=48),
                height=380,
            )
            st.plotly_chart(fig, use_container_width=True)
    st.divider()
    _, del_col = st.columns([3, 1])
    if del_col.button("🗑 Delete entry", type="secondary", use_container_width=True):
        delete_history_entry(uid, int(selected_id))
        st.success("Entry deleted.")
        st.rerun()
