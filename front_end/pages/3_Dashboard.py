"""Dashboard page with current air-quality summary and trends."""

from __future__ import annotations

import invoke_bootstrap  # noqa: F401 — ensures project root on sys.path

import streamlit as st

from front_end import charts as ch
from back_end import page_dashboard as pdash
from front_end.ui import configure_authenticated_workspace_page, hero_title

configure_authenticated_workspace_page(page_title="Dashboard · Urban air quality")
hero_title("Dashboard", "Current air-quality state and recent pollutant trends.")

prepared_df = st.session_state.get("prepared_df")
if prepared_df is None:
    with st.spinner("Loading air quality data..."):
        prepared_df = pdash.load_default_prepared()

pdash.render_dashboard_charts(prepared_df, ch)
