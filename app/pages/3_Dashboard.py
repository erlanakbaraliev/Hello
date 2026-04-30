"""Dashboard page with current air-quality summary and trends."""

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

import streamlit as st

from app import charts as ch
from app import page_dashboard as pdash
from app.ui import configure_authenticated_workspace_page, hero_title

configure_authenticated_workspace_page(page_title="Dashboard · Urban air quality")
hero_title("Dashboard", "Current air-quality state and recent pollutant behavior.")

prepared_df = st.session_state.get("prepared_df")
if prepared_df is None:
    prepared_df = pdash.load_default_prepared()

pdash.render_dashboard_charts(prepared_df, ch)
