"""Compatibility launcher for the modular Streamlit application."""

from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import streamlit as st

from app.logging_utils import setup_logging
from app.ui import page_icon_path

setup_logging()
st.set_page_config(
    page_title="Launcher · Urban air quality",
    page_icon=page_icon_path(),
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("Urban Air Quality Predictor")
st.info(
    "This entry point is deprecated. Use the modular app instead: "
    "`PYTHONPATH=. streamlit run app/Home.py` from the `thesis/` directory."
)
if st.button("Open modular app", type="primary", use_container_width=True):
    st.switch_page("app/Home.py")
