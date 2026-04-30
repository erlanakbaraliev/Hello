"""HelpDesk chat page powered by Gemini."""

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

from app import page_helpdesk as ph
from app.ui import configure_authenticated_workspace_page, current_user_id, hero_title
from helpdesk_gemini import get_gemini_api_key

configure_authenticated_workspace_page(page_title="HelpDesk · Urban air quality")
hero_title("HelpDesk", "Ask questions and receive Gemini-powered assistance.")

uid = current_user_id()
assert uid is not None
api_key = get_gemini_api_key(st.secrets)
ph.render_helpdesk_page(uid, api_key)
