"""HelpDesk chat page powered by Gemini."""

from __future__ import annotations

import invoke_bootstrap  # noqa: F401 — ensures project root on sys.path

import streamlit as st

from back_end import page_helpdesk as ph
from front_end.ui import configure_authenticated_workspace_page, current_user_id, hero_title
from back_end.helpdesk_gemini import get_gemini_api_key

configure_authenticated_workspace_page(page_title="HelpDesk · Urban air quality")
hero_title("HelpDesk", "Ask questions about air quality and get Gemini-powered answers.")

uid = current_user_id()
if uid is None:
    st.stop()
api_key = get_gemini_api_key(st.secrets)
ph.render_helpdesk_page(uid, api_key)
