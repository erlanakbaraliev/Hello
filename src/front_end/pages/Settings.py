"""User settings page: profile, default forecast model, and security."""

from __future__ import annotations

import invoke_bootstrap  # noqa: F401 — ensures project root on sys.path

import streamlit as st

from front_end.ui import configure_authenticated_workspace_page, current_user_id, hero_title
from db import (
    change_password,
    delete_user_account,
    get_user_profile,
    get_user_settings,
    update_user_profile,
    upsert_user_settings,
)

configure_authenticated_workspace_page(page_title="Settings · Air quality")
hero_title("Settings", "Profile, default forecast model, and security.")

uid = current_user_id()
if uid is None:
    st.stop()

try:
    profile = get_user_profile(uid)
    settings = get_user_settings(uid)
except Exception as exc:
    st.error(f"Could not load settings: {exc}")
    st.stop()

# ── General settings (single form) ───────────────────────────────────────────
with st.form("settings_form"):
    t_profile, t_pref = st.tabs(["Profile", "Preferences"])

    with t_profile:
        st.text_input(
            "Username",
            value=profile["username"],
            disabled=True,
            help="Cannot be changed after registration.",
        )
        email = st.text_input("Email (optional)", value=profile.get("email") or "", key="f_email")

    with t_pref:
        _models = ["LSTM", "XGBoost"]
        default_model = st.selectbox(
            "Default model",
            _models,
            key="f_model",
            index=(
                _models.index(settings["default_model"])
                if settings["default_model"] in _models
                else 0
            ),
            help="Used as the initial model on the Forecast page.",
        )

    save_all = st.form_submit_button("Save all settings", type="primary", use_container_width=False)

if save_all:
    try:
        update_user_profile(uid, email.strip() or None)
        upsert_user_settings(uid, default_model)
        st.success("All settings saved.")
    except Exception as exc:
        st.error(f"Could not save settings: {exc}")

# ── Security (separate) ──────────────────────────────────────────────────────
st.divider()
st.markdown("### Security")

st.markdown("##### Change password")
with st.form("change_password_form"):
    current_pw = st.text_input("Current password", type="password")
    new_pw = st.text_input("New password", type="password")
    confirm_pw = st.text_input("Confirm new password", type="password")
    submitted = st.form_submit_button("Update password", type="primary")
if submitted:
    if not current_pw or not new_pw:
        st.error("All password fields are required.")
    elif new_pw != confirm_pw:
        st.error("New password and confirmation do not match.")
    else:
        try:
            change_password(uid, current_pw, new_pw)
            st.success("Password updated.")
        except ValueError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Could not update password: {exc}")

st.divider()
st.markdown("##### Delete account")
st.warning("Permanently removes your account and prediction history. Cannot be undone.")
confirm_text = st.text_input("Type DELETE to confirm")
if st.button("Delete my account", type="secondary"):
    if confirm_text != "DELETE":
        st.error("Type DELETE exactly to confirm.")
    else:
        try:
            delete_user_account(uid)
            for key in list(st.session_state.keys()):
                st.session_state.pop(key, None)
            st.success("Account deleted.")
            st.rerun()
        except Exception as exc:
            st.error(f"Could not delete account: {exc}")
