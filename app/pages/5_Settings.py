"""User settings page: profile, preferences, display, notifications, and security."""

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

from app.ui import configure_authenticated_workspace_page, current_user_id, hero_title
from database import (
    change_password,
    delete_user_account,
    get_user_profile,
    get_user_settings,
    update_user_profile,
    upsert_user_settings,
)

configure_authenticated_workspace_page(page_title="Settings · Urban air quality")
hero_title(
    "Settings", "Manage your profile, preferences, charts, notifications, and account security."
)

uid = current_user_id()
assert uid is not None

try:
    profile = get_user_profile(uid)
    settings = get_user_settings(uid)
except Exception as exc:
    st.error(f"Could not load settings: {exc}")
    st.stop()

t_profile, t_pref, t_charts, t_notify, t_security = st.tabs(
    ["Profile", "Preferences", "Charts", "Notifications", "Security"]
)

# ── Profile ──────────────────────────────────────────────────────────────────
with t_profile:
    st.subheader("Profile")
    st.text_input(
        "Username",
        value=profile["username"],
        disabled=True,
        help="Username cannot be changed after registration.",
    )
    email = st.text_input("Email (optional)", value=profile.get("email") or "")
    if st.button("Save profile", key="save_profile"):
        try:
            update_user_profile(uid, email.strip() or None)
            st.success("Profile updated.")
        except Exception as exc:
            st.error(f"Could not update profile: {exc}")

# ── Preferences ───────────────────────────────────────────────────────────────
with t_pref:
    st.subheader("Forecast preferences")
    default_model = st.selectbox(
        "Default model",
        ["ARIMA", "LSTM", "XGBoost"],
        index=(
            ["ARIMA", "LSTM", "XGBoost"].index(settings["default_model"])
            if settings["default_model"] in ["ARIMA", "LSTM", "XGBoost"]
            else 0
        ),
    )
    granularity = st.selectbox(
        "Time granularity",
        ["hourly", "daily"],
        index=0 if settings["time_granularity"] == "hourly" else 1,
    )
    if st.button("Save preferences", key="save_prefs"):
        try:
            upsert_user_settings(
                uid,
                default_model,
                granularity,
                "light",
                settings["preferred_chart"],
                bool(settings["high_pollution_alerts"]),
            )
            st.success("Preferences saved.")
        except Exception as exc:
            st.error(f"Could not save preferences: {exc}")

# ── Charts ────────────────────────────────────────────────────────────────────
with t_charts:
    st.subheader("Chart preferences")
    chart = st.selectbox(
        "Preferred chart type",
        ["line", "area", "bar"],
        index=(
            ["line", "area", "bar"].index(settings["preferred_chart"])
            if settings["preferred_chart"] in ["line", "area", "bar"]
            else 0
        ),
    )
    if st.button("Save chart preferences", key="save_display"):
        try:
            upsert_user_settings(
                uid,
                settings["default_model"],
                settings["time_granularity"],
                "light",
                chart,
                bool(settings["high_pollution_alerts"]),
            )
            st.success("Chart preferences saved.")
        except Exception as exc:
            st.error(f"Could not save chart preferences: {exc}")

# ── Notifications ─────────────────────────────────────────────────────────────
with t_notify:
    st.subheader("Notifications")
    alerts = st.toggle(
        "Enable high pollution alerts",
        value=bool(settings["high_pollution_alerts"]),
        help="Show a warning banner when predicted PM2.5 exceeds the 'High' threshold (>35.4 µg/m³).",
    )
    if st.button("Save notification settings", key="save_notify"):
        try:
            upsert_user_settings(
                uid,
                settings["default_model"],
                settings["time_granularity"],
                "light",
                settings["preferred_chart"],
                alerts,
            )
            st.success("Notification settings saved.")
        except Exception as exc:
            st.error(f"Could not save notification settings: {exc}")

# ── Security ──────────────────────────────────────────────────────────────────
with t_security:
    st.subheader("Change password")
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
                st.success("Password updated successfully.")
            except ValueError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Could not update password: {exc}")

    st.divider()
    st.subheader("Delete account")
    st.warning(
        "This permanently removes your account, all prediction history, and chat sessions. "
        "This action cannot be undone."
    )
    confirm_text = st.text_input("Type DELETE to confirm account removal")
    if st.button("Delete my account", type="primary"):
        if confirm_text != "DELETE":
            st.error("Type DELETE exactly to confirm.")
        else:
            try:
                delete_user_account(uid)
                for key in list(st.session_state.keys()):
                    st.session_state.pop(key, None)
                st.success("Account deleted. You have been logged out.")
                st.rerun()
            except Exception as exc:
                st.error(f"Could not delete account: {exc}")
