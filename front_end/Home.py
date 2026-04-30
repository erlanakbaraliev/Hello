"""Home entry point: page configuration, authentication, and project overview."""

from __future__ import annotations

import invoke_bootstrap  # noqa: F401 — ensures project root on sys.path

import streamlit as st

from front_end.logging_utils import setup_logging
from front_end.ui import (
    configure_public_shell,
    hero_title,
    persist_session,
    prose_block,
    render_sidebar_branding,
    render_sidebar_workspace_nav,
    render_signed_in_sidebar_account,
    restore_session,
)
from db import AccountLockedError, add_user, init_db, verify_user

setup_logging()
configure_public_shell(page_title="Home · Urban air quality")
init_db()
if "user_id" not in st.session_state:
    st.session_state["user_id"] = None
restore_session()

render_sidebar_branding()
render_sidebar_workspace_nav()

if st.session_state.get("user_id") is None:
    hero_title(
        "Urban air quality platform",
        "Explore pollutant dashboards, run forecasts, and track prediction history.",
    )

    auth, intro = st.columns([0.9, 1.1], gap="large")
    with auth:
        st.markdown("##### Sign in")
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="your.username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log in", type="primary", use_container_width=True)
            if submitted:
                try:
                    uid = verify_user(username, password)
                except AccountLockedError as exc:
                    st.error(str(exc))
                else:
                    if uid is None:
                        st.error("Invalid username or password.")
                    else:
                        st.session_state["user_id"] = uid
                        st.session_state["username"] = username.strip()
                        persist_session(uid)
                        st.rerun()

        with st.expander("Create an account"):
            with st.form("register_form"):
                nu = st.text_input("Username", key="reg_u", placeholder="choose a unique name")
                pw1 = st.text_input("Password", type="password", key="reg_p")
                pw2 = st.text_input("Confirm password", type="password", key="reg_p2")
                reg_sub = st.form_submit_button("Register", use_container_width=True)
                if reg_sub:
                    if pw1 != pw2:
                        st.error("Passwords do not match.")
                    else:
                        try:
                            uid = add_user(nu, pw1)
                            st.session_state["user_id"] = uid
                            st.session_state["username"] = nu.strip()
                            persist_session(uid)
                            st.success("Account created — logging you in.")
                            st.rerun()
                        except ValueError as exc:
                            st.error(str(exc))
                        except RuntimeError as exc:
                            st.error(str(exc))

    with intro:
        st.markdown(
            """
<div class="thesis-feature-card">
  <p class="card-title">What you get</p>
  <ul>
    <li><strong>Data Explorer</strong> — upload CSVs, view correlations & distributions</li>
    <li><strong>Forecasting</strong> — 168-hour PM₂.₅ predictions (LSTM / XGBoost)</li>
    <li><strong>Dashboard</strong> — live pollutant metrics and trend charts</li>
    <li><strong>History</strong> — browse and download past prediction runs</li>
    <li><strong>HelpDesk</strong> — Gemini-powered Q&A assistant</li>
  </ul>
</div>
            """,
            unsafe_allow_html=True,
        )

    st.stop()

render_signed_in_sidebar_account()

hero_title(
    "About air quality",
    "Context for fine particulate matter (PM₂.₅) and how this platform supports your analysis.",
)

prose_block("""
<p><strong>Fine particulate matter (PM<sub>2.5</sub>)</strong> denotes airborne particles
with aerodynamic diameters ≤ 2.5 µm. Sustained exposure is linked to cardiovascular and
pulmonary stress — making it a key metric in environmental health monitoring.</p>

<p>This platform compares <strong>ARIMA, LSTM, and XGBoost</strong> models on hourly
multivariate panels from urban monitoring networks. Open <strong>Data Explorer</strong>
for distribution views, or <strong>Predict future quality</strong> to generate a
one-week hourly forecast.</p>
""")
