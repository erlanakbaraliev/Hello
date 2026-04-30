"""Home entry point: page configuration, authentication, and project overview."""

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

from app.logging_utils import setup_logging
from app.ui import (
    configure_public_shell,
    hero_title,
    prose_block,
    render_sidebar_branding,
    render_sidebar_workspace_nav,
    render_signed_in_sidebar_account,
)
from database import AccountLockedError, add_user, init_db, verify_user

setup_logging()
configure_public_shell(page_title="Home · Urban air quality")
init_db()
if "user_id" not in st.session_state:
    st.session_state["user_id"] = None

if st.session_state.get("user_id") is None:
    hero_title(
        "Urban air quality platform",
        "Sign in to access exploratory dashboards, forecasting tools, and prediction "
        "history stored for your account.",
    )

    intro, auth = st.columns([1.05, 0.95], gap="large")
    with intro:
        prose_block("""
<p style="margin-top:0;">
This workspace supports thesis work on <strong>hourly urban air-quality</strong> panels:
multivariate pollutants on a common timeline, classical and machine-learning baselines,
and reproducible logging of forecast runs.
</p>
<p>
Use the sidebar after authentication to open the <strong>Data Explorer</strong> or the
<strong>Predict future quality</strong> workflow.
</p>
            """)

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
                        st.rerun()

        st.markdown("")
        st.markdown("##### Create an account")
        with st.form("register_form"):
            nu = st.text_input("New username", key="reg_u", placeholder="choose a unique name")
            pw1 = st.text_input("New password", type="password", key="reg_p")
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
                        st.success("Account created. You are now logged in.")
                        st.rerun()
                    except ValueError as exc:
                        st.error(str(exc))
                    except RuntimeError as exc:
                        st.error(str(exc))

    st.stop()

render_sidebar_branding()
render_sidebar_workspace_nav()
render_signed_in_sidebar_account()

hero_title(
    "About air quality",
    "Context for fine particulate matter (PM₂.₅) and how this platform supports your analysis.",
)

prose_block("""
<p><strong>Fine particulate matter (PM<sub>2.5</sub>)</strong> denotes airborne particles
with aerodynamic diameters of roughly 2.5 micrometres or less. Because they can penetrate
deep into the respiratory tract, sustained exposure is linked to cardiovascular and
pulmonary stress and is routinely monitored in environmental health research.</p>

<p>This application is built around <strong>hourly</strong> multivariate panels typical of
urban networks: several gas and particle species aligned on one timeline. The modelling
track in the thesis compares classical seasonal ARIMA, recurrent neural networks, and
gradient-boosted trees—each encoding temporal structure in a different way.</p>

<p>Open <strong>Data Explorer</strong> for distribution and correlation views, or
<strong>Predict future quality</strong> to upload recent observations and obtain a
<strong>168-hour</strong> (one week, hourly) PM<sub>2.5</sub> forecast from the trained baselines.</p>
    """)
