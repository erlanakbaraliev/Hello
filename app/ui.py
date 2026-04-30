"""Shared Streamlit styling: typography, spacing, and light-theme component polish."""

from __future__ import annotations

import html
import logging
from pathlib import Path

import streamlit as st

logger = logging.getLogger(__name__)
_THEME_CSS_PATH = Path(__file__).resolve().parent / "static" / "theme.css"

AUTH_GUARD_MESSAGE = "Please sign in on the Home page to access this workspace."

# Multipage entry is `streamlit run app/Home.py`; Home sits next to `pages/` in `app/`.
_HOME_PAGE = "Home.py"
# ``st.switch_page`` uses rerun internally; calling it from ``on_click`` is a callback, where
# ``st.rerun()`` is disallowed and logs "Calling st.rerun() within a callback is a no-op."
_DEFER_SWITCH_HOME_AFTER_SIGN_OUT = "_thesis_deferred_switch_home_after_sign_out"


def sign_out() -> None:
    """Clear auth; navigation runs on the next script pass (not inside the click callback)."""
    st.session_state.pop("user_id", None)
    st.session_state.pop("username", None)
    st.session_state[_DEFER_SWITCH_HOME_AFTER_SIGN_OUT] = True


def _drain_deferred_switch_home_after_sign_out() -> None:
    if not st.session_state.pop(_DEFER_SWITCH_HOME_AFTER_SIGN_OUT, False):
        return
    _go_home_after_logout_query()


def _go_home_after_logout_query() -> None:
    try:
        st.switch_page(_HOME_PAGE)
    except Exception:
        st.rerun()


def current_user_id() -> int | None:
    """Signed-in user id from session state, or None if anonymous or invalid."""
    raw = st.session_state.get("user_id")
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _theme_css_text() -> str:
    try:
        return _THEME_CSS_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not load theme CSS from %s: %s", _THEME_CSS_PATH, exc)
        return ""


def configure_public_shell(
    *,
    page_title: str,
    layout: str = "wide",
    initial_sidebar_state: str = "expanded",
    page_icon: str | None = None,
) -> None:
    """First Streamlit calls for the Home page: config, logout query handling, and theme.

    Must be the first Streamlit API usage in the script (after imports), before other widgets.
    """
    st.set_page_config(
        page_title=page_title,
        page_icon=page_icon or page_icon_path(),
        layout=layout,
        initial_sidebar_state=initial_sidebar_state,
    )
    consume_logout_query()
    apply_theme()


def configure_authenticated_workspace_page(
    *,
    page_title: str,
    layout: str = "wide",
    initial_sidebar_state: str = "expanded",
    page_icon: str | None = None,
) -> None:
    """Page config, logout handling, auth guard, theme, and workspace sidebar for signed-in users.

    Must be the first Streamlit API usage in the script (after imports). Stops with an error if
    the viewer is not signed in (after ``consume_logout_query`` may have cleared the session).
    """
    st.set_page_config(
        page_title=page_title,
        page_icon=page_icon or page_icon_path(),
        layout=layout,
        initial_sidebar_state=initial_sidebar_state,
    )
    consume_logout_query()
    if current_user_id() is None:
        st.error(AUTH_GUARD_MESSAGE)
        st.stop()
    apply_theme()
    render_sidebar_branding()
    render_sidebar_workspace_nav()
    render_signed_in_sidebar_account()


def apply_theme() -> None:
    """Inject global light-theme CSS (safe to call once per script run)."""
    css = _theme_css_text()
    if css:
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def hero_title(text: str, subtitle: str) -> None:
    st.markdown(
        f"""
<div style="margin-bottom:2rem;padding-bottom:0.5rem;border-bottom:1px solid #e2e8f0;max-width:52rem;">
  <h1 style="margin:0;padding:0;border:none;color:#0f172a !important;">{text}</h1>
  <p style="margin:0.75rem 0 0 0;font-size:1.05rem;line-height:1.65;color:#475569;font-family:'Source Sans 3',sans-serif;">
    {subtitle}
  </p>
</div>
        """,
        unsafe_allow_html=True,
    )


def prose_block(html: str) -> None:
    st.markdown(
        f"""
<div class="thesis-prose" style="max-width:48rem;line-height:1.75;font-size:1.02rem;color:#334155;font-family:'Source Sans 3',sans-serif;">
{html}
</div>
        """,
        unsafe_allow_html=True,
    )


def page_icon_path() -> str:
    p = Path(__file__).resolve().parent / "static" / "favicon.png"
    return str(p) if p.is_file() else "🌿"


def render_sidebar_branding() -> None:
    st.sidebar.markdown(
        """
<div style="padding:0.15rem 0 1rem 0;margin-bottom:0.75rem;border-bottom:1px solid #e2e8f0;">
  <p style="margin:0;font-size:1rem;font-weight:700;color:#0f172a;font-family:'Source Sans 3',sans-serif;">Urban air quality</p>
  <p style="margin:0.25rem 0 0 0;font-size:0.78rem;color:#64748b;">Analysis workspace</p>
</div>
""",
        unsafe_allow_html=True,
    )


def render_sidebar_workspace_nav() -> None:
    """Custom multipage links in the sidebar (only when signed in).

    Requires ``client.showSidebarNavigation = false`` in ``.streamlit/config.toml`` so the
    default ``pages/`` list is hidden for anonymous users.
    """
    if current_user_id() is None:
        return
    st.sidebar.markdown(
        '<p style="margin:0.35rem 0 0.5rem 0;font-size:0.72rem;font-weight:600;text-transform:uppercase;letter-spacing:0.06em;color:#64748b;">Workspace</p>',
        unsafe_allow_html=True,
    )
    st.sidebar.page_link("Home.py", label="Home", icon="🏠")
    st.sidebar.page_link("pages/1_Data_Explorer.py", label="Data Explorer", icon="📊")
    st.sidebar.page_link(
        "pages/2_Predict_future_quality.py", label="Predict future quality", icon="🔮"
    )
    st.sidebar.page_link("pages/3_Dashboard.py", label="Dashboard", icon="📈")
    st.sidebar.page_link("pages/4_History.py", label="History", icon="🕘")
    st.sidebar.page_link("pages/5_Settings.py", label="Settings", icon="⚙️")
    st.sidebar.page_link("pages/6_HelpDesk.py", label="HelpDesk", icon="💬")


def consume_logout_query() -> None:
    """If the URL contains ?logout=1, sign out when applicable and strip the param (before auth guards)."""
    wants_logout = False
    if hasattr(st, "query_params"):
        v = st.query_params.get("logout")
        if v == "1" or (isinstance(v, list) and v and str(v[0]) == "1"):
            wants_logout = True
    else:
        qp = st.experimental_get_query_params()
        vals = qp.get("logout") or []
        wants_logout = bool(vals) and str(vals[0]) == "1"
    if not wants_logout:
        _drain_deferred_switch_home_after_sign_out()
        return
    if st.session_state.get("user_id") is not None:
        st.session_state.pop("user_id", None)
        st.session_state.pop("username", None)
    if hasattr(st, "query_params"):
        try:
            del st.query_params["logout"]
        except (KeyError, TypeError, AttributeError):
            try:
                st.query_params.clear()
            except Exception:
                pass
    else:
        remaining = {k: v for k, v in st.experimental_get_query_params().items() if k != "logout"}
        if remaining:
            flat = {}
            for k, vals in remaining.items():
                if len(vals) == 1:
                    flat[k] = vals[0]
                else:
                    flat[k] = vals
            st.experimental_set_query_params(**flat)
        else:
            st.experimental_set_query_params()
    _go_home_after_logout_query()


def render_signed_in_sidebar_account() -> None:
    """Username and Log out in the sidebar when authenticated (same-tab session)."""
    if current_user_id() is None:
        return
    user_disp = html.escape(str(st.session_state.get("username") or ""))
    st.sidebar.markdown(
        f"""
<style>
[data-testid="stSidebar"] .thesis-sidebar-account {{
  margin: 0.75rem 0 0.5rem 0;
  padding: 0.65rem 0.75rem;
  border: 1px solid #e2e8f0;
  border-radius: 12px;
  background: rgba(255, 255, 255, 0.92);
  box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
}}
[data-testid="stSidebar"] .thesis-sidebar-account p {{
  margin: 0;
  font-size: 0.92rem;
  font-weight: 600;
  color: #0f172a;
  word-break: break-word;
}}
[data-testid="stSidebar"] .thesis-sidebar-account .thesis-sidebar-account-label {{
  font-size: 0.72rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: #64748b;
  margin-bottom: 0.25rem;
}}
</style>
<div class="thesis-sidebar-account">
  <p class="thesis-sidebar-account-label">Signed in</p>
  <p>{user_disp}</p>
</div>
""",
        unsafe_allow_html=True,
    )
    st.sidebar.button(
        "Log out",
        key="thesis_sign_out_btn",
        type="secondary",
        use_container_width=True,
        on_click=sign_out,
    )
