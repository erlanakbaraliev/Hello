"""Shared Streamlit styling: typography, spacing, and light-theme component polish."""

from __future__ import annotations

import html
import logging
from pathlib import Path

import streamlit as st

logger = logging.getLogger(__name__)
_THEME_CSS_PATH = Path(__file__).resolve().parent / "static" / "theme.css"

AUTH_GUARD_MESSAGE = "Please sign in on the Home page to access this workspace."

_HOME_PAGE = "Home.py"
_DEFER_SWITCH_HOME_AFTER_SIGN_OUT = "_thesis_deferred_switch_home_after_sign_out"


# ─── Auth helpers ─────────────────────────────────────────────────────────────

_AUTH_COOKIE_NAME = "aq_session"


def _set_cookie(name: str, value: str, max_age_days: int = 30) -> None:
    """Set a browser cookie via injected JS."""
    import streamlit.components.v1 as components

    max_age = max_age_days * 86400
    js = f"""<script>document.cookie="{name}={value}; path=/; max-age={max_age}; SameSite=Strict";</script>"""
    components.html(js, height=0, width=0)


def _delete_cookie(name: str) -> None:
    """Delete a browser cookie via injected JS."""
    import streamlit.components.v1 as components

    js = f"""<script>document.cookie="{name}=; path=/; max-age=0";</script>"""
    components.html(js, height=0, width=0)


def _get_cookie(name: str) -> str | None:
    """Read a cookie from the current request."""
    cookies = st.context.cookies
    return cookies.get(name)


def restore_session() -> None:
    """On page load, restore user session from cookie if session_state is empty."""
    if st.session_state.get("user_id") is not None:
        return
    token = _get_cookie(_AUTH_COOKIE_NAME)
    if not token:
        return
    from db import verify_auth_token

    result = verify_auth_token(token)
    if result is None:
        _delete_cookie(_AUTH_COOKIE_NAME)
        return
    uid, username = result
    st.session_state["user_id"] = uid
    st.session_state["username"] = username


def persist_session(user_id: int) -> None:
    """After login, create a token and set it as a cookie."""
    from db import create_auth_token

    token = create_auth_token(user_id)
    _set_cookie(_AUTH_COOKIE_NAME, token)


def clear_persistent_session() -> None:
    """On logout, delete the token and cookie."""
    token = _get_cookie(_AUTH_COOKIE_NAME)
    if token:
        from db import delete_auth_token

        delete_auth_token(token)
    _delete_cookie(_AUTH_COOKIE_NAME)


def sign_out() -> None:
    """Clear auth; navigation runs on the next script pass."""
    clear_persistent_session()
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
    """Signed-in user id from session state, or None."""
    raw = st.session_state.get("user_id")
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


# ─── Theme ────────────────────────────────────────────────────────────────────


def _theme_css_text() -> str:
    try:
        return _THEME_CSS_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not load theme CSS from %s: %s", _THEME_CSS_PATH, exc)
        return ""


def apply_theme() -> None:
    """Inject global light-theme CSS."""
    css = _theme_css_text()
    if css:
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


# ─── Page configuration ──────────────────────────────────────────────────────


def page_icon_path() -> str:
    p = Path(__file__).resolve().parent / "static" / "favicon.png"
    return str(p) if p.is_file() else "🌿"


def configure_public_shell(
    *,
    page_title: str,
    layout: str = "wide",
    initial_sidebar_state: str = "expanded",
    page_icon: str | None = None,
) -> None:
    """First Streamlit calls for the Home page."""
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
    """Page config + auth guard + theme + sidebar for signed-in users."""
    st.set_page_config(
        page_title=page_title,
        page_icon=page_icon or page_icon_path(),
        layout=layout,
        initial_sidebar_state=initial_sidebar_state,
    )
    consume_logout_query()
    restore_session()
    if current_user_id() is None:
        st.error(AUTH_GUARD_MESSAGE)
        st.stop()
    apply_theme()
    render_sidebar_workspace_nav()
    render_signed_in_sidebar_account()


# ─── Layout components ────────────────────────────────────────────────────────


def hero_title(text: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="thesis-hero"><h1>{text}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def prose_block(content: str) -> None:
    st.markdown(
        f'<div class="thesis-prose">{content}</div>',
        unsafe_allow_html=True,
    )


def empty_state(icon: str, title: str, description: str) -> None:
    """Styled empty-state card with icon, title, and description."""
    st.markdown(
        f"""
<div class="thesis-empty-state">
  <div class="icon">{icon}</div>
  <h3>{html.escape(title)}</h3>
  <p>{html.escape(description)}</p>
</div>
""",
        unsafe_allow_html=True,
    )


# ─── Sidebar ─────────────────────────────────────────────────────────────────


def render_sidebar_workspace_nav() -> None:
    if current_user_id() is None:
        st.sidebar.markdown(
            '<p class="thesis-sidebar-guest-hint">Sign in to access the workspace.</p>',
            unsafe_allow_html=True,
        )


def render_signed_in_sidebar_account() -> None:
    if current_user_id() is None:
        return
    username = str(st.session_state.get("username") or "")
    with st.sidebar:
        with st.container():
            st.markdown(
                f'<p class="thesis-sidebar-auth-label">Signed in as <strong>{html.escape(username)}</strong></p>',
                unsafe_allow_html=True,
            )
            st.button(
                "Log out",
                key="thesis_sign_out_btn",
                type="secondary",
                use_container_width=True,
                on_click=sign_out,
            )


# ─── Logout query param handling ─────────────────────────────────────────────


def consume_logout_query() -> None:
    """If the URL contains ?logout=1, sign out and strip the param."""
    v = st.query_params.get("logout")
    wants_logout = v == "1" or (isinstance(v, list) and v and str(v[0]) == "1")
    if not wants_logout:
        _drain_deferred_switch_home_after_sign_out()
        return
    if st.session_state.get("user_id") is not None:
        st.session_state.pop("user_id", None)
        st.session_state.pop("username", None)
    try:
        del st.query_params["logout"]
    except (KeyError, TypeError, AttributeError):
        try:
            st.query_params.clear()
        except Exception:
            pass
    _go_home_after_logout_query()
