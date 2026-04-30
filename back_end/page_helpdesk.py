"""HelpDesk page body: Gemini chat UI and SQLite-backed sessions."""

from __future__ import annotations

import streamlit as st

from db import (
    append_chat_message,
    create_chat_session,
    delete_chat_session,
    get_chat_messages,
    list_chat_sessions,
)
from back_end.helpdesk_gemini import (
    GeminiClientError,
    generate_helpdesk_reply,
    sanitize_user_text,
)


def ensure_helpdesk_session_keys() -> None:
    if "helpdesk_messages" not in st.session_state:
        st.session_state["helpdesk_messages"] = []
    if "helpdesk_session_id" not in st.session_state:
        st.session_state["helpdesk_session_id"] = None
    if "helpdesk_save_history" not in st.session_state:
        st.session_state["helpdesk_save_history"] = True


def render_helpdesk_page(uid: int, api_key: str | None) -> None:
    """Single-column HelpDesk: conversations in expander, chat below."""
    ensure_helpdesk_session_keys()

    if not api_key:
        st.warning(
            "Gemini API key is missing. Add GEMINI_API_KEY in .streamlit/secrets.toml or environment variables."
        )

    # ── Conversation history (collapsible) ────────────────────────────────────
    with st.expander("Past conversations", expanded=False):
        sessions = list_chat_sessions(uid, limit=30)
        if sessions:
            options = {f"{s['title']} — {s['updated_at'][:16]}": s["id"] for s in sessions}
            selected_label = st.selectbox("Select conversation", list(options.keys()), label_visibility="collapsed")
            selected_id = options[selected_label]
            c1, c2 = st.columns([2, 1])
            if c1.button("Load conversation", type="primary", use_container_width=True):
                msgs = get_chat_messages(uid, selected_id)
                st.session_state["helpdesk_messages"] = [
                    {"role": m["role"], "content": m["message"]} for m in msgs
                ]
                st.session_state["helpdesk_session_id"] = selected_id
                st.rerun()
            if c2.button("Delete", type="secondary", use_container_width=True):
                delete_chat_session(uid, selected_id)
                if st.session_state.get("helpdesk_session_id") == selected_id:
                    st.session_state["helpdesk_session_id"] = None
                    st.session_state["helpdesk_messages"] = []
                st.rerun()
        else:
            st.caption("No saved conversations yet.")

    # ── Controls ──────────────────────────────────────────────────────────────
    st.session_state["helpdesk_save_history"] = st.toggle(
        "Save to history",
        value=bool(st.session_state["helpdesk_save_history"]),
    )

    # ── Chat messages ─────────────────────────────────────────────────────────
    for msg in st.session_state["helpdesk_messages"]:
        with st.chat_message("assistant" if msg["role"] == "assistant" else "user"):
            st.markdown(msg["content"])

    # ── Clear chat (bottom, near input) ───────────────────────────────────────
    if st.session_state["helpdesk_messages"]:
        _, clear_col = st.columns([4, 1])
        if clear_col.button("Clear chat", type="secondary", use_container_width=True):
            st.session_state["helpdesk_messages"] = []
            st.session_state["helpdesk_session_id"] = None
            st.rerun()

    # ── Chat input ────────────────────────────────────────────────────────────
    prompt = st.chat_input("Ask about air quality...")
    if prompt is not None:
        cleaned = sanitize_user_text(prompt, max_length=2000)
        if not cleaned:
            st.warning("Please enter a non-empty message.")
        else:
            st.session_state["helpdesk_messages"].append({"role": "user", "content": cleaned})
            with st.chat_message("user"):
                st.markdown(cleaned)

            with st.spinner("Thinking..."):
                try:
                    reply = generate_helpdesk_reply(
                        st.session_state["helpdesk_messages"], api_key=api_key or ""
                    )
                except GeminiClientError as exc:
                    reply = (
                        f"Unable to generate response: {exc}\n\n"
                        "Please verify GEMINI_API_KEY and network connectivity."
                    )
                except Exception as exc:
                    reply = f"Unexpected error: {exc}"

            st.session_state["helpdesk_messages"].append({"role": "assistant", "content": reply})
            with st.chat_message("assistant"):
                st.markdown(reply)

            if st.session_state["helpdesk_save_history"]:
                sid = st.session_state.get("helpdesk_session_id")
                if sid is None:
                    sid = create_chat_session(uid, title=cleaned[:60])
                    st.session_state["helpdesk_session_id"] = sid
                append_chat_message(uid, sid, "user", cleaned)
                append_chat_message(uid, sid, "assistant", reply)
