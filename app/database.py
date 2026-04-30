"""SQLite persistence for users, settings, prediction history, and chat.

This module is the canonical database layer for the modular ``app/`` entry
point.  It delegates to the project-root ``database`` module so that both
``app.py`` (monolithic) and ``app/`` (pages-based) share a single SQLite file
and a single implementation.
"""

from __future__ import annotations

# Re-export everything from the root-level database module so that pages
# under app/pages/ can import from ``app.database`` without caring which
# entry point is active.
from database import (  # noqa: F401  (re-exports)
    AccountLockedError,
    add_user,
    append_chat_message,
    change_password,
    create_chat_session,
    delete_chat_session,
    delete_history_entry,
    delete_user_account,
    get_chat_messages,
    get_history_filtered,
    get_user_history,
    get_user_profile,
    get_user_settings,
    init_db,
    list_chat_sessions,
    save_prediction,
    save_prediction_artifacts,
    update_user_profile,
    upsert_user_settings,
    verify_user,
)
