"""SQLite persistence for authentication, settings, and prediction history."""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import bcrypt

DB_PATH = Path(__file__).resolve().parent / "air_quality.db"
logger = logging.getLogger(__name__)
PASSWORD_MIN_LENGTH = 8
MAX_FAILED_ATTEMPTS = 5
LOCKOUT_MINUTES = 15
USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")


class AccountLockedError(RuntimeError):
    """Raised when login is blocked because the account is temporarily locked."""


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    ddl_users = """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL
        );
    """
    ddl_predictions = """
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            timestamp TEXT NOT NULL,
            model_used TEXT NOT NULL,
            prediction_results_json TEXT NOT NULL,
            dataset_name TEXT,
            upload_date TEXT,
            avg_aqi REAL,
            risk_level TEXT,
            dataset_csv_blob BLOB,
            prediction_csv_blob BLOB,
            details_json TEXT
        );
    """
    ddl_settings = """
        CREATE TABLE IF NOT EXISTS user_settings (
            user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
            default_model TEXT NOT NULL DEFAULT 'ARIMA',
            time_granularity TEXT NOT NULL DEFAULT 'daily',
            theme_mode TEXT NOT NULL DEFAULT 'light',
            preferred_chart TEXT NOT NULL DEFAULT 'line',
            high_pollution_alerts INTEGER NOT NULL DEFAULT 1,
            updated_at TEXT NOT NULL
        );
    """
    ddl_chat_sessions = """
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            title TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
    """
    ddl_chat_messages = """
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            role TEXT NOT NULL,
            message TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
    """
    ddl_auth_tokens = """
        CREATE TABLE IF NOT EXISTS auth_tokens (
            token TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            created_at TEXT NOT NULL
        );
    """
    with _connect() as conn:
        conn.executescript(
            ddl_users + ddl_predictions + ddl_settings + ddl_chat_sessions + ddl_chat_messages + ddl_auth_tokens
        )
        # Backward-compatible migration for existing databases.
        _ensure_column(conn, "users", "email", "TEXT")
        _ensure_column(conn, "users", "failed_attempts", "INTEGER NOT NULL DEFAULT 0")
        _ensure_column(conn, "users", "locked_until", "TEXT")
        _ensure_column(conn, "prediction_history", "dataset_name", "TEXT")
        _ensure_column(conn, "prediction_history", "upload_date", "TEXT")
        _ensure_column(conn, "prediction_history", "avg_aqi", "REAL")
        _ensure_column(conn, "prediction_history", "risk_level", "TEXT")
        _ensure_column(conn, "prediction_history", "dataset_csv_blob", "BLOB")
        _ensure_column(conn, "prediction_history", "prediction_csv_blob", "BLOB")
        _ensure_column(conn, "prediction_history", "details_json", "TEXT")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_updated ON chat_sessions(user_id, updated_at DESC)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chat_messages_session_created ON chat_messages(session_id, created_at)"
        )


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, column_type: str) -> None:
    cols = conn.execute(f"PRAGMA table_info({table})").fetchall()
    if not any(c["name"] == column for c in cols):
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")


def add_user(username: str, password: str) -> int:
    name = (username or "").strip()
    if not name:
        raise ValueError("Username must be non-empty.")
    if not USERNAME_PATTERN.match(name):
        raise ValueError(
            "Username must be 3-32 chars and only contain letters, numbers, ., _, or -."
        )
    if not password:
        raise ValueError("Password must be non-empty.")
    if len(password) < PASSWORD_MIN_LENGTH:
        raise ValueError(f"Password must be at least {PASSWORD_MIN_LENGTH} characters.")
    password_bytes = password.encode("utf-8")
    if len(password_bytes) > 72:
        raise ValueError("Password exceeds bcrypt length limit.")

    password_hash = bcrypt.hashpw(password_bytes, bcrypt.gensalt()).decode("ascii")
    try:
        with _connect() as conn:
            cur = conn.execute(
                "INSERT INTO users (username, password_hash, email) VALUES (?, ?, ?)",
                (name, password_hash, None),
            )
            return int(cur.lastrowid or 0)
    except sqlite3.IntegrityError as exc:
        raise ValueError("Username already exists.") from exc


def verify_user(username: str, password: str) -> int | None:
    name = (username or "").strip()
    if not name or not password:
        return None
    try:
        with _connect() as conn:
            row = conn.execute(
                "SELECT id, password_hash, failed_attempts, locked_until FROM users WHERE username = ?",
                (name,),
            ).fetchone()
    except sqlite3.Error:
        logger.exception("Database error while verifying user")
        return None

    if row is None:
        return None
    if row["locked_until"]:
        try:
            locked_until = datetime.fromisoformat(str(row["locked_until"]))
            if datetime.now(timezone.utc) < locked_until:
                raise AccountLockedError(
                    "Account temporarily locked after repeated failed logins. "
                    f"Try again after {locked_until.strftime('%Y-%m-%d %H:%M UTC')}."
                )
        except ValueError:
            logger.warning("Invalid locked_until value for user_id=%s", row["id"])
    if not bcrypt.checkpw(password.encode("utf-8"), row["password_hash"].encode("ascii")):
        _record_failed_login(int(row["id"]))
        return None
    _clear_failed_logins(int(row["id"]))
    return int(row["id"])


def change_password(user_id: int, current_password: str, new_password: str) -> None:
    if not current_password or not new_password:
        raise ValueError("Current and new password are required.")
    if len(new_password.encode("utf-8")) > 72:
        raise ValueError("New password exceeds bcrypt length limit.")
    if len(new_password) < PASSWORD_MIN_LENGTH:
        raise ValueError(f"New password must be at least {PASSWORD_MIN_LENGTH} characters.")

    with _connect() as conn:
        row = conn.execute(
            "SELECT password_hash FROM users WHERE id = ?",
            (int(user_id),),
        ).fetchone()
        if row is None:
            raise ValueError("User not found.")

        stored = row["password_hash"].encode("ascii")
        if not bcrypt.checkpw(current_password.encode("utf-8"), stored):
            raise ValueError("Current password is incorrect.")

        new_hash = bcrypt.hashpw(new_password.encode("utf-8"), bcrypt.gensalt()).decode("ascii")
        conn.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?",
            (new_hash, int(user_id)),
        )


def save_prediction(
    user_id: int,
    model_used: str,
    prediction_results: Any,
    timestamp: str | None = None,
) -> int:
    label = (model_used or "").strip()
    if not label:
        raise ValueError("Model label is required.")

    ts = timestamp or datetime.now(timezone.utc).isoformat()
    try:
        payload = json.dumps(prediction_results, ensure_ascii=False)
    except TypeError as exc:
        raise TypeError("Prediction results must be JSON serializable.") from exc

    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO prediction_history (user_id, timestamp, model_used, prediction_results_json)
            VALUES (?, ?, ?, ?)
            """,
            (int(user_id), ts, label, payload),
        )
        return int(cur.lastrowid or 0)


def save_prediction_artifacts(
    user_id: int,
    model_used: str,
    prediction_results: Any,
    dataset_name: str,
    avg_aqi: float,
    risk_level: str,
    dataset_csv: bytes,
    prediction_csv: bytes,
    details: dict[str, Any] | None = None,
) -> int:
    ts = datetime.now(timezone.utc).isoformat()
    payload = json.dumps(prediction_results, ensure_ascii=False)
    details_json = json.dumps(details or {}, ensure_ascii=False)
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO prediction_history
            (user_id, timestamp, model_used, prediction_results_json, dataset_name, upload_date,
             avg_aqi, risk_level, dataset_csv_blob, prediction_csv_blob, details_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(user_id),
                ts,
                model_used.strip(),
                payload,
                dataset_name,
                ts,
                float(avg_aqi),
                risk_level,
                dataset_csv,
                prediction_csv,
                details_json,
            ),
        )
        return int(cur.lastrowid or 0)


def get_user_history(user_id: int, limit: int = 50) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT id, timestamp, model_used, prediction_results_json,
                   dataset_name, upload_date, avg_aqi, risk_level,
                   dataset_csv_blob, prediction_csv_blob, details_json
            FROM prediction_history
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(user_id), int(limit)),
        ).fetchall()

    items: list[dict[str, Any]] = []
    for row in rows:
        raw_json = row["prediction_results_json"]
        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError:
            data = {"raw": raw_json}
        items.append(
            {
                "id": int(row["id"]),
                "timestamp": row["timestamp"],
                "model_used": row["model_used"],
                "results": data,
                "dataset_name": row["dataset_name"],
                "upload_date": row["upload_date"],
                "avg_aqi": row["avg_aqi"],
                "risk_level": row["risk_level"],
                "dataset_csv_blob": row["dataset_csv_blob"],
                "prediction_csv_blob": row["prediction_csv_blob"],
                "details": json.loads(row["details_json"] or "{}"),
            }
        )
    return items


def get_history_filtered(
    user_id: int,
    start_date: str | None = None,
    end_date: str | None = None,
    model_used: str | None = None,
) -> list[dict[str, Any]]:
    query = """
        SELECT id, timestamp, model_used, prediction_results_json,
               dataset_name, upload_date, avg_aqi, risk_level,
               dataset_csv_blob, prediction_csv_blob, details_json
        FROM prediction_history
        WHERE user_id = ?
    """
    params: list[Any] = [int(user_id)]
    if start_date:
        query += " AND date(timestamp) >= date(?)"
        params.append(start_date)
    if end_date:
        query += " AND date(timestamp) <= date(?)"
        params.append(end_date)
    if model_used and model_used != "All":
        query += " AND model_used = ?"
        params.append(model_used)
    query += " ORDER BY id DESC"
    with _connect() as conn:
        rows = conn.execute(query, tuple(params)).fetchall()
    return [
        {
            "id": int(r["id"]),
            "timestamp": r["timestamp"],
            "model_used": r["model_used"],
            "results": json.loads(r["prediction_results_json"] or "{}"),
            "dataset_name": r["dataset_name"],
            "upload_date": r["upload_date"],
            "avg_aqi": r["avg_aqi"],
            "risk_level": r["risk_level"],
            "dataset_csv_blob": r["dataset_csv_blob"],
            "prediction_csv_blob": r["prediction_csv_blob"],
            "details": json.loads(r["details_json"] or "{}"),
        }
        for r in rows
    ]


def get_user_profile(user_id: int) -> dict[str, Any]:
    with _connect() as conn:
        row = conn.execute(
            "SELECT id, username, email FROM users WHERE id = ?",
            (int(user_id),),
        ).fetchone()
    if row is None:
        raise ValueError("User not found.")
    return {"id": int(row["id"]), "username": row["username"], "email": row["email"] or ""}


def update_user_profile(user_id: int, email: str | None) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE users SET email = ? WHERE id = ?",
            ((email or "").strip() or None, int(user_id)),
        )


def get_user_settings(user_id: int) -> dict[str, Any]:
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT default_model, time_granularity, theme_mode, preferred_chart, high_pollution_alerts
            FROM user_settings WHERE user_id = ?
            """,
            (int(user_id),),
        ).fetchone()
    if row is None:
        return {
            "default_model": "ARIMA",
            "time_granularity": "daily",
            "theme_mode": "light",
            "preferred_chart": "line",
            "high_pollution_alerts": True,
        }
    return {
        "default_model": row["default_model"],
        "time_granularity": row["time_granularity"],
        "theme_mode": row["theme_mode"],
        "preferred_chart": row["preferred_chart"],
        "high_pollution_alerts": bool(row["high_pollution_alerts"]),
    }


def upsert_user_settings(
    user_id: int,
    default_model: str,
    time_granularity: str,
    theme_mode: str,
    preferred_chart: str,
    high_pollution_alerts: bool,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO user_settings (user_id, default_model, time_granularity, theme_mode, preferred_chart, high_pollution_alerts, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                default_model=excluded.default_model,
                time_granularity=excluded.time_granularity,
                theme_mode=excluded.theme_mode,
                preferred_chart=excluded.preferred_chart,
                high_pollution_alerts=excluded.high_pollution_alerts,
                updated_at=excluded.updated_at
            """,
            (
                int(user_id),
                default_model,
                time_granularity,
                theme_mode,
                preferred_chart,
                int(high_pollution_alerts),
                now,
            ),
        )


def delete_history_entry(user_id: int, history_id: int) -> None:
    with _connect() as conn:
        conn.execute(
            "DELETE FROM prediction_history WHERE id = ? AND user_id = ?",
            (int(history_id), int(user_id)),
        )


def delete_user_account(user_id: int) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM users WHERE id = ?", (int(user_id),))


def create_chat_session(user_id: int, title: str | None = None) -> int:
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO chat_sessions (user_id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (int(user_id), (title or "").strip() or "New conversation", now, now),
        )
        return int(cur.lastrowid or 0)


def list_chat_sessions(user_id: int, limit: int = 50) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT id, title, created_at, updated_at
            FROM chat_sessions
            WHERE user_id = ?
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (int(user_id), int(limit)),
        ).fetchall()
    return [
        {
            "id": int(r["id"]),
            "title": r["title"] or "New conversation",
            "created_at": r["created_at"],
            "updated_at": r["updated_at"],
        }
        for r in rows
    ]


def get_chat_messages(user_id: int, session_id: int) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT id, role, message, created_at
            FROM chat_messages
            WHERE user_id = ? AND session_id = ?
            ORDER BY created_at ASC, id ASC
            """,
            (int(user_id), int(session_id)),
        ).fetchall()
    return [
        {
            "id": int(r["id"]),
            "role": r["role"],
            "message": r["message"],
            "created_at": r["created_at"],
        }
        for r in rows
    ]


def append_chat_message(user_id: int, session_id: int, role: str, message: str) -> int:
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO chat_messages (session_id, user_id, role, message, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (int(session_id), int(user_id), role.strip().lower(), message, now),
        )
        conn.execute(
            "UPDATE chat_sessions SET updated_at = ? WHERE id = ? AND user_id = ?",
            (now, int(session_id), int(user_id)),
        )
        return int(cur.lastrowid or 0)


def _record_failed_login(user_id: int) -> None:
    now = datetime.now(timezone.utc)
    with _connect() as conn:
        row = conn.execute(
            "SELECT failed_attempts FROM users WHERE id = ?",
            (int(user_id),),
        ).fetchone()
        attempts = int((row["failed_attempts"] if row else 0) or 0) + 1
        lock_until = None
        if attempts >= MAX_FAILED_ATTEMPTS:
            lock_until = now.replace(microsecond=0) + timedelta(minutes=LOCKOUT_MINUTES)
            attempts = 0
        conn.execute(
            "UPDATE users SET failed_attempts = ?, locked_until = ? WHERE id = ?",
            (attempts, lock_until.isoformat() if lock_until else None, int(user_id)),
        )


def _clear_failed_logins(user_id: int) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE users SET failed_attempts = 0, locked_until = NULL WHERE id = ?",
            (int(user_id),),
        )


def delete_chat_session(user_id: int, session_id: int) -> None:
    with _connect() as conn:
        conn.execute(
            "DELETE FROM chat_sessions WHERE id = ? AND user_id = ?",
            (int(session_id), int(user_id)),
        )


# ─── Persistent session tokens ────────────────────────────────────────────────


def create_auth_token(user_id: int) -> str:
    """Create a random token for persistent login. Returns the token string."""
    import secrets

    token = secrets.token_urlsafe(32)
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO auth_tokens (token, user_id, created_at) VALUES (?, ?, ?)",
            (token, user_id, now),
        )
    return token


def verify_auth_token(token: str) -> tuple[int, str] | None:
    """Verify a token. Returns (user_id, username) or None if invalid."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT t.user_id, u.username FROM auth_tokens t JOIN users u ON t.user_id = u.id WHERE t.token = ?",
            (token,),
        ).fetchone()
    if row is None:
        return None
    return int(row[0]), str(row[1])


def delete_auth_token(token: str) -> None:
    """Remove a token (logout)."""
    with _connect() as conn:
        conn.execute("DELETE FROM auth_tokens WHERE token = ?", (token,))


def delete_user_tokens(user_id: int) -> None:
    """Remove all tokens for a user."""
    with _connect() as conn:
        conn.execute("DELETE FROM auth_tokens WHERE user_id = ?", (user_id,))
