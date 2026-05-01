"""Comprehensive tests for the SQLite persistence layer (``src/db/database.py``).

These tests redirect ``database.DB_PATH`` to a per-test SQLite file under
``tmp_path`` so the production ``air_quality.db`` is never touched.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

import pytest

from db import database

# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture()
def temp_db(tmp_path: Path) -> Iterator[Path]:
    """Point ``database.DB_PATH`` at a throw-away SQLite file."""
    original = database.DB_PATH
    test_db = tmp_path / "test_air_quality.db"
    database.DB_PATH = test_db
    database.init_db()
    try:
        yield test_db
    finally:
        database.DB_PATH = original


@pytest.fixture()
def user_id(temp_db: Path) -> int:
    _ = temp_db
    return database.add_user("alice", "pass12345")


# ─── add_user / verify_user / username + password validation ─────────────────


def test_add_user_returns_positive_id(temp_db: Path) -> None:
    _ = temp_db
    uid = database.add_user("tester", "pass12345")
    assert uid > 0


def test_add_user_strips_username(temp_db: Path) -> None:
    _ = temp_db
    uid = database.add_user("  spaced  ", "pass12345")
    assert database.verify_user("spaced", "pass12345") == uid


def test_add_user_rejects_empty_username(temp_db: Path) -> None:
    _ = temp_db
    with pytest.raises(ValueError, match="non-empty"):
        database.add_user("   ", "pass12345")


def test_add_user_rejects_invalid_username_pattern(temp_db: Path) -> None:
    _ = temp_db
    with pytest.raises(ValueError, match="3-32 chars"):
        database.add_user("ab", "pass12345")  # too short
    with pytest.raises(ValueError, match="3-32 chars"):
        database.add_user("bad name!", "pass12345")  # disallowed chars


def test_add_user_rejects_empty_password(temp_db: Path) -> None:
    _ = temp_db
    with pytest.raises(ValueError, match="non-empty"):
        database.add_user("tester", "")


def test_add_user_rejects_short_password(temp_db: Path) -> None:
    _ = temp_db
    with pytest.raises(ValueError, match="at least"):
        database.add_user("tester", "short")


def test_add_user_rejects_oversize_password(temp_db: Path) -> None:
    _ = temp_db
    with pytest.raises(ValueError, match="bcrypt length limit"):
        database.add_user("tester", "p" * 73)


def test_add_user_duplicate_username_raises(temp_db: Path) -> None:
    _ = temp_db
    database.add_user("dupe", "pass12345")
    with pytest.raises(ValueError, match="already exists"):
        database.add_user("dupe", "pass12345")


def test_verify_user_returns_none_for_unknown_user(temp_db: Path) -> None:
    _ = temp_db
    assert database.verify_user("ghost", "pass12345") is None


def test_verify_user_returns_none_for_missing_credentials(temp_db: Path) -> None:
    _ = temp_db
    assert database.verify_user("", "pass12345") is None
    assert database.verify_user("tester", "") is None
    assert database.verify_user(None, "pass12345") is None  # type: ignore[arg-type]


def test_verify_user_succeeds_clears_failed_attempts(temp_db: Path) -> None:
    _ = temp_db
    uid = database.add_user("tester", "pass12345")
    assert database.verify_user("tester", "wrong") is None
    assert database.verify_user("tester", "pass12345") == uid
    with sqlite3.connect(database.DB_PATH) as conn:
        row = conn.execute(
            "SELECT failed_attempts, locked_until FROM users WHERE id = ?", (uid,)
        ).fetchone()
    assert row[0] == 0
    assert row[1] is None


def test_verify_user_locks_after_max_failed_attempts(temp_db: Path) -> None:
    _ = temp_db
    database.add_user("lockme", "pass12345")
    for _ in range(database.MAX_FAILED_ATTEMPTS):
        assert database.verify_user("lockme", "wrong-pass") is None
    with pytest.raises(database.AccountLockedError):
        database.verify_user("lockme", "pass12345")


def test_verify_user_swallows_invalid_locked_until(temp_db: Path) -> None:
    """Garbage in ``locked_until`` must not crash login."""
    uid = database.add_user("tester", "pass12345")
    with sqlite3.connect(database.DB_PATH) as conn:
        conn.execute(
            "UPDATE users SET locked_until = ? WHERE id = ?",
            ("not-an-iso-date", uid),
        )
    # Should not raise; bad timestamp is logged and ignored.
    assert database.verify_user("tester", "pass12345") == uid


def test_verify_user_allows_login_after_lock_expires(temp_db: Path) -> None:
    uid = database.add_user("tester", "pass12345")
    past = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
    with sqlite3.connect(database.DB_PATH) as conn:
        conn.execute(
            "UPDATE users SET locked_until = ? WHERE id = ?", (past, uid)
        )
    assert database.verify_user("tester", "pass12345") == uid


def test_verify_user_returns_none_on_db_error(temp_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _ = temp_db

    def boom() -> None:
        raise sqlite3.OperationalError("forced failure")

    monkeypatch.setattr(database, "_connect", boom)
    assert database.verify_user("tester", "pass12345") is None


# ─── change_password ─────────────────────────────────────────────────────────


def test_change_password_success(user_id: int) -> None:
    database.change_password(user_id, "pass12345", "brand-new-pw")
    assert database.verify_user("alice", "brand-new-pw") == user_id
    assert database.verify_user("alice", "pass12345") is None


def test_change_password_requires_both_inputs(user_id: int) -> None:
    with pytest.raises(ValueError, match="required"):
        database.change_password(user_id, "", "new-password-123")
    with pytest.raises(ValueError, match="required"):
        database.change_password(user_id, "pass12345", "")


def test_change_password_rejects_oversize_new_password(user_id: int) -> None:
    with pytest.raises(ValueError, match="bcrypt length limit"):
        database.change_password(user_id, "pass12345", "p" * 73)


def test_change_password_rejects_short_new_password(user_id: int) -> None:
    with pytest.raises(ValueError, match="at least"):
        database.change_password(user_id, "pass12345", "tiny")


def test_change_password_rejects_unknown_user(temp_db: Path) -> None:
    _ = temp_db
    with pytest.raises(ValueError, match="not found"):
        database.change_password(9999, "pass12345", "new-password-123")


def test_change_password_rejects_wrong_current(user_id: int) -> None:
    with pytest.raises(ValueError, match="incorrect"):
        database.change_password(user_id, "wrong-current", "new-password-123")


# ─── save_prediction / save_prediction_artifacts / history ───────────────────


def test_save_prediction_returns_positive_id(user_id: int) -> None:
    pid = database.save_prediction(user_id, "LSTM", {"horizon": 24})
    assert pid > 0


def test_save_prediction_rejects_empty_label(user_id: int) -> None:
    with pytest.raises(ValueError, match="Model label"):
        database.save_prediction(user_id, "  ", {"x": 1})


def test_save_prediction_rejects_non_serializable_payload(user_id: int) -> None:
    with pytest.raises(TypeError, match="JSON serializable"):
        database.save_prediction(user_id, "LSTM", {"bad": object()})


def test_save_prediction_with_explicit_timestamp(user_id: int) -> None:
    ts = "2026-01-01T00:00:00+00:00"
    database.save_prediction(user_id, "LSTM", {"a": 1}, timestamp=ts)
    history = database.get_user_history(user_id)
    assert history[0]["timestamp"] == ts


def test_save_prediction_artifacts_round_trips_blobs(user_id: int) -> None:
    pid = database.save_prediction_artifacts(
        user_id=user_id,
        model_used="XGBoost",
        prediction_results={"horizon": 72},
        dataset_name="upload.csv",
        avg_aqi=18.5,
        risk_level="Moderate",
        dataset_csv=b"time,pm2_5\n2024-01-01,12.0\n",
        prediction_csv=b"time,pm2_5_pred\n2024-01-01,11.0\n",
        details={"notes": "test"},
    )
    assert pid > 0
    history = database.get_user_history(user_id)
    assert len(history) == 1
    item = history[0]
    assert item["dataset_name"] == "upload.csv"
    assert item["risk_level"] == "Moderate"
    assert item["avg_aqi"] == 18.5
    assert item["dataset_csv_blob"].startswith(b"time,pm2_5")
    assert item["prediction_csv_blob"].startswith(b"time,pm2_5_pred")
    assert item["details"] == {"notes": "test"}


def test_save_prediction_artifacts_without_details_uses_empty_dict(user_id: int) -> None:
    database.save_prediction_artifacts(
        user_id=user_id,
        model_used="LSTM",
        prediction_results={},
        dataset_name="d.csv",
        avg_aqi=5.0,
        risk_level="Low",
        dataset_csv=b"x",
        prediction_csv=b"y",
    )
    items = database.get_user_history(user_id)
    assert items[0]["details"] == {}


def test_get_user_history_orders_newest_first_and_respects_limit(user_id: int) -> None:
    for i in range(3):
        database.save_prediction(user_id, "LSTM", {"i": i})
    items = database.get_user_history(user_id, limit=2)
    assert len(items) == 2
    assert items[0]["results"]["i"] == 2
    assert items[1]["results"]["i"] == 1


def test_get_user_history_handles_corrupt_json(user_id: int) -> None:
    database.save_prediction(user_id, "LSTM", {"ok": True})
    with sqlite3.connect(database.DB_PATH) as conn:
        conn.execute(
            "UPDATE prediction_history SET prediction_results_json = ? WHERE user_id = ?",
            ("{not-valid-json", user_id),
        )
    items = database.get_user_history(user_id)
    assert items[0]["results"] == {"raw": "{not-valid-json"}


def test_get_user_history_for_unknown_user_returns_empty(temp_db: Path) -> None:
    _ = temp_db
    assert database.get_user_history(9999) == []


# ─── get_history_filtered ────────────────────────────────────────────────────


def test_get_history_filtered_by_model_and_dates(user_id: int) -> None:
    database.save_prediction(user_id, "LSTM", {"a": 1}, timestamp="2026-01-01T00:00:00+00:00")
    database.save_prediction(user_id, "XGBoost", {"a": 2}, timestamp="2026-01-05T00:00:00+00:00")
    database.save_prediction(user_id, "LSTM", {"a": 3}, timestamp="2026-02-01T00:00:00+00:00")

    only_lstm = database.get_history_filtered(user_id, model_used="LSTM")
    assert {item["results"]["a"] for item in only_lstm} == {1, 3}

    in_january = database.get_history_filtered(
        user_id, start_date="2026-01-01", end_date="2026-01-31"
    )
    assert {item["results"]["a"] for item in in_january} == {1, 2}

    all_models = database.get_history_filtered(user_id, model_used="All")
    assert len(all_models) == 3


def test_get_history_filtered_no_matches(user_id: int) -> None:
    database.save_prediction(user_id, "LSTM", {"a": 1}, timestamp="2026-01-01T00:00:00+00:00")
    items = database.get_history_filtered(
        user_id, start_date="2030-01-01", end_date="2030-12-31"
    )
    assert items == []


def test_get_history_filtered_handles_missing_details_json(user_id: int) -> None:
    database.save_prediction(user_id, "LSTM", {"a": 1})
    items = database.get_history_filtered(user_id)
    assert items[0]["details"] == {}


# ─── User profile / settings ─────────────────────────────────────────────────


def test_get_user_profile_returns_username(user_id: int) -> None:
    profile = database.get_user_profile(user_id)
    assert profile["username"] == "alice"
    assert profile["id"] == user_id
    assert profile["email"] == ""


def test_get_user_profile_unknown_raises(temp_db: Path) -> None:
    _ = temp_db
    with pytest.raises(ValueError, match="not found"):
        database.get_user_profile(9999)


def test_update_user_profile_sets_and_clears_email(user_id: int) -> None:
    database.update_user_profile(user_id, "alice@example.com")
    assert database.get_user_profile(user_id)["email"] == "alice@example.com"
    database.update_user_profile(user_id, "   ")
    assert database.get_user_profile(user_id)["email"] == ""
    database.update_user_profile(user_id, None)
    assert database.get_user_profile(user_id)["email"] == ""


def test_get_user_settings_default_when_unset(user_id: int) -> None:
    settings = database.get_user_settings(user_id)
    assert settings == {"default_model": "LSTM"}


def test_upsert_user_settings_normalizes_arima_to_lstm(user_id: int) -> None:
    database.upsert_user_settings(user_id, "ARIMA")
    assert database.get_user_settings(user_id)["default_model"] == "LSTM"


def test_upsert_user_settings_unknown_falls_back_to_lstm(user_id: int) -> None:
    database.upsert_user_settings(user_id, "Prophet")
    assert database.get_user_settings(user_id)["default_model"] == "LSTM"


def test_upsert_user_settings_accepts_xgboost(user_id: int) -> None:
    database.upsert_user_settings(user_id, "XGBoost")
    assert database.get_user_settings(user_id)["default_model"] == "XGBoost"
    # idempotent on update
    database.upsert_user_settings(user_id, "LSTM")
    assert database.get_user_settings(user_id)["default_model"] == "LSTM"


def test_get_user_settings_normalizes_legacy_arima_row(user_id: int) -> None:
    """If a legacy row stored ARIMA, the read API must coerce to LSTM."""
    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(database.DB_PATH) as conn:
        conn.execute(
            "INSERT INTO user_settings (user_id, default_model, updated_at) VALUES (?, ?, ?)",
            (user_id, "ARIMA", now),
        )
    assert database.get_user_settings(user_id)["default_model"] == "LSTM"


# ─── Delete history + account ────────────────────────────────────────────────


def test_delete_history_entry_removes_only_target_row(user_id: int) -> None:
    pid1 = database.save_prediction(user_id, "LSTM", {"a": 1})
    pid2 = database.save_prediction(user_id, "LSTM", {"a": 2})
    database.delete_history_entry(user_id, pid1)
    remaining = {item["id"] for item in database.get_user_history(user_id)}
    assert remaining == {pid2}


def test_delete_user_account_cascades_history(user_id: int) -> None:
    database.save_prediction(user_id, "LSTM", {"a": 1})
    database.delete_user_account(user_id)
    assert database.get_user_history(user_id) == []
    with sqlite3.connect(database.DB_PATH) as conn:
        rows = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchall()
    assert rows == []


# ─── Auth tokens ─────────────────────────────────────────────────────────────


def test_create_and_verify_auth_token(user_id: int) -> None:
    token = database.create_auth_token(user_id)
    assert isinstance(token, str) and len(token) > 16
    result = database.verify_auth_token(token)
    assert result == (user_id, "alice")


def test_verify_auth_token_unknown_returns_none(temp_db: Path) -> None:
    _ = temp_db
    assert database.verify_auth_token("not-a-real-token") is None


def test_delete_auth_token_invalidates_it(user_id: int) -> None:
    token = database.create_auth_token(user_id)
    database.delete_auth_token(token)
    assert database.verify_auth_token(token) is None


def test_delete_user_tokens_removes_all(user_id: int) -> None:
    t1 = database.create_auth_token(user_id)
    t2 = database.create_auth_token(user_id)
    database.delete_user_tokens(user_id)
    assert database.verify_auth_token(t1) is None
    assert database.verify_auth_token(t2) is None


# ─── Migrations / schema ─────────────────────────────────────────────────────


def test_init_db_is_idempotent(temp_db: Path) -> None:
    database.init_db()
    database.init_db()  # second call must not raise
    with sqlite3.connect(temp_db) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    expected = {"users", "prediction_history", "user_settings", "auth_tokens"}
    assert expected.issubset(tables)


def test_init_db_migrates_legacy_user_settings_table(tmp_path: Path) -> None:
    """A pre-existing legacy ``user_settings`` table with extra columns must be
    rebuilt without the legacy preference columns and lose ARIMA defaults."""
    legacy = tmp_path / "legacy.db"
    with sqlite3.connect(legacy) as conn:
        conn.executescript(
            """
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL
            );
            INSERT INTO users (username, password_hash) VALUES ('legacy', 'x');
            CREATE TABLE user_settings (
                user_id INTEGER PRIMARY KEY,
                default_model TEXT NOT NULL DEFAULT 'ARIMA',
                updated_at TEXT NOT NULL,
                theme TEXT,
                notify INTEGER
            );
            INSERT INTO user_settings (user_id, default_model, updated_at, theme, notify)
            VALUES (1, 'ARIMA', '2025-01-01T00:00:00+00:00', 'dark', 1);
            """
        )
    original = database.DB_PATH
    database.DB_PATH = legacy
    try:
        database.init_db()
        with sqlite3.connect(legacy) as conn:
            cols = {
                row[1]
                for row in conn.execute("PRAGMA table_info(user_settings)").fetchall()
            }
            row = conn.execute(
                "SELECT default_model FROM user_settings WHERE user_id = 1"
            ).fetchone()
        assert cols == {"user_id", "default_model", "updated_at"}
        # Legacy ARIMA value normalized to LSTM by the seed UPDATE.
        assert row[0] == "LSTM"
    finally:
        database.DB_PATH = original


# ─── Round-trip: complex JSON payload ───────────────────────────────────────


def test_save_prediction_round_trips_complex_payload(user_id: int) -> None:
    payload = {
        "model": "LSTM",
        "horizon": 168,
        "rows": [{"time": "2026-01-01T00:00:00+00:00", "pm2_5": 12.5}],
    }
    database.save_prediction(user_id, "LSTM", payload)
    items = database.get_user_history(user_id)
    assert items[0]["results"] == payload
    assert json.dumps(items[0]["results"])  # serialisable
