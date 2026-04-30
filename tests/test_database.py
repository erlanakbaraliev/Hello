from __future__ import annotations

from pathlib import Path

import pytest

import database


@pytest.fixture()
def temp_db(tmp_path: Path) -> Path:
    original = database.DB_PATH
    test_db = tmp_path / "test_air_quality.db"
    database.DB_PATH = test_db
    database.init_db()
    try:
        yield test_db
    finally:
        database.DB_PATH = original


def test_add_and_verify_user(temp_db: Path) -> None:
    _ = temp_db
    user_id = database.add_user("tester", "pass12345")
    assert user_id > 0
    assert database.verify_user("tester", "pass12345") == user_id
    assert database.verify_user("tester", "wrong-pass") is None


def test_verify_user_raises_when_locked(temp_db: Path) -> None:
    _ = temp_db
    database.add_user("lockme", "pass12345")
    for _ in range(database.MAX_FAILED_ATTEMPTS):
        assert database.verify_user("lockme", "wrong-pass") is None
    with pytest.raises(database.AccountLockedError):
        database.verify_user("lockme", "pass12345")


def test_duplicate_username_rejected(temp_db: Path) -> None:
    _ = temp_db
    database.add_user("tester", "pass12345")
    with pytest.raises(ValueError):
        database.add_user("tester", "pass12345")
