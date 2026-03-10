"""Tests for src.user_store — UserStore CRUD and password management."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.user_store import UserStore, init_user_store, get_user_store


@pytest.fixture()
def store(tmp_path: Path) -> UserStore:
    """Create a fresh UserStore with an isolated SQLite DB."""
    db_url = f"sqlite:///{(tmp_path / 'users.db').as_posix()}"
    s = UserStore(database_url=db_url)
    s.create_schema()
    return s


class TestUserStoreCRUD:
    """UserStore create/read operations."""

    def test_create_user(self, store: UserStore) -> None:
        result = store.create_user("admin", "password123")
        assert result is True

    def test_create_duplicate_user(self, store: UserStore) -> None:
        store.create_user("admin", "password123")
        result = store.create_user("admin", "other_pass")
        assert result is False

    def test_get_user_exists(self, store: UserStore) -> None:
        store.create_user("admin", "password123")
        user = store.get_user("admin")
        assert user is not None
        assert user["username"] == "admin"
        assert user["role"] == "admin"
        assert user["is_active"] is True

    def test_get_user_not_found(self, store: UserStore) -> None:
        user = store.get_user("nonexistent")
        assert user is None

    def test_list_users(self, store: UserStore) -> None:
        store.create_user("admin", "pass1")
        store.create_user("viewer", "pass2")
        users = store.list_users()
        assert len(users) == 2
        # password_hash should NOT be in list output
        for u in users:
            assert "password_hash" not in u


class TestPasswordManagement:
    """UserStore password hashing and verification."""

    def test_verify_correct_password(self, store: UserStore) -> None:
        store.create_user("admin", "secret123")
        assert store.verify_password("admin", "secret123") is True

    def test_verify_wrong_password(self, store: UserStore) -> None:
        store.create_user("admin", "secret123")
        assert store.verify_password("admin", "wrongpass") is False

    def test_verify_nonexistent_user(self, store: UserStore) -> None:
        assert store.verify_password("ghost", "anypass") is False

    def test_hash_password_produces_bcrypt_hash(self) -> None:
        hashed = UserStore.hash_password("test")
        assert hashed.startswith("$2b$") or hashed.startswith("$2a$")

    def test_hash_password_unique_per_call(self) -> None:
        h1 = UserStore.hash_password("same")
        h2 = UserStore.hash_password("same")
        assert h1 != h2  # Different salts


class TestCreateSchema:
    """UserStore.create_schema is idempotent."""

    def test_idempotent(self, store: UserStore) -> None:
        store.create_schema()  # second call should not raise
        store.create_user("test", "pass")
        assert store.get_user("test") is not None


class TestModuleLevel:
    """Module-level init/get helpers."""

    def test_get_user_store_without_init_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import src.user_store as mod

        monkeypatch.setattr(mod, "_user_store", None)
        with pytest.raises(RuntimeError):
            get_user_store()

    def test_init_and_get(self, tmp_path: Path) -> None:
        db_url = f"sqlite:///{(tmp_path / 'users2.db').as_posix()}"
        init_user_store(db_url)
        s = get_user_store()
        assert isinstance(s, UserStore)
