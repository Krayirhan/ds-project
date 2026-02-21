from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch
import types

import bcrypt
import pytest
from fastapi import HTTPException

import src.dashboard_auth as auth


class FakeRedis:
    def __init__(self) -> None:
        self.kv: dict[str, str] = {}
        self.zsets: dict[str, dict[str, float]] = {}

    def setex(self, key: str, _ttl: int, value: str) -> None:
        self.kv[key] = value

    def zadd(self, key: str, mapping: dict[str, float]) -> None:
        z = self.zsets.setdefault(key, {})
        z.update(mapping)

    def expire(self, _key: str, _ttl: int) -> None:
        return None

    def get(self, key: str) -> str | None:
        return self.kv.get(key)

    def delete(self, key: str) -> None:
        self.kv.pop(key, None)

    def zrem(self, key: str, member: str) -> None:
        self.zsets.setdefault(key, {}).pop(member, None)

    def zremrangebyscore(self, key: str, lo: float, hi: float) -> None:
        z = self.zsets.setdefault(key, {})
        for member, score in list(z.items()):
            if lo <= score <= hi:
                z.pop(member, None)

    def zcard(self, key: str) -> int:
        return len(self.zsets.setdefault(key, {}))

    def zrange(self, key: str, start: int, stop: int) -> list[str]:
        items = sorted(self.zsets.setdefault(key, {}).items(), key=lambda kv: kv[1])
        if stop < 0:
            return [k for k, _ in items[start:]]
        return [k for k, _ in items[start : stop + 1]]

    def zremrangebyrank(self, key: str, start: int, stop: int) -> None:
        members = self.zrange(key, start, stop)
        for member in members:
            self.zsets.setdefault(key, {}).pop(member, None)


@pytest.fixture(autouse=True)
def reset_auth_state(monkeypatch: pytest.MonkeyPatch) -> None:
    auth._token_store.clear()
    monkeypatch.setattr(auth, "_redis_client", None)
    for key in [
        "DASHBOARD_AUTH_ENABLED",
        "DASHBOARD_ADMIN_PASSWORD_ADMIN",
        "DASHBOARD_ALLOW_INSECURE_DEV_LOGIN",
        "DASHBOARD_ADMIN_USERNAME",
        "DASHBOARD_ADMIN_PASSWORD",
        "DASHBOARD_EXTRA_USERS",
        "DASHBOARD_TOKEN_TTL_MINUTES",
        "DS_ENV",
        "REDIS_URL",
    ]:
        monkeypatch.delenv(key, raising=False)


def test_auth_enabled_variants(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DASHBOARD_AUTH_ENABLED", "false")
    assert auth._auth_enabled() is False
    monkeypatch.setenv("DASHBOARD_AUTH_ENABLED", "0")
    assert auth._auth_enabled() is False
    monkeypatch.setenv("DASHBOARD_AUTH_ENABLED", "no")
    assert auth._auth_enabled() is False
    monkeypatch.setenv("DASHBOARD_AUTH_ENABLED", "true")
    assert auth._auth_enabled() is True


def test_get_users_requires_explicit_admin_password(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DS_ENV", "development")
    with pytest.raises(RuntimeError):
        auth._get_users()


def test_get_users_allows_insecure_login_only_with_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DS_ENV", "development")
    monkeypatch.setenv("DASHBOARD_ALLOW_INSECURE_DEV_LOGIN", "true")
    users = auth._get_users()
    assert users["admin"] == "admin123"


def test_get_users_rejects_placeholder_in_prod(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DS_ENV", "production")
    monkeypatch.setenv("DASHBOARD_ADMIN_PASSWORD_ADMIN", "replace-me")
    with pytest.raises(RuntimeError):
        auth._get_users()


def test_get_users_placeholder_in_dev_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DS_ENV", "development")
    monkeypatch.setenv("DASHBOARD_ADMIN_PASSWORD_ADMIN", "replace-me")
    with patch.object(auth.logger, "warning") as warn:
        users = auth._get_users()
    assert users["admin"] == "replace-me"
    warn.assert_called()


def test_get_users_missing_admin_in_prod_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DS_ENV", "staging")
    monkeypatch.delenv("DASHBOARD_ADMIN_PASSWORD_ADMIN", raising=False)
    with pytest.raises(RuntimeError, match="must be set"):
        auth._get_users()


def test_get_users_invalid_extra_users_json_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DASHBOARD_ADMIN_PASSWORD_ADMIN", "StrongPass!123")
    monkeypatch.setenv("DASHBOARD_EXTRA_USERS", "{invalid-json")
    users = auth._get_users()
    assert users["admin"] == "StrongPass!123"


def test_get_users_merges_legacy_and_extra_users(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DASHBOARD_ADMIN_PASSWORD_ADMIN", "StrongPass!123")
    monkeypatch.setenv("DASHBOARD_ADMIN_USERNAME", "legacy")
    monkeypatch.setenv("DASHBOARD_ADMIN_PASSWORD", "legacy-pass")
    monkeypatch.setenv("DASHBOARD_EXTRA_USERS", '{"alice":"a1","bob":"b2"}')
    users = auth._get_users()
    assert users["admin"] == "StrongPass!123"
    assert users["legacy"] == "legacy-pass"
    assert users["alice"] == "a1"
    assert users["bob"] == "b2"


def test_verify_credentials_plaintext(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DASHBOARD_ADMIN_PASSWORD_ADMIN", "PlainPass")
    assert auth._verify_credentials("admin", "PlainPass") is True
    assert auth._verify_credentials("admin", "wrong") is False
    assert auth._verify_credentials("nobody", "PlainPass") is False


def test_verify_credentials_bcrypt(monkeypatch: pytest.MonkeyPatch) -> None:
    hashed = bcrypt.hashpw(b"Secret$123", bcrypt.gensalt()).decode("utf-8")
    monkeypatch.setenv("DASHBOARD_ADMIN_PASSWORD_ADMIN", hashed)
    assert auth._verify_credentials("admin", "Secret$123") is True
    assert auth._verify_credentials("admin", "wrong") is False


def test_verify_credentials_bcrypt_exception_returns_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "DASHBOARD_ADMIN_PASSWORD_ADMIN",
        "$2b$12$abcdefghijklmnopqrstuv12345678901234567890123456789012",
    )
    with patch.object(auth.bcrypt, "checkpw", side_effect=ValueError("bad-hash")):
        assert auth._verify_credentials("admin", "Secret$123") is False


def test_token_ttl_and_parse_bearer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DASHBOARD_TOKEN_TTL_MINUTES", "1")
    assert auth._token_ttl_minutes() == 5
    monkeypatch.setenv("DASHBOARD_TOKEN_TTL_MINUTES", "abc")
    assert auth._token_ttl_minutes() == 480

    assert auth._parse_bearer_token(None) is None
    assert auth._parse_bearer_token("Basic abc") is None
    assert auth._parse_bearer_token("Bearer ") is None
    assert auth._parse_bearer_token("Bearer tok-123") == "tok-123"


def test_cleanup_expired_tokens_variants() -> None:
    now = datetime.now(timezone.utc)
    auth._token_store["expired"] = {"username": "u", "expires_at": now - timedelta(minutes=1)}
    auth._token_store["valid"] = {"username": "u", "expires_at": now + timedelta(minutes=10)}

    auth._cleanup_expired_tokens()
    assert "expired" not in auth._token_store
    assert "valid" in auth._token_store

    auth._token_store["expired2"] = {"username": "u", "expires_at": now - timedelta(minutes=1)}
    with auth._token_lock:
        auth._cleanup_expired_tokens_locked()
    assert "expired2" not in auth._token_store


def test_redis_helper_roundtrip() -> None:
    r = FakeRedis()
    expires = datetime.now(timezone.utc) + timedelta(minutes=10)
    auth._redis_add_token(r, "tok", "alice", expires)
    data = auth._redis_get_token(r, "tok")
    assert data is not None
    assert data["username"] == "alice"

    auth._redis_remove_token(r, "tok", "alice")
    assert auth._redis_get_token(r, "tok") is None


def test_redis_get_token_invalid_json() -> None:
    r = FakeRedis()
    r.kv[f"{auth._REDIS_TOKEN_PREFIX}tok"] = "not-json"
    assert auth._redis_get_token(r, "tok") is None


def test_redis_enforce_user_limit_evicts_oldest() -> None:
    r = FakeRedis()
    user_key = f"{auth._REDIS_USER_PREFIX}alice"
    base = datetime.now(timezone.utc).timestamp()
    for i in range(7):
        tok = f"tok-{i}"
        r.kv[f"{auth._REDIS_TOKEN_PREFIX}{tok}"] = json.dumps({"username": "alice"})
        r.zsets.setdefault(user_key, {})[tok] = base + i

    auth._redis_enforce_user_limit(r, "alice")
    assert len(r.zsets[user_key]) <= auth._MAX_TOKENS_PER_USER


def test_dashboard_login_auth_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DASHBOARD_AUTH_ENABLED", "false")
    payload = auth.LoginRequest(username="any", password="any")
    resp = auth.dashboard_login(payload)
    assert resp.access_token == "auth-disabled"
    assert resp.username == "anonymous"


def test_dashboard_login_invalid_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DASHBOARD_ADMIN_PASSWORD_ADMIN", "StrongPass!123")
    with patch.object(auth, "_verify_credentials", return_value=False):
        with pytest.raises(HTTPException) as ex:
            auth.dashboard_login(auth.LoginRequest(username="admin", password="bad"))
    assert ex.value.status_code == 401


def test_dashboard_login_in_memory_evicts_oldest(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DASHBOARD_ADMIN_PASSWORD_ADMIN", "StrongPass!123")
    now = datetime.now(timezone.utc)
    for i in range(auth._MAX_TOKENS_PER_USER):
        auth._token_store[f"tok-{i}"] = {"username": "admin", "expires_at": now + timedelta(minutes=i + 1)}

    with (
        patch.object(auth, "_verify_credentials", return_value=True),
        patch.object(auth, "_get_redis_client", return_value=None),
        patch.object(auth.secrets, "token_urlsafe", return_value="new-token"),
    ):
        resp = auth.dashboard_login(auth.LoginRequest(username="admin", password="ok"))
    assert resp.access_token == "new-token"
    user_tokens = [k for k, v in auth._token_store.items() if v.get("username") == "admin"]
    assert len(user_tokens) == auth._MAX_TOKENS_PER_USER
    assert "tok-0" not in auth._token_store


def test_dashboard_login_in_memory_capacity_full(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DASHBOARD_ADMIN_PASSWORD_ADMIN", "StrongPass!123")
    monkeypatch.setattr(auth, "_MAX_TOTAL_TOKENS", 2)
    now = datetime.now(timezone.utc)
    auth._token_store["a"] = {"username": "u1", "expires_at": now + timedelta(minutes=10)}
    auth._token_store["b"] = {"username": "u2", "expires_at": now + timedelta(minutes=10)}

    with (
        patch.object(auth, "_verify_credentials", return_value=True),
        patch.object(auth, "_get_redis_client", return_value=None),
    ):
        with pytest.raises(HTTPException) as ex:
            auth.dashboard_login(auth.LoginRequest(username="admin", password="ok"))
    assert ex.value.status_code == 503


def test_dashboard_login_redis_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DASHBOARD_ADMIN_PASSWORD_ADMIN", "StrongPass!123")
    redis_client = object()
    with (
        patch.object(auth, "_verify_credentials", return_value=True),
        patch.object(auth, "_get_redis_client", return_value=redis_client),
        patch.object(auth, "_redis_enforce_user_limit") as mock_limit,
        patch.object(auth, "_redis_add_token") as mock_add,
        patch.object(auth.secrets, "token_urlsafe", return_value="redis-token"),
    ):
        resp = auth.dashboard_login(auth.LoginRequest(username="admin", password="ok"))
    assert resp.access_token == "redis-token"
    mock_limit.assert_called_once()
    mock_add.assert_called_once()


def test_require_dashboard_user_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DASHBOARD_AUTH_ENABLED", "false")
    anon = auth.require_dashboard_user(None)
    assert anon["auth_enabled"] is False

    monkeypatch.setenv("DASHBOARD_AUTH_ENABLED", "true")
    with pytest.raises(HTTPException):
        auth.require_dashboard_user(None)

    token = "tok-1"
    auth._token_store[token] = {
        "username": "alice",
        "expires_at": datetime.now(timezone.utc) + timedelta(minutes=10),
    }
    user = auth.require_dashboard_user(f"Bearer {token}")
    assert user["username"] == "alice"

    with pytest.raises(HTTPException):
        auth.require_dashboard_user("Bearer missing")


def test_require_dashboard_user_redis_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DASHBOARD_AUTH_ENABLED", "true")
    with (
        patch.object(auth, "_get_redis_client", return_value=object()),
        patch.object(auth, "_redis_get_token", return_value={"username": "redis-user"}),
    ):
        user = auth.require_dashboard_user("Bearer redis-token")
    assert user["username"] == "redis-user"


def test_dashboard_logout_and_me(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DASHBOARD_AUTH_ENABLED", "true")
    auth._token_store["tok"] = {
        "username": "admin",
        "expires_at": datetime.now(timezone.utc) + timedelta(minutes=10),
    }
    out = auth.dashboard_logout("Bearer tok", {"username": "admin"})
    assert out["status"] == "ok"
    assert "tok" not in auth._token_store

    with (
        patch.object(auth, "_get_redis_client", return_value=object()),
        patch.object(auth, "_redis_remove_token") as mock_remove,
    ):
        out2 = auth.dashboard_logout("Bearer redis-tok", {"username": "admin"})
    assert out2["status"] == "ok"
    mock_remove.assert_called_once()

    me = auth.dashboard_me({"username": "admin", "auth_enabled": True})
    assert me["username"] == "admin"


def test_get_redis_client_short_circuits_when_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    cached = SimpleNamespace()
    monkeypatch.setattr(auth, "_redis_client", cached)
    assert auth._get_redis_client() is cached


def test_get_redis_client_import_and_ping_failure_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _BadClient:
        def ping(self):
            raise RuntimeError("redis down")

    class _Redis:
        @staticmethod
        def from_url(*_args, **_kwargs):
            return _BadClient()

    fake_module = types.ModuleType("redis")
    fake_module.Redis = _Redis

    monkeypatch.setattr(auth, "_redis_client", None)
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setitem(__import__("sys").modules, "redis", fake_module)

    out = auth._get_redis_client()
    assert out is None
