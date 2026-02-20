from __future__ import annotations

import hmac
import json
import logging
import os
import secrets
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

import bcrypt
from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router_dashboard_auth = APIRouter(prefix="/auth", tags=["dashboard-auth"])

_token_lock = threading.Lock()
_token_store: Dict[str, Dict[str, Any]] = {}
_MAX_TOKENS_PER_USER = 5
_MAX_TOTAL_TOKENS = 10_000  # hard global cap (#23)

# ── Redis token backend (#22) ──────────────────────────────────────────────────
_REDIS_TOKEN_PREFIX = "ds:auth:tok:"
_REDIS_USER_PREFIX = "ds:auth:usr:"
_redis_client: Any = None


def _get_redis_client() -> Any | None:
    """Return a Redis client for the token store, or None if unavailable."""
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    redis_url = os.getenv("REDIS_URL")
    if not redis_url:
        return None
    try:
        import redis as _redis  # type: ignore[import]
        client = _redis.Redis.from_url(redis_url, decode_responses=True, socket_timeout=2)
        client.ping()
        _redis_client = client
        logger.info("Auth token store: Redis backend active (%s)", redis_url)
        return _redis_client
    except Exception as exc:
        logger.warning("Auth token store: Redis unavailable, using in-memory fallback. reason=%s", exc)
        return None


def _redis_add_token(r: Any, token: str, username: str, expires: datetime) -> None:
    ttl = max(1, int((expires - datetime.now(timezone.utc)).total_seconds()))
    key = f"{_REDIS_TOKEN_PREFIX}{token}"
    r.setex(key, ttl, json.dumps({"username": username, "expires_at": expires.isoformat()}))
    user_key = f"{_REDIS_USER_PREFIX}{username}"
    r.zadd(user_key, {token: expires.timestamp()})
    r.expire(user_key, ttl + 60)


def _redis_get_token(r: Any, token: str) -> Dict[str, Any] | None:
    raw = r.get(f"{_REDIS_TOKEN_PREFIX}{token}")
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def _redis_remove_token(r: Any, token: str, username: str | None = None) -> None:
    r.delete(f"{_REDIS_TOKEN_PREFIX}{token}")
    if username:
        r.zrem(f"{_REDIS_USER_PREFIX}{username}", token)


def _redis_enforce_user_limit(r: Any, username: str) -> None:
    """Evict oldest tokens for `username` until count < _MAX_TOKENS_PER_USER."""
    user_key = f"{_REDIS_USER_PREFIX}{username}"
    # Remove already-expired entries from the sorted set
    r.zremrangebyscore(user_key, 0, datetime.now(timezone.utc).timestamp())
    count = r.zcard(user_key)
    if count >= _MAX_TOKENS_PER_USER:
        to_remove = r.zrange(user_key, 0, count - _MAX_TOKENS_PER_USER)
        for old_tok in to_remove:
            r.delete(f"{_REDIS_TOKEN_PREFIX}{old_tok}")
        r.zremrangebyrank(user_key, 0, count - _MAX_TOKENS_PER_USER - 1)


def _auth_enabled() -> bool:
    return os.getenv("DASHBOARD_AUTH_ENABLED", "true").strip().lower() not in {
        "false",
        "0",
        "no",
    }


def _is_bcrypt_hash(value: str) -> bool:
    """Check if a string looks like a bcrypt hash ($2b$, $2a$, $2y$)."""
    return bool(value) and value.startswith(("$2b$", "$2a$", "$2y$"))


def _get_users() -> Dict[str, str]:
    """Return dict of username->password from env with development fallback."""
    users: Dict[str, str] = {}
    admin_pass = os.getenv("DASHBOARD_ADMIN_PASSWORD_ADMIN")
    _PLACEHOLDER_PASSWORDS = {"dev-admin-change-me", "replace-me", "changeme", ""}
    _PROD_ENVS = {"production", "prod", "staging"}
    _current_env = os.getenv("DS_ENV", "").strip().lower()
    _is_prod = _current_env in _PROD_ENVS

    if admin_pass and admin_pass not in _PLACEHOLDER_PASSWORDS:
        users["admin"] = admin_pass
    elif _is_prod:
        # In production/staging, refuse to start with a placeholder or missing password.
        # This prevents accidental exposure of the dev-only admin/admin123 credentials.
        raise RuntimeError(
            "DASHBOARD_ADMIN_PASSWORD_ADMIN must be set to a strong non-placeholder value "
            f"when DS_ENV={_current_env!r}. Refusing to activate insecure dev credentials."
        )
    else:
        # Development fallback only: admin/admin123
        logger.warning(
            "Using insecure development credentials for admin user (admin/admin123). "
            "Set DASHBOARD_ADMIN_PASSWORD_ADMIN and DS_ENV=prod to disable this."
        )
        users["admin"] = "admin123"
    # Legacy single-user env vars
    legacy_user = os.getenv("DASHBOARD_ADMIN_USERNAME")
    legacy_pass = os.getenv("DASHBOARD_ADMIN_PASSWORD")
    if legacy_user and legacy_pass:
        users[legacy_user] = legacy_pass
    # Extra users via JSON: DASHBOARD_EXTRA_USERS='{"user2":"pass2"}'
    extra_raw = os.getenv("DASHBOARD_EXTRA_USERS", "")
    if extra_raw:
        try:
            import json
            extra = json.loads(extra_raw)
            if isinstance(extra, dict):
                users.update(extra)
        except Exception:
            pass
    return users


def _verify_credentials(username: str, password: str) -> bool:
    """
    Check username/password against configured users.

    Supports both bcrypt-hashed and plaintext passwords in env vars:
      - Bcrypt hash ($2b$…): verified with bcrypt.checkpw  (recommended)
      - Plaintext: verified with hmac.compare_digest + deprecation warning
    """
    users = _get_users()
    expected_pass = users.get(username)
    if expected_pass is None:
        return False
    if _is_bcrypt_hash(expected_pass):
        try:
            return bcrypt.checkpw(
                password.encode("utf-8"), expected_pass.encode("utf-8")
            )
        except Exception:
            return False
    # Legacy plaintext — timing-safe but passwords visible if env vars leak
    logger.warning(
        "Plaintext password detected for user '%s'. Use a bcrypt hash instead.",
        username,
    )
    return hmac.compare_digest(password, expected_pass)


def _token_ttl_minutes() -> int:
    try:
        v = int(os.getenv("DASHBOARD_TOKEN_TTL_MINUTES", "480"))
        return max(v, 5)
    except Exception:
        return 480


def _cleanup_expired_tokens() -> None:
    now = datetime.now(timezone.utc)
    with _token_lock:
        expired = [k for k, v in _token_store.items() if v.get("expires_at") <= now]
        for key in expired:
            _token_store.pop(key, None)


def _cleanup_expired_tokens_locked() -> None:
    """Like _cleanup_expired_tokens but assumes _token_lock is already held."""
    now = datetime.now(timezone.utc)
    expired = [k for k, v in _token_store.items() if v.get("expires_at") <= now]
    for key in expired:
        _token_store.pop(key, None)


def _parse_bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    prefix = "Bearer "
    if not authorization.startswith(prefix):
        return None
    token = authorization[len(prefix) :].strip()
    return token or None


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_at: str
    username: str


@router_dashboard_auth.post("/login", response_model=LoginResponse)
def dashboard_login(payload: LoginRequest) -> LoginResponse:
    if not _auth_enabled():
        expires = datetime.now(timezone.utc) + timedelta(minutes=_token_ttl_minutes())
        return LoginResponse(
            access_token="auth-disabled",
            expires_at=expires.isoformat(),
            username="anonymous",
        )

    if not _verify_credentials(payload.username, payload.password):
        raise HTTPException(status_code=401, detail="Geçersiz kullanıcı adı veya şifre")

    token = secrets.token_urlsafe(32)
    expires = datetime.now(timezone.utc) + timedelta(minutes=_token_ttl_minutes())

    r = _get_redis_client()
    if r is not None:
        # Redis path: per-user limit enforced via sorted set
        _redis_enforce_user_limit(r, payload.username)
        _redis_add_token(r, token, payload.username, expires)
    else:
        # In-memory path: enforce both per-user and global cap (#23)
        with _token_lock:
            if len(_token_store) >= _MAX_TOTAL_TOKENS:
                _cleanup_expired_tokens_locked()
                if len(_token_store) >= _MAX_TOTAL_TOKENS:
                    raise HTTPException(
                        status_code=503,
                        detail="Token store kapasitesi dolu. Lütfen daha sonra tekrar deneyin.",
                    )
            user_tokens = [
                k for k, v in _token_store.items()
                if v.get("username") == payload.username
            ]
            if len(user_tokens) >= _MAX_TOKENS_PER_USER:
                oldest = min(user_tokens, key=lambda k: _token_store[k]["expires_at"])
                _token_store.pop(oldest, None)
            _token_store[token] = {"username": payload.username, "expires_at": expires}

    return LoginResponse(
        access_token=token,
        expires_at=expires.isoformat(),
        username=payload.username,
    )


def require_dashboard_user(
    authorization: str | None = Header(default=None),
) -> Dict[str, Any]:
    if not _auth_enabled():
        return {
            "username": "anonymous",
            "auth_enabled": False,
        }

    token = _parse_bearer_token(authorization)
    if token is None:
        raise HTTPException(status_code=401, detail="Oturum gerekli")

    r = _get_redis_client()
    if r is not None:
        data = _redis_get_token(r, token)
    else:
        _cleanup_expired_tokens()
        with _token_lock:
            data = _token_store.get(token)

    if data is None:
        raise HTTPException(status_code=401, detail="Geçersiz veya süresi dolmuş oturum")

    return {
        "username": data.get("username", "unknown"),
        "auth_enabled": True,
    }


@router_dashboard_auth.post("/logout")
def dashboard_logout(
    authorization: str | None = Header(default=None),
    _user: Dict[str, Any] = Depends(require_dashboard_user),
):
    token = _parse_bearer_token(authorization)
    if token is not None:
        r = _get_redis_client()
        if r is not None:
            username = _user.get("username")
            _redis_remove_token(r, token, username)
        else:
            with _token_lock:
                _token_store.pop(token, None)

    return {"status": "ok", "message": "Oturum kapatıldı"}


@router_dashboard_auth.get("/me")
def dashboard_me(user: Dict[str, Any] = Depends(require_dashboard_user)):
    return {
        "status": "ok",
        "username": user.get("username"),
        "auth_enabled": user.get("auth_enabled", True),
    }
