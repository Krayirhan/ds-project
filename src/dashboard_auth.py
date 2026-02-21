"""dashboard_auth.py — PostgreSQL-backed authentication (single admin user).

Flow:
  POST /auth/login   → verify against `users` table → issue bearer token (Redis / in-memory)
  GET  /auth/me      → validate token, return username
  POST /auth/logout  → revoke token
"""
from __future__ import annotations

import json
import logging
import os
import secrets
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel

from .user_store import get_user_store

logger = logging.getLogger(__name__)

router_dashboard_auth = APIRouter(prefix="/auth", tags=["dashboard-auth"])

# ── In-memory token store (fallback when Redis unavailable) ──────────────────
_token_lock = threading.Lock()
_token_store: Dict[str, Dict[str, Any]] = {}
_MAX_TOKENS_PER_USER = 5
_MAX_TOTAL_TOKENS = 10_000

# ── Redis token backend ───────────────────────────────────────────────────────
_REDIS_TOKEN_PREFIX = "ds:auth:tok:"  # nosec B105
_REDIS_USER_PREFIX  = "ds:auth:usr:"
_redis_client: Any  = None


def _get_redis_client() -> Any | None:
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
    r.setex(
        f"{_REDIS_TOKEN_PREFIX}{token}",
        ttl,
        json.dumps({"username": username, "expires_at": expires.isoformat()}),
    )
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
    user_key = f"{_REDIS_USER_PREFIX}{username}"
    r.zremrangebyscore(user_key, 0, datetime.now(timezone.utc).timestamp())
    count = r.zcard(user_key)
    if count >= _MAX_TOKENS_PER_USER:
        to_remove = r.zrange(user_key, 0, count - _MAX_TOKENS_PER_USER)
        for old_tok in to_remove:
            r.delete(f"{_REDIS_TOKEN_PREFIX}{old_tok}")
        r.zremrangebyrank(user_key, 0, count - _MAX_TOKENS_PER_USER - 1)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _auth_enabled() -> bool:
    return os.getenv("DASHBOARD_AUTH_ENABLED", "true").strip().lower() not in {"false", "0", "no"}


def _token_ttl_minutes() -> int:
    try:
        return max(int(os.getenv("DASHBOARD_TOKEN_TTL_MINUTES", "480")), 5)
    except Exception:
        return 480


def _verify_credentials(username: str, password: str) -> bool:
    store = get_user_store()
    if store is None:
        logger.warning("UserStore unavailable; credential check failed.")
        return False
    return store.verify_password(username, password)


def _cleanup_expired_tokens() -> None:
    now = datetime.now(timezone.utc)
    with _token_lock:
        expired = [k for k, v in _token_store.items() if v.get("expires_at") <= now]
        for key in expired:
            _token_store.pop(key, None)


def _cleanup_expired_tokens_locked() -> None:
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
    token = authorization[len(prefix):].strip()
    return token or None


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_at: str
    username: str


# ── Dependency ────────────────────────────────────────────────────────────────

def require_dashboard_user(
    authorization: str | None = Header(default=None),
) -> Dict[str, Any]:
    """Validate bearer token and return user dict."""
    if not _auth_enabled():
        return {"username": "admin", "auth_enabled": False}

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

    return {"username": data.get("username", "admin"), "auth_enabled": True}


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router_dashboard_auth.post("/login", response_model=LoginResponse)
def dashboard_login(payload: LoginRequest) -> LoginResponse:
    if not _auth_enabled():
        expires = datetime.now(timezone.utc) + timedelta(minutes=_token_ttl_minutes())
        return LoginResponse(
            access_token="auth-disabled",  # nosec B106
            expires_at=expires.isoformat(),
            username="admin",
        )

    if not _verify_credentials(payload.username, payload.password):
        raise HTTPException(status_code=401, detail="Geçersiz kullanıcı adı veya şifre")

    token = secrets.token_urlsafe(32)
    expires = datetime.now(timezone.utc) + timedelta(minutes=_token_ttl_minutes())
    token_data: Dict[str, Any] = {"username": payload.username, "expires_at": expires}

    r = _get_redis_client()
    if r is not None:
        _redis_enforce_user_limit(r, payload.username)
        _redis_add_token(r, token, payload.username, expires)
    else:
        with _token_lock:
            if len(_token_store) >= _MAX_TOTAL_TOKENS:
                _cleanup_expired_tokens_locked()
                if len(_token_store) >= _MAX_TOTAL_TOKENS:
                    raise HTTPException(status_code=503, detail="Token store kapasitesi dolu.")
            user_tokens = [k for k, v in _token_store.items() if v.get("username") == payload.username]
            if len(user_tokens) >= _MAX_TOKENS_PER_USER:
                oldest = min(user_tokens, key=lambda k: _token_store[k]["expires_at"])
                _token_store.pop(oldest, None)
            _token_store[token] = token_data

    return LoginResponse(
        access_token=token,
        expires_at=expires.isoformat(),
        username=payload.username,
    )


@router_dashboard_auth.get("/me")
def dashboard_me(user: Dict[str, Any] = Depends(require_dashboard_user)):
    return {
        "status": "ok",
        "username": user.get("username"),
        "auth_enabled": user.get("auth_enabled", True),
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
            _redis_remove_token(r, token, _user.get("username"))
        else:
            with _token_lock:
                _token_store.pop(token, None)
    return {"status": "ok", "message": "Oturum kapatıldı"}
