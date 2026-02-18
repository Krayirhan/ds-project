from __future__ import annotations

import hmac
import os
import secrets
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel

router_dashboard_auth = APIRouter(prefix="/auth", tags=["dashboard-auth"])

_token_lock = threading.Lock()
_token_store: Dict[str, Dict[str, Any]] = {}


def _auth_enabled() -> bool:
    return os.getenv("DASHBOARD_AUTH_ENABLED", "true").strip().lower() not in {
        "false",
        "0",
        "no",
    }


def _get_users() -> Dict[str, str]:
    """Return dict of username->password from env.  Always includes 'admin'."""
    users: Dict[str, str] = {
        "admin": os.getenv("DASHBOARD_ADMIN_PASSWORD_ADMIN", "ChangeMe123!"),
    }
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
    """Check username/password against all configured users."""
    users = _get_users()
    expected_pass = users.get(username)
    if expected_pass is None:
        return False
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
    with _token_lock:
        _token_store[token] = {
            "username": payload.username,
            "expires_at": expires,
        }

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

    _cleanup_expired_tokens()
    token = _parse_bearer_token(authorization)
    if token is None:
        raise HTTPException(status_code=401, detail="Oturum gerekli")

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
