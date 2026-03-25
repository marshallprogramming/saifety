"""
Dashboard authentication.

Set DASHBOARD_PASSWORD env var to enable login protection.
If unset, the dashboard is open (dev mode — consistent with proxy key auth).

Sessions are in-memory (cleared on restart). Tokens are cryptographically
random and validated with constant-time comparison.
"""

import os
import secrets
import hmac
from typing import Optional

_PASSWORD: Optional[str] = os.environ.get("DASHBOARD_PASSWORD")
_SESSIONS: set[str] = set()

# Routes that are always public (no auth check)
PUBLIC_PATHS = {"/login", "/health", "/favicon.ico"}

# Routes used by the proxy itself — auth handled separately by proxy keys
def is_proxy_path(path: str) -> bool:
    return path.startswith("/v1/")


def auth_enabled() -> bool:
    return bool(_PASSWORD)


def check_password(password: str) -> bool:
    if not _PASSWORD:
        return True
    return hmac.compare_digest(password, _PASSWORD)


def create_session() -> str:
    token = secrets.token_urlsafe(32)
    _SESSIONS.add(token)
    return token


def validate_session(token: Optional[str]) -> bool:
    if not _PASSWORD:
        return True
    if not token:
        return False
    # Use constant-time check to prevent timing attacks
    return any(hmac.compare_digest(token, s) for s in _SESSIONS)


def revoke_session(token: str) -> None:
    _SESSIONS.discard(token)
