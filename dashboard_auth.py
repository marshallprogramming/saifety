"""
Dashboard authentication.

Two auth modes coexist:
  Admin session  — set DASHBOARD_PASSWORD; login with password only; sees all tenants
  User session   — sign up / log in with email + password; scoped to own tenant

Sessions map token → identity string:
  "admin"    — admin session (full access)
  <user_id>  — user session (scoped to that user's tenant)

If DASHBOARD_PASSWORD is not set and no user accounts exist, the dashboard is
open with full access (dev mode — consistent with proxy key auth behaviour).

Sessions are in-memory and cleared on server restart. Cookie max-age is 30 days
for user sessions and 24 hours for admin sessions.
"""

import os
import secrets
import hmac
from typing import Optional

_PASSWORD: Optional[str] = os.environ.get("DASHBOARD_PASSWORD")

# token → "admin" or user_id
_SESSIONS: dict[str, str] = {}

# Routes that bypass dashboard auth entirely
PUBLIC_PATHS = {
    "/login", "/signup", "/health", "/favicon.ico", "/billing/webhook",
    "/forgot-password", "/reset-password",
}


def is_proxy_path(path: str) -> bool:
    return path.startswith("/v1/")


def auth_enabled() -> bool:
    """True when DASHBOARD_PASSWORD is set. User signup always works regardless."""
    return bool(_PASSWORD)


def check_password(password: str) -> bool:
    """Validate the admin password."""
    if not _PASSWORD:
        return False
    return hmac.compare_digest(password, _PASSWORD)


def create_session(user_id: str = "admin") -> str:
    """
    Create a new session for a user_id or for the admin ("admin").
    Returns the session token to store in a cookie.
    """
    token = secrets.token_urlsafe(32)
    _SESSIONS[token] = user_id
    return token


def get_session_user(token: Optional[str]) -> Optional[str]:
    """
    Return the identity for a session token.
    Returns "admin", a user_id string, or None if the token is invalid.
    Uses constant-time comparison to prevent timing attacks.
    """
    if not token:
        return None
    for s, uid in _SESSIONS.items():
        if hmac.compare_digest(token, s):
            return uid
    return None


def validate_session(token: Optional[str]) -> bool:
    """Returns True if the token belongs to any valid session."""
    if not _PASSWORD:
        return True  # dev mode — open access
    return get_session_user(token) is not None


def revoke_session(token: str) -> None:
    _SESSIONS.pop(token, None)
