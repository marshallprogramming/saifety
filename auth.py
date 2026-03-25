"""
Proxy authentication — validates caller-supplied proxy keys against keys.yaml.

Keys are separate from AI provider API keys (OpenAI/Anthropic). Clients
authenticate to the proxy with a proxy key; the proxy then uses its own
stored AI credentials when forwarding requests upstream.

Auth modes:
  Disabled (default) — keys.yaml absent or empty. All requests pass through.
                       Tenant comes from X-Tenant-ID header. AI key forwarded
                       as-is from the client. Safe for local development.

  Enabled            — keys.yaml contains at least one key. Every request
                       must supply a valid proxy key. Tenant is derived from
                       the key definition (X-Tenant-ID is ignored). AI key
                       comes from policy.yaml, not the client.

keys.yaml is re-read on every request so key changes take effect immediately
without restarting the proxy.
"""

import os
import secrets
import yaml
from dataclasses import dataclass
from typing import Optional

_KEYS_FILE = os.path.join(os.path.dirname(__file__), "keys.yaml")


@dataclass
class ProxyKey:
    key: str
    name: str
    tenant_id: str
    enabled: bool = True


class KeyStore:
    @property
    def auth_enabled(self) -> bool:
        return bool(self._load_raw())

    def validate(self, raw_key: Optional[str]) -> Optional[ProxyKey]:
        """
        Validate a proxy key.
        Returns the ProxyKey if valid and enabled.
        Returns a permissive dev key if auth is disabled.
        Returns None if auth is enabled but the key is invalid/missing.
        """
        keys = self._load_raw()

        if not keys:
            # Auth disabled — return a permissive placeholder
            return ProxyKey(key="", name="dev-passthrough", tenant_id="__from_header__")

        if not raw_key:
            return None

        cfg = keys.get(raw_key)
        if cfg and cfg.get("enabled", True):
            return ProxyKey(
                key=raw_key,
                name=cfg.get("name", "unnamed"),
                tenant_id=cfg.get("tenant_id", "default"),
                enabled=True,
            )

        return None

    def _load_raw(self) -> dict:
        if not os.path.exists(_KEYS_FILE):
            return {}
        with open(_KEYS_FILE) as f:
            data = yaml.safe_load(f) or {}
        return data.get("keys", {})


def generate_key() -> str:
    """Generate a new random proxy key."""
    return "sk-saifety-" + secrets.token_hex(16)


def _extract_bearer(authorization: Optional[str]) -> Optional[str]:
    """Pull the token from 'Bearer <token>'."""
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return None
