"""
User accounts — sign-up, authentication, per-user tenant provisioning.

Each signed-up user gets:
  • A unique tenant_id written to policy.yaml with default guardrail rules
  • A proxy key for authenticating to the proxy
  • Optional encrypted storage for their upstream AI API key

AI key encryption requires ENCRYPTION_KEY env var (Fernet base64 key).
Generate one with:
    python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

If ENCRYPTION_KEY is not set, keys are stored with a plain: prefix (dev mode only).
"""

import os
import sqlite3
import secrets
import time
import threading
from dataclasses import dataclass
from typing import Optional

import bcrypt
import yaml

from auth import generate_key

_DB_PATH    = os.path.join(os.path.dirname(__file__), "users.db")
_POLICY_FILE = os.path.join(os.path.dirname(__file__), "policy.yaml")

# Shared lock for all policy.yaml reads+writes (imported by main.py too)
_POLICY_LOCK = threading.Lock()

# ── Fernet encryption for stored AI keys ─────────────────────────────────────

_fernet = None
_ENCRYPTION_KEY = os.environ.get("ENCRYPTION_KEY")
if _ENCRYPTION_KEY:
    try:
        from cryptography.fernet import Fernet
        _fernet = Fernet(_ENCRYPTION_KEY.encode())
    except Exception:
        raise RuntimeError(
            "ENCRYPTION_KEY is set but invalid.\n"
            "Generate a valid key with:\n"
            "  python3 -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\""
        )

# ── Plans ─────────────────────────────────────────────────────────────────────

PLANS = {
    "free":    {"price_monthly": 0,   "monthly_requests": 10_000,    "rpm": 10,  "rph": 200},
    "starter": {"price_monthly": 49,  "monthly_requests": 100_000,   "rpm": 60,  "rph": 1_000},
    "growth":  {"price_monthly": 199, "monthly_requests": 1_000_000, "rpm": 200, "rph": 10_000},
}

# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class User:
    id: str
    email: str
    tenant_id: str
    proxy_key: str
    plan: str
    stripe_customer_id: Optional[str]
    stripe_subscription_id: Optional[str]
    ai_api_key_encrypted: Optional[str]
    created_at: float

    @property
    def has_ai_key(self) -> bool:
        return bool(self.ai_api_key_encrypted)


def _row_to_user(row) -> User:
    return User(
        id=row["id"],
        email=row["email"],
        tenant_id=row["tenant_id"],
        proxy_key=row["proxy_key"],
        plan=row["plan"],
        stripe_customer_id=row["stripe_customer_id"],
        stripe_subscription_id=row["stripe_subscription_id"],
        ai_api_key_encrypted=row["ai_api_key_encrypted"],
        created_at=row["created_at"],
    )

# ── Encryption helpers ────────────────────────────────────────────────────────

def _encrypt(plaintext: str) -> str:
    if _fernet:
        return _fernet.encrypt(plaintext.encode()).decode()
    return f"plain:{plaintext}"   # dev mode only


def _decrypt(ciphertext: Optional[str]) -> Optional[str]:
    if not ciphertext:
        return None
    if ciphertext.startswith("plain:"):
        return ciphertext[6:]
    if _fernet:
        try:
            return _fernet.decrypt(ciphertext.encode()).decode()
        except Exception:
            return None
    return None

# ── Policy.yaml helpers (use _POLICY_LOCK for all reads+writes) ───────────────

def _write_default_policy(tenant_id: str) -> None:
    """Add a new tenant entry to policy.yaml with free-plan defaults."""
    plan = PLANS["free"]
    with _POLICY_LOCK:
        with open(_POLICY_FILE) as f:
            raw = yaml.safe_load(f) or {}
        if "tenants" not in raw:
            raw["tenants"] = {}
        if tenant_id not in raw["tenants"]:
            raw["tenants"][tenant_id] = {
                "upstream_url": "https://api.openai.com/v1/chat/completions",
                "input": {
                    "pii": {"enabled": True, "action": "redact",
                            "types": ["email", "phone", "ssn", "credit_card"]},
                    "prompt_injection": {"enabled": True, "action": "block"},
                    "topic_filter": {"enabled": False, "action": "block", "blocked_topics": []},
                },
                "output": {
                    "max_length": None,
                    "toxicity": {"enabled": True, "provider": "wordlist"},
                },
                "rate_limit": {
                    "enabled": True,
                    "requests_per_minute": plan["rpm"],
                    "requests_per_hour": plan["rph"],
                },
            }
            with open(_POLICY_FILE, "w") as f:
                yaml.safe_dump(raw, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def _update_policy_upstream_key(tenant_id: str, api_key: Optional[str]) -> None:
    """Write the user's AI API key into their tenant entry in policy.yaml."""
    with _POLICY_LOCK:
        with open(_POLICY_FILE) as f:
            raw = yaml.safe_load(f) or {}
        if tenant_id in raw.get("tenants", {}):
            raw["tenants"][tenant_id]["upstream_api_key"] = api_key
            with open(_POLICY_FILE, "w") as f:
                yaml.safe_dump(raw, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def _update_policy_rate_limit(tenant_id: str, plan: str) -> None:
    """Update the rate limits for a tenant when their plan changes."""
    cfg = PLANS.get(plan, PLANS["free"])
    with _POLICY_LOCK:
        with open(_POLICY_FILE) as f:
            raw = yaml.safe_load(f) or {}
        if tenant_id in raw.get("tenants", {}):
            raw["tenants"][tenant_id]["rate_limit"] = {
                "enabled": True,
                "requests_per_minute": cfg["rpm"],
                "requests_per_hour": cfg["rph"],
            }
            with open(_POLICY_FILE, "w") as f:
                yaml.safe_dump(raw, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

# ── UserStore ─────────────────────────────────────────────────────────────────

class UserStore:

    def __init__(self):
        self._init_db()

    def _conn(self):
        conn = sqlite3.connect(_DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id                      TEXT PRIMARY KEY,
                    email                   TEXT UNIQUE NOT NULL,
                    password_hash           TEXT NOT NULL,
                    tenant_id               TEXT UNIQUE NOT NULL,
                    proxy_key               TEXT UNIQUE NOT NULL,
                    plan                    TEXT NOT NULL DEFAULT 'free',
                    stripe_customer_id      TEXT,
                    stripe_subscription_id  TEXT,
                    ai_api_key_encrypted    TEXT,
                    created_at              REAL NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_proxy_key ON users(proxy_key)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_stripe_customer ON users(stripe_customer_id)")

    def create_user(self, email: str, password: str) -> Optional[User]:
        """Register a new user. Returns None if the email is already taken."""
        user_id   = secrets.token_hex(16)
        tenant_id = f"user_{secrets.token_hex(6)}"
        proxy_key = generate_key()
        pw_hash   = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

        try:
            with self._conn() as conn:
                conn.execute(
                    "INSERT INTO users (id, email, password_hash, tenant_id, proxy_key, plan, created_at) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (user_id, email.lower().strip(), pw_hash, tenant_id, proxy_key, "free", time.time()),
                )
        except sqlite3.IntegrityError:
            return None  # email already registered

        _write_default_policy(tenant_id)
        return self.get_by_id(user_id)

    def authenticate(self, email: str, password: str) -> Optional[User]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE email = ?", (email.lower().strip(),)
            ).fetchone()
        if row is None:
            return None
        if not bcrypt.checkpw(password.encode(), row["password_hash"].encode()):
            return None
        return _row_to_user(row)

    def get_by_id(self, user_id: str) -> Optional[User]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return _row_to_user(row) if row else None

    def get_by_proxy_key(self, proxy_key: str) -> Optional[User]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM users WHERE proxy_key = ?", (proxy_key,)).fetchone()
        return _row_to_user(row) if row else None

    def get_by_stripe_customer(self, customer_id: str) -> Optional[User]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE stripe_customer_id = ?", (customer_id,)
            ).fetchone()
        return _row_to_user(row) if row else None

    def set_ai_key(self, user_id: str, plaintext_key: str) -> None:
        """Encrypt and store the user's AI API key, and update policy.yaml."""
        encrypted = _encrypt(plaintext_key)
        with self._conn() as conn:
            conn.execute(
                "UPDATE users SET ai_api_key_encrypted = ? WHERE id = ?", (encrypted, user_id)
            )
        user = self.get_by_id(user_id)
        if user:
            _update_policy_upstream_key(user.tenant_id, plaintext_key)

    def get_ai_key(self, user_id: str) -> Optional[str]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT ai_api_key_encrypted FROM users WHERE id = ?", (user_id,)
            ).fetchone()
        return _decrypt(row["ai_api_key_encrypted"]) if row else None

    def set_plan(self, user_id: str, plan: str) -> None:
        if plan not in PLANS:
            raise ValueError(f"Unknown plan: {plan}")
        with self._conn() as conn:
            conn.execute("UPDATE users SET plan = ? WHERE id = ?", (plan, user_id))
        user = self.get_by_id(user_id)
        if user:
            _update_policy_rate_limit(user.tenant_id, plan)

    def set_stripe_ids(self, user_id: str, customer_id: Optional[str],
                       subscription_id: Optional[str]) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE users SET stripe_customer_id = ?, stripe_subscription_id = ? WHERE id = ?",
                (customer_id, subscription_id, user_id),
            )
