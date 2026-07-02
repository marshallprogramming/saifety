"""Tests for proxy key generation, extraction, and validation."""

import os
import sys
import yaml
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auth import generate_key, _extract_bearer, KeyStore


class TestGenerateKey:
    def test_prefix(self):
        key = generate_key()
        assert key.startswith("sk-saifety-")

    def test_length(self):
        key = generate_key()
        # "sk-saifety-" (11 chars) + 32 hex chars = 43
        assert len(key) == 43

    def test_uniqueness(self):
        keys = {generate_key() for _ in range(50)}
        assert len(keys) == 50


class TestExtractBearer:
    def test_valid_bearer(self):
        assert _extract_bearer("Bearer sk-test-123") == "sk-test-123"

    def test_case_insensitive(self):
        assert _extract_bearer("bearer sk-test-123") == "sk-test-123"

    def test_extra_whitespace(self):
        assert _extract_bearer("Bearer  sk-test-123 ") == "sk-test-123"

    def test_none_input(self):
        assert _extract_bearer(None) is None

    def test_empty_string(self):
        assert _extract_bearer("") is None

    def test_no_bearer_prefix(self):
        assert _extract_bearer("sk-test-123") is None


class TestKeyStore:
    def test_auth_disabled_no_file(self, tmp_path, monkeypatch):
        """When keys.yaml doesn't exist, auth is disabled and any key passes."""
        import auth
        monkeypatch.setattr(auth, "_KEYS_FILE", str(tmp_path / "nonexistent.yaml"))
        ks = KeyStore()
        assert not ks.auth_enabled
        result = ks.validate(None)
        assert result is not None
        assert result.name == "dev-passthrough"

    def test_auth_enabled_valid_key(self, tmp_path, monkeypatch):
        import auth
        keys_file = tmp_path / "keys.yaml"
        keys_file.write_text(yaml.safe_dump({
            "keys": {
                "sk-saifety-testkey": {
                    "name": "test-key",
                    "tenant_id": "my_tenant",
                    "enabled": True,
                }
            }
        }))
        monkeypatch.setattr(auth, "_KEYS_FILE", str(keys_file))
        ks = KeyStore()
        assert ks.auth_enabled
        result = ks.validate("sk-saifety-testkey")
        assert result is not None
        assert result.tenant_id == "my_tenant"

    def test_auth_enabled_invalid_key(self, tmp_path, monkeypatch):
        import auth
        keys_file = tmp_path / "keys.yaml"
        keys_file.write_text(yaml.safe_dump({
            "keys": {"sk-valid": {"name": "ok", "tenant_id": "t", "enabled": True}}
        }))
        monkeypatch.setattr(auth, "_KEYS_FILE", str(keys_file))
        ks = KeyStore()
        assert ks.validate("sk-wrong") is None

    def test_disabled_key_rejected(self, tmp_path, monkeypatch):
        import auth
        keys_file = tmp_path / "keys.yaml"
        keys_file.write_text(yaml.safe_dump({
            "keys": {"sk-disabled": {"name": "off", "tenant_id": "t", "enabled": False}}
        }))
        monkeypatch.setattr(auth, "_KEYS_FILE", str(keys_file))
        ks = KeyStore()
        assert ks.validate("sk-disabled") is None
