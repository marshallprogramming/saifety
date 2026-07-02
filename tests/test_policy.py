"""Tests for policy loading, merging, and environment variable resolution."""

import os
import sys
import yaml
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from policy import PolicyEngine, _resolve_env


class TestResolveEnv:
    def test_none_passthrough(self):
        assert _resolve_env(None) is None

    def test_empty_string_passthrough(self):
        assert _resolve_env("") == ""

    def test_plain_string_passthrough(self):
        assert _resolve_env("sk-abc123") == "sk-abc123"

    def test_full_placeholder_resolved(self, monkeypatch):
        monkeypatch.setenv("MY_KEY", "secret-value")
        assert _resolve_env("${MY_KEY}") == "secret-value"

    def test_full_placeholder_unset_returns_none(self, monkeypatch):
        monkeypatch.delenv("NONEXISTENT_VAR_XYZ", raising=False)
        assert _resolve_env("${NONEXISTENT_VAR_XYZ}") is None

    def test_partial_placeholder_resolved(self, monkeypatch):
        monkeypatch.setenv("HOST", "example.com")
        assert _resolve_env("https://${HOST}/api") == "https://example.com/api"


class TestPolicyEngine:
    @pytest.fixture(autouse=True)
    def _setup_policy(self, tmp_path):
        policy = {
            "tenants": {
                "default": {
                    "upstream_url": "https://api.openai.com/v1/chat/completions",
                    "upstream_api_key": "sk-default-key",
                    "input": {
                        "pii": {"enabled": True, "action": "redact"},
                        "prompt_injection": {"enabled": True},
                    },
                    "rate_limit": {"enabled": False},
                },
                "user_abc": {
                    "upstream_api_key": "sk-user-key",
                    "input": {
                        "pii": {"enabled": True, "action": "block"},
                    },
                },
                "user_no_key": {
                    "input": {
                        "pii": {"enabled": False},
                    },
                },
            }
        }
        pf = tmp_path / "policy.yaml"
        pf.write_text(yaml.safe_dump(policy))
        PolicyEngine._policy_file = str(pf)
        yield

    def test_loads_default_tenant(self):
        p = PolicyEngine.load_for_tenant("default")
        assert p.tenant_id == "default"
        assert p.upstream_api_key == "sk-default-key"
        assert p.input.pii.action == "redact"

    def test_user_tenant_overrides(self):
        p = PolicyEngine.load_for_tenant("user_abc")
        assert p.upstream_api_key == "sk-user-key"
        assert p.input.pii.action == "block"
        # Inherited from default
        assert p.input.injection.enabled is True

    def test_user_tenant_does_not_inherit_api_keys(self):
        """user_no_key has no upstream_api_key — it should NOT inherit from default."""
        p = PolicyEngine.load_for_tenant("user_no_key")
        assert p.upstream_api_key is None

    def test_unknown_tenant_falls_back_to_default(self):
        p = PolicyEngine.load_for_tenant("nonexistent_tenant")
        assert p.upstream_api_key == "sk-default-key"

    def test_anthropic_key_not_inherited(self):
        p = PolicyEngine.load_for_tenant("user_no_key")
        assert p.upstream_anthropic_key is None
