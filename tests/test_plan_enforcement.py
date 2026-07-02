"""Tests for monthly request limits and the Powered-By response header."""

import os
import sys
from unittest.mock import patch, MagicMock, AsyncMock

import pytest
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.conftest import (
    OPENAI_RESPONSE, ANTHROPIC_RESPONSE,
    FakeHTTPResponse, FakeStreamResponse, OPENAI_STREAM_CHUNKS,
    make_test_user, POLICY_YAML,
)


@pytest.fixture(autouse=True)
def _isolate_policy(tmp_path, monkeypatch):
    pf = tmp_path / "policy.yaml"
    pf.write_text(yaml.safe_dump(POLICY_YAML))
    from policy import PolicyEngine
    monkeypatch.setattr(PolicyEngine, "_policy_file", str(pf))


def _make_client(user):
    """Build a TestClient with the given user mock."""
    import main

    mock_userstore = MagicMock()
    mock_userstore.get_by_proxy_key.return_value = user
    mock_userstore.get_by_tenant_id.return_value = user
    mock_userstore.get_by_id.return_value = user

    mock_audit = MagicMock()
    mock_audit.get_monthly_request_count.return_value = 0

    mock_webhook = MagicMock()
    mock_webhook.dispatch = MagicMock()
    mock_toxicity = MagicMock()
    mock_toxicity.check = AsyncMock(return_value=None)

    originals = (main.userstore, main.audit, main.webhook, main.toxicity)
    main.userstore = mock_userstore
    main.audit = mock_audit
    main.webhook = mock_webhook
    main.toxicity = mock_toxicity

    from starlette.testclient import TestClient
    tc = TestClient(main.app, raise_server_exceptions=False)
    tc._mock_audit = mock_audit
    tc._originals = originals
    return tc


def _teardown_client(tc):
    import main
    main.userstore, main.audit, main.webhook, main.toxicity = tc._originals


def _post_openai(tc, mock_upstream=True):
    """Helper: POST to OpenAI route, optionally mocking the upstream call."""
    if mock_upstream:
        with patch("main.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post = AsyncMock(return_value=FakeHTTPResponse(OPENAI_RESPONSE))
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance
            return tc.post(
                "/v1/chat/completions",
                json={"model": "gpt-4o-mini",
                      "messages": [{"role": "user", "content": "Hi"}]},
                headers={"Authorization": "Bearer sk-saifety-testkey123"},
            )
    return tc.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o-mini",
              "messages": [{"role": "user", "content": "Hi"}]},
        headers={"Authorization": "Bearer sk-saifety-testkey123"},
    )


# ── Monthly limit tests ──────────────────────────────────────────────────────

class TestMonthlyLimits:
    def test_free_user_below_limit_passes(self):
        user = make_test_user(plan="free")
        tc = _make_client(user)
        try:
            tc._mock_audit.get_monthly_request_count.return_value = 50
            resp = _post_openai(tc)
            assert resp.status_code == 200
        finally:
            _teardown_client(tc)

    def test_free_user_at_limit_blocked(self):
        user = make_test_user(plan="free")
        tc = _make_client(user)
        try:
            tc._mock_audit.get_monthly_request_count.return_value = 200
            resp = _post_openai(tc, mock_upstream=False)
            assert resp.status_code == 429
            data = resp.json()
            assert "monthly_limit_reached" in data["detail"]["error"]
        finally:
            _teardown_client(tc)

    def test_starter_user_high_usage_passes(self):
        user = make_test_user(plan="starter")
        tc = _make_client(user)
        try:
            tc._mock_audit.get_monthly_request_count.return_value = 50_000
            resp = _post_openai(tc)
            assert resp.status_code == 200
        finally:
            _teardown_client(tc)

    def test_starter_user_at_limit_blocked(self):
        user = make_test_user(plan="starter")
        tc = _make_client(user)
        try:
            tc._mock_audit.get_monthly_request_count.return_value = 100_000
            resp = _post_openai(tc, mock_upstream=False)
            assert resp.status_code == 429
        finally:
            _teardown_client(tc)


# ── Powered-By header tests ──────────────────────────────────────────────────

class TestPoweredByHeader:
    def test_free_user_gets_powered_by(self):
        user = make_test_user(plan="free")
        tc = _make_client(user)
        try:
            resp = _post_openai(tc)
            assert resp.status_code == 200
            assert resp.headers.get("x-powered-by") == "sAIfety (saifety.dev)"
        finally:
            _teardown_client(tc)

    def test_starter_user_no_powered_by(self):
        user = make_test_user(plan="starter")
        tc = _make_client(user)
        try:
            resp = _post_openai(tc)
            assert resp.status_code == 200
            assert "x-powered-by" not in resp.headers
        finally:
            _teardown_client(tc)

    def test_growth_user_no_powered_by(self):
        user = make_test_user(plan="growth")
        tc = _make_client(user)
        try:
            resp = _post_openai(tc)
            assert resp.status_code == 200
            assert "x-powered-by" not in resp.headers
        finally:
            _teardown_client(tc)

    def test_free_user_streaming_gets_powered_by(self):
        user = make_test_user(plan="free")
        tc = _make_client(user)
        try:
            with patch("streaming.httpx.AsyncClient") as MockClient:
                instance = AsyncMock()
                instance.stream = MagicMock(
                    return_value=FakeStreamResponse(OPENAI_STREAM_CHUNKS)
                )
                instance.__aenter__ = AsyncMock(return_value=instance)
                instance.__aexit__ = AsyncMock(return_value=False)
                MockClient.return_value = instance

                resp = tc.post(
                    "/v1/chat/completions",
                    json={"model": "gpt-4o-mini", "stream": True,
                          "messages": [{"role": "user", "content": "Hi"}]},
                    headers={"Authorization": "Bearer sk-saifety-testkey123"},
                )
            assert resp.status_code == 200
            assert resp.headers.get("x-powered-by") == "sAIfety (saifety.dev)"
        finally:
            _teardown_client(tc)
