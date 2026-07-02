import json
import os
import sys
import tempfile
from unittest.mock import MagicMock, AsyncMock, patch

import pytest
import yaml

# Ensure the project root is on sys.path so test imports resolve
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Minimal policy YAML used by most tests
# ---------------------------------------------------------------------------

POLICY_YAML = {
    "tenants": {
        "default": {
            "upstream_url": "https://api.openai.com/v1/chat/completions",
            "upstream_api_key": "sk-test-upstream-key",
            "input": {
                "pii": {"enabled": True, "action": "redact",
                        "types": ["email", "phone", "ssn", "credit_card"]},
                "prompt_injection": {"enabled": True, "action": "block"},
                "topic_filter": {"enabled": True, "action": "block",
                                 "blocked_topics": ["competitor", "lawsuit"]},
            },
            "output": {"max_length": 5000},
            "rate_limit": {"enabled": False},
        },
        "test_tenant": {
            "upstream_url": "https://api.openai.com/v1/chat/completions",
            "upstream_api_key": "sk-test-upstream-key",
            "input": {
                "pii": {"enabled": True, "action": "redact",
                        "types": ["email", "phone", "ssn", "credit_card"]},
                "prompt_injection": {"enabled": True, "action": "block"},
                "topic_filter": {"enabled": True, "action": "block",
                                 "blocked_topics": ["competitor"]},
            },
            "output": {"max_length": 5000},
            "rate_limit": {"enabled": True, "requests_per_minute": 60,
                           "requests_per_hour": 1000},
        },
        "tokenize_tenant": {
            "upstream_url": "https://api.openai.com/v1/chat/completions",
            "upstream_api_key": "sk-test-upstream-key",
            "input": {
                "pii": {"enabled": True, "action": "tokenize",
                        "types": ["email", "phone", "ssn", "credit_card"]},
                "prompt_injection": {"enabled": True, "action": "block"},
                "topic_filter": {"enabled": False},
            },
            "output": {"max_length": 5000},
            "rate_limit": {"enabled": False},
        },
    }
}


@pytest.fixture()
def tmp_data_dir(tmp_path):
    """Write a test policy.yaml into a temp dir and point DATA_DIR at it."""
    policy_file = tmp_path / "policy.yaml"
    policy_file.write_text(yaml.safe_dump(POLICY_YAML))
    old = os.environ.get("DATA_DIR")
    os.environ["DATA_DIR"] = str(tmp_path)
    yield tmp_path
    if old is None:
        os.environ.pop("DATA_DIR", None)
    else:
        os.environ["DATA_DIR"] = old


# ---------------------------------------------------------------------------
# Canned upstream responses
# ---------------------------------------------------------------------------

OPENAI_RESPONSE = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "gpt-4o-mini",
    "choices": [{
        "index": 0,
        "message": {"role": "assistant", "content": "The answer is 4."},
        "finish_reason": "stop",
    }],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}

ANTHROPIC_RESPONSE = {
    "id": "msg-test",
    "type": "message",
    "role": "assistant",
    "content": [{"type": "text", "text": "The answer is 4."}],
    "model": "claude-sonnet-4-20250514",
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 10, "output_tokens": 5},
}

OPENAI_STREAM_CHUNKS = [
    'data: {"id":"chatcmpl-test","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}\n\n',
    'data: {"id":"chatcmpl-test","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}\n\n',
    'data: {"id":"chatcmpl-test","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}\n\n',
    'data: {"id":"chatcmpl-test","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
    "data: [DONE]\n\n",
]

ANTHROPIC_STREAM_CHUNKS = [
    'event: message_start\ndata: {"type":"message_start","message":{"id":"msg-test","type":"message","role":"assistant","content":[],"model":"claude-sonnet-4-20250514"}}\n\n',
    'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n',
    'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello world"}}\n\n',
    'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n',
    'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}\n\n',
    'event: message_stop\ndata: {"type":"message_stop"}\n\n',
]


# ---------------------------------------------------------------------------
# Fake httpx response helpers for mocking upstream AI calls
# ---------------------------------------------------------------------------

class FakeHTTPResponse:
    """Mimics httpx.Response for non-streaming calls."""
    def __init__(self, data, status_code=200):
        self.status_code = status_code
        self._data = data

    def json(self):
        return self._data

    def raise_for_status(self):
        if self.status_code >= 400:
            import httpx
            raise httpx.HTTPStatusError(
                "error", request=MagicMock(), response=self
            )


class FakeStreamResponse:
    """Mimics httpx streaming context manager for SSE tests."""
    def __init__(self, chunks):
        self._chunks = chunks
        self.status_code = 200
        self.headers = {"content-type": "text/event-stream"}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def aiter_lines(self):
        for chunk in self._chunks:
            for line in chunk.strip().split("\n"):
                yield line
            yield ""

    def raise_for_status(self):
        pass


# ---------------------------------------------------------------------------
# Test user fixture for proxy/plan tests
# ---------------------------------------------------------------------------

def make_test_user(plan="free", tenant_id="test_tenant",
                   proxy_key="sk-saifety-testkey123"):
    """Build a mock User object with sensible defaults."""
    user = MagicMock()
    user.id = "user-test-001"
    user.email = "test@example.com"
    user.tenant_id = tenant_id
    user.proxy_key = proxy_key
    user.plan = plan
    user.stripe_customer_id = None
    user.stripe_subscription_id = None
    user.has_ai_key = True
    user.has_anthropic_key = False
    user.created_at = 1700000000.0
    return user
