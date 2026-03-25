"""
Webhook dispatcher — fires an HTTP POST when a guardrail blocks a request.

Design:
  - Fire-and-forget using asyncio.create_task() — never delays the proxy response
  - Retries up to 3 times with exponential backoff (1s, 2s) on 5xx or network errors
  - Optional HMAC-SHA256 request signing via X-Saifety-Signature header
  - Events are filtered by the tenant's `on` list before dispatching

Payload shape:
  {
    "event":           "input_blocked",
    "tenant_id":       "strict",
    "api":             "openai",
    "guardrail":       "pii",
    "reason":          "Request contains PII: email",
    "timestamp":       1712345678.123,
    "message_preview": "My email is jo..."   // first 200 chars of last user message
  }

Verifying signatures on your receiver:
  import hmac, hashlib
  expected = hmac.new(secret.encode(), body_bytes, hashlib.sha256).hexdigest()
  assert header == f"sha256={expected}"
"""

import asyncio
import hashlib
import hmac as _hmac
import json
import time
import httpx
from typing import Optional

from policy import WebhookConfig


class WebhookDispatcher:

    def dispatch(
        self,
        config: WebhookConfig,
        event: str,
        tenant_id: str,
        api: str,
        guardrail: Optional[str],
        reason: Optional[str],
        messages: list,
    ) -> None:
        """
        Schedule webhook delivery as a background task.
        Returns immediately — never blocks the caller.
        """
        if not config.enabled or not config.url:
            return
        if event not in config.on:
            return

        payload = {
            "event":           event,
            "tenant_id":       tenant_id,
            "api":             api,
            "guardrail":       guardrail,
            "reason":          reason,
            "timestamp":       time.time(),
            "message_preview": self._preview(messages),
        }

        asyncio.create_task(
            self._deliver_with_retry(config.url, config.secret, payload)
        )

    async def _deliver_with_retry(
        self,
        url: str,
        secret: Optional[str],
        payload: dict,
    ) -> None:
        body_bytes = json.dumps(payload).encode()
        headers = {"Content-Type": "application/json", "User-Agent": "saifety-webhook/1.0"}

        if secret:
            sig = _hmac.new(secret.encode(), body_bytes, hashlib.sha256).hexdigest()
            headers["X-Saifety-Signature"] = f"sha256={sig}"

        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    resp = await client.post(url, content=body_bytes, headers=headers)
                if resp.status_code < 500:
                    return   # success or a 4xx we shouldn't retry
            except Exception:
                pass

            if attempt < 2:
                await asyncio.sleep(2 ** attempt)   # 1s then 2s

        # All retries exhausted — log quietly, never raise
        print(f"[webhook] delivery failed after 3 attempts → {url} ({payload['event']})")

    def _preview(self, messages: list, max_len: int = 200) -> str:
        """Return the last user message, truncated."""
        for msg in reversed(messages):
            content = msg.get("content", "")
            if isinstance(content, str) and content:
                return content[:max_len]
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text", "")
                        if text:
                            return text[:max_len]
        return ""
