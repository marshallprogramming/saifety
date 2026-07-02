"""
Streaming handlers for OpenAI and Anthropic APIs.

Both handlers:
1. Forward chunks to the client as they arrive (real streaming experience).
2. Accumulate the full text and run incremental output guardrails after each chunk.
3. If a guardrail triggers mid-stream, send an error event and terminate.

Note: JSON schema output validation is not applied to streaming responses —
partial responses are never valid JSON. Use non-streaming if you need it.
"""

import json
import httpx
from fastapi.responses import StreamingResponse
from typing import Optional

from pipeline import GuardrailPipeline
from policy import WebhookConfig, ToxicityConfig
from audit import AuditLogger
from webhooks import WebhookDispatcher
from toxicity import ToxicityChecker
from guardrails.pii import PIIVault, StreamRestorer

# Read timeout is None so long generations don't time out mid-stream.
_STREAM_TIMEOUT = httpx.Timeout(connect=10.0, read=None, write=10.0, pool=10.0)


# ── OpenAI ────────────────────────────────────────────────────────────────────

def stream_openai(
    upstream_url: str,
    body: dict,
    headers: dict,
    pipeline: GuardrailPipeline,
    tenant_id: str,
    audit: AuditLogger,
    webhook: WebhookDispatcher,
    webhook_config: WebhookConfig,
    toxicity_checker: ToxicityChecker = None,
    toxicity_config: ToxicityConfig = None,
    response_headers: Optional[dict] = None,
    vault: Optional[PIIVault] = None,
) -> StreamingResponse:

    restorer = StreamRestorer(vault) if (vault is not None and vault.has_tokens) else None

    async def generate():
        accumulated = ""
        tox_last_len = 0
        last_chunk = None

        try:
            async with httpx.AsyncClient(timeout=_STREAM_TIMEOUT) as client:
                async with client.stream("POST", upstream_url, json=body, headers=headers) as resp:
                    resp.raise_for_status()

                    async for raw_line in resp.aiter_lines():
                        if not raw_line:
                            yield "\n"
                            continue

                        if not raw_line.startswith("data: "):
                            yield f"{raw_line}\n\n"
                            continue

                        data_str = raw_line[6:]

                        if data_str == "[DONE]":
                            # Fallback flush in case no finish_reason chunk arrived
                            if restorer is not None and last_chunk is not None:
                                tail = restorer.flush()
                                if tail:
                                    yield _openai_tail_chunk(last_chunk, tail)
                            audit.log(tenant_id, "passed", None, body, "openai")
                            yield "data: [DONE]\n\n"
                            return

                        try:
                            chunk = json.loads(data_str)
                        except json.JSONDecodeError:
                            yield f"{raw_line}\n\n"
                            continue

                        delta_text = (
                            chunk.get("choices", [{}])[0]
                                 .get("delta", {})
                                 .get("content") or ""
                        )

                        if delta_text:
                            accumulated += delta_text
                            error = pipeline.check_stream_chunk(accumulated)
                            if error:
                                audit.log(tenant_id, "output_blocked", error, body, "openai")
                                webhook.dispatch(webhook_config, "output_blocked", tenant_id,
                                                 "openai", "output_validator", error,
                                                 body.get("messages", []))
                                err_chunk = {"error": {"type": "guardrail_blocked", "message": error}}
                                yield f"data: {json.dumps(err_chunk)}\n\n"
                                yield "data: [DONE]\n\n"
                                return

                            if toxicity_checker and toxicity_config:
                                tox_error, tox_last_len = await toxicity_checker.check_stream(
                                    accumulated, toxicity_config, tox_last_len
                                )
                                if tox_error:
                                    audit.log(tenant_id, "output_blocked", tox_error, body, "openai")
                                    webhook.dispatch(webhook_config, "output_blocked", tenant_id,
                                                     "openai", "toxicity", tox_error,
                                                     body.get("messages", []))
                                    err_chunk = {"error": {"type": "guardrail_blocked", "message": tox_error}}
                                    yield f"data: {json.dumps(err_chunk)}\n\n"
                                    yield "data: [DONE]\n\n"
                                    return

                        # PII reinjection: rewrite the delta with restored text.
                        # A token split across chunks is held back until complete.
                        out_line = raw_line
                        if restorer is not None and chunk.get("choices"):
                            choice = chunk["choices"][0]
                            if delta_text:
                                choice["delta"]["content"] = restorer.feed(delta_text)
                            if choice.get("finish_reason"):
                                # emit any held-back text before the finish chunk
                                tail = restorer.flush()
                                if tail:
                                    yield _openai_tail_chunk(chunk, tail)
                            last_chunk = chunk
                            out_line = f"data: {json.dumps(chunk)}"

                        yield f"{out_line}\n\n"

        except httpx.HTTPStatusError as e:
            err = {"error": {"type": "upstream_error", "message": str(e)}}
            yield f"data: {json.dumps(err)}\n\n"
            yield "data: [DONE]\n\n"
        except httpx.RequestError as e:
            err = {"error": {"type": "upstream_error", "message": str(e)}}
            yield f"data: {json.dumps(err)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream", headers=response_headers or {})


def _openai_tail_chunk(template: dict, tail: str) -> str:
    """Build a synthetic delta chunk carrying held-back restored text."""
    synth = {k: v for k, v in template.items() if k != "choices"}
    synth["choices"] = [{"index": 0, "delta": {"content": tail}, "finish_reason": None}]
    return f"data: {json.dumps(synth)}\n\n"


# ── Anthropic ─────────────────────────────────────────────────────────────────

def stream_anthropic(
    upstream_url: str,
    body: dict,
    headers: dict,
    pipeline: GuardrailPipeline,
    tenant_id: str,
    audit: AuditLogger,
    webhook: WebhookDispatcher,
    webhook_config: WebhookConfig,
    toxicity_checker: ToxicityChecker = None,
    toxicity_config: ToxicityConfig = None,
    response_headers: Optional[dict] = None,
    vault: Optional[PIIVault] = None,
) -> StreamingResponse:
    """
    Anthropic streams Server-Sent Events with explicit event: lines.
    Text content arrives in content_block_delta events.
    """

    restorer = StreamRestorer(vault) if (vault is not None and vault.has_tokens) else None

    async def generate():
        accumulated = ""
        current_event: Optional[str] = None
        tox_last_len = 0
        last_index = 0

        try:
            async with httpx.AsyncClient(timeout=_STREAM_TIMEOUT) as client:
                async with client.stream("POST", upstream_url, json=body, headers=headers) as resp:
                    resp.raise_for_status()

                    async for raw_line in resp.aiter_lines():
                        if not raw_line:
                            yield "\n"
                            current_event = None
                            continue

                        if raw_line.startswith("event: "):
                            current_event = raw_line[7:]
                            # PII reinjection: emit any held-back restored text as
                            # one last delta before the content block closes.
                            if current_event == "content_block_stop" and restorer is not None:
                                tail = restorer.flush()
                                if tail:
                                    payload = json.dumps({
                                        "type": "content_block_delta",
                                        "index": last_index,
                                        "delta": {"type": "text_delta", "text": tail},
                                    })
                                    yield f"event: content_block_delta\ndata: {payload}\n\n"
                            yield f"{raw_line}\n"
                            continue

                        if raw_line.startswith("data: "):
                            data_str = raw_line[6:]
                            out_line = raw_line

                            if current_event == "content_block_delta":
                                try:
                                    chunk = json.loads(data_str)
                                    delta = chunk.get("delta", {})
                                    if delta.get("type") == "text_delta":
                                        text = delta.get("text", "")
                                        if text:
                                            accumulated += text
                                            error = pipeline.check_stream_chunk(accumulated)
                                            if error:
                                                audit.log(tenant_id, "output_blocked",
                                                          error, body, "anthropic")
                                                webhook.dispatch(webhook_config, "output_blocked",
                                                                 tenant_id, "anthropic",
                                                                 "output_validator", error,
                                                                 body.get("messages", []))
                                                yield _anthropic_error_event(error)
                                                return

                                            if toxicity_checker and toxicity_config:
                                                tox_error, tox_last_len = await toxicity_checker.check_stream(
                                                    accumulated, toxicity_config, tox_last_len
                                                )
                                                if tox_error:
                                                    audit.log(tenant_id, "output_blocked",
                                                              tox_error, body, "anthropic")
                                                    webhook.dispatch(webhook_config, "output_blocked",
                                                                     tenant_id, "anthropic",
                                                                     "toxicity", tox_error,
                                                                     body.get("messages", []))
                                                    yield _anthropic_error_event(tox_error)
                                                    return

                                            if restorer is not None:
                                                last_index = chunk.get("index", last_index)
                                                chunk["delta"]["text"] = restorer.feed(text)
                                                out_line = f"data: {json.dumps(chunk)}"
                                except json.JSONDecodeError:
                                    pass

                            elif current_event == "message_stop":
                                audit.log(tenant_id, "passed", None, body, "anthropic")

                            yield f"{out_line}\n"
                            continue

                        yield f"{raw_line}\n"

        except httpx.HTTPStatusError as e:
            yield _anthropic_error_event(f"Upstream error: {e}")
        except httpx.RequestError as e:
            yield _anthropic_error_event(f"Upstream error: {e}")

    return StreamingResponse(generate(), media_type="text/event-stream", headers=response_headers or {})


def _anthropic_error_event(message: str) -> str:
    payload = json.dumps({
        "type": "error",
        "error": {"type": "guardrail_blocked", "message": message},
    })
    return f"event: error\ndata: {payload}\n\n"
