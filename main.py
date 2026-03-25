"""
AI Guardrail Proxy
Drop-in proxy for OpenAI-compatible APIs AND the Anthropic Messages API.
Change your base_url — no other code changes needed.

OpenAI route:    POST /v1/chat/completions
Anthropic route: POST /v1/messages

Usage:
    uvicorn main:app --port 8000

OpenAI clients:
    client = OpenAI(base_url="http://localhost:8000/v1")

Anthropic clients:
    client = Anthropic(base_url="http://localhost:8000")
"""

import time
import os
import httpx
import yaml
from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from typing import Optional

from policy import PolicyEngine
from pipeline import GuardrailPipeline
from audit import AuditLogger
from streaming import stream_openai, stream_anthropic
from rate_limiter import RateLimiter
from auth import KeyStore, _extract_bearer
from webhooks import WebhookDispatcher
from toxicity import ToxicityChecker
from guardrails.content_utils import get_text

app = FastAPI(title="AI Guardrail Proxy", version="0.6.0")
audit = AuditLogger()
limiter = RateLimiter()
keystore = KeyStore()
webhook = WebhookDispatcher()
toxicity = ToxicityChecker()

_DASHBOARD_DIR = os.path.join(os.path.dirname(__file__), "dashboard")


@app.get("/")
async def dashboard():
    return FileResponse(os.path.join(_DASHBOARD_DIR, "index.html"))

ANTHROPIC_API_BASE = "https://api.anthropic.com"
ANTHROPIC_DEFAULT_VERSION = "2023-06-01"


# ── OpenAI route ──────────────────────────────────────────────────────────────

@app.post("/v1/chat/completions")
async def proxy_openai(
    request: Request,
    authorization: Optional[str] = Header(None),
    x_tenant_id: Optional[str] = Header(None),
):
    # ── Auth ──
    proxy_key = keystore.validate(_extract_bearer(authorization))
    if proxy_key is None:
        raise HTTPException(status_code=401, detail={"error": "invalid_api_key",
                                                      "message": "Invalid or missing proxy key."})

    tenant_id = (
        proxy_key.tenant_id
        if proxy_key.tenant_id != "__from_header__"
        else (x_tenant_id or "default")
    )
    body = await request.json()

    policy = PolicyEngine.load_for_tenant(tenant_id)
    pipeline = GuardrailPipeline(policy)

    rl = limiter.check(tenant_id, policy.rate_limit)
    if rl.limited:
        audit.log(tenant_id, "rate_limited", rl.reason, body, "openai")
        webhook.dispatch(policy.webhook, "rate_limited", tenant_id, "openai", None, rl.reason, body.get("messages", []))
        raise HTTPException(
            status_code=429,
            detail={"error": "rate_limited", "reason": rl.reason},
            headers={"Retry-After": str(rl.retry_after)},
        )

    input_result = pipeline.run_input(body.get("messages", []))
    if input_result.blocked:
        audit.log(tenant_id, "input_blocked", input_result.reason, body, "openai")
        webhook.dispatch(policy.webhook, "input_blocked", tenant_id, "openai", input_result.guardrail, input_result.reason, body.get("messages", []))
        raise HTTPException(status_code=400, detail={
            "error": "request_blocked",
            "reason": input_result.reason,
            "guardrail": input_result.guardrail,
        })

    body["messages"] = input_result.messages

    # Use the policy's stored upstream key if configured, else forward the client's key
    upstream_auth = (
        f"Bearer {policy.upstream_api_key}"
        if policy.upstream_api_key
        else authorization
    )
    forward_headers = {
        "Content-Type": "application/json",
        **({"Authorization": upstream_auth} if upstream_auth else {}),
    }

    # Streaming path
    if body.get("stream"):
        return stream_openai(policy.upstream_url, body, forward_headers, pipeline, tenant_id, audit, webhook, policy.webhook, toxicity, policy.output.toxicity)

    # Non-streaming path
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(policy.upstream_url, json=body, headers=forward_headers)
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Upstream error: {e}")

    response_data = resp.json()

    if "choices" in response_data:
        output_result = pipeline.run_output_openai(response_data["choices"])
        if output_result.blocked:
            audit.log(tenant_id, "output_blocked", output_result.reason, body, "openai")
            webhook.dispatch(policy.webhook, "output_blocked", tenant_id, "openai", output_result.guardrail, output_result.reason, body.get("messages", []))
            raise HTTPException(status_code=502, detail={
                "error": "response_blocked",
                "reason": output_result.reason,
                "guardrail": output_result.guardrail,
            })
        response_data["choices"] = output_result.data

        # Async toxicity check (wordlist, OpenAI Moderation, or Perspective)
        response_text = get_text(response_data["choices"][0].get("message", {}).get("content", "")) if response_data["choices"] else ""
        tox_error = await toxicity.check(response_text, policy.output.toxicity)
        if tox_error:
            audit.log(tenant_id, "output_blocked", tox_error, body, "openai")
            webhook.dispatch(policy.webhook, "output_blocked", tenant_id, "openai", "toxicity", tox_error, body.get("messages", []))
            raise HTTPException(status_code=502, detail={"error": "response_blocked", "reason": tox_error, "guardrail": "toxicity"})

    audit.log(tenant_id, "passed", None, body, "openai")
    return JSONResponse(response_data)


# ── Anthropic route ───────────────────────────────────────────────────────────

@app.post("/v1/messages")
async def proxy_anthropic(
    request: Request,
    x_api_key: Optional[str] = Header(None),
    x_tenant_id: Optional[str] = Header(None),
    anthropic_version: Optional[str] = Header(None),
):
    # ── Auth ──
    proxy_key = keystore.validate(x_api_key)
    if proxy_key is None:
        raise HTTPException(status_code=401, detail={
            "type": "error",
            "error": {"type": "authentication_error", "message": "Invalid or missing proxy key."},
        })

    tenant_id = (
        proxy_key.tenant_id
        if proxy_key.tenant_id != "__from_header__"
        else (x_tenant_id or "default")
    )
    body = await request.json()

    policy = PolicyEngine.load_for_tenant(tenant_id)
    pipeline = GuardrailPipeline(policy)

    rl = limiter.check(tenant_id, policy.rate_limit)
    if rl.limited:
        audit.log(tenant_id, "rate_limited", rl.reason, body, "anthropic")
        webhook.dispatch(policy.webhook, "rate_limited", tenant_id, "anthropic", None, rl.reason, body.get("messages", []))
        raise HTTPException(
            status_code=429,
            detail={
                "type": "error",
                "error": {"type": "rate_limit_error", "message": rl.reason},
            },
            headers={"Retry-After": str(rl.retry_after)},
        )

    # Anthropic puts the system prompt as a top-level field, not in messages
    system_prompt = body.get("system")

    input_result = pipeline.run_input(body.get("messages", []), system=system_prompt)
    if input_result.blocked:
        audit.log(tenant_id, "input_blocked", input_result.reason, body, "anthropic")
        webhook.dispatch(policy.webhook, "input_blocked", tenant_id, "anthropic", input_result.guardrail, input_result.reason, body.get("messages", []))
        raise HTTPException(status_code=400, detail={
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": f"Request blocked by guardrail ({input_result.guardrail}): {input_result.reason}",
            }
        })

    body["messages"] = input_result.messages
    if input_result.system is not None:
        body["system"] = input_result.system

    # Use the policy's stored upstream key if configured, else forward the client's key
    upstream_api_key = policy.upstream_api_key or x_api_key
    forward_headers = {
        "Content-Type": "application/json",
        "anthropic-version": anthropic_version or ANTHROPIC_DEFAULT_VERSION,
        **({"x-api-key": upstream_api_key} if upstream_api_key else {}),
    }

    upstream_url = f"{ANTHROPIC_API_BASE}/v1/messages"

    # Streaming path
    if body.get("stream"):
        return stream_anthropic(upstream_url, body, forward_headers, pipeline, tenant_id, audit, webhook, policy.webhook, toxicity, policy.output.toxicity)

    # Non-streaming path
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(upstream_url, json=body, headers=forward_headers)
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Upstream error: {e}")

    response_data = resp.json()

    # Anthropic response: {"content": [{"type": "text", "text": "..."}], ...}
    if "content" in response_data and isinstance(response_data["content"], list):
        output_result = pipeline.run_output_anthropic(response_data["content"])
        if output_result.blocked:
            audit.log(tenant_id, "output_blocked", output_result.reason, body, "anthropic")
            webhook.dispatch(policy.webhook, "output_blocked", tenant_id, "anthropic", output_result.guardrail, output_result.reason, body.get("messages", []))
            raise HTTPException(status_code=502, detail={
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": f"Response blocked by guardrail ({output_result.guardrail}): {output_result.reason}",
                }
            })
        response_data["content"] = output_result.data

    # Async toxicity check
    if "content" in response_data and isinstance(response_data["content"], list):
        response_text = " ".join(
            b.get("text", "") for b in response_data["content"] if b.get("type") == "text"
        )
        tox_error = await toxicity.check(response_text, policy.output.toxicity)
        if tox_error:
            audit.log(tenant_id, "output_blocked", tox_error, body, "anthropic")
            webhook.dispatch(policy.webhook, "output_blocked", tenant_id, "anthropic", "toxicity", tox_error, body.get("messages", []))
            raise HTTPException(status_code=502, detail={
                "type": "error",
                "error": {"type": "api_error", "message": f"Response blocked by guardrail (toxicity): {tox_error}"},
            })

    audit.log(tenant_id, "passed", None, body, "anthropic")
    return JSONResponse(response_data)


# ── Utility routes ────────────────────────────────────────────────────────────

@app.get("/audit")
async def get_audit_log(
    limit: int = 50,
    tenant_id: Optional[str] = None,
    api: Optional[str] = None,
):
    """View recent audit log entries. Filter by tenant_id or api (openai|anthropic)."""
    return audit.get_recent(limit=limit, tenant_id=tenant_id, api=api)


@app.get("/rate-limits")
async def get_rate_limits(tenant_id: str = "default"):
    """Current rate limit usage for a tenant."""
    policy = PolicyEngine.load_for_tenant(tenant_id)
    return limiter.status(tenant_id, policy.rate_limit)


@app.get("/stats")
async def get_stats(tenant_id: Optional[str] = None):
    """Aggregated stats for the dashboard."""
    return audit.get_stats(tenant_id=tenant_id)


@app.get("/auth-status")
async def auth_status():
    """Shows whether proxy auth is enabled. Does not expose keys."""
    return {
        "auth_enabled": keystore.auth_enabled,
        "message": (
            "Auth is enforced. All requests require a valid proxy key."
            if keystore.auth_enabled
            else "Auth is disabled (dev mode). Create keys.yaml to enable it."
        ),
    }


@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.2.0", "timestamp": time.time()}


# ── Policy editor routes ───────────────────────────────────────────────────────

_POLICY_FILE = os.path.join(os.path.dirname(__file__), "policy.yaml")


def _load_raw_policy() -> dict:
    with open(_POLICY_FILE) as f:
        return yaml.safe_load(f) or {}


def _save_raw_policy(raw: dict) -> None:
    with open(_POLICY_FILE, "w") as f:
        yaml.safe_dump(raw, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


@app.get("/policy")
async def list_policy():
    """Return all tenant names and their raw configs."""
    raw = _load_raw_policy()
    return {"tenants": list(raw.get("tenants", {}).keys()), "config": raw.get("tenants", {})}


@app.get("/policy/{tenant_id}")
async def get_tenant_policy(tenant_id: str):
    """Return raw config for a specific tenant."""
    raw = _load_raw_policy()
    tenants = raw.get("tenants", {})
    if tenant_id not in tenants:
        raise HTTPException(status_code=404, detail=f"Tenant '{tenant_id}' not found")
    return tenants[tenant_id]


@app.put("/policy/{tenant_id}")
async def update_tenant_policy(tenant_id: str, request: Request):
    """Overwrite config for a tenant in policy.yaml. Creates tenant if it doesn't exist."""
    updates = await request.json()
    raw = _load_raw_policy()
    if "tenants" not in raw:
        raw["tenants"] = {}
    raw["tenants"][tenant_id] = updates
    _save_raw_policy(raw)
    return {"status": "ok", "tenant_id": tenant_id}


@app.delete("/policy/{tenant_id}")
async def delete_tenant_policy(tenant_id: str):
    """Remove a tenant from policy.yaml. Cannot delete 'default'."""
    if tenant_id == "default":
        raise HTTPException(status_code=400, detail="Cannot delete the default tenant policy")
    raw = _load_raw_policy()
    tenants = raw.get("tenants", {})
    if tenant_id not in tenants:
        raise HTTPException(status_code=404, detail=f"Tenant '{tenant_id}' not found")
    del raw["tenants"][tenant_id]
    _save_raw_policy(raw)
    return {"status": "ok", "tenant_id": tenant_id}
