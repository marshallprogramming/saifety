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
from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from typing import Optional

from policy import PolicyEngine
from pipeline import GuardrailPipeline
from audit import AuditLogger

app = FastAPI(title="AI Guardrail Proxy", version="0.2.0")
audit = AuditLogger()

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
    tenant_id = x_tenant_id or "default"
    body = await request.json()

    policy = PolicyEngine.load_for_tenant(tenant_id)
    pipeline = GuardrailPipeline(policy)

    input_result = pipeline.run_input(body.get("messages", []))
    if input_result.blocked:
        audit.log(tenant_id, "input_blocked", input_result.reason, body, "openai")
        raise HTTPException(status_code=400, detail={
            "error": "request_blocked",
            "reason": input_result.reason,
            "guardrail": input_result.guardrail,
        })

    body["messages"] = input_result.messages

    forward_headers = {
        "Content-Type": "application/json",
        **({"Authorization": authorization} if authorization else {}),
    }

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
            raise HTTPException(status_code=502, detail={
                "error": "response_blocked",
                "reason": output_result.reason,
                "guardrail": output_result.guardrail,
            })
        response_data["choices"] = output_result.data

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
    tenant_id = x_tenant_id or "default"
    body = await request.json()

    policy = PolicyEngine.load_for_tenant(tenant_id)
    pipeline = GuardrailPipeline(policy)

    # Anthropic puts the system prompt as a top-level field, not in messages
    system_prompt = body.get("system")

    input_result = pipeline.run_input(body.get("messages", []), system=system_prompt)
    if input_result.blocked:
        audit.log(tenant_id, "input_blocked", input_result.reason, body, "anthropic")
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

    forward_headers = {
        "Content-Type": "application/json",
        "anthropic-version": anthropic_version or ANTHROPIC_DEFAULT_VERSION,
        **({"x-api-key": x_api_key} if x_api_key else {}),
    }

    upstream_url = f"{ANTHROPIC_API_BASE}/v1/messages"

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
            raise HTTPException(status_code=502, detail={
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": f"Response blocked by guardrail ({output_result.guardrail}): {output_result.reason}",
                }
            })
        response_data["content"] = output_result.data

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


@app.get("/stats")
async def get_stats(tenant_id: Optional[str] = None):
    """Aggregated stats for the dashboard."""
    return audit.get_stats(tenant_id=tenant_id)


@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.2.0", "timestamp": time.time()}
