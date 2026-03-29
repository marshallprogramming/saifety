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
import shutil
import httpx
import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException, Header

load_dotenv()  # no-op when .env doesn't exist (e.g. production)
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from typing import Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import RedirectResponse

from policy import PolicyEngine
from pipeline import GuardrailPipeline
from audit import AuditLogger
from streaming import stream_openai, stream_anthropic
from rate_limiter import RateLimiter
from auth import KeyStore, ProxyKey, _extract_bearer
from webhooks import WebhookDispatcher
from toxicity import ToxicityChecker
from guardrails.content_utils import get_text
import dashboard_auth as dash_auth
from users import UserStore, PLANS, _POLICY_LOCK
from billing import create_checkout_session, create_billing_portal_session
from billing import handle_webhook as _stripe_webhook
from email_utils import send_password_reset

app = FastAPI(title="AI Guardrail Proxy", version="0.7.0")


@app.on_event("startup")
async def _seed_data_dir():
    """
    When DATA_DIR points to a fresh Fly volume, copy the bundled policy.yaml
    into it so the app has a working default config on first boot.
    """
    data_dir = os.environ.get("DATA_DIR")
    if not data_dir:
        return
    os.makedirs(data_dir, exist_ok=True)
    dest = os.path.join(data_dir, "policy.yaml")
    src  = os.path.join(os.path.dirname(__file__), "policy.yaml")
    if not os.path.exists(dest) and os.path.exists(src):
        shutil.copy(src, dest)
        print(f"[startup] Seeded policy.yaml into {data_dir}")


class DashboardAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        if dash_auth.is_proxy_path(path) or path in dash_auth.PUBLIC_PATHS:
            return await call_next(request)

        # Dev mode — no password set and no user accounts required
        if not dash_auth.auth_enabled():
            request.state.is_admin = True
            request.state.user_id = None
            request.state.tenant_id = None
            return await call_next(request)

        token = request.cookies.get("dash_session")
        session_user = dash_auth.get_session_user(token)

        if session_user is None:
            accept = request.headers.get("accept", "")
            if "text/html" in accept:
                return RedirectResponse(url="/login", status_code=302)
            return JSONResponse({"error": "unauthenticated"}, status_code=401)

        if session_user == "admin":
            request.state.is_admin = True
            request.state.user_id = None
            request.state.tenant_id = None
        else:
            request.state.is_admin = False
            request.state.user_id = session_user
            user = userstore.get_by_id(session_user)
            request.state.tenant_id = user.tenant_id if user else None

        return await call_next(request)


app.add_middleware(DashboardAuthMiddleware)
audit = AuditLogger()
limiter = RateLimiter()
keystore = KeyStore()
webhook = WebhookDispatcher()
toxicity = ToxicityChecker()
userstore = UserStore()

_DASHBOARD_DIR = os.path.join(os.path.dirname(__file__), "dashboard")


@app.get("/")
async def dashboard():
    return FileResponse(os.path.join(_DASHBOARD_DIR, "index.html"))


@app.get("/login")
async def login_page():
    return FileResponse(os.path.join(_DASHBOARD_DIR, "login.html"))


@app.post("/login")
async def login(request: Request):
    form = await request.form()
    email    = form.get("email", "").strip().lower()
    password = form.get("password", "")

    # User email+password login
    if email:
        user = userstore.authenticate(email, password)
        if user:
            token = dash_auth.create_session(user.id)
            response = RedirectResponse(url="/account", status_code=302)
            response.set_cookie("dash_session", token, httponly=True, samesite="lax", max_age=86400 * 30)
            return response
        return FileResponse(os.path.join(_DASHBOARD_DIR, "login.html"), status_code=401)

    # Admin password-only login
    if dash_auth.check_password(password):
        token = dash_auth.create_session("admin")
        response = RedirectResponse(url="/", status_code=302)
        response.set_cookie("dash_session", token, httponly=True, samesite="lax", max_age=86400)
        return response

    return FileResponse(os.path.join(_DASHBOARD_DIR, "login.html"), status_code=401)


@app.get("/signup")
async def signup_page():
    return FileResponse(os.path.join(_DASHBOARD_DIR, "signup.html"))


@app.post("/signup")
async def signup(request: Request):
    form = await request.form()
    email    = form.get("email", "").strip().lower()
    password = form.get("password", "")

    if not email or not password or len(password) < 8:
        return RedirectResponse(url="/signup?error=invalid", status_code=302)

    user = userstore.create_user(email, password)
    if user is None:
        return RedirectResponse(url="/signup?error=email_taken", status_code=302)

    token = dash_auth.create_session(user.id)
    response = RedirectResponse(url="/account?new=1", status_code=302)
    response.set_cookie("dash_session", token, httponly=True, samesite="lax", max_age=86400 * 30)
    return response


@app.get("/logout")
async def logout(request: Request):
    token = request.cookies.get("dash_session")
    if token:
        dash_auth.revoke_session(token)
    response = RedirectResponse(url="/login", status_code=302)
    response.delete_cookie("dash_session")
    return response


@app.get("/forgot-password")
async def forgot_password_page():
    return FileResponse(os.path.join(_DASHBOARD_DIR, "forgot-password.html"))


@app.post("/forgot-password")
async def forgot_password(request: Request):
    form = await request.form()
    email = form.get("email", "").strip().lower()
    if not email:
        return RedirectResponse(url="/forgot-password?error=invalid", status_code=302)

    token = userstore.create_reset_token(email)
    if token:
        base = str(request.base_url).rstrip("/")
        reset_url = f"{base}/reset-password?token={token}"
        try:
            send_password_reset(email, reset_url)
        except Exception:
            pass  # fail silently — don't reveal email validity

    # Always redirect to the same confirmation page to avoid email enumeration
    return RedirectResponse(url="/forgot-password?sent=1", status_code=302)


@app.get("/reset-password")
async def reset_password_page():
    return FileResponse(os.path.join(_DASHBOARD_DIR, "reset-password.html"))


@app.post("/reset-password")
async def reset_password(request: Request):
    form = await request.form()
    token    = form.get("token", "").strip()
    password = form.get("password", "")

    if not token or len(password) < 8:
        return RedirectResponse(
            url=f"/reset-password?token={token}&error=invalid", status_code=302
        )

    success = userstore.use_reset_token(token, password)
    if not success:
        return RedirectResponse(
            url=f"/reset-password?token={token}&error=expired", status_code=302
        )

    return RedirectResponse(url="/login?reset=1", status_code=302)


@app.get("/account")
async def account_page():
    return FileResponse(os.path.join(_DASHBOARD_DIR, "account.html"))


@app.get("/account/data")
async def account_data(request: Request):
    user_id = getattr(request.state, "user_id", None)
    if not user_id:
        raise HTTPException(status_code=403, detail="Not a user session")
    user = userstore.get_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404)
    stats = audit.get_stats(tenant_id=user.tenant_id)
    monthly_used = audit.get_monthly_request_count(user.tenant_id)
    return {
        "email":              user.email,
        "proxy_key":          user.proxy_key,
        "tenant_id":          user.tenant_id,
        "plan":               user.plan,
        "plan_info":          PLANS.get(user.plan, PLANS["free"]),
        "has_ai_key":         user.has_ai_key,
        "stats":              stats,
        "monthly_used":       monthly_used,
    }


@app.post("/account/ai-key")
async def save_ai_key(request: Request):
    user_id = getattr(request.state, "user_id", None)
    if not user_id:
        raise HTTPException(status_code=403)
    form = await request.form()
    ai_key = form.get("ai_key", "").strip()
    if not ai_key:
        raise HTTPException(status_code=400, detail="AI key required")
    userstore.set_ai_key(user_id, ai_key)
    return RedirectResponse(url="/account?saved=1", status_code=302)


@app.post("/billing/checkout")
async def billing_checkout(request: Request):
    user_id = getattr(request.state, "user_id", None)
    if not user_id:
        raise HTTPException(status_code=403)
    form = await request.form()
    plan = form.get("plan", "")
    if plan not in ("starter", "growth"):
        raise HTTPException(status_code=400, detail="Invalid plan")
    user = userstore.get_by_id(user_id)
    base = str(request.base_url).rstrip("/")
    try:
        url = create_checkout_session(
            user, plan,
            success_url=f"{base}/account?checkout=success",
            cancel_url=f"{base}/account",
        )
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=503, detail=str(e))
    return RedirectResponse(url=url, status_code=303)


@app.post("/billing/webhook")
async def billing_webhook(request: Request):
    """Stripe webhook — must be in PUBLIC_PATHS (no session auth)."""
    payload = await request.body()
    sig = request.headers.get("stripe-signature", "")
    try:
        result = _stripe_webhook(payload, sig, userstore)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return JSONResponse(result)


@app.get("/billing/portal")
async def billing_portal(request: Request):
    user_id = getattr(request.state, "user_id", None)
    if not user_id:
        raise HTTPException(status_code=403)
    user = userstore.get_by_id(user_id)
    if not user or not user.stripe_customer_id:
        raise HTTPException(status_code=400, detail="No billing account found. Complete a checkout first.")
    base = str(request.base_url).rstrip("/")
    try:
        url = create_billing_portal_session(user, return_url=f"{base}/account")
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=503, detail=str(e))
    return RedirectResponse(url=url, status_code=303)


@app.get("/auth-config")
async def auth_config():
    """Tells the dashboard whether password auth is active."""
    return {"dashboard_auth_enabled": dash_auth.auth_enabled()}

ANTHROPIC_API_BASE = "https://api.anthropic.com"
ANTHROPIC_DEFAULT_VERSION = "2023-06-01"


def _check_monthly_limit(tenant_id: str) -> Optional[str]:
    """
    Returns an error message if the tenant has hit their monthly request cap,
    or None if they are within limits (or are a self-hosted / admin tenant).
    Only applies to user-provisioned tenants that have an associated account.
    """
    user = userstore.get_by_tenant_id(tenant_id)
    if user is None:
        return None  # not a SaaS user — no cap
    plan = PLANS.get(user.plan, PLANS["free"])
    cap = plan.get("monthly_requests")
    if not cap:
        return None
    used = audit.get_monthly_request_count(tenant_id)
    if used >= cap:
        return (
            f"Monthly request limit reached ({cap:,} requests). "
            f"Upgrade your plan at saifety.dev/account."
        )
    return None


# ── OpenAI route ──────────────────────────────────────────────────────────────

@app.post("/v1/chat/completions")
async def proxy_openai(
    request: Request,
    authorization: Optional[str] = Header(None),
    x_tenant_id: Optional[str] = Header(None),
):
    # ── Auth ──
    bearer = _extract_bearer(authorization)
    proxy_key = keystore.validate(bearer)
    if proxy_key is None:
        # Fallback: check user-provisioned proxy keys
        user = userstore.get_by_proxy_key(bearer) if bearer else None
        if user is None:
            raise HTTPException(status_code=401, detail={"error": "invalid_api_key",
                                                          "message": "Invalid or missing proxy key."})
        proxy_key = ProxyKey(key=bearer, name=f"user:{user.email}", tenant_id=user.tenant_id)

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

    monthly_err = _check_monthly_limit(tenant_id)
    if monthly_err:
        audit.log(tenant_id, "rate_limited", monthly_err, body, "openai")
        raise HTTPException(status_code=429, detail={"error": "monthly_limit_reached", "reason": monthly_err})

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

    usage = response_data.get("usage", {})
    openai_usage = {
        "prompt_tokens":    usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
    } if usage else None
    audit.log(tenant_id, "passed", None, body, "openai", openai_usage)
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
        # Fallback: check user-provisioned proxy keys
        user = userstore.get_by_proxy_key(x_api_key) if x_api_key else None
        if user is None:
            raise HTTPException(status_code=401, detail={
                "type": "error",
                "error": {"type": "authentication_error", "message": "Invalid or missing proxy key."},
            })
        proxy_key = ProxyKey(key=x_api_key, name=f"user:{user.email}", tenant_id=user.tenant_id)

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

    monthly_err = _check_monthly_limit(tenant_id)
    if monthly_err:
        audit.log(tenant_id, "rate_limited", monthly_err, body, "anthropic")
        raise HTTPException(status_code=429, detail={
            "type": "error",
            "error": {"type": "rate_limit_error", "message": monthly_err},
        })

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

    usage = response_data.get("usage", {})
    anthropic_usage = {
        "prompt_tokens":    usage.get("input_tokens"),
        "completion_tokens": usage.get("output_tokens"),
    } if usage else None
    audit.log(tenant_id, "passed", None, body, "anthropic", anthropic_usage)
    return JSONResponse(response_data)


# ── Utility routes ────────────────────────────────────────────────────────────

@app.get("/audit")
async def get_audit_log(
    request: Request,
    limit: int = 50,
    tenant_id: Optional[str] = None,
    api: Optional[str] = None,
):
    """View recent audit log entries. Filter by tenant_id or api (openai|anthropic)."""
    scoped = getattr(request.state, "tenant_id", None)
    return audit.get_recent(limit=limit, tenant_id=scoped or tenant_id, api=api)


@app.get("/rate-limits")
async def get_rate_limits(request: Request, tenant_id: str = "default"):
    """Current rate limit usage for a tenant."""
    scoped = getattr(request.state, "tenant_id", None)
    tid = scoped or tenant_id
    policy = PolicyEngine.load_for_tenant(tid)
    return limiter.status(tid, policy.rate_limit)


@app.get("/stats")
async def get_stats(request: Request, tenant_id: Optional[str] = None):
    """Aggregated stats for the dashboard."""
    scoped = getattr(request.state, "tenant_id", None)
    return audit.get_stats(tenant_id=scoped or tenant_id)


@app.get("/metrics")
async def get_metrics(request: Request, tenant_id: Optional[str] = None, days: int = 7):
    """Token usage and request counts per tenant over the last N days."""
    scoped = getattr(request.state, "tenant_id", None)
    return audit.get_token_metrics(tenant_id=scoped or tenant_id, days=days)


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

_DATA_DIR    = os.environ.get("DATA_DIR", os.path.dirname(__file__))
_POLICY_FILE = os.path.join(_DATA_DIR, "policy.yaml")


def _load_raw_policy() -> dict:
    with _POLICY_LOCK:
        with open(_POLICY_FILE) as f:
            return yaml.safe_load(f) or {}


def _save_raw_policy(raw: dict) -> None:
    with _POLICY_LOCK:
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
