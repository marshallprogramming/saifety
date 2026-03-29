# sAIfety

A lightweight server that sits between your application and an AI API, automatically applying safety guardrails to every request and response — without changing your existing code.

Change one line (`base_url`) and every AI call in your app is covered.

> **Live demo & docs:** [saifety.dev](https://saifety.dev)

---

## What it does

- **PII detection** — redacts or blocks emails, phone numbers, SSNs, credit card numbers before they reach the model
- **Prompt injection blocking** — catches jailbreak attempts, instruction-override patterns, and persona hijacking
- **Topic filtering** — blocks requests mentioning keywords you define (competitors, legal topics, pricing, etc.)
- **Toxicity detection** — checks model output for slurs and harmful content (word list, OpenAI Moderation API, or Google Perspective)
- **Output length and schema enforcement** — cap response length, or require responses to match a JSON schema
- **Per-tenant policies** — different rule sets for different apps, environments, or customers in a single YAML file
- **Rate limiting** — per-tenant request budgets (per minute and per hour)
- **Proxy authentication** — clients hold proxy keys; the proxy holds your AI API keys so they never leave your infrastructure
- **Streaming support** — full SSE pass-through with incremental guardrail checks mid-stream
- **Webhook alerts** — HTTP callbacks when guardrails fire, with HMAC-SHA256 payload signing
- **Audit log** — every request recorded to SQLite or Postgres
- **Dashboard** — live traffic, block rates, per-tenant token usage, and a policy editor UI

---

## How it works

```
Your App  ──►  Guardrail Proxy  ──►  OpenAI / Anthropic
               (localhost:8000)
                      │
              ┌───────▼────────┐
              │  Input checks  │   runs before forwarding
              │  • PII         │   block → 400 back to app
              │  • Injection   │
              │  • Topics      │
              └───────┬────────┘
                      │ (passes)
              ┌───────▼────────┐
              │ Output checks  │   runs before returning
              │  • Toxicity    │   block → 502 back to app
              │  • Max length  │
              │  • JSON schema │
              └───────┬────────┘
                      │
              SQLite / Postgres audit log
              + live dashboard at /
```

The proxy speaks the **exact same wire format** as OpenAI and Anthropic — swapping `base_url` is the only code change required.

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your AI API key

```bash
export OPENAI_API_KEY=sk-...
# or for Anthropic:
export ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Start the proxy

```bash
uvicorn main:app --port 8000
```

### 4. Open the dashboard

Visit [http://localhost:8000](http://localhost:8000)

### 5. Point your app at the proxy

Change one line in your existing code — nothing else:

```python
# Before
client = OpenAI(api_key="sk-...")

# After
client = OpenAI(api_key="sk-...", base_url="http://localhost:8000/v1")
```

---

## Integration examples

### Python — OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-...",
    base_url="http://localhost:8000/v1",
)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

### Python — Anthropic SDK

```python
from anthropic import Anthropic

client = Anthropic(
    api_key="sk-ant-...",
    base_url="http://localhost:8000",   # no /v1 — Anthropic adds it
)

response = client.messages.create(
    model="claude-opus-4-6",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}],
)
```

### JavaScript / TypeScript — OpenAI SDK

```typescript
import OpenAI from "openai";

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  baseURL: "http://localhost:8000/v1",
});
```

### JavaScript / TypeScript — Anthropic SDK

```typescript
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
  baseURL: "http://localhost:8000",
});
```

### Streaming

Streaming works exactly as before — pass `stream: true` and the proxy forwards chunks to your client while running incremental checks in the background. If a guardrail fires mid-stream, an error event is sent and the stream terminates.

```python
stream = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")
```

---

## React / frontend apps

AI API keys must never be exposed in browser code. The correct architecture is:

```
Browser (React)  ──►  Your backend server  ──►  Guardrail Proxy  ──►  AI API
```

Your backend holds the proxy key; the proxy holds the AI API key. Both never reach the browser.

```typescript
// server.ts — Express backend
import OpenAI from "openai";

const ai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  baseURL: "http://localhost:8000/v1",
});

app.post("/api/chat", async (req, res) => {
  try {
    const response = await ai.chat.completions.create({
      model: "gpt-4o",
      messages: req.body.messages,
    });
    res.json(response);
  } catch (err: any) {
    // Proxy returns 400 when a guardrail blocks the request
    res.status(err.status || 500).json({ error: err.error?.reason || "error" });
  }
});
```

```tsx
// ChatComponent.tsx
async function sendMessage(text: string) {
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      messages: [...history, { role: "user", content: text }],
    }),
  });

  if (!res.ok) {
    const { error } = await res.json();
    setError(error); // e.g. "Request contains PII: email address"
    return;
  }

  const data = await res.json();
  setHistory((h) => [
    ...h,
    { role: "assistant", content: data.choices[0].message.content },
  ]);
}
```

---

## Guardrails reference

### Input guardrails

| Guardrail            | What it detects                                                                                  | Actions                                                                                                                                       |
| -------------------- | ------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------- |
| **PII detection**    | Email addresses, US phone numbers, SSNs (`XXX-XX-XXXX`), credit card numbers                     | `redact` — replaces in-place (e.g. `[REDACTED_EMAIL]`) before the request reaches the model<br>`block` — rejects the request with a 400 error |
| **Prompt injection** | "Ignore all previous instructions", jailbreak patterns, persona hijacking, instruction overrides | `block`                                                                                                                                       |
| **Topic filter**     | Any keywords you list as off-limits                                                              | `block`                                                                                                                                       |

### Output guardrails

| Guardrail       | What it checks                                | Action        |
| --------------- | --------------------------------------------- | ------------- |
| **Max length**  | Response character count exceeds your limit   | `block` (502) |
| **Toxicity**    | Slurs, hate speech, self-harm encouragement   | `block` (502) |
| **JSON schema** | Response doesn't match the schema you defined | `block` (502) |

---

## Policy configuration

All guardrail rules live in `policy.yaml`. No code changes needed to update a rule.

```yaml
tenants:
  default:
    upstream_url: "https://api.openai.com/v1/chat/completions"
    upstream_api_key: "${OPENAI_API_KEY}" # reads from environment

    input:
      pii:
        enabled: true
        action: redact # silently strip PII before it reaches the model
        types: [email, phone, ssn, credit_card]

      prompt_injection:
        enabled: true
        action: block

      topic_filter:
        enabled: false
        action: block
        blocked_topics: []

    output:
      max_length: null # null = no limit
      toxicity:
        enabled: true
        provider: wordlist # "wordlist" | "openai" | "perspective"
      json_schema: null # null = no schema enforcement

    rate_limit:
      enabled: false
      requests_per_minute: 60
      requests_per_hour: 1000

    webhook:
      enabled: false
      url: "${WEBHOOK_URL}"
      secret: "${WEBHOOK_SECRET}"
      on: [input_blocked, output_blocked, rate_limited]
```

### Multiple tenants

Add as many tenants as you need. Each tenant can have a completely different policy:

```yaml
tenants:
  default:
    # ... base config ...

  customer_chatbot: # strict — public-facing
    input:
      pii:
        enabled: true
        action: block # reject rather than silently redact
      topic_filter:
        enabled: true
        blocked_topics: [competitor, lawsuit, refund, pricing]
    output:
      max_length: 2000
      toxicity:
        enabled: true
        provider: openai # ML-based for customer-facing tenants
        api_key: "${OPENAI_API_KEY}"
    rate_limit:
      enabled: true
      requests_per_minute: 20

  internal_tools: # permissive — internal use only
    input:
      pii:
        enabled: false
      prompt_injection:
        enabled: false
    output:
      toxicity:
        enabled: false

  structured_output: # enforce a JSON response schema
    output:
      json_schema:
        type: object
        required: [answer, confidence]
        properties:
          answer:
            type: string
          confidence:
            type: number
```

### Selecting a tenant at call time

Pass the `X-Tenant-ID` header. If omitted, `default` is used.

```python
client.chat.completions.create(
    model="gpt-4o",
    messages=[...],
    extra_headers={"X-Tenant-ID": "customer_chatbot"},
)
```

### Environment variable substitution

Any value in `policy.yaml` can reference an environment variable:

```yaml
upstream_api_key: "${OPENAI_API_KEY}"
webhook:
  url: "${WEBHOOK_URL}"
  secret: "${WEBHOOK_SECRET}"
```

---

## Proxy authentication

By default the proxy is open (useful for local dev). To require clients to authenticate, create a `keys.yaml` file:

```yaml
# keys.yaml  (gitignored — never commit this)
keys:
  - key: "sk-proxy-abc123"
    tenant_id: "customer_chatbot" # always routes to this tenant
    description: "Production app"

  - key: "sk-proxy-def456"
    tenant_id: "__from_header__" # tenant resolved from X-Tenant-ID header
    description: "Internal tools"
```

Once `keys.yaml` exists, all requests must include a valid proxy key:

```python
# OpenAI route — proxy key in Authorization header
client = OpenAI(api_key="sk-proxy-abc123", base_url="http://localhost:8000/v1")

# Anthropic route — proxy key in x-api-key header
client = Anthropic(api_key="sk-proxy-abc123", base_url="http://localhost:8000")
```

The proxy extracts the upstream AI API key from its own config (`upstream_api_key` in `policy.yaml`) and uses it to call OpenAI/Anthropic. Your AI API keys never travel to clients.

Generate a new key:

```bash
python3 -c "from auth import generate_key; print(generate_key())"
```

---

## Dashboard

Visit [http://localhost:8000](http://localhost:8000) after starting the proxy.

**Overview tab:**

- Total requests, blocked count, and pass rate
- Top block reasons — bar chart of the most common guardrail triggers
- Live activity feed — every request with timestamp, tenant, API (OpenAI / Anthropic), outcome, and block reason
- Token usage table — input tokens, output tokens, and total per tenant over 24h / 7d / 30d
- Filter by tenant or API; auto-refreshes every 10 seconds

**Policy Editor tab:**

- View and edit every tenant's guardrail rules directly in the browser
- Toggle guardrails on/off, change actions, update blocked topics, adjust rate limits
- Create new tenants or delete existing ones
- Changes write to `policy.yaml` immediately — no server restart needed

### Protecting the dashboard

Set `DASHBOARD_PASSWORD` to require a login before accessing the dashboard or any of its API endpoints:

```bash
export DASHBOARD_PASSWORD=your-secret-password
uvicorn main:app --port 8000
```

The dashboard will redirect to a login page. Proxy routes (`/v1/*`) are not affected — they use proxy key auth independently.

Sessions expire after 24 hours and are cleared on server restart. If `DASHBOARD_PASSWORD` is not set, the dashboard is open (dev mode).

---

## Rate limiting

Per-tenant sliding window rate limits. Configure in `policy.yaml`:

```yaml
rate_limit:
  enabled: true
  requests_per_minute: 20
  requests_per_hour: 200
```

When a tenant exceeds their limit, the proxy returns HTTP 429 with a `Retry-After` header. Check current usage:

```
GET /rate-limits?tenant_id=customer_chatbot
```

---

## Webhook alerts

Receive an HTTP POST whenever a guardrail fires:

```yaml
webhook:
  enabled: true
  url: "${WEBHOOK_URL}"
  secret: "${WEBHOOK_SECRET}" # optional — enables HMAC-SHA256 payload signing
  on:
    - input_blocked
    - output_blocked
    - rate_limited
```

Payload format:

```json
{
  "event": "input_blocked",
  "tenant_id": "customer_chatbot",
  "api": "openai",
  "guardrail": "pii",
  "reason": "PII detected: email address",
  "timestamp": 1711234567.89,
  "messages": [{ "role": "user", "content": "..." }]
}
```

When `secret` is set, every request includes an `X-Saifety-Signature: sha256=<hmac>` header so you can verify the payload hasn't been tampered with. Webhooks are delivered asynchronously (fire-and-forget with 3 retries) — they never slow down the request path.

---

## Toxicity detection

Three providers, configured per tenant:

| Provider      | API key required            | Notes                                                                                                       |
| ------------- | --------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `wordlist`    | No                          | Built-in regex patterns. Fast, zero cost. Default.                                                          |
| `openai`      | Yes — `OPENAI_API_KEY`      | [OpenAI Moderation API](https://platform.openai.com/docs/guides/moderation). Free with your OpenAI account. |
| `perspective` | Yes — `PERSPECTIVE_API_KEY` | [Google Perspective API](https://perspectiveapi.com). Configurable score threshold.                         |

```yaml
output:
  toxicity:
    enabled: true
    provider: openai
    api_key: "${OPENAI_API_KEY}"

# Perspective with custom threshold:
output:
  toxicity:
    enabled: true
    provider: perspective
    api_key: "${PERSPECTIVE_API_KEY}"
    threshold: 0.75    # 0.0–1.0, default 0.8
```

All providers **fail open** — if an external API call errors, the proxy logs a warning and passes the content through rather than blocking legitimate requests.

---

## Token usage metrics

Token counts are captured from every successful non-streaming response and stored in the audit log. Query aggregated usage via the API:

```
GET /metrics?days=7
GET /metrics?days=30&tenant_id=customer_chatbot
```

Response:

```json
{
  "period_days": 7,
  "tenants": [
    {
      "tenant_id": "customer_chatbot",
      "requests": 142,
      "input_tokens": 48320,
      "output_tokens": 21150,
      "total_tokens": 69470,
      "by_api": {
        "openai": {
          "requests": 142,
          "input_tokens": 48320,
          "output_tokens": 21150
        }
      }
    }
  ],
  "totals": {
    "requests": 142,
    "input_tokens": 48320,
    "output_tokens": 21150,
    "total_tokens": 69470
  }
}
```

Token usage is also shown in the dashboard Overview tab with a 24h / 7d / 30d selector.

---

## Postgres support

By default, audit logs are written to a local `audit.db` SQLite file. For production, set `DATABASE_URL` to use Postgres:

```bash
export DATABASE_URL=postgresql://user:password@host:5432/dbname
```

Both `postgres://` and `postgresql://` URL schemes are accepted. The table is created automatically on first run.

---

## API reference

| Method   | Path                   | Description                                            |
| -------- | ---------------------- | ------------------------------------------------------ |
| `POST`   | `/v1/chat/completions` | OpenAI-compatible proxy endpoint                       |
| `POST`   | `/v1/messages`         | Anthropic-compatible proxy endpoint                    |
| `GET`    | `/`                    | Dashboard UI                                           |
| `GET`    | `/login`               | Dashboard login page (when auth enabled)               |
| `GET`    | `/logout`              | Clear dashboard session                                |
| `GET`    | `/audit`               | Audit log entries. Params: `limit`, `tenant_id`, `api` |
| `GET`    | `/stats`               | Aggregated pass/block stats. Param: `tenant_id`        |
| `GET`    | `/metrics`             | Token usage by tenant. Params: `days`, `tenant_id`     |
| `GET`    | `/rate-limits`         | Current rate limit usage. Param: `tenant_id`           |
| `GET`    | `/policy`              | All tenant configs                                     |
| `GET`    | `/policy/{tenant_id}`  | Single tenant config                                   |
| `PUT`    | `/policy/{tenant_id}`  | Update a tenant's config                               |
| `DELETE` | `/policy/{tenant_id}`  | Delete a tenant                                        |
| `GET`    | `/auth-status`         | Whether proxy key auth is enabled                      |
| `GET`    | `/health`              | Health check                                           |

---

## File structure

```
ai-guardrail-proxy/
│
├── main.py               # FastAPI app — proxy routes, middleware, dashboard routes
├── pipeline.py           # Orchestrates guardrail execution
├── policy.py             # Loads and parses policy.yaml into typed config objects
├── audit.py              # Dual-backend audit logger (SQLite + Postgres)
├── rate_limiter.py       # Sliding window rate limiter
├── auth.py               # Proxy key store and validation
├── webhooks.py           # Async fire-and-forget webhook dispatcher
├── toxicity.py           # Async toxicity checker (wordlist / OpenAI / Perspective)
├── streaming.py          # SSE streaming pass-through with incremental guardrail checks
├── dashboard_auth.py     # Dashboard session management
│
├── policy.yaml           # All guardrail rules — edit this, not the code
├── keys.yaml.example     # Template for proxy key configuration
├── requirements.txt
│
├── guardrails/
│   ├── pii.py            # PII detection and redaction
│   ├── injection.py      # Prompt injection pattern matching
│   ├── topic_filter.py   # Keyword-based topic blocking
│   ├── output_validator.py  # Max length and JSON schema validation
│   └── content_utils.py  # Normalises OpenAI string vs Anthropic block-array content
│
├── dashboard/
│   ├── index.html        # Single-file dashboard (no build step)
│   └── login.html        # Login page
│
└── site/
    └── index.html        # Marketing landing page (saifety.dev)
```

---

## Environment variables

| Variable              | Required           | Description                                                                |
| --------------------- | ------------------ | -------------------------------------------------------------------------- |
| `OPENAI_API_KEY`      | If using OpenAI    | Forwarded to OpenAI when `upstream_api_key: "${OPENAI_API_KEY}"` in policy |
| `ANTHROPIC_API_KEY`   | If using Anthropic | Same pattern for Anthropic upstream key                                    |
| `DATABASE_URL`        | No                 | Postgres connection string. SQLite used if unset.                          |
| `DASHBOARD_PASSWORD`  | No                 | Enables dashboard login. Dashboard is open if unset.                       |
| `WEBHOOK_URL`         | No                 | Webhook delivery endpoint. Referenced from `policy.yaml`.                  |
| `WEBHOOK_SECRET`      | No                 | HMAC signing secret for webhook payloads.                                  |
| `PERSPECTIVE_API_KEY` | No                 | Required only when using the Perspective toxicity provider.                |

---

## Deployment

The proxy should run on an internal network — accessible to your backend services but not exposed to the public internet.

```
Internet  ──►  Your app servers  ──►  Guardrail Proxy (internal)  ──►  AI API
```

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

```bash
docker build -t guardrail-proxy .
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=sk-... \
  -e DASHBOARD_PASSWORD=secret \
  -e DATABASE_URL=postgresql://... \
  guardrail-proxy
```

### Minimum production checklist

- [ ] Set `DASHBOARD_PASSWORD` so the policy editor isn't open to the network
- [ ] Create `keys.yaml` with proxy keys for each calling service
- [ ] Set `DATABASE_URL` to a Postgres instance for a durable audit log
- [ ] Add each tenant to `policy.yaml` with appropriate rules
- [ ] Run with `--workers 4` (or more) for concurrency
- [ ] Put Nginx or a load balancer in front for TLS

---

## License

Apache 2.0 — free to use, modify, and distribute. See [LICENSE](LICENSE).
