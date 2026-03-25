# AI Guardrail Proxy

A lightweight server that sits between your application and an AI API (OpenAI or Anthropic/Claude), automatically applying safety rules to every request and response — without changing your existing code.

---

## The problem it solves

When you add AI to an application, you immediately face questions like:

- What if a user pastes their credit card number into the chat?
- What if someone tries to manipulate the AI into ignoring your instructions?
- What if the AI says something toxic or off-brand?
- What if a user asks about a competitor, a lawsuit, or pricing you haven't approved?
- How do I audit what's being sent to the AI, and what's coming back?

Every team adds AI ends up building some version of these checks. This proxy means you only build them once, configure them in a YAML file, and every AI call in every app is covered automatically.

---

## How it works

```
Your App  ──►  Guardrail Proxy  ──►  OpenAI / Anthropic
               (localhost:8000)
                      │
                      ▼
               ┌─────────────────┐
               │  Input checks   │  ← runs before forwarding
               │  • PII          │
               │  • Injection    │
               │  • Topic filter │
               └────────┬────────┘
                        │ (blocked = 400 error back to app)
                        ▼
               ┌─────────────────┐
               │  Output checks  │  ← runs before returning
               │  • Max length   │
               │  • Toxicity     │
               │  • JSON schema  │
               └────────┬────────┘
                        │
                        ▼
                  SQLite audit log
                  + dashboard UI
```

The proxy is **transparent** — it speaks the exact same API format as OpenAI and Anthropic, so changing your `base_url` is the only code change needed.

---

## Quickstart

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

**2. Start the proxy**

```bash
uvicorn main:app --port 8000
```

**3. Open the dashboard**

Visit [http://localhost:8000](http://localhost:8000) to see live traffic, block rates, and audit logs.

**4. Point your app at the proxy instead of the AI API directly**

No other code changes needed — see the integration examples below.

---

## Integration examples

### Python — OpenAI SDK

```python
from openai import OpenAI

# Before
client = OpenAI(api_key="sk-...")

# After — one line change
client = OpenAI(api_key="sk-...", base_url="http://localhost:8000/v1")

# All existing calls work exactly the same
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Python — Anthropic SDK

```python
from anthropic import Anthropic

# Before
client = Anthropic(api_key="sk-ant-...")

# After — one line change
client = Anthropic(api_key="sk-ant-...", base_url="http://localhost:8000")

response = client.messages.create(
    model="claude-opus-4-6",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### JavaScript / TypeScript — OpenAI SDK

```typescript
import OpenAI from "openai";

// Before
const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// After — one line change
const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  baseURL: "http://localhost:8000/v1",
});
```

### JavaScript / TypeScript — Anthropic SDK

```typescript
import Anthropic from "@anthropic-ai/sdk";

// Before
const client = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });

// After — one line change
const client = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
  baseURL: "http://localhost:8000",
});
```

---

## Does this work with a React app?

**Yes, but there's an important catch you need to understand first.**

In a typical React app, you have two options for calling an AI API:

### Option A — Via your own backend (recommended)

```
React app  ──►  Your server (Node/Python/etc.)  ──►  Guardrail Proxy  ──►  AI API
```

This is the right architecture. Your server holds the API key (never exposed to the browser), and the proxy sits between your server and the AI. All guardrails apply. This is the setup you'd use in production.

### Option B — Directly from the browser (development only)

```
React app  ──►  Guardrail Proxy  ──►  AI API
```

This can work during development if you run the proxy on your local machine. However, **don't do this in production** — the AI API key travels through the browser, and you can't trust client-side code to honour guardrails (a user could bypass them by calling the AI directly).

### React example (via backend)

If your backend is in Node.js with Express:

```typescript
// server.ts — your backend
import OpenAI from "openai";

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  baseURL: "http://localhost:8000/v1",   // proxy, not OpenAI directly
});

app.post("/api/chat", async (req, res) => {
  const { messages } = req.body;

  try {
    const response = await client.chat.completions.create({
      model: "gpt-4o",
      messages,
    });
    res.json(response);
  } catch (err: any) {
    // The proxy returns 400 if a guardrail blocks the request
    if (err.status === 400) {
      res.status(400).json({ error: err.error.reason });
    } else {
      res.status(500).json({ error: "AI unavailable" });
    }
  }
});
```

```tsx
// ChatComponent.tsx — your React component
export function Chat() {
  const [messages, setMessages] = useState([]);
  const [error, setError] = useState(null);

  async function sendMessage(text: string) {
    const updated = [...messages, { role: "user", content: text }];
    setMessages(updated);

    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: updated }),
    });

    if (!res.ok) {
      const { error } = await res.json();
      setError(error); // e.g. "Request contains PII: email"
      return;
    }

    const data = await res.json();
    setMessages([...updated, data.choices[0].message]);
  }

  return (
    // ...your chat UI
  );
}
```

The React component never talks to the AI API directly — only to your own backend, which routes through the proxy.

---

## Guardrails reference

### Input guardrails (run before forwarding to AI)

| Guardrail | What it catches | Actions |
|---|---|---|
| **PII detection** | Email addresses, US phone numbers, SSNs, credit card numbers | `redact` (replace with `[REDACTED_EMAIL]` etc.) or `block` (reject the request) |
| **Prompt injection** | Attempts to override system instructions — "ignore all previous instructions", jailbreak patterns, persona hijacking | `block` |
| **Topic filter** | Any keywords you configure as off-limits — competitors, legal topics, pricing, etc. | `block` |

### Output guardrails (run before returning to your app)

| Guardrail | What it catches | Actions |
|---|---|---|
| **Max length** | Responses over a character limit you set | `block` |
| **Toxicity** | Slurs, hate speech, self-harm encouragement | `block` |
| **JSON schema** | If you need structured output, validates the model's response matches a schema you define | `block` |

---

## Policy configuration (`policy.yaml`)

Guardrail rules live in `policy.yaml`, not in code. You can have different rule sets for different tenants, environments, or use cases.

```yaml
tenants:

  default:                         # applies to all apps unless overridden
    upstream_url: "https://api.openai.com/v1/chat/completions"
    input:
      pii:
        enabled: true
        action: redact             # silently strip PII before it reaches the model
      prompt_injection:
        enabled: true
        action: block
    output:
      toxicity:
        enabled: true
        action: block

  customer_chatbot:                # stricter rules for a public-facing product
    input:
      pii:
        enabled: true
        action: block              # reject rather than silently redact
      topic_filter:
        enabled: true
        blocked_topics:
          - competitor
          - lawsuit
          - refund
    output:
      max_length: 2000

  internal_tools:                  # permissive rules for internal use
    input:
      pii:
        enabled: false
      prompt_injection:
        enabled: false
    output:
      toxicity:
        enabled: false
```

To select a policy at call time, pass the `X-Tenant-ID` header:

```python
client.chat.completions.create(
    model="gpt-4o",
    messages=[...],
    extra_headers={"X-Tenant-ID": "customer_chatbot"}
)
```

If no header is sent, the `default` policy applies.

---

## API routes

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat endpoint |
| `POST` | `/v1/messages` | Anthropic-compatible messages endpoint |
| `GET` | `/` | Dashboard UI |
| `GET` | `/audit` | Raw audit log (JSON). Query params: `limit`, `tenant_id`, `api` |
| `GET` | `/stats` | Aggregated stats for dashboard. Query param: `tenant_id` |
| `GET` | `/health` | Health check |

---

## Dashboard

Open [http://localhost:8000](http://localhost:8000) after starting the proxy.

The dashboard shows:
- **Total requests, blocked count, and pass rate** — updated every 10 seconds
- **Top block reasons** — bar chart of what's being caught most often
- **Live activity feed** — every request with timestamp, tenant, API used (OpenAI vs Anthropic), outcome, and the reason if blocked
- **Filters** — narrow the feed by tenant or API

---

## File structure

```
ai-guardrail-proxy/
│
├── main.py                  # FastAPI app — two proxy routes + dashboard + audit/stats endpoints
├── pipeline.py              # Orchestrates guardrail execution in order
├── policy.py                # Loads and parses policy.yaml per tenant
├── audit.py                 # SQLite audit logger
├── policy.yaml              # All guardrail rules — edit this, not the code
├── requirements.txt
│
├── guardrails/
│   ├── pii.py               # PII detection and redaction
│   ├── injection.py         # Prompt injection pattern matching
│   ├── topic_filter.py      # Keyword-based topic blocking
│   ├── output_validator.py  # Max length, toxicity, JSON schema
│   └── content_utils.py     # Shared helpers for OpenAI/Anthropic content formats
│
└── dashboard/
    └── index.html           # Single-file dashboard (no build step)
```

---

## Deploying beyond localhost

For production, run the proxy on a server your backend can reach. It does not need to be — and should not be — exposed to the public internet, only to your own backend services.

```
Internet  ──►  Your app servers  ──►  Guardrail Proxy (internal)  ──►  AI API
```

A minimal deployment:

```bash
# On your server
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

For production you'd also want to:
- Put Nginx or a load balancer in front
- Swap the SQLite audit log for PostgreSQL (edit `audit.py`)
- Add authentication on the proxy itself so only your own services can use it
- Run with `--workers 4` for concurrency

---

## What's not included (yet)

- **Streaming responses** — `stream: true` calls currently get the full response before guardrails run
- **Rate limiting** — per-tenant request budgets
- **ML-based toxicity** — the current toxicity check is a word list; swap in a real classifier for production
- **Auth on the proxy itself** — right now anyone who can reach the proxy can use it
