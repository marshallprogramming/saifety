"""
Policy engine — loads tenant config from policy.yaml and exposes it
as a typed object the pipeline can query.
"""

from __future__ import annotations
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional
import yaml


def _resolve_env(value: Optional[str]) -> Optional[str]:
    """Replace ${VAR_NAME} placeholders with environment variable values."""
    if not value:
        return value
    return re.sub(r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), m.group(0)), value)


@dataclass
class PIIConfig:
    enabled: bool = True
    action: str = "redact"          # "redact" | "block"
    types: list[str] = field(default_factory=lambda: ["email", "phone", "ssn", "credit_card"])


@dataclass
class InjectionConfig:
    enabled: bool = True
    action: str = "block"


@dataclass
class TopicFilterConfig:
    enabled: bool = False
    action: str = "block"
    blocked_topics: list[str] = field(default_factory=list)


@dataclass
class InputPolicy:
    pii: PIIConfig = field(default_factory=PIIConfig)
    injection: InjectionConfig = field(default_factory=InjectionConfig)
    topic_filter: TopicFilterConfig = field(default_factory=TopicFilterConfig)


@dataclass
class OutputPolicy:
    max_length: Optional[int] = None
    toxicity_enabled: bool = False
    toxicity_action: str = "block"
    json_schema: Optional[dict] = None     # if set, validate response against this schema


@dataclass
class RateLimitConfig:
    enabled: bool = False
    requests_per_minute: Optional[int] = None
    requests_per_hour: Optional[int] = None


@dataclass
class WebhookConfig:
    enabled: bool = False
    url: Optional[str] = None
    on: list = field(default_factory=lambda: ["input_blocked", "output_blocked", "rate_limited"])
    secret: Optional[str] = None   # if set, signs payloads with HMAC-SHA256


@dataclass
class Policy:
    tenant_id: str
    upstream_url: str
    upstream_api_key: Optional[str] = None   # if set, used instead of the client's key
    input: InputPolicy = field(default_factory=InputPolicy)
    output: OutputPolicy = field(default_factory=OutputPolicy)
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    webhook: WebhookConfig = field(default_factory=WebhookConfig)


class PolicyEngine:
    _cache: dict[str, Policy] = {}
    _policy_file = os.path.join(os.path.dirname(__file__), "policy.yaml")

    @classmethod
    def load_for_tenant(cls, tenant_id: str) -> Policy:
        raw = cls._load_yaml()
        tenants: dict = raw.get("tenants", {})

        tenant_cfg = tenants.get(tenant_id) or tenants.get("default") or {}
        defaults = tenants.get("default") or {}

        # Merge: tenant overrides default
        merged = {**defaults, **tenant_cfg}

        return cls._parse(tenant_id, merged)

    @classmethod
    def _load_yaml(cls) -> dict:
        with open(cls._policy_file) as f:
            return yaml.safe_load(f) or {}

    @classmethod
    def _parse(cls, tenant_id: str, cfg: dict) -> Policy:
        input_cfg = cfg.get("input", {})
        output_cfg = cfg.get("output", {})
        rl_raw = cfg.get("rate_limit", {})
        wh_raw = cfg.get("webhook", {})

        pii_raw = input_cfg.get("pii", {})
        inj_raw = input_cfg.get("prompt_injection", {})
        topic_raw = input_cfg.get("topic_filter", {})
        tox_raw = output_cfg.get("toxicity", {})

        return Policy(
            tenant_id=tenant_id,
            upstream_url=cfg.get("upstream_url", "https://api.openai.com/v1/chat/completions"),
            upstream_api_key=_resolve_env(cfg.get("upstream_api_key")),
            input=InputPolicy(
                pii=PIIConfig(
                    enabled=pii_raw.get("enabled", True),
                    action=pii_raw.get("action", "redact"),
                    types=pii_raw.get("types", ["email", "phone", "ssn", "credit_card"]),
                ),
                injection=InjectionConfig(
                    enabled=inj_raw.get("enabled", True),
                    action=inj_raw.get("action", "block"),
                ),
                topic_filter=TopicFilterConfig(
                    enabled=topic_raw.get("enabled", False),
                    action=topic_raw.get("action", "block"),
                    blocked_topics=[t.lower() for t in topic_raw.get("blocked_topics", [])],
                ),
            ),
            output=OutputPolicy(
                max_length=output_cfg.get("max_length"),
                toxicity_enabled=tox_raw.get("enabled", False),
                toxicity_action=tox_raw.get("action", "block"),
                json_schema=output_cfg.get("json_schema"),
            ),
            rate_limit=RateLimitConfig(
                enabled=rl_raw.get("enabled", False),
                requests_per_minute=rl_raw.get("requests_per_minute"),
                requests_per_hour=rl_raw.get("requests_per_hour"),
            ),
            webhook=WebhookConfig(
                enabled=wh_raw.get("enabled", False),
                url=_resolve_env(wh_raw.get("url")),
                on=wh_raw.get("on", ["input_blocked", "output_blocked", "rate_limited"]),
                secret=_resolve_env(wh_raw.get("secret")),
            ),
        )
