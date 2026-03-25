"""
Policy engine — loads tenant config from policy.yaml and exposes it
as a typed object the pipeline can query.
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Any, Optional
import yaml


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
class Policy:
    tenant_id: str
    upstream_url: str
    input: InputPolicy = field(default_factory=InputPolicy)
    output: OutputPolicy = field(default_factory=OutputPolicy)


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

        pii_raw = input_cfg.get("pii", {})
        inj_raw = input_cfg.get("prompt_injection", {})
        topic_raw = input_cfg.get("topic_filter", {})
        tox_raw = output_cfg.get("toxicity", {})

        return Policy(
            tenant_id=tenant_id,
            upstream_url=cfg.get("upstream_url", "https://api.openai.com/v1/chat/completions"),
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
        )
