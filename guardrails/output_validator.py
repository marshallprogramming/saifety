"""
Output validator — runs synchronous checks after the upstream model responds.
Handles both OpenAI (choices array) and Anthropic (content array) response formats.

Checks: max length, optional JSON schema.

Toxicity checking has been moved to toxicity.py (async, ML-capable).
"""

import json
import copy
from typing import Optional
from guardrails.base import GuardrailResult
from guardrails.content_utils import get_text
from policy import OutputPolicy


class OutputValidator:
    def __init__(self, config: OutputPolicy):
        self.config = config

    # ── OpenAI format ─────────────────────────────────────────────────────────

    def check(self, choices: list[dict]) -> GuardrailResult:
        """Validate OpenAI-format choices array."""
        working = copy.deepcopy(choices)
        for choice in working:
            text = get_text(choice.get("message", {}).get("content", ""))
            error = self._validate_text(text)
            if error:
                return GuardrailResult(blocked=True, reason=error, choices=choices)
        return GuardrailResult(blocked=False, choices=working)

    # ── Anthropic format ──────────────────────────────────────────────────────

    def check_anthropic(self, content_blocks: list[dict]) -> GuardrailResult:
        """Validate Anthropic-format content block array."""
        working = copy.deepcopy(content_blocks)
        text = get_text(working)
        error = self._validate_text(text)
        if error:
            return GuardrailResult(blocked=True, reason=error)
        return GuardrailResult(blocked=False, choices=working)

    # ── Streaming (incremental, sync only) ───────────────────────────────────

    def check_stream(self, accumulated_text: str) -> Optional[str]:
        """
        Sync checks on accumulated streaming text (max_length only).
        Toxicity is checked separately via ToxicityChecker.check_stream().
        """
        if self.config.max_length and len(accumulated_text) > self.config.max_length:
            return f"Response exceeds max length ({len(accumulated_text)} > {self.config.max_length} chars)"
        return None

    # ── Shared validation logic ───────────────────────────────────────────────

    def _validate_text(self, text: str) -> Optional[str]:
        if self.config.max_length and len(text) > self.config.max_length:
            return f"Response exceeds max length ({len(text)} > {self.config.max_length} chars)"

        if self.config.json_schema:
            try:
                parsed = json.loads(text)
                return _validate_schema(parsed, self.config.json_schema)
            except json.JSONDecodeError:
                return "Response is not valid JSON (schema validation required)"

        return None


def _validate_schema(data: dict, schema: dict) -> Optional[str]:
    """Minimal JSON schema validator (type + required fields only)."""
    required = schema.get("required", [])
    properties = schema.get("properties", {})

    if not isinstance(data, dict):
        return "expected object at root"

    for key in required:
        if key not in data:
            return f"missing required field: '{key}'"

    type_map = {"string": str, "number": (int, float), "integer": int,
                "boolean": bool, "array": list, "object": dict}

    for key, prop_schema in properties.items():
        if key in data:
            expected_type = prop_schema.get("type")
            if expected_type and expected_type in type_map:
                if not isinstance(data[key], type_map[expected_type]):
                    return f"field '{key}' expected type {expected_type}"

    return None
