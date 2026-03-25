"""
PII detection and redaction.
Detects: email, US phone, SSN, credit card numbers.
Actions: redact (replace with placeholder) or block the request entirely.
Works with both OpenAI (string content) and Anthropic (content block array) formats.
"""

import re
import copy
from guardrails.base import GuardrailResult
from guardrails.content_utils import get_text, apply_text_transform
from policy import PIIConfig

PII_PATTERNS = {
    "email": (
        r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
        "[REDACTED_EMAIL]",
    ),
    "phone": (
        r"\b(?:\+?1[\s\-.]?)?\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{4}\b",
        "[REDACTED_PHONE]",
    ),
    "ssn": (
        r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b",
        "[REDACTED_SSN]",
    ),
    "credit_card": (
        r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|"
        r"6(?:011|5[0-9]{2})[0-9]{12})\b",
        "[REDACTED_CARD]",
    ),
}


class PIIGuardrail:
    def __init__(self, config: PIIConfig):
        self.config = config
        self.active_patterns = {
            ptype: (re.compile(pattern), placeholder)
            for ptype, (pattern, placeholder) in PII_PATTERNS.items()
            if ptype in config.types
        }

    def check(self, messages: list[dict]) -> GuardrailResult:
        redacted_messages = copy.deepcopy(messages)
        found_types: list[str] = []

        for msg in redacted_messages:
            content = msg.get("content")
            if content is None:
                continue

            text = get_text(content)
            msg_found: list[str] = []

            for ptype, (pattern, _) in self.active_patterns.items():
                if pattern.search(text):
                    msg_found.append(ptype)

            if msg_found:
                found_types.extend(msg_found)
                if self.config.action == "redact":
                    def make_redactor(patterns):
                        def redact(t):
                            for _, (pat, placeholder) in patterns.items():
                                t = pat.sub(placeholder, t)
                            return t
                        return redact

                    msg["content"] = apply_text_transform(content, make_redactor(self.active_patterns))

        if found_types:
            if self.config.action == "block":
                return GuardrailResult(
                    blocked=True,
                    reason=f"Request contains PII: {', '.join(set(found_types))}",
                )
            return GuardrailResult(blocked=False, messages=redacted_messages)

        return GuardrailResult(blocked=False, messages=redacted_messages)
