"""
PII detection, redaction, and reversible tokenization.

Detects: email, US phone, SSN, credit card numbers (Luhn-validated).

Actions:
  redact   — replace with a static placeholder (e.g. [REDACTED_EMAIL]); one-way
  tokenize — replace with a unique placeholder (e.g. [PII_EMAIL_1]) and keep the
             original value in a per-request PIIVault so it can be reinjected
             into the model's response. The real value never reaches the LLM.
  block    — reject the request entirely

Works with both OpenAI (string content) and Anthropic (content block array) formats.
"""

import re
import copy
from guardrails.base import GuardrailResult
from guardrails.content_utils import get_text, apply_text_transform
from policy import PIIConfig

# Ordered by priority: when two patterns match overlapping text, the earlier
# type claims the span (e.g. digits inside an email aren't also flagged as a phone).
PII_PATTERNS = {
    "email": (
        r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
        "[REDACTED_EMAIL]",
    ),
    "credit_card": (
        r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|"
        r"6(?:011|5[0-9]{2})[0-9]{12})\b",
        "[REDACTED_CARD]",
    ),
    "ssn": (
        r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b",
        "[REDACTED_SSN]",
    ),
    "phone": (
        r"\b(?:\+?1[\s\-.]?)?\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{4}\b",
        "[REDACTED_PHONE]",
    ),
}


def _luhn_ok(number: str) -> bool:
    digits = [int(d) for d in number if d.isdigit()]
    if len(digits) < 13:
        return False
    checksum = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


# Secondary validation applied on top of the regex match.
_VALIDATORS = {"credit_card": _luhn_ok}


def _find_pii(text: str, active_patterns: dict) -> list[tuple[int, int, str, str]]:
    """Return non-overlapping matches as (start, end, ptype, value), in text order."""
    claimed: list[tuple[int, int]] = []
    found: list[tuple[int, int, str, str]] = []
    for ptype, (pattern, _placeholder) in active_patterns.items():
        validator = _VALIDATORS.get(ptype)
        for m in pattern.finditer(text):
            if validator and not validator(m.group()):
                continue
            if any(s < m.end() and m.start() < e for s, e in claimed):
                continue
            claimed.append((m.start(), m.end()))
            found.append((m.start(), m.end(), ptype, m.group()))
    found.sort()
    return found


_TOKEN_RE = re.compile(r"\[PII_([A-Z_]+)_(\d+)\]")


class PIIVault:
    """Per-request mapping of placeholder tokens to original PII values.

    The vault must never outlive a single request — reuse across requests
    would leak one caller's PII into another caller's response.
    """

    def __init__(self):
        self._token_for_value: dict[tuple[str, str], str] = {}
        self._value_for_token: dict[str, str] = {}
        self._counters: dict[str, int] = {}

    @property
    def has_tokens(self) -> bool:
        return bool(self._value_for_token)

    @property
    def mapping(self) -> dict[str, str]:
        return dict(self._value_for_token)

    def reserve_existing(self, text: str) -> None:
        """Bump counters past token-shaped strings already present in the source
        text, so a client-supplied literal [PII_EMAIL_1] can never alias a
        vault entry and be swapped for someone's real value on restore."""
        for m in _TOKEN_RE.finditer(text):
            ptype, n = m.group(1).lower(), int(m.group(2))
            self._counters[ptype] = max(self._counters.get(ptype, 0), n)

    def tokenize(self, ptype: str, value: str) -> str:
        key = (ptype, value)
        token = self._token_for_value.get(key)
        if token is None:
            n = self._counters.get(ptype, 0) + 1
            self._counters[ptype] = n
            token = f"[PII_{ptype.upper()}_{n}]"
            self._token_for_value[key] = token
            self._value_for_token[token] = value
        return token

    def restore(self, text: str) -> str:
        if not self._value_for_token or not text:
            return text
        return _TOKEN_RE.sub(
            lambda m: self._value_for_token.get(m.group(0), m.group(0)), text
        )

    def restore_content(self, content):
        """Restore tokens in string or content-block-array message content."""
        return apply_text_transform(content, self.restore)


class StreamRestorer:
    """Incrementally restore vault tokens in streamed text.

    Holds back any trailing text that could be the start of a token, so a
    placeholder split across SSE chunks is still restored. The holdback is
    bounded by the token length (~25 chars).
    """

    _PARTIAL = re.compile(r"\[(?:P(?:I(?:I(?:_[A-Z_]*\d*)?)?)?)?$")

    def __init__(self, vault: PIIVault):
        self.vault = vault
        self._buf = ""

    def feed(self, text: str) -> str:
        self._buf += text
        m = self._PARTIAL.search(self._buf)
        holdback = m.start() if m else len(self._buf)
        emit, self._buf = self._buf[:holdback], self._buf[holdback:]
        return self.vault.restore(emit)

    def flush(self) -> str:
        out = self.vault.restore(self._buf)
        self._buf = ""
        return out


class PIIGuardrail:
    def __init__(self, config: PIIConfig):
        self.config = config
        self.active_patterns = {
            ptype: (re.compile(pattern), placeholder)
            for ptype, (pattern, placeholder) in PII_PATTERNS.items()
            if ptype in config.types
        }
        # One vault per guardrail instance; the pipeline creates a fresh
        # instance per request, so tokens never cross requests.
        self.vault = PIIVault() if config.action == "tokenize" else None

    def _sanitize(self, text: str) -> tuple[str, list[str]]:
        matches = _find_pii(text, self.active_patterns)
        if not matches:
            return text, []
        if self.vault is not None:
            self.vault.reserve_existing(text)
        out, last, types = [], 0, []
        for start, end, ptype, value in matches:
            out.append(text[last:start])
            if self.vault is not None:
                out.append(self.vault.tokenize(ptype, value))
            else:
                out.append(self.active_patterns[ptype][1])
            types.append(ptype)
            last = end
        out.append(text[last:])
        return "".join(out), types

    def check(self, messages: list[dict]) -> GuardrailResult:
        working = copy.deepcopy(messages)
        found_types: list[str] = []

        for msg in working:
            content = msg.get("content")
            if content is None:
                continue

            msg_found: list[str] = []

            def transform(t: str) -> str:
                new_t, types = self._sanitize(t)
                msg_found.extend(types)
                return new_t

            new_content = apply_text_transform(content, transform)

            if msg_found:
                found_types.extend(msg_found)
                if self.config.action in ("redact", "tokenize"):
                    msg["content"] = new_content

        if found_types and self.config.action == "block":
            return GuardrailResult(
                blocked=True,
                reason=f"Request contains PII: {', '.join(sorted(set(found_types)))}",
            )
        return GuardrailResult(blocked=False, messages=working)
