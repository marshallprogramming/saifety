"""
Prompt injection detection — fully deterministic, no LLM in the loop.

Defence layers, applied to a normalized copy of the text (the forwarded
message is never modified):

1. Normalization — NFKC unicode fold (fullwidth chars, mathematical
   alphanumerics), strip zero-width and bidi control characters, map common
   Latin-lookalike homoglyphs (Cyrillic/Greek), collapse whitespace.
2. Weighted signatures — each matched category adds its weight; the request
   is blocked when the total score reaches the configured threshold
   (default 1.0, so one strong signal or two weak ones).
3. Obfuscation probes — base64 blobs are decoded and re-scanned, a leetspeak
   fold is re-scanned, and the mere presence of bidi overrides or zero-width
   characters spliced inside words scores as a signal on its own.

Works with both string and Anthropic content block array formats.
"""

import re
import base64
import binascii
import unicodedata
from guardrails.base import GuardrailResult
from guardrails.content_utils import get_text
from policy import InjectionConfig

# ── Normalization ─────────────────────────────────────────────────────────────

# Zero-width / invisible characters used to splice trigger words apart.
_INVISIBLE = {0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF, 0x00AD, 0x180E}
# Bidirectional control characters (trojan-source style reordering).
_BIDI = set(range(0x202A, 0x202F)) | set(range(0x2066, 0x206A))
_STRIP = _INVISIBLE | _BIDI

# Common Latin-lookalike homoglyphs (Cyrillic, Greek). Detection-only —
# mangling legitimate Cyrillic/Greek prose is fine because we never forward
# the normalized text.
_HOMOGLYPHS = str.maketrans({
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "х": "x", "у": "y",
    "і": "i", "ѕ": "s", "ј": "j", "һ": "h", "ԁ": "d", "ɡ": "g",
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O",
    "Р": "P", "С": "C", "Т": "T", "Х": "X",
    "α": "a", "ο": "o", "ι": "i", "ν": "v", "κ": "k", "τ": "t", "ρ": "p",
    "υ": "u", "ε": "e",
})

# Leetspeak fold, applied as a second scan pass only.
_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s",
                       "7": "t", "@": "a", "$": "s", "!": "i"})

_ZW_INSIDE_WORD = re.compile(
    "[A-Za-z][\u200B\u200C\u200D\u2060\uFEFF\u00AD]+[A-Za-z]"
)
_BIDI_RE = re.compile("[\u202A-\u202E\u2066-\u2069]")


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = "".join(ch for ch in text if ord(ch) not in _STRIP)
    text = text.translate(_HOMOGLYPHS)
    return re.sub(r"\s+", " ", text)


# ── Weighted signatures ───────────────────────────────────────────────────────
# (category, weight, pattern). Weight 1.0 blocks on its own at the default
# threshold; 0.5 signals need to co-occur with another signal.

_SIGNATURES = [
    ("instruction override", 1.0,
     r"\b(ignore|disregard|bypass|override|forget|discard)\b[^.!?\n]{0,40}"
     r"\b(previous|prior|above|earlier|initial|original|system|all)\b[^.!?\n]{0,30}"
     r"\b(instructions?|prompts?|directives?|rules?|guidelines?|constraints?|messages?|context)\b"),
    ("instruction override", 1.0,
     r"\bforget (everything|all previous|all prior|your instructions)\b"),
    ("instruction override", 1.0,
     r"\byour new (instructions?|persona|role|task|identity) (is|are)\b"),
    ("instruction override", 1.0,
     r"\b(new|updated|revised) (instructions?|rules?|directives?) *(:|is\b|are\b|follow\b)"),
    ("instruction override", 0.5,
     r"\bfrom now on,? you (are|will|must)\b"),
    ("instruction override", 0.5,
     r"\byou must (obey|comply with) (me|my|all)\b"),

    ("persona hijack", 1.0, r"\bDAN\b"),
    ("persona hijack", 1.0, r"\bdo anything now\b"),
    ("persona hijack", 1.0, r"\bjail\s*break(ing|s|ed)?\b"),
    ("persona hijack", 1.0, r"\bdeveloper mode\b"),
    ("persona hijack", 1.0,
     r"\bpretend (you('re| are)|to be) (not |a different |an? )?"
     r"(unrestricted|unfiltered|uncensored|evil|malicious|amoral)"),
    ("persona hijack", 1.0,
     r"\bact as (an? )?(unrestricted|unfiltered|uncensored|amoral|evil)\b"),
    ("persona hijack", 0.5,
     r"\byou are now (a|an|the|no longer|free|unrestricted|unfiltered)\b"),
    ("persona hijack", 0.5,
     r"\bwithout (any )?(restrictions?|filters?|limitations?|censorship|guidelines?)\b"),
    ("persona hijack", 0.5,
     r"\bno (ethical|moral) (guidelines?|constraints?|restrictions?|limits?)\b"),

    ("prompt extraction", 1.0,
     r"\b(print|repeat|output|show|reveal|display|share|paste|leak|tell me)\b[^.!?\n]{0,30}"
     r"\b(system prompt|initial instructions?|base prompt|hidden (prompt|instructions?)"
     r"|your (instructions?|prompt|guidelines))"),
    ("prompt extraction", 1.0,
     r"\bwhat (are|were) your (original |initial |system |hidden )?instructions\b"),
    ("prompt extraction", 0.5,
     r"\brepeat (everything|all( of)? the text|the words) above\b"),

    ("delimiter injection", 1.0, r"</?(system|user|assistant|instructions?)>"),
    ("delimiter injection", 1.0, r"<\|im_(start|end)\|>"),
    ("delimiter injection", 1.0, r"\[/?INST\]"),
    ("delimiter injection", 1.0, r"(?m)^#{1,4} *(system|instructions?) *$"),
    ("delimiter injection", 1.0,
     r"\b(BEGIN|END) (SYSTEM|HIDDEN|SECRET) (PROMPT|MESSAGE|INSTRUCTIONS?)\b"),
]

_COMPILED = [(cat, w, re.compile(p, re.IGNORECASE)) for cat, w, p in _SIGNATURES]

# Long base64-looking runs; decoded and re-scanned for hidden payloads.
_BASE64_RE = re.compile(r"[A-Za-z0-9+/]{24,}={0,2}")


def _scan(text: str) -> tuple[float, list[str]]:
    """Score text against the signature set. Each pattern counts once."""
    score, categories = 0.0, []
    for category, weight, pattern in _COMPILED:
        if pattern.search(text):
            score += weight
            if category not in categories:
                categories.append(category)
    return score, categories


def _decoded_base64_blobs(text: str) -> list[str]:
    decoded = []
    for m in _BASE64_RE.finditer(text):
        blob = m.group()
        try:
            raw = base64.b64decode(blob + "=" * (-len(blob) % 4), validate=True)
            candidate = raw.decode("utf-8")
        except (binascii.Error, UnicodeDecodeError, ValueError):
            continue
        # Only consider blobs that decode to plausible text.
        if candidate and all(ch.isprintable() or ch in "\n\r\t" for ch in candidate):
            decoded.append(candidate)
    return decoded


class InjectionGuardrail:
    def __init__(self, config: InjectionConfig):
        self.config = config
        self.threshold = getattr(config, "threshold", 1.0) or 1.0

    def check(self, messages: list[dict]) -> GuardrailResult:
        for msg in messages:
            if msg.get("role") != "user":
                continue
            raw = get_text(msg.get("content", ""))
            if not raw:
                continue
            score, reasons = self._analyze(raw)
            if score >= self.threshold:
                return GuardrailResult(
                    blocked=True,
                    reason=f"Prompt injection attempt detected ({', '.join(reasons)})",
                    messages=messages,
                )
        return GuardrailResult(blocked=False, messages=messages)

    def _analyze(self, raw: str) -> tuple[float, list[str]]:
        text = _normalize(raw)
        score, categories = _scan(text)
        reasons = list(categories)

        # Leetspeak fold: only score signals the plain scan didn't already find.
        folded = text.translate(_LEET)
        if folded != text:
            leet_score, leet_cats = _scan(folded)
            new = [c for c in leet_cats if c not in categories]
            if leet_score > 0 and new:
                score += leet_score
                reasons.extend(f"{c} (leetspeak)" for c in new)

        # Base64 payloads: decode and re-scan once (no recursion).
        for decoded in _decoded_base64_blobs(text):
            blob_score, _ = _scan(_normalize(decoded))
            if blob_score > 0:
                score += max(blob_score, 1.0)
                reasons.append("base64-encoded payload")
                break

        # Character-level obfuscation is suspicious in its own right.
        if _BIDI_RE.search(raw):
            score += 0.5
            reasons.append("bidirectional control characters")
        if _ZW_INSIDE_WORD.search(raw):
            score += 0.5
            reasons.append("zero-width characters inside words")

        return score, reasons
