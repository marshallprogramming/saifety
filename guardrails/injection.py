"""
Prompt injection detection.
Looks for classic jailbreak/override patterns in user messages.
Works with both string and Anthropic content block array formats.
"""

import re
from guardrails.base import GuardrailResult
from guardrails.content_utils import get_text
from policy import InjectionConfig

INJECTION_PATTERNS = [
    r"ignore (all |your )?(previous|prior|above|system|original) (instructions?|prompts?|directives?)",
    r"disregard (all |your )?(previous|prior|above|system|original)",
    r"forget (everything|all|your instructions)",
    r"you are now",
    r"your new (instructions?|persona|role|task) (is|are)",
    r"\bDAN\b",
    r"do anything now",
    r"jailbreak",
    r"pretend you('re| are) (not|a different|an? (unrestricted|unfiltered|evil|malicious))",
    r"(print|repeat|output|show|reveal|tell me) (your |the )?(system prompt|initial instructions|base prompt|instructions)",
    r"what (are|were) your (original |initial |system )?instructions",
    r"</?(system|user|assistant|instructions?)>",
    r"\[INST\]|\[/INST\]",
]

_compiled = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]


class InjectionGuardrail:
    def __init__(self, config: InjectionConfig):
        self.config = config

    def check(self, messages: list[dict]) -> GuardrailResult:
        for msg in messages:
            if msg.get("role") != "user":
                continue
            text = get_text(msg.get("content", ""))
            for pattern in _compiled:
                if pattern.search(text):
                    return GuardrailResult(
                        blocked=True,
                        reason="Prompt injection attempt detected",
                        messages=messages,
                    )

        return GuardrailResult(blocked=False, messages=messages)
