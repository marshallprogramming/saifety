from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GuardrailResult:
    blocked: bool = False
    reason: Optional[str] = None
    messages: list = field(default_factory=list)   # for input guardrails
    choices: list = field(default_factory=list)    # for output guardrails
