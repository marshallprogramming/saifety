"""
Guardrail pipeline — runs input and output guardrails in order,
short-circuiting on the first block.
Supports both OpenAI (choices) and Anthropic (content blocks) output formats.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

from policy import Policy
from guardrails.pii import PIIGuardrail
from guardrails.injection import InjectionGuardrail
from guardrails.topic_filter import TopicFilterGuardrail
from guardrails.output_validator import OutputValidator


@dataclass
class InputResult:
    blocked: bool = False
    reason: Optional[str] = None
    guardrail: Optional[str] = None
    messages: list = field(default_factory=list)
    # Anthropic-specific: system prompt may also be redacted
    system: Optional[str] = None


@dataclass
class OutputResult:
    blocked: bool = False
    reason: Optional[str] = None
    guardrail: Optional[str] = None
    data: list = field(default_factory=list)   # works for both choices and content blocks


class GuardrailPipeline:
    def __init__(self, policy: Policy):
        self.policy = policy

    def run_input(self, messages: list[dict], system: Optional[str] = None) -> InputResult:
        """
        Run input guardrails on messages (+ optional system prompt for Anthropic).
        system prompt is checked for PII/injection but treated as trusted for topic filter.
        """
        working_messages = [m.copy() for m in messages]
        working_system = system

        # 1. Prompt injection
        if self.policy.input.injection.enabled:
            result = InjectionGuardrail(self.policy.input.injection).check(working_messages)
            if result.blocked:
                return InputResult(blocked=True, reason=result.reason,
                                   guardrail="prompt_injection", messages=messages)

        # 2. Topic filter (user messages only)
        if self.policy.input.topic_filter.enabled:
            result = TopicFilterGuardrail(self.policy.input.topic_filter).check(working_messages)
            if result.blocked:
                return InputResult(blocked=True, reason=result.reason,
                                   guardrail="topic_filter", messages=messages)

        # 3. PII (may redact instead of block; also check system prompt)
        if self.policy.input.pii.enabled:
            pii = PIIGuardrail(self.policy.input.pii)
            result = pii.check(working_messages)
            if result.blocked:
                return InputResult(blocked=True, reason=result.reason,
                                   guardrail="pii", messages=messages)
            working_messages = result.messages

            # Also redact PII from the system prompt if present
            if working_system:
                sys_result = pii.check([{"role": "system", "content": working_system}])
                if sys_result.blocked:
                    return InputResult(blocked=True, reason=sys_result.reason,
                                       guardrail="pii", messages=messages)
                if sys_result.messages:
                    working_system = sys_result.messages[0].get("content", working_system)

        return InputResult(blocked=False, messages=working_messages, system=working_system)

    def check_stream_chunk(self, accumulated_text: str) -> Optional[str]:
        """Check accumulated streaming text. Returns an error string if blocked."""
        return OutputValidator(self.policy.output).check_stream(accumulated_text)

    def run_output_openai(self, choices: list[dict]) -> OutputResult:
        validator = OutputValidator(self.policy.output)
        result = validator.check(choices)
        if result.blocked:
            return OutputResult(blocked=True, reason=result.reason,
                                guardrail="output_validator", data=choices)
        return OutputResult(blocked=False, data=result.choices)

    def run_output_anthropic(self, content_blocks: list[dict]) -> OutputResult:
        validator = OutputValidator(self.policy.output)
        result = validator.check_anthropic(content_blocks)
        if result.blocked:
            return OutputResult(blocked=True, reason=result.reason,
                                guardrail="output_validator", data=content_blocks)
        return OutputResult(blocked=False, data=result.choices)
