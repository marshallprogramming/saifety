"""
Guardrail pipeline — runs input and output guardrails in order,
short-circuiting on the first block.
Supports both OpenAI (choices) and Anthropic (content blocks) output formats.
"""

from __future__ import annotations
import copy
from dataclasses import dataclass, field
from typing import Optional

from policy import Policy
from guardrails.pii import PIIGuardrail, PIIVault
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
    # Set when PII action is "tokenize": maps placeholder tokens back to the
    # original values so they can be reinjected into the model's response.
    vault: Optional[PIIVault] = None


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

        # 3. PII (may redact/tokenize instead of block; also check system prompt)
        vault = None
        if self.policy.input.pii.enabled:
            pii = PIIGuardrail(self.policy.input.pii)
            result = pii.check(working_messages)
            if result.blocked:
                return InputResult(blocked=True, reason=result.reason,
                                   guardrail="pii", messages=messages)
            working_messages = result.messages

            # Also redact PII from the system prompt if present
            # (same guardrail instance, so tokens are shared with the messages)
            if working_system:
                sys_result = pii.check([{"role": "system", "content": working_system}])
                if sys_result.blocked:
                    return InputResult(blocked=True, reason=sys_result.reason,
                                       guardrail="pii", messages=messages)
                if sys_result.messages:
                    working_system = sys_result.messages[0].get("content", working_system)

            if pii.vault is not None and pii.vault.has_tokens:
                vault = pii.vault

        return InputResult(blocked=False, messages=working_messages,
                           system=working_system, vault=vault)

    def check_stream_chunk(self, accumulated_text: str) -> Optional[str]:
        """Check accumulated streaming text. Returns an error string if blocked."""
        return OutputValidator(self.policy.output).check_stream(accumulated_text)

    def run_output_openai(self, choices: list[dict],
                          vault: Optional[PIIVault] = None) -> OutputResult:
        # Reinject tokenized PII first so validation sees what the client will see
        if vault is not None and vault.has_tokens:
            choices = copy.deepcopy(choices)
            for choice in choices:
                message = choice.get("message")
                if message and message.get("content") is not None:
                    message["content"] = vault.restore_content(message["content"])

        validator = OutputValidator(self.policy.output)
        result = validator.check(choices)
        if result.blocked:
            return OutputResult(blocked=True, reason=result.reason,
                                guardrail="output_validator", data=choices)
        return OutputResult(blocked=False, data=result.choices)

    def run_output_anthropic(self, content_blocks: list[dict],
                             vault: Optional[PIIVault] = None) -> OutputResult:
        if vault is not None and vault.has_tokens:
            content_blocks = vault.restore_content(copy.deepcopy(content_blocks))

        validator = OutputValidator(self.policy.output)
        result = validator.check_anthropic(content_blocks)
        if result.blocked:
            return OutputResult(blocked=True, reason=result.reason,
                                guardrail="output_validator", data=content_blocks)
        return OutputResult(blocked=False, data=result.choices)
