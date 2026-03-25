"""
Topic filter — blocks requests that mention configured off-limits topics.
Works with both string and Anthropic content block array formats.
"""

import re
from guardrails.base import GuardrailResult
from guardrails.content_utils import get_text
from policy import TopicFilterConfig


class TopicFilterGuardrail:
    def __init__(self, config: TopicFilterConfig):
        self.config = config
        self._patterns = [
            (topic, re.compile(rf"\b{re.escape(topic)}\b", re.IGNORECASE))
            for topic in config.blocked_topics
        ]

    def check(self, messages: list[dict]) -> GuardrailResult:
        full_text = " ".join(
            get_text(msg.get("content", ""))
            for msg in messages
        )

        for topic, pattern in self._patterns:
            if pattern.search(full_text):
                return GuardrailResult(
                    blocked=True,
                    reason=f"Request mentions blocked topic: '{topic}'",
                    messages=messages,
                )

        return GuardrailResult(blocked=False, messages=messages)
