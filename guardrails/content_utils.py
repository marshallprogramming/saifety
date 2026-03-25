"""
Utilities for normalizing message content between OpenAI and Anthropic formats.

OpenAI content:  string  OR  list of {"type": "text"|"image_url", ...}
Anthropic content: string  OR  list of {"type": "text"|"image", ...}

All guardrails use these helpers so they work with both APIs.
"""

from typing import Callable


def get_text(content) -> str:
    """Return the plain-text representation of any content value."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""


def apply_text_transform(content, transform: Callable[[str], str]):
    """Apply a string→string transform to content, preserving its original format."""
    if isinstance(content, str):
        return transform(content)
    if isinstance(content, list):
        result = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                result.append({**block, "text": transform(block.get("text", ""))})
            else:
                result.append(block)
        return result
    return content
