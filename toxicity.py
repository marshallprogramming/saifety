"""
Async toxicity checker supporting three providers:

  wordlist    — fast regex, no API key needed (default, always available)
  openai      — OpenAI Moderation API, free, uses your existing OpenAI key
  perspective — Google Perspective API, requires a PERSPECTIVE_API_KEY

Fails open: if an external API call fails, a warning is logged and the
content is NOT blocked. A moderation API outage should not take down
the proxy or block legitimate requests.

Streaming optimisation: ML providers are only called every
_STREAM_CHECK_INTERVAL characters to limit API usage. The wordlist
checks every chunk as it's just regex with no cost.
"""

import re
import httpx
from typing import Optional

from policy import ToxicityConfig

_STREAM_CHECK_INTERVAL = 500   # only call ML API after this many new chars in a stream

_WORDLIST_PATTERNS = [
    r"\b(kill yourself|kys|go die)\b",
    r"\b(n[i1]gg[ae]r|f[a4]gg[o0]t|ch[i1]nk|sp[i1]c)\b",
]
_compiled_wordlist = [re.compile(p, re.IGNORECASE) for p in _WORDLIST_PATTERNS]


class ToxicityChecker:

    async def check(self, text: str, config: ToxicityConfig) -> Optional[str]:
        """
        Check text for toxicity. Returns an error string if blocked, None if clean.
        Called for complete (non-streaming) responses.
        """
        if not config.enabled or not text:
            return None

        if config.provider == "wordlist":
            return _check_wordlist(text)
        if config.provider == "openai":
            return await self._check_openai(text, config)
        if config.provider == "perspective":
            return await self._check_perspective(text, config)

        return None

    async def check_stream(
        self,
        accumulated: str,
        config: ToxicityConfig,
        last_checked_len: int,
    ) -> tuple:
        """
        Check accumulated streaming text.
        Returns (error_or_None, new_last_checked_len).

        Wordlist: checked on every call (free).
        ML providers: checked only when text has grown by _STREAM_CHECK_INTERVAL chars.
        """
        if not config.enabled:
            return None, last_checked_len

        if config.provider == "wordlist":
            return _check_wordlist(accumulated), last_checked_len

        # ML providers — debounce to limit API calls
        if len(accumulated) - last_checked_len < _STREAM_CHECK_INTERVAL:
            return None, last_checked_len

        error = await self.check(accumulated, config)
        return error, len(accumulated)

    # ── Providers ─────────────────────────────────────────────────────────────

    async def _check_openai(self, text: str, config: ToxicityConfig) -> Optional[str]:
        if not config.api_key:
            print("[toxicity/openai] no api_key configured — skipping ML check")
            return None
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(
                    "https://api.openai.com/v1/moderations",
                    json={"input": text},
                    headers={"Authorization": f"Bearer {config.api_key}"},
                )
                resp.raise_for_status()
                data = resp.json()

            result = data["results"][0]
            if result["flagged"]:
                flagged = [cat for cat, hit in result["categories"].items() if hit]
                return f"Content flagged by moderation API: {', '.join(flagged)}"
            return None

        except Exception as e:
            print(f"[toxicity/openai] check failed, failing open: {e}")
            return None

    async def _check_perspective(self, text: str, config: ToxicityConfig) -> Optional[str]:
        if not config.api_key:
            print("[toxicity/perspective] no api_key configured — skipping ML check")
            return None
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(
                    "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze",
                    params={"key": config.api_key},
                    json={
                        "comment": {"text": text},
                        "requestedAttributes": {"TOXICITY": {}},
                    },
                )
                resp.raise_for_status()
                data = resp.json()

            score = data["attributeScores"]["TOXICITY"]["summaryScore"]["value"]
            if score >= config.threshold:
                return f"Content flagged as toxic (score: {score:.2f}, threshold: {config.threshold})"
            return None

        except Exception as e:
            print(f"[toxicity/perspective] check failed, failing open: {e}")
            return None


def _check_wordlist(text: str) -> Optional[str]:
    for pattern in _compiled_wordlist:
        if pattern.search(text):
            return "Response contains toxic content"
    return None
