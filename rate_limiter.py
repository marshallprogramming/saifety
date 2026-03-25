"""
Sliding window rate limiter — in-memory, per-tenant.

Tracks request timestamps in a deque per tenant per window.
On each check, old entries outside the window are evicted, then
the current count is compared against the configured limit.

Note: in-memory means limits reset on restart and don't
share state across multiple proxy processes. For multi-process
deployments, swap the deque store for Redis (same interface).
"""

import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Optional

from policy import RateLimitConfig


@dataclass
class RateLimitResult:
    limited: bool
    reason: Optional[str] = None
    retry_after: Optional[int] = None   # seconds until the oldest request exits the window


class RateLimiter:
    def __init__(self):
        # tenant_id -> deque of float timestamps
        self._per_minute: dict[str, deque] = defaultdict(deque)
        self._per_hour:   dict[str, deque] = defaultdict(deque)

    def check(self, tenant_id: str, config: RateLimitConfig) -> RateLimitResult:
        """
        Check whether tenant_id is within their rate limits.
        If allowed, records the request. If denied, does NOT record it.
        """
        if not config.enabled:
            return RateLimitResult(limited=False)

        now = time.time()

        if config.requests_per_minute is not None:
            result = self._check_window(
                store=self._per_minute[tenant_id],
                now=now,
                window_seconds=60,
                limit=config.requests_per_minute,
                label="minute",
            )
            if result.limited:
                return result

        if config.requests_per_hour is not None:
            result = self._check_window(
                store=self._per_hour[tenant_id],
                now=now,
                window_seconds=3600,
                limit=config.requests_per_hour,
                label="hour",
            )
            if result.limited:
                return result

        # Within limits — record the request in each active window
        if config.requests_per_minute is not None:
            self._per_minute[tenant_id].append(now)
        if config.requests_per_hour is not None:
            self._per_hour[tenant_id].append(now)

        return RateLimitResult(limited=False)

    def _check_window(
        self,
        store: deque,
        now: float,
        window_seconds: int,
        limit: int,
        label: str,
    ) -> RateLimitResult:
        cutoff = now - window_seconds

        # Evict timestamps outside the window
        while store and store[0] < cutoff:
            store.popleft()

        if len(store) >= limit:
            # Retry after the oldest request in this window expires
            retry_after = max(1, int(store[0] + window_seconds - now) + 1)
            return RateLimitResult(
                limited=True,
                reason=f"Rate limit exceeded: {limit} requests per {label}",
                retry_after=retry_after,
            )

        return RateLimitResult(limited=False)

    def status(self, tenant_id: str, config: RateLimitConfig) -> dict:
        """Return current usage counts for a tenant — used by /rate-limits endpoint."""
        now = time.time()
        result = {"tenant_id": tenant_id}

        if config.requests_per_minute is not None:
            store = self._per_minute[tenant_id]
            cutoff = now - 60
            count = sum(1 for t in store if t >= cutoff)
            result["per_minute"] = {
                "limit": config.requests_per_minute,
                "used": count,
                "remaining": max(0, config.requests_per_minute - count),
            }

        if config.requests_per_hour is not None:
            store = self._per_hour[tenant_id]
            cutoff = now - 3600
            count = sum(1 for t in store if t >= cutoff)
            result["per_hour"] = {
                "limit": config.requests_per_hour,
                "used": count,
                "remaining": max(0, config.requests_per_hour - count),
            }

        return result
