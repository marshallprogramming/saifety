"""Tests for the sliding-window rate limiter."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from policy import RateLimitConfig
from rate_limiter import RateLimiter


class TestRateLimiter:
    def test_disabled_always_allows(self):
        limiter = RateLimiter()
        cfg = RateLimitConfig(enabled=False)
        for _ in range(100):
            result = limiter.check("tenant1", cfg)
            assert not result.limited

    def test_within_rpm_limit(self):
        limiter = RateLimiter()
        cfg = RateLimitConfig(enabled=True, requests_per_minute=5)
        for _ in range(5):
            result = limiter.check("tenant1", cfg)
            assert not result.limited

    def test_exceeds_rpm_limit(self):
        limiter = RateLimiter()
        cfg = RateLimitConfig(enabled=True, requests_per_minute=3)
        for _ in range(3):
            limiter.check("tenant1", cfg)
        result = limiter.check("tenant1", cfg)
        assert result.limited
        assert "minute" in result.reason
        assert result.retry_after is not None
        assert result.retry_after > 0

    def test_different_tenants_independent(self):
        limiter = RateLimiter()
        cfg = RateLimitConfig(enabled=True, requests_per_minute=2)
        limiter.check("tenant_a", cfg)
        limiter.check("tenant_a", cfg)
        # tenant_a is at limit
        assert limiter.check("tenant_a", cfg).limited
        # tenant_b should still be fine
        assert not limiter.check("tenant_b", cfg).limited

    def test_within_rph_limit(self):
        limiter = RateLimiter()
        cfg = RateLimitConfig(enabled=True, requests_per_hour=5)
        for _ in range(5):
            result = limiter.check("tenant1", cfg)
            assert not result.limited

    def test_exceeds_rph_limit(self):
        limiter = RateLimiter()
        cfg = RateLimitConfig(enabled=True, requests_per_hour=2)
        limiter.check("tenant1", cfg)
        limiter.check("tenant1", cfg)
        result = limiter.check("tenant1", cfg)
        assert result.limited
        assert "hour" in result.reason
