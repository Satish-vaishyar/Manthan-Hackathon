"""bot.execution.rate_limiter — token bucket + 429 penalty box.

Per docs/06 §4.3:
    - 120 req/min hard cap; we plan for 100/min
    - 429 -> 30s penalty box + refill rate reduced by 20% for 2 min
    - priority queue (heartbeat > orders)
"""
from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import structlog

_log = structlog.get_logger("rate")


@dataclass
class RateLimiterStats:
    acquired: int = 0
    waited: int = 0
    waited_total_ms: float = 0.0
    penalties: int = 0
    refused: int = 0
    by_priority: dict[str, int] = field(default_factory=dict)


class RateLimiter:
    """Async token bucket with priority + penalty-box on 429."""

    def __init__(self, rate_per_min: int = 100, capacity: int = 10) -> None:
        self.rate_per_sec = rate_per_min / 60.0
        self.capacity = capacity
        self._tokens = float(capacity)
        self._last = time.perf_counter()
        self._lock = asyncio.Lock()
        # Penalty box
        self._penalty_until: float = 0.0
        self._penalty_rate_factor: float = 1.0
        self._penalty_factor_until: float = 0.0
        self.stats = RateLimiterStats()

    async def acquire(self, *, priority: str = "normal") -> bool:
        """Try to take 1 token. Heartbeat priority bypasses if forced."""
        async with self._lock:
            now = time.perf_counter()
            self._refill(now)
            # Penalty box
            if now < self._penalty_until:
                self.stats.refused += 1
                _log.warning("rate_penalty_active",
                             until=self._penalty_until - now,
                             priority=priority,
                             component="rate")
                return False
            # Effective rate factor (reduced after a 429 for 2 min)
            if now > self._penalty_factor_until:
                self._penalty_rate_factor = 1.0
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                self.stats.acquired += 1
                self.stats.by_priority[priority] = self.stats.by_priority.get(priority, 0) + 1
                return True
            # Need to wait
            deficit = 1.0 - self._tokens
            wait_s = deficit / (self.rate_per_sec * self._penalty_rate_factor)
            self.stats.waited += 1
            self.stats.waited_total_ms += wait_s * 1000.0
        # Wait outside the lock so other coroutines can refill
        await asyncio.sleep(wait_s)
        # Retry once
        async with self._lock:
            now = time.perf_counter()
            self._refill(now)
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                self.stats.acquired += 1
                self.stats.by_priority[priority] = self.stats.by_priority.get(priority, 0) + 1
                return True
            self.stats.refused += 1
            return False

    def _refill(self, now: float) -> None:
        elapsed = now - self._last
        if elapsed <= 0:
            return
        self._tokens = min(
            self.capacity,
            self._tokens + elapsed * self.rate_per_sec * self._penalty_rate_factor,
        )
        self._last = now

    async def note_429(self, *, penalty_seconds: float = 30.0,
                       rate_reduction: float = 0.8, reduction_seconds: float = 120.0) -> None:
        """Call this when the gateway returns 429. Triggers penalty box
        and a temporary reduction in refill rate."""
        async with self._lock:
            now = time.perf_counter()
            self._penalty_until = max(self._penalty_until, now + penalty_seconds)
            self._penalty_rate_factor = min(self._penalty_rate_factor, rate_reduction)
            self._penalty_factor_until = max(self._penalty_factor_until, now + reduction_seconds)
            self.stats.penalties += 1
            _log.warning(
                "rate_429_penalty",
                **{
                    "penalty_s": penalty_seconds,
                    "rate_factor": rate_reduction,
                    "reduction_s": reduction_seconds,
                },
                component="rate",
            )

    @property
    def available_tokens(self) -> float:
        return self._tokens

    @property
    def in_penalty(self) -> bool:
        return time.perf_counter() < self._penalty_until