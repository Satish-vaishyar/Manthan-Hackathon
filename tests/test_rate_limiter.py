"""tests/test_rate_limiter.py — token bucket + 429 penalty."""
from __future__ import annotations

import asyncio
import time

import pytest

from bot.execution.rate_limiter import RateLimiter


@pytest.mark.asyncio
async def test_burst_capacity_then_throttle():
    rl = RateLimiter(rate_per_min=60, capacity=5)  # 1 token/sec
    # Burst 5 should succeed immediately
    start = time.perf_counter()
    for _ in range(5):
        assert await rl.acquire(priority="normal") is True
    burst_ms = (time.perf_counter() - start) * 1000.0
    assert burst_ms < 100.0  # should be fast
    # 6th should wait
    start = time.perf_counter()
    assert await rl.acquire(priority="normal") is True
    waited_ms = (time.perf_counter() - start) * 1000.0
    assert waited_ms > 800.0  # ~1 second wait


@pytest.mark.asyncio
async def test_429_penalty_blocks_then_unblocks():
    rl = RateLimiter(rate_per_min=600, capacity=10)
    # Consume some
    for _ in range(5):
        assert await rl.acquire() is True
    # Trigger 429
    await rl.note_429(penalty_seconds=0.3, rate_reduction=0.5, reduction_seconds=0.5)
    # Within penalty window: should refuse
    assert await rl.acquire() is False
    # Wait out penalty
    await asyncio.sleep(0.4)
    assert rl.in_penalty is False
    # Should be allowed again (with reduced rate, so maybe wait)
    assert await rl.acquire() is True


@pytest.mark.asyncio
async def test_priority_tracking():
    rl = RateLimiter(rate_per_min=600, capacity=10)
    await rl.acquire(priority="heartbeat")
    await rl.acquire(priority="order")
    assert rl.stats.by_priority.get("heartbeat") == 1
    assert rl.stats.by_priority.get("order") == 1