"""tests/test_rate_limiter.py — token bucket + 429 penalty."""
from __future__ import annotations

import asyncio
import time

import pytest

from bot.execution.rate_limiter import RateLimiter


@pytest.mark.asyncio
async def test_burst_capacity_then_throttle():
    # 120/min = 2 tokens/sec, capacity 3. After burst of 3 the next token
    # should take ~500 ms to refill (deterministic under Windows scheduler).
    rl = RateLimiter(rate_per_min=120, capacity=3)
    # Burst 3 should succeed immediately
    start = time.perf_counter()
    for _ in range(3):
        assert await rl.acquire(priority="normal") is True
    burst_ms = (time.perf_counter() - start) * 1000.0
    assert burst_ms < 100.0  # should be fast
    # 4th should wait ~500 ms
    start = time.perf_counter()
    assert await rl.acquire(priority="normal") is True
    waited_ms = (time.perf_counter() - start) * 1000.0
    # 400 ms lower bound (500 ms target + 50 ms slack); still proves the
    # throttling behavior — without throttling, the 4th acquire would
    # complete in <50 ms.
    assert waited_ms > 400.0, f"expected wait > 400 ms, got {waited_ms:.1f}"


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