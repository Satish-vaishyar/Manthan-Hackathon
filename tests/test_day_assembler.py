"""tests/test_day_assembler.py — complete-by-count, deadline, loop, out-of-order."""
from __future__ import annotations

import asyncio
from datetime import date, datetime, timezone
from typing import List

import pytest

from bot.config import Settings
from bot.contracts import (
    DayBundle, NewsFlashData, NewsFlashEvent, PriceTickData, PriceTickEvent,
)
from bot.ingest.day_assembler import DayAssembler


def _settings() -> Settings:
    return Settings(env="sandbox", server_ip="127.0.0.1", team_username="test",
                    bot_access_token="x")


def _tick(ticker: str, sd: date, close: float = 100.0) -> PriceTickEvent:
    return PriceTickEvent(
        event_type="price_tick",
        timestamp=datetime(2024, 5, sd.day, 10, 0, 0, tzinfo=timezone.utc),
        simulated_date=sd,
        data=PriceTickData(ticker=ticker, open=close, high=close, low=close, close=close, volume=1000),
    )


def _news(ticker: str, sd: date, headline: str = "earnings beat") -> NewsFlashEvent:
    return NewsFlashEvent(
        event_type="news_flash",
        timestamp=datetime(2024, 5, sd.day, 10, 0, 0, tzinfo=timezone.utc),
        simulated_date=sd,
        data=NewsFlashData(ticker=ticker, headline=headline),
    )


@pytest.mark.asyncio
async def test_complete_by_count_triggers_callback():
    s = _settings()
    s.timing.hard_deadline_ms = 5000  # 5s
    s.timing.news_window_ms = 50      # 50ms — let news window elapse quickly
    q: asyncio.Queue = asyncio.Queue()
    bundles: List[DayBundle] = []
    async def cb(b: DayBundle):
        bundles.append(b)
    a = DayAssembler(s, q, cb, lambda: {"AAA", "BBB"})
    await a.start()
    # Push ticks for one day, two tickers
    sd = date(2024, 5, 1)
    await q.put(_tick("AAA", sd))
    await q.put(_tick("BBB", sd))
    # Wait for news_window_ms to elapse and bundle to fire
    for _ in range(20):
        if bundles:
            break
        await asyncio.sleep(0.05)
    await a.stop()
    assert len(bundles) == 1
    assert bundles[0].simulated_date == sd
    assert bundles[0].complete_by_count is True


@pytest.mark.asyncio
async def test_complete_by_deadline_when_incomplete():
    s = _settings()
    s.timing.hard_deadline_ms = 200
    s.timing.news_window_ms = 100
    q: asyncio.Queue = asyncio.Queue()
    bundles: List[DayBundle] = []
    async def cb(b: DayBundle):
        bundles.append(b)
    # Expected tickers include BBB so AAA alone doesn't trigger complete-by-count
    a = DayAssembler(s, q, cb, lambda: {"AAA", "BBB"})
    await a.start()
    sd = date(2024, 5, 1)
    await q.put(_tick("AAA", sd))
    await asyncio.sleep(0.4)  # exceed hard_deadline
    await a.stop()
    assert len(bundles) == 1
    assert bundles[0].complete_by_deadline is True


@pytest.mark.asyncio
async def test_news_attached_to_current_day():
    s = _settings()
    s.timing.hard_deadline_ms = 2000
    s.timing.news_window_ms = 100
    q: asyncio.Queue = asyncio.Queue()
    bundles: List[DayBundle] = []
    async def cb(b: DayBundle):
        bundles.append(b)
    # Single expected ticker so we get exactly one bundle (complete by count)
    a = DayAssembler(s, q, cb, lambda: {"AAA"})
    await a.start()
    sd = date(2024, 5, 1)
    await q.put(_tick("AAA", sd))
    await q.put(_news("AAA", sd))
    await asyncio.sleep(0.2)
    await a.stop()
    assert len(bundles) == 1
    assert len(bundles[0].news) == 1


@pytest.mark.asyncio
async def test_loop_detected():
    s = _settings()
    s.timing.hard_deadline_ms = 200
    q: asyncio.Queue = asyncio.Queue()
    bundles: List[DayBundle] = []
    async def cb(b: DayBundle):
        bundles.append(b)
    a = DayAssembler(s, q, cb, lambda: set())
    await a.start()
    a.stats.last_sim_date = date(2024, 5, 10)
    # Send a tick for an earlier date
    await q.put(_tick("AAA", date(2024, 5, 1)))
    await asyncio.sleep(0.3)
    await a.stop()
    assert a.stats.loop_detected >= 1