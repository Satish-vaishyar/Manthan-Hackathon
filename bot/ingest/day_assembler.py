"""bot.ingest.day_assembler — group Kafka events into DayBundle objects.

Per docs/02 §4 (timing budget) and docs/11 §1 (test cases):
    - complete-by-count: all expected tickers for `simulated_date` have arrived
    - complete-by-deadline: hard_deadline_ms has elapsed since first tick
    - news window: news_window_ms after the last tick — include those news
    - out-of-order partitions: if a later-day event arrives while we wait, stash it
    - loop detection: if simulated_date goes BACKWARDS, log + tag

The assembler reads from the queue populated by kafka_consumer and emits
finished DayBundles via a callback. It tracks one in-progress day at a time
(plus a small spillover buffer for out-of-order events).
"""
from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date as _date, datetime, timezone
from typing import Awaitable, Callable, Optional

import structlog

from ..config import Settings
from ..contracts import (
    DayBundle,
    KafkaEvent,
    NewsFlashEvent,
    NewsFlashData,
    PriceTickData,
    PriceTickEvent,
)

_log = structlog.get_logger("assembler")

# Type for the completion callback
DayCallback = Callable[[DayBundle], Awaitable[None]]


@dataclass
class AssemblerStats:
    days_assembled: int = 0
    days_by_count: int = 0
    days_by_deadline: int = 0
    loop_detected: int = 0
    out_of_order_stashed: int = 0
    last_sim_date: Optional[_date] = None
    avg_latency_ms: float = 0.0


class DayAssembler:
    """Drains the Kafka event queue and emits one DayBundle per simulated day."""

    def __init__(
        self,
        settings: Settings,
        queue: asyncio.Queue[KafkaEvent],
        on_day: DayCallback,
        expected_tickers_provider: Callable[[], set[str]],
    ) -> None:
        self.settings = settings
        self.queue = queue
        self.on_day = on_day
        self._expected_tickers_provider = expected_tickers_provider
        self.stats = AssemblerStats()

        # Current day being built
        self._cur_date: Optional[_date] = None
        self._cur_ticks: dict[str, PriceTickData] = {}
        self._cur_news: list[NewsFlashData] = []
        self._cur_first_tick_at: Optional[float] = None
        self._cur_last_tick_at: Optional[float] = None
        self._cur_count_complete: bool = False  # all expected tickers in

        # Out-of-order stash (sim_date -> [event])
        self._stash: dict[_date, list[KafkaEvent]] = defaultdict(list)

        self._stop = asyncio.Event()
        self._task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="assembler")

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass

    async def _run(self) -> None:
        while not self._stop.is_set():
            try:
                event = await asyncio.wait_for(self.queue.get(), timeout=0.05)
            except asyncio.TimeoutError:
                # Check if current day has expired by deadline (poll frequently)
                await self._maybe_fire_by_deadline()
                continue
            except asyncio.CancelledError:
                return
            await self._handle(event)
            # After every event, also check deadline (cheap)
            await self._maybe_fire_by_deadline()

    async def _handle(self, event: KafkaEvent) -> None:
        # Preflight ticks are NOT day events — ignore here (heartbeat handles them)
        if event.event_type == "preflight_tick":
            return

        if isinstance(event, PriceTickEvent):
            sd = event.simulated_date
            data = event.data

            # Loop detection
            if self.stats.last_sim_date and sd < self.stats.last_sim_date:
                self.stats.loop_detected += 1
                _log.warning(
                    "sandbox_loop_detected",
                    **{"prev": self.stats.last_sim_date.isoformat(), "now": sd.isoformat()},
                    component="assembler",
                )

            # Out-of-order: this tick is for a date we already finished
            if self._cur_date is not None and sd < self._cur_date:
                self._stash[sd].append(event)
                self.stats.out_of_order_stashed += 1
                return

            # New day boundary: finalize current, start new
            async with self._lock:
                if self._cur_date is not None and sd != self._cur_date:
                    await self._finalize(complete_by_count=False)

                if self._cur_date is None or sd != self._cur_date:
                    self._cur_date = sd
                    self._cur_ticks.clear()
                    self._cur_news.clear()
                    self._cur_first_tick_at = time.perf_counter()
                    self._cur_count_complete = False
                self._cur_ticks[data.ticker] = data
                self._cur_last_tick_at = time.perf_counter()

                # Update last seen
                if not self.stats.last_sim_date or sd > self.stats.last_sim_date:
                    self.stats.last_sim_date = sd

                # Mark count-complete if all expected tickers have arrived
                expected = self._expected_tickers_provider()
                if expected and expected.issubset(self._cur_ticks.keys()):
                    self._cur_count_complete = True
                # The actual finalize decision happens in _maybe_fire, which
                # checks both count-complete AND news_window elapsed.

        elif isinstance(event, NewsFlashEvent):
            sd = event.simulated_date
            async with self._lock:
                # If this news is for the current day, attach it
                if self._cur_date is not None and sd == self._cur_date:
                    self._cur_news.append(event.data)
                else:
                    # Stash — belongs to a day we already finalized (or future)
                    self._stash[sd].append(event)
                    self.stats.out_of_order_stashed += 1

    async def _maybe_fire_by_deadline(self) -> None:
        """If hard_deadline_ms elapsed since the first tick of the current day, finalize.

        Two-stage completion:
        - count-complete AND news_window elapsed -> finalize by count
        - hard_deadline elapsed -> finalize by deadline (count may be incomplete)
        """
        if self._cur_date is None or self._cur_first_tick_at is None:
            return
        elapsed_ms = (time.perf_counter() - self._cur_first_tick_at) * 1000.0
        news_window = self.settings.timing.news_window_ms
        async with self._lock:
            if self._cur_date is None:
                return
            if self._cur_count_complete and elapsed_ms >= news_window:
                await self._finalize(complete_by_count=True)
            elif elapsed_ms >= self.settings.timing.hard_deadline_ms:
                await self._finalize(complete_by_count=False)

    async def _finalize(self, *, complete_by_count: bool) -> None:
        """Emit the current DayBundle, drain the stash, and reset."""
        if self._cur_date is None:
            return
        now = time.perf_counter()
        assembly_latency_ms = (now - (self._cur_first_tick_at or now)) * 1000.0

        # Pull any stashed events for this day
        stashed = self._stash.pop(self._cur_date, [])
        for ev in stashed:
            if isinstance(ev, PriceTickEvent):
                # Only add if not already present (count is more recent)
                if ev.data.ticker not in self._cur_ticks:
                    self._cur_ticks[ev.data.ticker] = ev.data
            elif isinstance(ev, NewsFlashEvent):
                self._cur_news.append(ev.data)

        bundle = DayBundle(
            simulated_date=self._cur_date,
            ticks=dict(self._cur_ticks),
            news=list(self._cur_news),
            expected_tickers=set(self._expected_tickers_provider()),
            assembled_at=datetime.now(timezone.utc),
            assembly_latency_ms=assembly_latency_ms,
            complete_by_count=complete_by_count,
            complete_by_deadline=not complete_by_count,
        )

        self.stats.days_assembled += 1
        if complete_by_count:
            self.stats.days_by_count += 1
        else:
            self.stats.days_by_deadline += 1
        # Rolling average latency
        n = self.stats.days_assembled
        self.stats.avg_latency_ms = (
            self.stats.avg_latency_ms * (n - 1) + assembly_latency_ms
        ) / n

        _log.info(
            "day_assembled",
            **{
                "sim_date": self._cur_date.isoformat(),
                "tickers": len(self._cur_ticks),
                "news": len(self._cur_news),
                "complete_by": "count" if complete_by_count else "deadline",
                "latency_ms": round(assembly_latency_ms, 1),
            },
            component="assembler",
        )

        # Reset
        self._cur_date = None
        self._cur_ticks.clear()
        self._cur_news.clear()
        self._cur_first_tick_at = None
        self._cur_last_tick_at = None

        # Hand off (don't await user code under lock)
        try:
            await self.on_day(bundle)
        except Exception as e:  # noqa: BLE001
            _log.error("day_callback_failed", err=repr(e), component="assembler")