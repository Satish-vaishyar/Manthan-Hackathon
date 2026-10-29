"""bot.ingest.stage_watcher — poll /api/v1/stage + drive the Lifecycle.

Per docs/02 §3 + docs/10 §1: drives the BOOT -> DISCOVER -> PREFLIGHT -> TRADING
transitions and opens/closes the TradingGate.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Optional

import structlog

from ..config import Settings
from ..contracts import StageInfo

_log = structlog.get_logger("stage")


@dataclass
class StageWatcherStats:
    polls: int = 0
    last_stage: Optional[str] = None
    last_poll_at: Optional[float] = None
    consecutive_errors: int = 0


class StageWatcher:
    """Polls /api/v1/stage every settings.timing.stage_poll_*_s.

    Emits StageInfo via a callback (typically Lifecycle.on_stage_change).
    Cadence: 2.5 s during preflight, 30 s during simulation (per docs/02 §4).
    """

    def __init__(self, settings, api_get, on_stage) -> None:
        """
        Args:
            settings: bot Settings.
            api_get: async callable `await api_get(path) -> dict`.
            on_stage: async callable `await on_stage(StageInfo)`.
        """
        self.settings = settings
        self._api_get = api_get
        self._on_stage = on_stage
        self.stats = StageWatcherStats()
        self._stop = asyncio.Event()
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="stage-watcher")

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass

    def _interval_for(self, stage: Optional[str]) -> float:
        """Poll faster during readiness stages (preflight cadence) than
        during simulation (P3-14: <=10s detection of stage flips)."""
        s = self.settings
        if stage in ("development", "ready", "preflight"):
            return s.timing.stage_poll_preflight_s
        return s.timing.stage_poll_simulation_s

    async def _run(self) -> None:
        """Poll loop: GET /stage -> StageInfo -> on_stage callback; logs
        transitions and counts consecutive failures without crashing."""
        # First poll: short delay so we let other components come up
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=2.0)
            return
        except asyncio.TimeoutError:
            pass

        last_stage: Optional[str] = None
        while not self._stop.is_set():
            try:
                raw = await self._api_get("/api/v1/stage")
                stage = StageInfo.model_validate(raw) if isinstance(raw, dict) else None
                if stage is None:
                    raise RuntimeError("invalid stage payload")
                self.stats.polls += 1
                self.stats.last_poll_at = time.time()
                self.stats.consecutive_errors = 0
                if stage.stage != self.stats.last_stage:
                    self.stats.last_stage = stage.stage
                    _log.info(
                        "stage_changed",
                        **{
                            "stage": stage.stage,
                            "sim_date": stage.simulated_date.isoformat() if stage.simulated_date else None,
                            "interval_s": stage.interval_seconds,
                            "short_allowed": stage.short_selling_enabled,
                        },
                        component="stage",
                    )
                try:
                    await self._on_stage(stage)
                except Exception as e:  # noqa: BLE001
                    _log.warning("on_stage_callback_failed", err=repr(e), component="stage")
                last_stage = stage.stage
            except Exception as e:  # noqa: BLE001
                self.stats.consecutive_errors += 1
                _log.warning(
                    "stage_poll_failed",
                    **{"err": repr(e), "consecutive": self.stats.consecutive_errors},
                    component="stage",
                )

            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self._interval_for(last_stage))
                return
            except asyncio.TimeoutError:
                continue