"""bot.lifecycle — state machine for the trading bot.

Per docs/02 §1 (design principles) and docs/02 §3 (control flow):
    BOOT -> DISCOVER -> PREFLIGHT -> TRADING -> PAUSED / COMPLETED -> SAFE_MODE

The lifecycle object owns the TradingGate (which StrategyStack checks before
emitting orders) and broadcasts stage transitions to other components via an
asyncio.Event per state.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import date as _date
from enum import Enum
from typing import Optional

import structlog

from .config import Settings
from .contracts import StageInfo


class BotState(str, Enum):
    BOOT = "BOOT"               # starting up, loading config + contracts
    DISCOVER = "DISCOVER"       # fetching universe from /stocks
    PREFLIGHT = "PREFLIGHT"     # sending ACKs to readiness tracker
    TRADING = "TRADING"         # simulation running, daily pipeline active
    PAUSED = "PAUSED"           # platform paused; orders blocked
    COMPLETED = "COMPLETED"     # platform finished
    SAFE_MODE = "SAFE_MODE"     # degraded: only HOLD allowed
    SHUTDOWN = "SHUTDOWN"       # graceful exit


# State transitions that are legal
_LEGAL: dict[BotState, set[BotState]] = {
    BotState.BOOT: {BotState.DISCOVER, BotState.SAFE_MODE, BotState.SHUTDOWN},
    BotState.DISCOVER: {BotState.PREFLIGHT, BotState.SAFE_MODE, BotState.SHUTDOWN},
    BotState.PREFLIGHT: {BotState.TRADING, BotState.PAUSED, BotState.SAFE_MODE, BotState.SHUTDOWN},
    BotState.TRADING: {BotState.PAUSED, BotState.COMPLETED, BotState.SAFE_MODE, BotState.SHUTDOWN},
    BotState.PAUSED: {BotState.TRADING, BotState.COMPLETED, BotState.SAFE_MODE, BotState.SHUTDOWN},
    BotState.COMPLETED: {BotState.SHUTDOWN},
    # SAFE_MODE must be able to re-enter PREFLIGHT: a false-positive DD breaker
    # (e.g. reconcile failed during a network blip → equity=0 → dd=100%) would
    # otherwise trap the bot forever while the server says stage=preflight.
    BotState.SAFE_MODE: {BotState.PREFLIGHT, BotState.TRADING, BotState.PAUSED, BotState.COMPLETED, BotState.SHUTDOWN},
    BotState.SHUTDOWN: set(),
}


@dataclass
class LifecycleStats:
    state: BotState = BotState.BOOT
    entered_state_at: float = field(default_factory=time.time)
    transitions: int = 0
    days_traded: int = 0
    orders_sent: int = 0
    orders_rejected: int = 0
    safe_mode_reason: Optional[str] = None
    last_transition_reason: Optional[str] = None


class TradingGate:
    """Lightweight guard checked by StrategyStack / OrderManager.

    Single source of truth for "is it OK to send orders right now?"
    """

    def __init__(self) -> None:
        self._open = False
        self._sim_date: Optional[_date] = None
        self._orders_today: int = 0
        self._short_allowed: bool = True
        self._lock = asyncio.Lock()

    @property
    def is_open(self) -> bool:
        return self._open

    @property
    def sim_date(self) -> Optional[_date]:
        return self._sim_date

    @property
    def orders_today(self) -> int:
        return self._orders_today

    @property
    def short_allowed(self) -> bool:
        return self._short_allowed

    async def open_for_day(self, sim_date: _date, max_orders: int, short_allowed: bool) -> None:
        async with self._lock:
            if sim_date != self._sim_date:
                self._sim_date = sim_date
                self._orders_today = 0
            self._open = True
            self._short_allowed = short_allowed

    async def close_for_day(self) -> None:
        async with self._lock:
            self._open = False

    async def record_order(self) -> None:
        async with self._lock:
            self._orders_today += 1

    async def orders_remaining(self, max_orders: int) -> int:
        return max(0, max_orders - self._orders_today)


class Lifecycle:
    """Owns the state machine + TradingGate + per-state asyncio.Events.

    Other components `await lifecycle.wait_for(BotState.TRADING)` or
    `await lifecycle.events[BotState.PAUSED].wait()`.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.stats = LifecycleStats()
        self.gate = TradingGate()
        self.events: dict[BotState, asyncio.Event] = {
            s: asyncio.Event() for s in BotState
        }
        # Mark BOOT as already-set on construction
        self.events[BotState.BOOT].set()
        self._log = structlog.get_logger("lifecycle")

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------
    async def transition(self, to: BotState, *, reason: str = "") -> bool:
        from_state = self.stats.state
        if to == from_state:
            return True
        if to not in _LEGAL[from_state]:
            self._log.error(
                "illegal_transition",
                **{"from": from_state.value, "to": to.value, "reason": reason},
                component="lifecycle",
            )
            return False

        now = time.time()
        # Clear old state's event, set new one's
        self.events[from_state].clear()
        self.events[to].set()

        self.stats.state = to
        self.stats.entered_state_at = now
        self.stats.transitions += 1
        self.stats.last_transition_reason = reason or None
        if to == BotState.SAFE_MODE and reason:
            self.stats.safe_mode_reason = reason

        self._log.info(
            "state_transition",
            **{"from": from_state.value, "to": to.value, "reason": reason},
            component="lifecycle",
        )

        # Open/close the trading gate based on state
        if to == BotState.TRADING:
            pass  # gate is opened per-day by the day assembler
        elif to in (BotState.PAUSED, BotState.COMPLETED, BotState.SAFE_MODE, BotState.SHUTDOWN):
            await self.gate.close_for_day()

        return True

    async def enter_safe_mode(self, reason: str) -> None:
        await self.transition(BotState.SAFE_MODE, reason=reason)

    # ------------------------------------------------------------------
    # Convenience helpers for common transitions
    # ------------------------------------------------------------------
    async def on_stage_change(self, stage: StageInfo) -> None:
        """Drive the lifecycle from a StageInfo payload.

        Per docs/03 §2.2: "Trading gate OPEN iff stage == 'simulation' and
        status == 'RUNNING' (in sandbox: stage == 'development' and status
        == 'RUNNING', verify)." So in the sandbox (stage=development), we
        open the trading gate immediately once we know status is RUNNING.
        """
        cur = self.stats.state

        # In sandbox mode the stage "development" IS the trading stage.
        # Once the platform reports status=RUNNING we go straight to TRADING.
        if stage.stage == "development" and (stage.status or "").upper() == "RUNNING":
            if cur in (BotState.BOOT,):
                await self.transition(BotState.DISCOVER, reason="stage_watcher_online")
                cur = BotState.DISCOVER
            if cur == BotState.DISCOVER:
                await self.transition(BotState.PREFLIGHT, reason="sandbox_dev_running")
                cur = BotState.PREFLIGHT
            if cur == BotState.PREFLIGHT:
                await self.transition(BotState.TRADING, reason="sandbox_dev_running")
            return

        target = {
            "development": BotState.PREFLIGHT,
            "ready": BotState.PREFLIGHT,
            "preflight": BotState.PREFLIGHT,
            "simulation": BotState.TRADING,
            "paused": BotState.PAUSED,
            "completed": BotState.COMPLETED,
        }.get(stage.stage)

        if target is None:
            return

        # Skip through intermediate states if we're in BOOT/DISCOVER
        if cur in (BotState.BOOT,):
            await self.transition(BotState.DISCOVER, reason="stage_watcher_online")
            cur = BotState.DISCOVER
        if cur == BotState.DISCOVER and target in (BotState.PREFLIGHT, BotState.TRADING):
            await self.transition(BotState.PREFLIGHT, reason=f"stage={stage.stage}")
            cur = BotState.PREFLIGHT
        if target != cur:
            await self.transition(target, reason=f"stage={stage.stage}")

    async def wait_for(self, state: BotState, *, timeout: Optional[float] = None) -> bool:
        try:
            await asyncio.wait_for(self.events[state].wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    @property
    def state(self) -> BotState:
        return self.stats.state

    def snapshot(self) -> dict:
        return {
            "state": self.stats.state.value,
            "entered_state_at": self.stats.entered_state_at,
            "transitions": self.stats.transitions,
            "days_traded": self.stats.days_traded,
            "orders_sent": self.stats.orders_sent,
            "orders_rejected": self.stats.orders_rejected,
            "safe_mode_reason": self.stats.safe_mode_reason,
            "last_transition_reason": self.stats.last_transition_reason,
            "gate_open": self.gate.is_open,
            "sim_date": self.gate.sim_date.isoformat() if self.gate.sim_date else None,
            "orders_today": self.gate.orders_today,
        }