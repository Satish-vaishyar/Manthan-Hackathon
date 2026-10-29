"""bot.ingest.heartbeat — ACK every preflight_tick + watchdog.

Per docs/01 §5 and docs/03 §2.6:
    - portal requires >=1 ACK per 15 s during preflight
    - watchdog re-sends the LAST seq at settings.timing.heartbeat_interval_s
      even between preflight_ticks, so the portal sees us as alive
    - POST /api/v1/bot/heartbeat with {"seq": <int>}
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional

import structlog

from ..config import Settings
from ..contracts import HeartbeatResponse, PreflightTickEvent

_log = structlog.get_logger("heartbeat")


@dataclass
class HeartbeatStats:
    acks_sent: int = 0
    ack_failures: int = 0
    invalid_seq: int = 0
    watchdog_repeats: int = 0
    last_seq: Optional[int] = None
    last_seq_at: Optional[float] = None
    last_ack_at: Optional[float] = None


class Heartbeat:
    """Two-pronged liveness:

    1. Event-driven: caller invokes `on_preflight(ev)` whenever a preflight_tick
       arrives. We POST the seq immediately.
    2. Watchdog: every `settings.timing.heartbeat_interval_s`, re-send the
       last seq (or empty body) so the portal keeps us green.
    """

    def __init__(self, settings: Settings, api_post) -> None:
        """
        Args:
            settings: bot Settings.
            api_post: async callable `await api_post(path, json)` returning
                      a dict (parsed response body) or raising on transport error.
        """
        self.settings = settings
        self._api_post = api_post
        self.stats = HeartbeatStats()
        self._stop = asyncio.Event()
        self._watchdog_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        self._watchdog_task = asyncio.create_task(self._watchdog_loop(), name="heartbeat-watchdog")

    async def stop(self) -> None:
        self._stop.set()
        if self._watchdog_task:
            self._watchdog_task.cancel()
            try:
                await self._watchdog_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass

    async def on_preflight(self, ev: PreflightTickEvent) -> bool:
        """ACK a preflight_tick immediately. Returns True on success."""
        return await self._send(ev.seq)

    async def _watchdog_loop(self) -> None:
        interval = self.settings.timing.heartbeat_interval_s
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=interval)
                return
            except asyncio.TimeoutError:
                if self.stats.last_seq is not None:
                    # Re-send the last seq (matches platform's "last-seen" model)
                    ok = await self._send(self.stats.last_seq, is_watchdog=True)
                    if ok:
                        self.stats.watchdog_repeats += 1

    async def _send(self, seq: Optional[int], *, is_watchdog: bool = False) -> bool:
        body = {"seq": seq} if seq is not None else {}
        path = "/api/v1/bot/heartbeat"
        try:
            resp = await self._api_post(path, body)
            # Try pydantic parse but tolerate shape drift
            try:
                parsed = HeartbeatResponse.model_validate(resp) if isinstance(resp, dict) else None
            except Exception:  # noqa: BLE001
                parsed = None

            # Platform returns either {"accepted": true} OR a wrapped
            # {"_status": 200, "_body": {"status": "ok", ...}} — accept both.
            body = resp.get("_body") if isinstance(resp, dict) else None
            accepted = (
                isinstance(resp, dict) and resp.get("accepted") is True
            ) or (
                isinstance(body, dict) and body.get("status") == "ok"
            ) or (
                isinstance(resp, dict) and resp.get("status") == "ok"
            )
            if accepted:
                self.stats.acks_sent += 1
                self.stats.last_ack_at = time.time()
                if seq is not None:
                    self.stats.last_seq = seq
                    self.stats.last_seq_at = time.time()
                _log.info(
                    "heartbeat_ok",
                    **{"seq": seq, "watchdog": is_watchdog},
                    component="heartbeat",
                )
                return True
            if isinstance(resp, dict) and resp.get("error") == "INVALID_SEQ":
                # Normal in `development` stage — no preflight_tick has fired yet.
                self.stats.invalid_seq += 1
                _log.debug(
                    "heartbeat_invalid_seq",
                    **{"seq": seq, "watchdog": is_watchdog},
                    component="heartbeat",
                )
                return False
            self.stats.ack_failures += 1
            _log.warning(
                "heartbeat_failed",
                **{"seq": seq, "watchdog": is_watchdog, "resp": resp},
                component="heartbeat",
            )
            return False
        except Exception as e:  # noqa: BLE001
            self.stats.ack_failures += 1
            _log.warning(
                "heartbeat_transport_error",
                **{"err": repr(e), "seq": seq, "watchdog": is_watchdog},
                component="heartbeat",
            )
            return False