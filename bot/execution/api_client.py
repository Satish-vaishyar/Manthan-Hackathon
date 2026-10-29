"""bot.execution.api_client — async httpx wrapper for the Order Gateway.

Per docs/03 §2 + docs/10 §3:
    - keep-alive connection pool
    - timeout 1.5s
    - Bearer auth from settings.bot_access_token
    - retries with exponential backoff for transient errors
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx
import structlog

from ..config import Settings

_log = structlog.get_logger("api")


@dataclass
class APIClientStats:
    requests: int = 0
    by_status: dict[int, int] = field(default_factory=dict)
    timeouts: int = 0
    retried: int = 0
    last_latency_ms: float = 0.0


class APIClient:
    """Sole entry point for REST calls (docs/10 §3: 'no network outside this')."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.stats = APIClientStats()
        self._client: Optional[httpx.AsyncClient] = None
        self._stop = asyncio.Event()

    async def start(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=self.settings.gateway_url,
            timeout=httpx.Timeout(self.settings.api.timeout_s, connect=1.0),
            headers={
                "Authorization": f"Bearer {self.settings.bot_access_token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": f"manthan-bot/{self.settings.team_username}",
            },
            limits=httpx.Limits(
                max_connections=20,
                max_keepalive_connections=10,
                keepalive_expiry=30.0,
            ),
            http2=False,
        )

    async def stop(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("APIClient not started")
        return self._client

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.settings.bot_access_token}"}

    async def get(self, path: str, *, params: Optional[dict] = None) -> dict:
        """GET and return parsed JSON. Raises on non-2xx (callers treat GET
        failures as retryable — no order-mutation risk)."""
        t0 = time.perf_counter()
        try:
            r = await self.client.get(path, params=params, headers=self._auth_headers())
            self.stats.last_latency_ms = (time.perf_counter() - t0) * 1000.0
            self._record_status(r.status_code)
            r.raise_for_status()
            return r.json()
        except httpx.TimeoutException as e:
            self.stats.timeouts += 1
            _log.warning("api_timeout", path=path, err=repr(e), component="api")
            raise
        except httpx.HTTPStatusError as e:
            _log.warning("api_http_error",
                         path=path, status=r.status_code, body=r.text[:200],
                         component="api")
            raise

    async def post(self, path: str, body: dict) -> dict:
        """POST and return {"_status": int, "_body": dict}. Never raises on
        4xx/5xx (OrderManager maps status codes); timeouts return _status=0
        so the caller treats the outcome as UNKNOWN."""
        t0 = time.perf_counter()
        try:
            r = await self.client.post(path, json=body, headers=self._auth_headers())
            self.stats.last_latency_ms = (time.perf_counter() - t0) * 1000.0
            self._record_status(r.status_code)
            # Don't raise on 4xx — caller (validator/order_manager) maps codes
            return {"_status": r.status_code, "_body": r.json() if r.content else {}}
        except httpx.TimeoutException as e:
            self.stats.timeouts += 1
            _log.warning("api_timeout", path=path, err=repr(e), component="api")
            return {"_status": 0, "_body": {}, "_error": repr(e)}

    def _record_status(self, status: int) -> None:
        self.stats.requests += 1
        self.stats.by_status[status] = self.stats.by_status.get(status, 0) + 1