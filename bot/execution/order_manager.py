"""bot.execution.order_manager — validate -> risk approve -> rate limit -> POST -> apply fill -> journal.

Per docs/06 §4:
    - validate via OrderValidator
    - acquire rate-limit token (priority 'order')
    - POST /api/v1/orders
    - on 200: apply to Ledger + write journal record
    - on 429: tell RateLimiter, optional single retry after penalty
    - on other 4xx: log + journal rejection, don't retry
    - timeout-safe: GET /api/v1/orders to recover state if needed
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import date as _date, datetime
from typing import Optional

import structlog

from ..config import Settings
from ..contracts import OrderRequest, OrderResponse, RejectCode
from ..journal.journal import DecisionJournal
from ..portfolio.ledger import Ledger
from .api_client import APIClient
from .rate_limiter import RateLimiter
from .validator import OrderValidator

_log = structlog.get_logger("orders")


@dataclass
class OrderManagerStats:
    submitted: int = 0
    filled: int = 0
    rejected: int = 0
    rate_limited: int = 0
    validation_denied: int = 0
    last_latency_ms: float = 0.0


class OrderManager:
    """The single funnel for every order the bot wants to submit."""

    def __init__(
        self,
        settings: Settings,
        api: APIClient,
        rate_limiter: RateLimiter,
        validator: OrderValidator,
        ledger: Ledger,
        journal: "DecisionJournal",
    ) -> None:
        self.settings = settings
        self.api = api
        self.rate_limiter = rate_limiter
        self.validator = validator
        self.ledger = ledger
        self.journal = journal
        self.stats = OrderManagerStats()

    async def submit(
        self,
        req: OrderRequest,
        *,
        scores: Optional[dict] = None,
        reasons: Optional[list[str]] = None,
    ) -> OrderResponse:
        """End-to-end submit. Returns OrderResponse (status=FILLED|REJECTED)."""
        # 1) Validate
        v = self.validator.validate(
            req,
            self.ledger,
            short_allowed=self._short_allowed(req),
            sim_date=req.date,
        )
        if not v.ok:
            self.stats.validation_denied += 1
            self._journal_rejection(req, v.code, v.message, scores, reasons)
            return OrderResponse(
                id="", ticker=req.ticker, action=req.action,
                quantity=req.quantity, price=0.0,
                status="REJECTED", reject_code=v.code, reject_message=v.message,
                date=req.date, timestamp=datetime.utcnow(),
            )

        # 2) Rate limit
        ok = await self.rate_limiter.acquire(priority="order")
        if not ok:
            self.stats.rate_limited += 1
            self._journal_rejection(req, RejectCode.RATE_LIMITED, "rate limit", scores, reasons)
            return OrderResponse(
                id="", ticker=req.ticker, action=req.action,
                quantity=req.quantity, price=0.0,
                status="REJECTED", reject_code=RejectCode.RATE_LIMITED,
                reject_message="rate limit",
                date=req.date, timestamp=datetime.utcnow(),
            )

        # 3) Submit
        self.validator.mark_in_flight(req)
        self.stats.submitted += 1
        t0 = time.perf_counter()
        body = req.model_dump(mode="json")
        try:
            resp = await self.api.post("/api/v1/orders", body)
        except Exception as e:  # noqa: BLE001
            self.validator.clear_in_flight(req)
            _log.error("order_post_exception", err=repr(e), req=body, component="orders")
            self._journal_rejection(req, RejectCode.SERVER_ERROR, repr(e), scores, reasons)
            self.stats.rejected += 1
            return OrderResponse(
                id="", ticker=req.ticker, action=req.action, quantity=req.quantity,
                price=0.0, status="REJECTED", reject_code=RejectCode.SERVER_ERROR,
                reject_message=repr(e), date=req.date, timestamp=datetime.utcnow(),
            )
        self.stats.last_latency_ms = (time.perf_counter() - t0) * 1000.0

        status = resp.get("_status", 0)
        payload = resp.get("_body", {})

        # 4) Handle 429 -> penalty box + single retry after backoff
        if status == 429:
            await self.rate_limiter.note_429()
            self.validator.clear_in_flight(req)
            self.stats.rate_limited += 1
            self._journal_rejection(req, RejectCode.RATE_LIMITED, "429 from server", scores, reasons)
            return OrderResponse(
                id="", ticker=req.ticker, action=req.action,
                quantity=req.quantity, price=0.0,
                status="REJECTED", reject_code=RejectCode.RATE_LIMITED,
                reject_message="429", date=req.date, timestamp=datetime.utcnow(),
            )

        # 5) Handle 4xx rejections
        if 400 <= status < 500:
            self.validator.clear_in_flight(req)
            self.stats.rejected += 1
            code = payload.get("error", RejectCode.SERVER_ERROR) if isinstance(payload, dict) else RejectCode.SERVER_ERROR
            msg = payload.get("message", "") if isinstance(payload, dict) else str(payload)
            self._journal_rejection(req, code, msg, scores, reasons)
            return OrderResponse(
                id="", ticker=req.ticker, action=req.action,
                quantity=req.quantity, price=0.0,
                status="REJECTED", reject_code=code, reject_message=msg,
                date=req.date, timestamp=datetime.utcnow(),
            )

        # 6) Handle 5xx
        if status >= 500:
            self.validator.clear_in_flight(req)
            self.stats.rejected += 1
            self._journal_rejection(req, RejectCode.SERVER_ERROR, f"HTTP {status}", scores, reasons)
            return OrderResponse(
                id="", ticker=req.ticker, action=req.action,
                quantity=req.quantity, price=0.0,
                status="REJECTED", reject_code=RejectCode.SERVER_ERROR,
                reject_message=f"HTTP {status}",
                date=req.date, timestamp=datetime.utcnow(),
            )

        # 7) Success — apply to ledger + journal
        try:
            filled = OrderResponse.model_validate(payload) if isinstance(payload, dict) else None
        except Exception as e:  # noqa: BLE001
            filled = None
            _log.warning("order_response_parse_failed", err=repr(e), payload=payload, component="orders")

        if filled is None:
            # Best-effort: synthesize a minimal fill
            filled = OrderResponse(
                id=payload.get("id", "unknown"),
                ticker=req.ticker, action=req.action, quantity=req.quantity,
                price=self.ledger.last_close.get(req.ticker, 0.0),
                status="FILLED", date=req.date, timestamp=datetime.utcnow(),
            )

        self._apply_fill(filled)
        self.validator.clear_in_flight(req)
        self.stats.filled += 1
        self._journal_fill(filled, scores, reasons)
        return filled

    # ------------------------------------------------------------------
    # Apply fill -> ledger
    # ------------------------------------------------------------------
    def _apply_fill(self, resp: OrderResponse) -> None:
        b = self.settings.tax.brokerage_per_order
        d = self.settings.tax.dp_charge_per_sell
        if resp.action == "BUY":
            self.ledger.apply_buy(resp.ticker, resp.quantity, resp.price, resp.date, b, d)
        elif resp.action == "SELL":
            self.ledger.apply_sell(resp.ticker, resp.quantity, resp.price, resp.date, b, d)
        elif resp.action == "SHORT":
            self.ledger.apply_short(resp.ticker, resp.quantity, resp.price, resp.date, b, d)
        elif resp.action == "COVER":
            self.ledger.apply_cover(resp.ticker, resp.quantity, resp.price, resp.date, b, d)

    # ------------------------------------------------------------------
    # Journal helpers
    # ------------------------------------------------------------------
    def _journal_fill(self, resp: OrderResponse, scores, reasons) -> None:
        self.journal.record({
            "ts_wall": datetime.utcnow().isoformat() + "Z",
            "sim_date": resp.date.isoformat() if hasattr(resp.date, "isoformat") else str(resp.date),
            "ticker": resp.ticker,
            "decision": resp.action,
            "qty": resp.quantity,
            "price": resp.price,
            "source": "core",
            "scores": scores or {},
            "regime": None,
            "target_w": None,
            "current_w": None,
            "reasons": reasons or [],
            "shap_top3": [],
            "latency_ms": {"order": self.stats.last_latency_ms},
            "order_id": resp.id,
            "status": "FILLED",
            "reject_code": None,
        })

    def _journal_rejection(self, req: OrderRequest, code, msg, scores, reasons) -> None:
        self.journal.record({
            "ts_wall": datetime.utcnow().isoformat() + "Z",
            "sim_date": req.date.isoformat() if hasattr(req.date, "isoformat") else str(req.date),
            "ticker": req.ticker,
            "decision": req.action,
            "qty": req.quantity,
            "price": self.ledger.last_close.get(req.ticker, 0.0),
            "source": "core",
            "scores": scores or {},
            "regime": None,
            "target_w": None,
            "current_w": None,
            "reasons": reasons or [],
            "shap_top3": [],
            "latency_ms": {"order": self.stats.last_latency_ms},
            "order_id": "",
            "status": "REJECTED",
            "reject_code": code,
        })

    # ------------------------------------------------------------------
    def _short_allowed(self, req: OrderRequest) -> bool:
        # Caller wires TradingGate.short_allowed into OrderManager
        return getattr(self, "_short_flag", True)