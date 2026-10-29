"""bot.execution.validator — pure pre-submission checks.

Per docs/03 §3 + docs/06 + docs/11 §1:
    - each rejection code is prevented client-side
    - duplicate-in-flight detection
    - integer/limits
    - BUY cash check; SELL long check; SHORT margin + settlement check; COVER short check
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date as _date, datetime
from typing import Optional

from ..config import Settings
from ..contracts import OrderRequest, RejectCode
from ..portfolio.ledger import Ledger
from ..portfolio.risk import RiskManager


@dataclass
class ValidationResult:
    ok: bool
    code: Optional[str] = None
    message: Optional[str] = None

    @classmethod
    def allow(cls) -> "ValidationResult":
        return cls(ok=True)

    @classmethod
    def deny(cls, code: str, message: str) -> "ValidationResult":
        return cls(ok=False, code=code, message=message)


class OrderValidator:
    """Pure pre-submission gate.

    Does NOT touch the network. The OrderManager calls validate() and only
    submits if ok=True.
    """

    def __init__(self, settings: Settings, risk: RiskManager) -> None:
        self.settings = settings
        self.risk = risk
        self._in_flight: set[tuple[str, str, int]] = set()  # (ticker, action, qty)

    def validate(
        self,
        req: OrderRequest,
        ledger: Ledger,
        *,
        short_allowed: bool = True,
        sim_date: Optional[_date] = None,
        universe: Optional[set[str]] = None,
    ) -> ValidationResult:
        # 1) Basic field checks
        if not req.ticker or not isinstance(req.ticker, str):
            return ValidationResult.deny(RejectCode.INVALID_TICKER, "ticker missing or wrong type")
        if universe is not None and req.ticker not in universe:
            return ValidationResult.deny(RejectCode.INVALID_TICKER,
                f"{req.ticker} not in active universe")
        if not isinstance(req.quantity, int) or req.quantity <= 0:
            return ValidationResult.deny(RejectCode.INVALID_QUANTITY,
                f"quantity must be positive int, got {req.quantity}")
        if req.action not in ("BUY", "SELL", "SHORT", "COVER"):
            return ValidationResult.deny(RejectCode.INVALID_QUANTITY,
                f"action {req.action} not recognised")
        # Duplicate-in-flight
        key = (req.ticker, req.action, req.quantity)
        if key in self._in_flight:
            return ValidationResult.deny(RejectCode.DUPLICATE_ORDER,
                f"duplicate {req.action} {req.quantity} {req.ticker} in flight")

        # 2) Action-specific
        price = ledger.last_close.get(req.ticker, 0.0)
        if req.action == "BUY":
            return self._validate_buy(req, ledger, price)
        if req.action == "SELL":
            return self._validate_sell(req, ledger)
        if req.action == "SHORT":
            return self._validate_short(req, ledger, price, short_allowed, sim_date)
        if req.action == "COVER":
            return self._validate_cover(req, ledger)
        return ValidationResult.deny(RejectCode.SERVER_ERROR, f"unhandled action {req.action}")

    # ------------------------------------------------------------------
    # Per-action checks
    # ------------------------------------------------------------------
    def _validate_buy(self, req: OrderRequest, ledger: Ledger, price: float) -> ValidationResult:
        # Cash check (BUY cash impact = q*p + brokerage; if covers a short, less)
        notional = req.quantity * price
        brokerage = self.settings.tax.brokerage_per_order
        # Conservative: assume worst-case full long (no short to cover)
        if ledger.cash < notional + brokerage:
            return ValidationResult.deny(RejectCode.INSUFFICIENT_CASH,
                f"cash {ledger.cash:.2f} < notional+fees {notional + brokerage:.2f}")
        # Min notional
        if notional < self.settings.strategy.min_trade_notional:
            return ValidationResult.deny(RejectCode.RISK_LIMIT,
                f"notional {notional:.0f} < min {self.settings.strategy.min_trade_notional:.0f}")
        # Risk: per-name / sector / cash buffer
        v = self.risk.check_buy(ledger, req.ticker, req.quantity, price, req.date)
        if not v.ok:
            return ValidationResult.deny(v.code or RejectCode.RISK_LIMIT, v.message or "risk check failed")
        return ValidationResult.allow()

    def _validate_sell(self, req: OrderRequest, ledger: Ledger) -> ValidationResult:
        if ledger.holding_qty(req.ticker) < req.quantity:
            return ValidationResult.deny(RejectCode.INSUFFICIENT_HOLDINGS,
                f"hold {ledger.holding_qty(req.ticker)} < sell {req.quantity}")
        v = self.risk.check_sell(ledger, req.ticker, req.quantity)
        if not v.ok:
            return ValidationResult.deny(v.code or RejectCode.RISK_LIMIT, v.message or "risk check failed")
        return ValidationResult.allow()

    def _validate_short(self, req: OrderRequest, ledger: Ledger, price: float,
                        short_allowed: bool, sim_date: Optional[_date]) -> ValidationResult:
        if not short_allowed:
            return ValidationResult.deny(RejectCode.SHORT_NOT_ALLOWED,
                "short selling disabled in current stage")
        notional = req.quantity * price
        brokerage = self.settings.tax.brokerage_per_order
        if ledger.cash < notional + brokerage:
            return ValidationResult.deny(RejectCode.INSUFFICIENT_MARGIN,
                f"margin requires {notional + brokerage:.2f}, cash {ledger.cash:.2f}")
        if sim_date is not None:
            v = self.risk.check_short(ledger, req.ticker, req.quantity, price, sim_date, short_allowed)
            if not v.ok:
                return ValidationResult.deny(v.code or RejectCode.RISK_LIMIT, v.message or "risk check failed")
        return ValidationResult.allow()

    def _validate_cover(self, req: OrderRequest, ledger: Ledger) -> ValidationResult:
        sp = ledger.shorts.get(req.ticker)
        if sp is None or sp.qty < req.quantity:
            return ValidationResult.deny(RejectCode.INSUFFICIENT_HOLDINGS,
                f"short {sp.qty if sp else 0} < cover {req.quantity}")
        v = self.risk.check_cover(ledger, req.ticker, req.quantity)
        if not v.ok:
            return ValidationResult.deny(v.code or RejectCode.RISK_LIMIT, v.message or "risk check failed")
        return ValidationResult.allow()

    # ------------------------------------------------------------------
    # In-flight tracking
    # ------------------------------------------------------------------
    def mark_in_flight(self, req: OrderRequest) -> None:
        self._in_flight.add((req.ticker, req.action, req.quantity))

    def clear_in_flight(self, req: OrderRequest) -> None:
        self._in_flight.discard((req.ticker, req.action, req.quantity))

    @property
    def inflight_count(self) -> int:
        return len(self._in_flight)