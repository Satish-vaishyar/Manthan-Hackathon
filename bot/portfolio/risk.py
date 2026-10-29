"""bot.portfolio.risk — pre-trade risk checks.

Pure logic. Called by validator and/or order_manager BEFORE submitting.
Per docs/06 §3 + docs/01 §9:
    - per-name cap (default 15%)
    - per-sector cap (default 30%)
    - gross long / gross short caps
    - cash buffer (5% equity or 15% short notional, whichever higher)
    - DD breakers (-8% soft, -15% hard)
    - settlement guard (no new SHORT if days_to_settlement < 3)
    - per-stop ATR-based stops
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date as _date
from typing import Optional

from ..config import RiskSettings
from .ledger import Ledger


@dataclass
class RiskVerdict:
    ok: bool
    code: Optional[str] = None
    message: Optional[str] = None

    @classmethod
    def allow(cls) -> "RiskVerdict":
        return cls(ok=True)

    @classmethod
    def deny(cls, code: str, message: str) -> "RiskVerdict":
        return cls(ok=False, code=code, message=message)


@dataclass
class DrawdownState:
    """Tracks running peak equity for DD breakers."""
    peak_equity: float
    current_equity: float

    @property
    def drawdown(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return (self.peak_equity - self.current_equity) / self.peak_equity

    def update(self, equity: float) -> None:
        self.current_equity = equity
        if equity > self.peak_equity:
            self.peak_equity = equity


class RiskManager:
    """All pre-trade checks. Pure functions over Ledger + inputs."""

    def __init__(self, settings: RiskSettings, settlement_calendar) -> None:
        self.settings = settings
        self.calendar = settlement_calendar
        # P3-14: seed peak_equity at 0 so the first reconcile can set the
        # real value. The previous default of 1_000_000 caused a "drawdown
        # 100%" false-positive at boot and a needless SAFE_MODE bounce.
        self.dd = DrawdownState(peak_equity=0.0, current_equity=0.0)

    def seed_equity(self, equity: float) -> None:
        """P3-14: call after the first reconcile so peak_equity starts at
        the real portfolio value (no false-positive DD breaker on boot)."""
        if equity <= 0:
            return
        if self.dd.peak_equity <= 0:
            self.dd.peak_equity = equity
            self.dd.current_equity = equity

    # ------------------------------------------------------------------
    # DD breakers
    # ------------------------------------------------------------------
    def update_equity(self, equity: float) -> RiskVerdict:
        """Feed the latest equity mark into the DD tracker; deny with
        RISK_DD_SOFT / RISK_DD_HARD once drawdown crosses those levels."""
        self.dd.update(equity)
        if self.dd.drawdown >= self.settings.dd_hard:
            return RiskVerdict.deny("RISK_DD_HARD",
                f"drawdown {self.dd.drawdown:.2%} >= hard {self.settings.dd_hard:.2%}")
        if self.dd.drawdown >= self.settings.dd_soft:
            return RiskVerdict.deny("RISK_DD_SOFT",
                f"drawdown {self.dd.drawdown:.2%} >= soft {self.settings.dd_soft:.2%}")
        return RiskVerdict.allow()

    # ------------------------------------------------------------------
    # Pre-trade checks
    # ------------------------------------------------------------------
    def check_buy(
        self,
        ledger: Ledger,
        ticker: str,
        qty: int,
        price: float,
        today: _date,
        sector: Optional[str] = None,
        sector_exposures: Optional[dict[str, float]] = None,
    ) -> RiskVerdict:
        """BUY checks: post-trade per-name cap, sector cap (if given), and
        cash buffer (the greater of cash_buffer*equity and 15% of short
        notional must remain after the trade)."""
        notional = qty * price
        equity = ledger.equity or 1.0
        # Per-name cap on POST-trade weight for THIS ticker
        cur_qty = ledger.holdings.get(ticker, 0)
        cur_w = (cur_qty * price) / equity if equity > 0 else 0
        new_exposure = cur_w * equity + notional
        new_w = new_exposure / equity if equity > 0 else 0
        if new_w > self.settings.max_name_weight:
            return RiskVerdict.deny("RISK_NAME_CAP",
                f"{ticker}: post-trade weight {new_w:.2%} > cap {self.settings.max_name_weight:.2%}")
        # Sector cap
        if sector is not None and sector_exposures is not None:
            sec_w = sector_exposures.get(sector, 0.0) * equity
            new_sec = (sec_w + notional) / equity
            if new_sec > self.settings.max_sector_weight:
                return RiskVerdict.deny("RISK_SECTOR_CAP",
                    f"{sector}: post-trade {new_sec:.2%} > cap {self.settings.max_sector_weight:.2%}")
        # Cash buffer: enough cash after trade?
        cash_after = ledger.cash - notional
        buffer_needed = max(
            self.settings.cash_buffer * equity,
            0.15 * ledger.short_notional,
        )
        if cash_after < buffer_needed:
            return RiskVerdict.deny("RISK_CASH_BUFFER",
                f"cash after {cash_after:.2f} < buffer {buffer_needed:.2f}")
        return RiskVerdict.allow()

    def check_sell(self, ledger: Ledger, ticker: str, qty: int) -> RiskVerdict:
        """SELL check: we actually hold enough shares."""
        if ledger.holding_qty(ticker) < qty:
            return RiskVerdict.deny("RISK_NO_HOLDING",
                f"SELL {qty} of {ticker} but only hold {ledger.holding_qty(ticker)}")
        return RiskVerdict.allow()

    def check_short(
        self,
        ledger: Ledger,
        ticker: str,
        qty: int,
        price: float,
        today: _date,
        short_allowed: bool = True,
    ) -> RiskVerdict:
        """SHORT checks: stage allows it, settlement is >=3 trading days away
        (forced square-off risk), gross short cap, and 5%-of-equity per-name
        short cap."""
        if not short_allowed:
            return RiskVerdict.deny("RISK_SHORT_DISABLED",
                f"short selling not allowed in current stage")
        # Settlement guard
        if self.calendar is not None:
            dt = self.calendar.days_to_next_settlement(today)
            if dt is not None and dt < 3:
                return RiskVerdict.deny("RISK_SETTLEMENT_LOCK",
                    f"settlement in {dt} trading days — no new shorts")
        # Gross short cap
        equity = ledger.equity or 1.0
        new_short_notional = ledger.short_notional + qty * price
        if new_short_notional / equity > self.settings.max_short_gross:
            return RiskVerdict.deny("RISK_SHORT_GROSS",
                f"gross short {new_short_notional/equity:.2%} > cap {self.settings.max_short_gross:.2%}")
        # Per-short cap: 5% of equity per single name
        if (qty * price) / equity > 0.05:
            return RiskVerdict.deny("RISK_SINGLE_SHORT",
                f"single short notional {qty*price/equity:.2%} > 5% cap")
        return RiskVerdict.allow()

    def check_cover(self, ledger: Ledger, ticker: str, qty: int) -> RiskVerdict:
        """COVER check: the open short is large enough."""
        sp = ledger.shorts.get(ticker)
        if sp is None or sp.qty < qty:
            return RiskVerdict.deny("RISK_NO_SHORT",
                f"COVER {qty} but only short {sp.qty if sp else 0}")
        return RiskVerdict.allow()

    # ------------------------------------------------------------------
    # ATR stop helper
    # ------------------------------------------------------------------
    @staticmethod
    def atr_stop_price(entry_price: float, atr: float, side: str, mult: float = 2.5) -> float:
        """Stop price for ATR-based stop.

        side: 'long' -> stop below entry; 'short' -> stop above entry.
        """
        if atr <= 0:
            return entry_price
        if side == "long":
            return max(0.01, entry_price - mult * atr)
        return entry_price + mult * atr