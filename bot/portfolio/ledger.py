"""bot.portfolio.ledger — pure local mirror of platform holdings.

Per docs/06 §1: BUY / SELL / SHORT / COVER apply-fill rules, FIFO lots,
FY (1 Apr -> 31 Mar) tracking, brokerage + DP allocation, settlement P&L.

NO network I/O — this module is pure. OrderManager calls into it after each
confirmed fill from the gateway.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import date as _date, datetime, timedelta, timezone
from typing import Optional

import structlog

_log = structlog.get_logger("ledger")


@dataclass
class Lot:
    """One per BUY fill (FIFO consumption on SELL)."""
    ticker: str
    qty: int
    price: float          # fill price (close of buy day)
    buy_date: _date       # simulated date
    fees: float           # brokerage allocated to this lot


@dataclass
class ShortPos:
    """Open short position — must be COVERed or settled."""
    ticker: str
    qty: int
    entry_price: float
    open_date: _date
    margin_locked: float  # = qty * entry_price


@dataclass
class RealisedGain:
    """One closed-trade line, used by tax engine."""
    ticker: str
    qty: int
    buy_date: _date
    sell_date: _date
    buy_price: float
    sell_price: float
    gross_pnl: float       # (sell - buy) * qty
    fees: float            # brokerage + DP allocated to this line
    fy_label: str          # "FY2025-26" — sell-date FY


@dataclass
class Ledger:
    """The single source of truth for local position state.

    Modelled on docs/06 §1. All methods are pure (no I/O) and operate on
    this object's fields.
    """
    cash: float
    initial_cash: float
    lots: dict[str, deque[Lot]] = field(default_factory=lambda: defaultdict(deque))
    shorts: dict[str, ShortPos] = field(default_factory=dict)
    realised: dict[str, list[RealisedGain]] = field(default_factory=lambda: defaultdict(list))
    last_close: dict[str, float] = field(default_factory=dict)

    # --- helpers ---
    @property
    def holdings(self) -> dict[str, int]:
        """Net long qty per ticker (sum of all open lots)."""
        return {t: sum(l.qty for l in dq) for t, dq in self.lots.items() if dq}

    @property
    def long_market_value(self) -> float:
        total = 0.0
        for t, dq in self.lots.items():
            if not dq:
                continue
            qty = sum(l.qty for l in dq)
            price = self.last_close.get(t)
            if price is None:
                # Fall back to latest lot price if no mark available
                price = dq[-1].price
            total += qty * price
        return total

    @property
    def short_notional(self) -> float:
        return sum(abs(sp.qty) * sp.entry_price for sp in self.shorts.values())

    @property
    def margin_locked(self) -> float:
        return sum(sp.margin_locked for sp in self.shorts.values())

    @property
    def equity(self) -> float:
        """cash + long MV - margin_locked."""
        return self.cash + self.long_market_value - self.margin_locked

    @property
    def realised_total(self) -> float:
        return sum(r.gross_pnl - r.fees for gains in self.realised.values() for r in gains)

    def holding_qty(self, ticker: str) -> int:
        return sum(l.qty for l in self.lots.get(ticker, deque()))

    def has_short(self, ticker: str) -> bool:
        sp = self.shorts.get(ticker)
        return sp is not None and sp.qty > 0

    # ------------------------------------------------------------------
    # Apply-fill rules — docs/06 §1 table
    # ------------------------------------------------------------------
    def apply_buy(self, ticker: str, qty: int, price: float, fill_date: _date,
                  brokerage: float, dp_charge: float) -> dict:
        """BUY q@p: cash -= q*p + brokerage. If short open: cover min(q, |short|)
        first, remainder becomes a new Lot.
        """
        if qty <= 0:
            raise ValueError("qty must be positive")
        if price <= 0:
            raise ValueError("price must be positive")
        # 1) Cover short first
        cover_qty = 0
        realised = 0.0
        sp = self.shorts.get(ticker)
        if sp and sp.qty > 0:
            cover_qty = min(qty, sp.qty)
            entry = sp.entry_price
            realised = (entry - price) * cover_qty
            # Cash: spent to buy at price, but margin was locked at entry;
            # net cash change = -(qty * price) - brokerage + release of margin
            self.cash += sp.margin_locked * (cover_qty / sp.qty) - cover_qty * price
            # Reduce / close short
            sp.qty -= cover_qty
            sp.margin_locked = sp.qty * entry
            if sp.qty == 0:
                del self.shorts[ticker]
            # Allocate fees to the closed-trade line (sell-side equivalent)
            buy_fees = (brokerage + dp_charge) * (cover_qty / qty)
            self.realised[ticker].append(RealisedGain(
                ticker=ticker, qty=cover_qty,
                buy_date=sp.open_date if sp else fill_date,
                sell_date=fill_date,
                buy_price=entry, sell_price=price,
                gross_pnl=realised, fees=buy_fees,
                fy_label=self._fy_label(fill_date),
            ))
        # 2) Remainder -> new long lot
        long_qty = qty - cover_qty
        if long_qty > 0:
            lot_brokerage = brokerage * (long_qty / qty) if qty else 0
            self.cash -= long_qty * price + lot_brokerage
            self.lots[ticker].append(Lot(
                ticker=ticker, qty=long_qty, price=price,
                buy_date=fill_date, fees=lot_brokerage,
            ))
        self.last_close[ticker] = price
        return {"covered": cover_qty, "long_added": long_qty, "realised": realised}

    def apply_sell(self, ticker: str, qty: int, price: float, fill_date: _date,
                   brokerage: float, dp_charge: float) -> dict:
        """SELL q@p: cash += q*p - brokerage - dp_charge.
        FIFO consumption of long lots; each closed lot -> RealisedGain.
        Refuses if long holding < qty (caller should have validated).
        """
        if qty <= 0:
            raise ValueError("qty must be positive")
        if self.holding_qty(ticker) < qty:
            raise ValueError(f"SELL {qty} of {ticker} exceeds holding {self.holding_qty(ticker)}")
        dq = self.lots[ticker]
        remaining = qty
        proceeds = 0.0
        realised_net = 0.0
        while remaining > 0 and dq:
            lot = dq[0]
            take = min(remaining, lot.qty)
            pnl = (price - lot.price) * take
            # Allocate fees pro-rata across lots consumed
            fee_share = (brokerage + dp_charge) * (take / qty)
            lot_fees = lot.fees * (take / lot.qty)
            total_fees_for_line = fee_share + lot_fees
            proceeds += take * price
            self.realised[ticker].append(RealisedGain(
                ticker=ticker, qty=take,
                buy_date=lot.buy_date, sell_date=fill_date,
                buy_price=lot.price, sell_price=price,
                gross_pnl=pnl, fees=total_fees_for_line,
                fy_label=self._fy_label(fill_date),
            ))
            lot.qty -= take
            lot.fees -= lot_fees
            realised_net += pnl - total_fees_for_line
            if lot.qty == 0:
                dq.popleft()
            remaining -= take
        # Cash: proceeds + release of buy-fees (already spent, so add back the lot's share)
        self.cash += proceeds - brokerage - dp_charge
        self.last_close[ticker] = price
        return {"sold": qty, "proceeds": proceeds - brokerage - dp_charge,
                "realised": realised_net}

    def apply_short(self, ticker: str, qty: int, price: float, fill_date: _date,
                    brokerage: float, dp_charge: float) -> dict:
        """SHORT q@p: cash += q*p - brokerage - dp_charge.
        Lock margin = q*p from cash (100% of notional).
        """
        if qty <= 0:
            raise ValueError("qty must be positive")
        notional = qty * price
        if self.cash < notional + brokerage:
            raise ValueError("insufficient cash to lock margin")
        sp = self.shorts.get(ticker)
        if sp:
            # Add to existing short at blended price (simple)
            total_qty = sp.qty + qty
            sp.entry_price = (sp.entry_price * sp.qty + price * qty) / total_qty
            sp.qty = total_qty
            sp.margin_locked = sp.qty * sp.entry_price
        else:
            self.shorts[ticker] = ShortPos(
                ticker=ticker, qty=qty, entry_price=price,
                open_date=fill_date, margin_locked=notional,
            )
        self.cash += notional - brokerage - dp_charge
        self.last_close[ticker] = price
        return {"shorted": qty, "margin_locked": notional}

    def apply_cover(self, ticker: str, qty: int, price: float, fill_date: _date,
                    brokerage: float, dp_charge: float) -> dict:
        """COVER q@p: cash -= q*p + brokerage. Release margin for covered qty.
        Realise P&L = (entry - cover) * q.
        """
        if qty <= 0:
            raise ValueError("qty must be positive")
        sp = self.shorts.get(ticker)
        if sp is None or sp.qty < qty:
            raise ValueError(f"COVER {qty} exceeds short {sp.qty if sp else 0}")
        entry = sp.entry_price
        realised = (entry - price) * qty
        # Release margin pro-rata; pay cover cost
        released = sp.margin_locked * (qty / sp.qty)
        self.cash += released - qty * price - brokerage - dp_charge
        sp.qty -= qty
        sp.margin_locked -= released
        if sp.qty == 0:
            del self.shorts[ticker]
        self.last_close[ticker] = price
        # Record realised line so tax engine + P&L stats work
        from datetime import date as _date
        self.realised[ticker].append(RealisedGain(
            ticker=ticker, qty=qty,
            buy_date=sp.open_date if sp else fill_date,
            sell_date=fill_date,
            buy_price=entry, sell_price=price,
            gross_pnl=realised, fees=brokerage + dp_charge,
            fy_label=self._fy_label(fill_date),
        ))
        return {"covered": qty, "realised": realised}

    # ------------------------------------------------------------------
    # Reconciliation helpers
    # ------------------------------------------------------------------
    def mark_to_market(self, prices: dict[str, float]) -> dict:
        """Update last_close for tickers and return snapshot metrics."""
        self.last_close.update(prices)
        return {
            "cash": self.cash,
            "equity": self.equity,
            "long_mv": self.long_market_value,
            "short_notional": self.short_notional,
            "margin_locked": self.margin_locked,
            "realised_total": self.realised_total,
            "holdings": dict(self.holdings),
            "shorts": {t: {"qty": sp.qty, "entry": sp.entry_price,
                            "margin": sp.margin_locked, "open": sp.open_date.isoformat()}
                       for t, sp in self.shorts.items()},
        }

    @staticmethod
    def _fy_label(d: _date) -> str:
        """Indian FY = 1 Apr -> 31 Mar. Label like 'FY2025-26'."""
        if d.month >= 4:
            start = d.year
            end = d.year + 1
        else:
            start = d.year - 1
            end = d.year
        return f"FY{str(start)[-4:]}-{str(end)[-2:]}"

    def fy_labels(self) -> list[str]:
        """All distinct FYs that have realised gains."""
        seen: set[str] = set()
        for gains in self.realised.values():
            for g in gains:
                seen.add(g.fy_label)
        return sorted(seen)