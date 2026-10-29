"""bot.portfolio.settlement — every-10-trading-days calendar + guard.

Per docs/01 §7 + docs/06 §1:
    - Settlement occurs every 10 TRADING DAYS
    - On settlement, shorts must be covered; if deficit -> longs liquidated
    - force_close_before_days: cover manually N trading days early to retain
      control of the exit price
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as _date, timedelta
from typing import Optional


@dataclass
class SettlementCalendar:
    """Computes settlement dates given the trading-day calendar.

    The interval is `interval_trading_days` (default 10). The actual calendar
    must be supplied externally (the platform emits a price_tick only on
    trading days, so we derive the calendar by observing ticks).
    """
    interval_trading_days: int = 10
    # All observed trading days (in chronological order)
    _observed: list[_date] = field(default_factory=list)
    _observed_set: set[_date] = field(default_factory=set)
    # Anchor: the first forced square-off date we discovered from the platform
    anchor: Optional[_date] = None

    def observe_trading_day(self, d: _date) -> None:
        if d in self._observed_set:
            return
        # Insert sorted
        idx = 0
        for i, x in enumerate(self._observed):
            if d < x:
                idx = i
                break
            idx = i + 1
        self._observed.insert(idx, d)
        self._observed_set.add(d)

    def trading_day_index(self, d: _date) -> Optional[int]:
        """1-based index of `d` in the observed trading days, or None."""
        try:
            return self._observed.index(d) + 1
        except ValueError:
            return None

    def is_settlement_day(self, d: _date) -> bool:
        idx = self.trading_day_index(d)
        if idx is None or self.anchor is None:
            return False
        # First settlement on/after anchor when trading_day_index % interval == 0
        anchor_idx = self.trading_day_index(self.anchor)
        if anchor_idx is None or idx < anchor_idx:
            return False
        return (idx - anchor_idx) % self.interval_trading_days == 0

    def next_settlement_day(self, after: _date) -> Optional[_date]:
        for d in self._observed:
            if d <= after:
                continue
            if self.is_settlement_day(d):
                return d
        return None

    def days_to_next_settlement(self, today: _date) -> Optional[int]:
        nxt = self.next_settlement_day(today)
        if nxt is None:
            return None
        # Count trading days from today (inclusive of today if trading day) to next settlement (inclusive)
        idx_today = self.trading_day_index(today)
        idx_next = self.trading_day_index(nxt)
        if idx_today is None or idx_next is None:
            return None
        return max(0, idx_next - idx_today)

    def should_force_close_short(self, today: _date, lead_days: int = 2) -> bool:
        """Cover shorts `lead_days` trading days before settlement."""
        dt = self.days_to_next_settlement(today)
        if dt is None:
            return False
        return 0 < dt <= lead_days

    def discover_anchor_from_event(self, settlement_date: _date) -> None:
        """If the platform emits a settlement event, set anchor there."""
        self.anchor = settlement_date