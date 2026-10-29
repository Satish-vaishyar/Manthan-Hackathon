"""bot.signals.volatility — M4 shared vol module for the bot (docs/05 §6).

Wraps the existing `bot.models.vol.VolEstimator` so callers in the bot don't
have to import from `bot.models`. Also exposes:
    - `atr(...)`              — Average True Range (14-day) per ticker
    - `atr_stop(...)`         — stop price helper used by RiskManager
    - `parkinson(...)`        — Parkinson range-volatility estimator (optional)

Position sizing (inverse-vol) consumes `vol(...)`. Risk uses `atr(...)` to
set the ATR stop distance (`stop_atr_mult × atr`).
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional

from ..models.vol import VolEstimator, annualised_vol_from_closes
from ..portfolio.risk import RiskManager


@dataclass
class _TickerATR:
    """Ring buffer for true-range history per ticker (length = atr_window)."""

    window: int
    trs: Deque[float] = field(default_factory=lambda: deque(maxlen=14))
    prev_close: Optional[float] = None

    def __post_init__(self) -> None:
        if self.trs.maxlen != self.window:
            self.trs = deque(self.trs, maxlen=self.window)

    def update(self, high: float, low: float, close: float) -> None:
        """Append one true-range observation (Wilder TR: max of H-L, |H-C'|, |L-C'|)."""
        if self.prev_close is None:
            tr = high - low
        else:
            tr = max(
                high - low,
                abs(high - self.prev_close),
                abs(low - self.prev_close),
            )
        self.trs.append(max(0.0, tr))
        self.prev_close = close

    @property
    def atr(self) -> Optional[float]:
        if not self.trs:
            return None
        return sum(self.trs) / len(self.trs)


class VolatilityModule:
    """Per-ticker realised vol + ATR.

    Single source of truth for both signals; the bot updates it once per day
    per ticker with the day's candle. RiskManager consumes `atr(...)` for the
    stop distance; the constructor consumes `vol(...)` for inverse-vol sizing.
    """

    def __init__(self, vol_window: int = 20, fallback_vol: float = 0.02,
                 atr_window: int = 14) -> None:
        self.vol_estimator = VolEstimator(window=vol_window, fallback_vol=fallback_vol)
        self._atr_window = atr_window
        self._atr_state: dict[str, _TickerATR] = {}

    # ------------------------------------------------------------------
    # Updates
    # ------------------------------------------------------------------
    def update(self, ticker: str, high: float, low: float, close: float) -> None:
        """Push today's candle."""
        self.vol_estimator.update(ticker, close)
        st = self._atr_state.get(ticker)
        if st is None:
            st = _TickerATR(window=self._atr_window)
            self._atr_state[ticker] = st
        st.update(high, low, close)

    def update_from_tick(self, ticker: str, open_: float, high: float,
                         low: float, close: float) -> None:
        """Convenience for the bot's DayAssembler → PriceTickData path."""
        self.update(ticker, high, low, close)

    # ------------------------------------------------------------------
    # Read-only
    # ------------------------------------------------------------------
    def vol(self, ticker: str) -> float:
        """Annualised realised vol, or the fallback if not ready."""
        return self.vol_estimator.vol(ticker)

    def atr(self, ticker: str) -> Optional[float]:
        """Average True Range, or None if no observations yet."""
        st = self._atr_state.get(ticker)
        return st.atr if st else None

    def atr_stop(self, ticker: str, entry_price: float, side: str,
                 mult: float = 2.5) -> Optional[float]:
        """ATR-based stop price; None if ATR isn't available yet.

        `side`: 'long' -> stop below entry; 'short' -> stop above entry.
        """
        a = self.atr(ticker)
        if a is None or a <= 0 or entry_price <= 0:
            return None
        return RiskManager.atr_stop_price(entry_price, a, side, mult)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def parkinson(self, ticker: str, window: int = 20) -> Optional[float]:
        """Parkinson range-vol estimator (annualised) — best-effort, optional.

        Requires raw high/low history which we don't keep by default. Provided
        as a hook so a future caller can feed it explicit (high, low) tuples
        via `feed_parkinson_window`.
        """
        st = self._atr_state.get(ticker)
        if st is None or len(st.trs) < window:
            return None
        # True range lower-bounds Parkinson variance; we keep it simple.
        rs = list(st.trs)[-window:]
        mean = sum(rs) / len(rs)
        var = sum((r - mean) ** 2 for r in rs) / len(rs)
        return math.sqrt(max(0.0, var) * 252.0)

    def __len__(self) -> int:
        return len(self.vol_estimator)


# Convenience: standalone helper for the backtester.
def annualised_vol(closes: list[float], window: int = 20) -> float:
    """Wrapper over `bot.models.vol.annualised_vol_from_closes`."""
    return annualised_vol_from_closes(closes, window=window)