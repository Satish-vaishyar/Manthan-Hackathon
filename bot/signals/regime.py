"""bot.signals.regime — M3 rule-based risk-on / neutral / risk-off (docs/05 §5).

The HMM option (Option B in the doc) is intentionally not implemented here:
the rule filter is robust, explainable, and the HMM only earns its place if it
beats the rules in the backtester. We provide a clean `RegimeFilter` API that
the (future) HMM can implement behind the same interface.

Inputs (computed by the bot from the DayBundle, kept incrementally):
    market_return_1d       equal-weight cross-sectional 1-day return
    market_return_20d      equal-weight cross-sectional 20-day return
    market_vol_20d         equal-weight cross-sectional 20d realised vol (annualised)
    breadth_pct            fraction of tickers with close > SMA(breadth_window_days)
    market_vol_median_120d median of market_vol_20d over the last 120 days

Rules (Option A):
    risk-off   if (breadth_pct < breadth_risk_off AND market_return_20d < 0)
               OR (market_vol_20d > vol_spike_mult * market_vol_median_120d)
    risk-on    if breadth_pct > breadth_risk_on AND market_return_20d > 0
    neutral    otherwise

Hysteresis: a new label must be observed for `hold_days` consecutive days
before the filter switches. This prevents flip-flopping around thresholds.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field as _field
from datetime import date as _date
from typing import Deque, Iterable, Optional

from ..config import RegimeSettings


# Allowed regime labels (also matches RiskSettings.gross_long_target keys).
RISK_ON = "risk_on"
RISK_OFF = "risk_off"
NEUTRAL = "neutral"

ALL_REGIMES = (RISK_ON, RISK_OFF, NEUTRAL)


@dataclass
class MarketState:
    """Bot-maintained incremental state feeding the regime filter.

    `market_returns` is the bot's per-day equal-weight return series.
    `closes_by_ticker` is the per-ticker ring of closes used to compute the
    breadth signal. `market_vol_history` is the trailing annualised vol of the
    equal-weight market return series, one observation per simulated day.
    """

    market_returns: Deque[float] = _field(default_factory=lambda: deque(maxlen=260))
    market_vol_history: Deque[float] = _field(default_factory=lambda: deque(maxlen=260))
    closes_by_ticker: dict[str, Deque[float]] = _field(default_factory=dict)

    def update(self, sim_date: _date, ticks: dict) -> None:
        """Push one day of equal-weight market state.

        Args:
            sim_date: simulated date (for logging only).
            ticks:    ticker -> PriceTickData for the day.
        """
        if not ticks:
            return
        # 1-day return per ticker (vs its previous close in our ring)
        rets: list[float] = []
        for ticker, tick in ticks.items():
            buf = self.closes_by_ticker.setdefault(ticker, deque(maxlen=260))
            prev = buf[-1] if buf else None
            buf.append(tick.close)
            if prev is not None and prev > 0:
                rets.append(tick.close / prev - 1.0)
        if rets:
            self.market_returns.append(sum(rets) / len(rets))
        # Recompute trailing 20d vol after enough history
        if len(self.market_returns) >= 21:
            window = list(self.market_returns)[-20:]
            mean = sum(window) / len(window)
            var = sum((r - mean) ** 2 for r in window) / len(window)
            import math
            self.market_vol_history.append(math.sqrt(max(0.0, var) * 252.0))

    # --- derived features for the rule ---
    def market_return_1d(self) -> Optional[float]:
        if not self.market_returns:
            return None
        return self.market_returns[-1]

    def market_return_20d(self) -> Optional[float]:
        if len(self.market_returns) < 21:
            return None
        # product of (1+r) over the last 20 days minus 1
        prod = 1.0
        for r in list(self.market_returns)[-20:]:
            prod *= (1.0 + r)
        return prod - 1.0

    def market_vol_20d(self) -> Optional[float]:
        if not self.market_vol_history:
            return None
        return self.market_vol_history[-1]

    def market_vol_median_120d(self) -> Optional[float]:
        if len(self.market_vol_history) < 5:
            return None
        window = list(self.market_vol_history)[-120:]
        s = sorted(window)
        return s[len(s) // 2]

    def breadth_pct(self, window: int) -> Optional[float]:
        """Fraction of tickers with last close > SMA(window)."""
        if not self.closes_by_ticker:
            return None
        above = total = 0
        for buf in self.closes_by_ticker.values():
            if len(buf) < window:
                continue
            total += 1
            sma = sum(list(buf)[-window:]) / window
            if buf[-1] > sma:
                above += 1
        if total == 0:
            return None
        return above / total


@dataclass
class RegimeVerdict:
    regime: str = NEUTRAL
    candidate: str = NEUTRAL      # what the rules are saying today (pre-hysteresis)
    held_since: Optional[_date] = None
    days_in_current: int = 0
    confidence: float = 0.0       # 0..1 heuristic from how far past thresholds we are
    reason: str = ""


class RegimeFilter:
    """Rule-based M3 with hysteresis.

    Usage:
        rf = RegimeFilter(settings)
        # each simulated day, after ticks are in:
        market_state.update(sim_date, ticks)
        verdict = rf.update(sim_date, market_state)
        # verdict.regime in {RISK_ON, NEUTRAL, RISK_OFF}
    """

    def __init__(self, settings: RegimeSettings) -> None:
        self.settings = settings
        self._candidate: str = NEUTRAL
        self._candidate_streak: int = 0
        self._current: str = NEUTRAL
        self._days_in_current: int = 0
        self._held_since: Optional[_date] = None
        self._last_date: Optional[_date] = None

    @property
    def current(self) -> str:
        return self._current

    # ------------------------------------------------------------------
    def _classify(self, state: MarketState) -> tuple[str, float, str]:
        """Return (candidate, confidence, reason) from today's market state."""
        s = self.settings
        ret_20d = state.market_return_20d()
        breadth = state.breadth_pct(s.breadth_window_days)
        vol_20d = state.market_vol_20d()
        vol_med = state.market_vol_median_120d()

        if ret_20d is None or breadth is None:
            return NEUTRAL, 0.0, "warmup"

        # risk-off: breadth < breadth_risk_off AND ret_20d < 0
        breadth_off = breadth < s.breadth_risk_off and ret_20d < 0
        # Vol spike only counts if the median itself is meaningful — a near-zero
        # median (calm pre-launch warmup) would otherwise trigger risk_off on
        # any single noise bar.
        _VOL_FLOOR = 0.02  # 2% annualised — sane "calm market" floor
        vol_spike = (
            vol_20d is not None and vol_med is not None
            and vol_med >= _VOL_FLOOR
            and vol_20d > s.vol_spike_mult * vol_med
        )
        if breadth_off or vol_spike:
            reason = []
            if breadth_off:
                reason.append(f"breadth {breadth:.0%} < {s.breadth_risk_off:.0%} & ret_20d {ret_20d:+.2%}")
            if vol_spike:
                reason.append(f"vol {vol_20d:.2f} > {s.vol_spike_mult:.1f}x median {vol_med:.2f}")
            return RISK_OFF, 0.8 if vol_spike else 0.5, "; ".join(reason)

        # risk-on: breadth > breadth_risk_on AND ret_20d > 0
        if breadth > s.breadth_risk_on and ret_20d > 0:
            margin = min(1.0, (breadth - s.breadth_risk_on) / max(1e-9, 1.0 - s.breadth_risk_on))
            return RISK_ON, 0.5 + 0.5 * margin, f"breadth {breadth:.0%} > {s.breadth_risk_on:.0%} & ret_20d {ret_20d:+.2%}"

        return NEUTRAL, 0.3, f"breadth {breadth:.0%}, ret_20d {ret_20d:+.2%}"

    # ------------------------------------------------------------------
    def update(self, sim_date: _date, state: MarketState) -> RegimeVerdict:
        """Feed today's market state; returns the current (post-hysteresis) regime."""
        candidate, conf, reason = self._classify(state)

        # Hysteresis: same candidate `hold_days` in a row -> switch
        if candidate != self._candidate:
            self._candidate = candidate
            self._candidate_streak = 1
        else:
            self._candidate_streak += 1

        if candidate != self._current and self._candidate_streak >= self.settings.hold_days:
            self._current = candidate
            self._days_in_current = 1
            self._held_since = sim_date
        else:
            self._days_in_current += 1

        return RegimeVerdict(
            regime=self._current,
            candidate=candidate,
            held_since=self._held_since,
            days_in_current=self._days_in_current,
            confidence=conf,
            reason=reason,
        )

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._candidate = NEUTRAL
        self._candidate_streak = 0
        self._current = NEUTRAL
        self._days_in_current = 0
        self._held_since = None


# Convenience: map regime -> gross long target (RiskSettings.gross_long_target).
def gross_long_target(regime: str, gross_table: dict[str, float],
                      default: float = 0.75) -> float:
    """Look up the regime's gross long target with a sane fallback."""
    if regime not in gross_table:
        return default
    v = gross_table[regime]
    return max(0.0, min(1.0, float(v)))