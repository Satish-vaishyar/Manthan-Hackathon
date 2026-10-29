"""bot.models.regime — market regime detector (M5).

Per docs/04 §4: regime gates the gross long exposure.
    - risk_on   -> multiplier 1.0  (full deployment)
    - neutral   -> multiplier 0.8  (mild de-risking)
    - risk_off  -> multiplier 0.5  (defensive)

Per docs/02 §3 + docs/04 §6: regime is computed once per simulated day from
the day's tick bundle and a rolling history of market aggregates.

Algorithm (simple, robust, no GPU):
    1. Compute today's equal-weight market return from bundle.ticks
       (if we have D-1 close from history).
    2. Maintain a rolling window of:
       - market returns (e.g. 60 days)
       - cross-sectional return dispersion (std of ticker returns)
       - realised vol (annualised) of the market
    3. Classifier:
       - risk_off if realised_vol > VOL_HIGH OR recent cumulative return < -DD
       - risk_on  if recent cumulative return > +TREND AND vol < VOL_LOW
       - neutral  otherwise

Why simple? Per docs/05 §3 + the bot must stay within the 800 ms hard
deadline. HMM/Markov-switching needs EM which is too slow and overfits on
limited data. Thresholds on rolling stats are explainable, fast, and tunable.

Optional model upgrade (M5+): train an HMM on the captured Kafka data and
replace this with .predict(). Until then, the rules detector ships.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import date as _date
from typing import Optional

import structlog

from ..contracts import DayBundle, PriceTickData

_log = structlog.get_logger("regime")

# Regime labels
RISK_ON = "risk_on"
NEUTRAL = "neutral"
RISK_OFF = "risk_off"
ALL_REGIMES = (RISK_ON, NEUTRAL, RISK_OFF)

# Default gross long target per regime (matches docs/04 §4 multipliers)
DEFAULT_GROSS_TARGET = {
    RISK_ON: 1.00,
    NEUTRAL: 0.80,
    RISK_OFF: 0.50,
}


@dataclass
class RegimeConfig:
    """Tunable thresholds. All in normalised daily-return / annualised-vol units."""
    # Lookback windows (trading days)
    market_window: int = 60
    disp_window: int = 20

    # Realised vol thresholds (annualised)
    vol_low: float = 0.10    # below = risk_on candidate
    vol_high: float = 0.30   # above = risk_off

    # Cumulative-return thresholds over market_window
    trend_up: float = 0.05   # +5% over 60d -> risk_on
    trend_dn: float = -0.10  # -10% over 60d -> risk_off

    # Cross-sectional dispersion threshold (std of daily returns across universe)
    disp_high: float = 0.025  # 2.5% daily cross-sectional stdev = choppy

    # Warm-up: how many days of history before we trust the regime
    min_history_days: int = 20


@dataclass
class RegimeState:
    """Per-day market aggregates buffer."""
    market_returns: deque[float] = field(default_factory=lambda: deque(maxlen=60))
    cross_dispersion: deque[float] = field(default_factory=lambda: deque(maxlen=20))
    last_close: dict[str, float] = field(default_factory=dict)
    last_sim_date: Optional[_date] = None
    days_observed: int = 0
    current_regime: str = NEUTRAL  # until we have enough data
    days_in_current_regime: int = 0


class RegimeDetector:
    """Computes one regime label per simulated day from the bundle + history."""

    def __init__(self, config: Optional[RegimeConfig] = None) -> None:
        self.config = config or RegimeConfig()
        self.state = RegimeState(
            market_returns=deque(maxlen=self.config.market_window),
            cross_dispersion=deque(maxlen=self.config.disp_window),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update(self, bundle: DayBundle) -> str:
        """Push today's bundle; return the regime label for this day.

        Side effect: updates internal state. Same bundle should NOT be
        fed twice.
        """
        if not bundle.ticks:
            return self.state.current_regime

        # Compute today's market return + dispersion from the bundle
        market_ret = self._market_return(bundle)
        dispersion = self._cross_dispersion(bundle)

        if market_ret is not None:
            self.state.market_returns.append(market_ret)
        if dispersion is not None:
            self.state.cross_dispersion.append(dispersion)

        # Update last_close for next-day delta
        for ticker, tick in bundle.ticks.items():
            self.state.last_close[ticker] = tick.close

        self.state.days_observed += 1
        self.state.last_sim_date = bundle.simulated_date

        # Classify
        new_regime = self._classify()
        if new_regime != self.state.current_regime:
            self.state.days_in_current_regime = 1
            _log.info(
                "regime_change",
                **{
                    "from": self.state.current_regime,
                    "to": new_regime,
                    "sim_date": bundle.simulated_date.isoformat(),
                    "market_ret": round(market_ret, 5) if market_ret is not None else None,
                    "vol": round(self._market_vol() or 0.0, 4),
                },
                component="regime",
            )
            self.state.current_regime = new_regime
        else:
            self.state.days_in_current_regime += 1
        return new_regime

    @property
    def current(self) -> str:
        return self.state.current_regime

    @property
    def days_observed(self) -> int:
        return self.state.days_observed

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _market_return(self, bundle: DayBundle) -> Optional[float]:
        """Equal-weight log return of the universe, today vs last_close."""
        last = self.state.last_close
        rets: list[float] = []
        for ticker, tick in bundle.ticks.items():
            if ticker in last and last[ticker] > 0:
                rets.append(math.log(tick.close / last[ticker]))
        if not rets:
            return None
        return sum(rets) / len(rets)

    def _cross_dispersion(self, bundle: DayBundle) -> Optional[float]:
        """Std-dev of ticker returns today (cross-sectional dispersion)."""
        last = self.state.last_close
        rets: list[float] = []
        for ticker, tick in bundle.ticks.items():
            if ticker in last and last[ticker] > 0:
                rets.append(math.log(tick.close / last[ticker]))
        if len(rets) < 2:
            return None
        mean = sum(rets) / len(rets)
        var = sum((r - mean) ** 2 for r in rets) / len(rets)
        return math.sqrt(var)

    def _market_vol(self) -> Optional[float]:
        """Annualised realised vol from the market_returns window."""
        n = len(self.state.market_returns)
        if n < 5:
            return None
        mean = sum(self.state.market_returns) / n
        var = sum((r - mean) ** 2 for r in self.state.market_returns) / n
        return math.sqrt(max(0.0, var) * 252.0)

    def _cumulative_return(self) -> Optional[float]:
        """Sum of log returns in the market window (= cumulative log return)."""
        if not self.state.market_returns:
            return None
        return sum(self.state.market_returns)

    def _avg_dispersion(self) -> Optional[float]:
        """Mean cross-sectional dispersion over the disp_window."""
        if not self.state.cross_dispersion:
            return None
        return sum(self.state.cross_dispersion) / len(self.state.cross_dispersion)

    def _classify(self) -> str:
        cfg = self.config
        if self.state.days_observed < cfg.min_history_days:
            return NEUTRAL  # warm-up: stay neutral until we have history

        vol = self._market_vol()
        cumret = self._cumulative_return()
        disp = self._avg_dispersion()

        # Risk-off first (most conservative)
        # Trigger on ANY of: high vol, deep drawdown, choppy dispersion
        if vol is not None and vol >= cfg.vol_high:
            return RISK_OFF
        if cumret is not None and cumret <= cfg.trend_dn:
            return RISK_OFF
        if disp is not None and disp >= cfg.disp_high and vol is not None and vol >= cfg.vol_low:
            return RISK_OFF

        # Risk-on: positive trend with low vol
        if (cumret is not None and cumret >= cfg.trend_up
                and vol is not None and vol <= cfg.vol_low):
            return RISK_ON

        # Otherwise neutral
        return NEUTRAL

    # ------------------------------------------------------------------
    # Snapshot for the journal
    # ------------------------------------------------------------------
    def snapshot(self) -> dict:
        return {
            "regime": self.state.current_regime,
            "days_in_regime": self.state.days_in_current_regime,
            "days_observed": self.state.days_observed,
            "market_vol_annualised": self._market_vol(),
            "cumulative_return": self._cumulative_return(),
            "avg_dispersion": self._avg_dispersion(),
        }