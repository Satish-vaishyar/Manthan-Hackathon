"""bot.models.vol — realized-vol estimator for inverse-vol sizing (M4).

Replaces the hardcoded `vol=0.02` placeholder in bot.main with a real estimate
from the per-ticker rolling window. The constructor in
bot/portfolio/constructor.py already accepts vol per ticker; this module
just supplies a better value.

Per docs/04 §4: weights ∝ score / vol. A short-vol ticker with the same
score gets a larger weight than a high-vol ticker — the bot takes less
idiosyncratic risk per dollar.

Per docs/05 §3: features must be incremental (no per-tick pandas). We use a
Welford-style online estimator over a sliding window of returns, O(1) per
update.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

EPS = 1e-12


@dataclass
class _Window:
    """Rolling-window realized vol estimator.

    Maintains `window` log-returns and exposes the annualised volatility.
    Uses an O(window) sum + sum-of-squares refresh on each push (window ≤ 252
    so this is cheap; if it ever becomes hot we can switch to Welford).
    """
    window: int = 20
    _ret: deque[float] = field(default_factory=lambda: deque(maxlen=20))
    _prev_close: Optional[float] = None
    _sum: float = 0.0
    _sumsq: float = 0.0
    _n_obs: int = 0

    def __post_init__(self) -> None:
        # Ensure deque maxlen matches window
        if self._ret.maxlen != self.window:
            self._ret = deque(self._ret, maxlen=self.window)

    def update(self, close: float) -> None:
        """Push today's close; updates the rolling-window vol."""
        if self._prev_close is not None and self._prev_close > 0:
            r = math.log(close / self._prev_close)
            if len(self._ret) == self.window:
                # Slide window: drop oldest
                old = self._ret[0]
                self._sum -= old
                self._sumsq -= old * old
            self._ret.append(r)
            self._sum += r
            self._sumsq += r * r
            self._n_obs += 1
        else:
            self._n_obs += 1
        self._prev_close = close

    @property
    def realized_vol(self) -> float:
        """Annualised realized vol over the window. sqrt(252) scaling."""
        n = len(self._ret)
        if n < 2:
            return float("nan")
        mean = self._sum / n
        var = max(0.0, self._sumsq / n - mean * mean)
        return math.sqrt(var * 252.0)

    @property
    def n(self) -> int:
        return len(self._ret)

    @property
    def n_obs(self) -> int:
        """Total observations pushed (including the first, which produces no return)."""
        return self._n_obs

    @property
    def ready(self) -> bool:
        # Ready once we have at least 5 valid returns
        return self.n >= 5


class VolEstimator:
    """Per-ticker ring buffer of realized-vol windows.

    Usage:
        est = VolEstimator(window=20)
        est.update("AAA", 100.0)
        est.update("AAA", 101.5)
        v = est.vol("AAA")  # annualised vol, or fallback if not ready
    """

    def __init__(self, window: int = 20, fallback_vol: float = 0.02) -> None:
        self.window = window
        self.fallback = fallback_vol
        self._state: dict[str, _Window] = {}

    def _state_for(self, ticker: str) -> _Window:
        s = self._state.get(ticker)
        if s is None:
            s = _Window(window=self.window)
            self._state[ticker] = s
        return s

    def update(self, ticker: str, close: float) -> None:
        self._state_for(ticker).update(close)

    def vol(self, ticker: str) -> float:
        """Annualised realised vol, or `fallback_vol` if the ticker hasn't
        accumulated enough observations yet.

        For a ticker with >= 5 returns whose variance is exactly 0 (constant
        price series), this returns 0.0 — that's a legitimate measurement,
        not a fallback.
        """
        s = self._state.get(ticker)
        if s is None or not s.ready:
            return self.fallback
        v = s.realized_vol
        if math.isnan(v):
            return self.fallback
        return v

    def ready(self, ticker: str) -> bool:
        s = self._state.get(ticker)
        return s is not None and s.ready

    def __len__(self) -> int:
        return len(self._state)


# -----------------------------------------------------------------------------
# Standalone helper for backtest usage
# -----------------------------------------------------------------------------
def annualised_vol_from_closes(closes: list[float], window: int = 20) -> float:
    """Compute annualised vol over the last `window` log-returns.

    Used by the backtester; the live bot uses VolEstimator above.
    """
    if len(closes) < window + 1:
        return float("nan")
    rets = [math.log(closes[i] / closes[i - 1]) for i in range(-window, 0)]
    n = len(rets)
    mean = sum(rets) / n
    var = sum((r - mean) ** 2 for r in rets) / n
    return math.sqrt(max(0.0, var) * 252.0)