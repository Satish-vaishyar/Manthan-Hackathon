"""bot.signals.overlays — opt-in strategy-improvement wrappers (2026-10-29).

Each helper is a small, individually-disable-able overlay on top of the
sub-scorer closures wired in bot.main:

    - make_horizon_blend_mom_fn: blend 5d/10d/20d momentum into the mom
      sub-score (config: strategy.mom_horizon_weights).
    - make_mean_reversion_mom_fn: replace momentum with a mean-reversion
      sub-score (negative 10d return z-score, clipped) when the regime is
      risk_off (config: strategy.mean_reversion_in_risk_off).
    - make_decayed_news_fn: exponential decay of per-ticker news sentiment
      with a configurable half-life in sim days
      (config: strategy.news_half_life_days; 0 disables).
    - strong_disagreement: sign-mismatch gate between rank and mom
      sub-scores (config: strategy.disagreement_gate).
    - passes_entry_timing: universe-relative entry timing check against a
      rolling per-ticker score history (config: strategy.entry_timing).

All overlays are fail-soft: missing history / missing closures make the
feature a no-op and fall back to the base scorer.
"""
from __future__ import annotations

import math
from collections import deque
from statistics import median
from typing import Callable, Optional

import numpy as np
import structlog

from ..contracts import DayBundle

_log = structlog.get_logger("signals.overlays")


# ---------------------------------------------------------------------------
# Momentum helpers
# ---------------------------------------------------------------------------
def _horizon_z(ring_buffer, ticker: str, days: int) -> Optional[float]:
    """Vol-normalised `days`-day log return, clipped to [-1, +1] (z/2).

    Returns None when the ring buffer lacks `days` candles (fail-soft).
    """
    try:
        f = ring_buffer.frame(ticker, n=days)
        if f is None or len(f) < days:
            return None
        closes = f["close"].to_numpy(dtype=np.float64)
        log_c = np.log(closes)
        r = log_c[-1] - log_c[0]
        vol = float(np.std(np.diff(log_c))) + 1e-9
        z = float(r / (vol * math.sqrt(float(days))))
        return float(max(-1.0, min(1.0, z / 2.0)))
    except Exception:  # noqa: BLE001
        return None


def make_horizon_blend_mom_fn(base_fn, ring_buffer,
                              weights: dict[str, float]):
    """Blend 5d/10d/20d momentum into the mom sub-score.

    `weights` maps {"d5": w5, "d10": w10, "d20": w20}. Per ticker, the
    blended score is the weighted mean of the available horizon z-scores
    (renormalised over horizons that have enough history). Tickers with no
    usable horizon fall back to `base_fn` (which may itself be None).
    """
    horizons = (("d5", 5), ("d10", 10), ("d20", 20))

    def _mom_fn(bundle: DayBundle) -> dict[str, float]:
        base: dict[str, float] = {}
        if base_fn is not None:
            try:
                base = dict(base_fn(bundle) or {})
            except Exception as e:  # noqa: BLE001
                _log.warning("mom_blend_base_failed", err=repr(e),
                             component="overlays")
        out: dict[str, float] = {}
        for t in bundle.ticks.keys():
            num = 0.0
            den = 0.0
            for key, days in horizons:
                w = float(weights.get(key, 0.0) or 0.0)
                if w <= 0:
                    continue
                z = _horizon_z(ring_buffer, t, days)
                if z is None:
                    continue
                num += w * z
                den += w
            if den > 0:
                out[t] = float(max(-1.0, min(1.0, num / den)))
            elif t in base:
                out[t] = base[t]
        return out

    return _mom_fn


def make_mean_reversion_mom_fn(base_fn, ring_buffer, regime_getter,
                               days: int = 10):
    """Replace the mom sub-score with mean reversion when regime==risk_off.

    Momentum is anti-predictive in risk-off regimes, so we return the
    NEGATIVE of the 10d return z-score (clipped to [-1, +1]): beaten-down
    names score high. In any other regime the base mom_fn is used.
    """

    def _mom_fn(bundle: DayBundle) -> dict[str, float]:
        regime = "neutral"
        try:
            regime = regime_getter() or "neutral"
        except Exception:  # noqa: BLE001
            pass
        if regime != "risk_off":
            if base_fn is None:
                return {}
            return dict(base_fn(bundle) or {})
        out: dict[str, float] = {}
        base: dict[str, float] = {}
        for t in bundle.ticks.keys():
            z = _horizon_z(ring_buffer, t, days)
            if z is not None:
                out[t] = float(-z)
        if not out and base_fn is not None:
            # No history at all — fail-soft to the base scorer.
            try:
                base = dict(base_fn(bundle) or {})
            except Exception:  # noqa: BLE001
                base = {}
            return base
        return out

    return _mom_fn


# ---------------------------------------------------------------------------
# News decay
# ---------------------------------------------------------------------------
class NewsDecay:
    """Per-ticker exponential decay of news sentiment over sim days.

    effective(today) = max over recent observations of
        score * 0.5 ** (age_in_sim_days / half_life_days)

    A half-life <= 0 disables decay (caller should bypass the wrapper).
    Observations older than `keep_days` are dropped.
    """

    def __init__(self, half_life_days: float, keep_days: int = 20) -> None:
        self.half_life = float(half_life_days)
        self.keep_days = int(keep_days)
        # ticker -> deque[(sim_date, raw_score)]
        self._hist: dict[str, deque] = {}

    def update(self, sim_date, scores: dict[str, float]) -> None:
        for t, s in scores.items():
            dq = self._hist.get(t)
            if dq is None:
                dq = deque(maxlen=self.keep_days)
                self._hist[t] = dq
            # Idempotent on the same date (replace).
            if dq and dq[-1][0] == sim_date:
                dq[-1] = (sim_date, float(s))
            else:
                dq.append((sim_date, float(s)))

    def effective(self, ticker: str, sim_date) -> Optional[float]:
        dq = self._hist.get(ticker)
        if not dq:
            return None
        # Pick the decayed observation with the largest ABSOLUTE value and
        # keep its sign. (A plain max() would bias negative sentiment toward
        # zero: -0.8 decaying to -0.28 would lose to a fresher -0.1, even
        # though the strong negative signal should dominate.)
        best: Optional[float] = None
        for d, s in dq:
            age = max(0, (sim_date - d).days)
            decayed = s * (0.5 ** (age / self.half_life))
            if best is None or abs(decayed) > abs(best):
                best = decayed
        return best


def make_decayed_news_fn(base_fn, half_life_days: float):
    """Wrap a news_fn with exponential per-ticker decay (sim-day half-life)."""
    tracker = NewsDecay(half_life_days)

    def _news_fn(bundle: DayBundle) -> dict[str, float]:
        raw: dict[str, float] = {}
        if base_fn is not None:
            try:
                raw = dict(base_fn(bundle) or {})
            except Exception as e:  # noqa: BLE001
                _log.warning("news_decay_base_failed", err=repr(e),
                             component="overlays")
        sim_date = bundle.simulated_date
        tracker.update(sim_date, raw)
        tickers = set(raw.keys()) | set(tracker._hist.keys())
        out: dict[str, float] = {}
        for t in tickers:
            eff = tracker.effective(t, sim_date)
            if eff is not None:
                out[t] = float(max(-1.0, min(1.0, eff)))
        return out

    return _news_fn


# ---------------------------------------------------------------------------
# Entry gates (pure helpers — main applies them to BUY deltas)
# ---------------------------------------------------------------------------
def min_hold_allows_exit(age_days: Optional[int], min_hold_days: int,
                         reason: str = "") -> bool:
    """Minimum holding period gate. Returns False (block the exit) when the
    lot/short is younger than `min_hold_days` sim days. Stop-loss and
    settlement-guard exits are always allowed; unknown age -> allow
    (fail-soft)."""
    if min_hold_days <= 0:
        return True
    reason_l = (reason or "").lower()
    if "stop" in reason_l or "settlement" in reason_l:
        return True
    if age_days is None:
        return True
    return age_days >= min_hold_days


def strong_disagreement(rank_score: float, mom_score: float,
                        threshold: float = 0.1) -> bool:
    """True when rank and mom point in OPPOSITE directions and both are
    strong (|score| > threshold). Used to veto BUY entries."""
    if abs(rank_score) <= threshold or abs(mom_score) <= threshold:
        return False
    return (rank_score > 0) != (mom_score > 0)


def passes_entry_timing(score: float, history: list[float],
                        min_history: int = 3) -> bool:
    """Score must be >= its own rolling median. Fail-soft: with fewer than
    `min_history` observations there is no basis to gate -> allow."""
    hist = [h for h in history if h is not None]
    if len(hist) < max(1, int(min_history)):
        return True
    return score >= median(hist)
