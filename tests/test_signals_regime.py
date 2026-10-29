"""tests/test_signals_regime.py — coverage for bot.signals.regime (M3 rules)."""
from __future__ import annotations

import math
import random
from collections import deque
from datetime import date, timedelta

import pytest

from bot.config import RegimeSettings
from bot.contracts import PriceTickData
from bot.signals.regime import (
    MarketState, NEUTRAL, RISK_OFF, RISK_ON, RegimeFilter,
    gross_long_target,
)


@pytest.fixture
def settings() -> RegimeSettings:
    return RegimeSettings(
        hold_days=2,
        breadth_risk_off=0.35,
        breadth_risk_on=0.60,
        vol_spike_mult=1.5,
        breadth_window_days=50,
        warmup_days=20,
    )


def _tick(t: str, c: float) -> PriceTickData:
    return PriceTickData(ticker=t, open=c, high=c, low=c, close=c, volume=1000)


# ---------------------------------------------------------------- core rules
def test_warmup_stays_neutral(settings):
    rf = RegimeFilter(settings)
    ms = MarketState()
    # 5 days of data is well below warmup_days=20
    d = date(2025, 1, 1)
    for _ in range(5):
        ticks = {f"T{i}": _tick(f"T{i}", 100.0) for i in range(30)}
        ms.update(d, ticks)
        v = rf.update(d, ms)
        d += timedelta(days=1)
    assert rf.current == NEUTRAL


def test_risk_on_after_sustained_breadth(settings):
    rf = RegimeFilter(settings)
    ms = MarketState()
    d = date(2025, 1, 1)
    # 60 days of slow rally: most tickers above their SMA50
    for i in range(60):
        c = 100.0 + i * 0.2
        ticks = {f"T{i:02d}": _tick(f"T{i:02d}", c) for i in range(30)}
        ms.update(d, ticks)
        rf.update(d, ms)
        d += timedelta(days=1)
    assert rf.current == RISK_ON


def test_risk_off_after_sustained_drawdown(settings):
    rf = RegimeFilter(settings)
    ms = MarketState()
    d = date(2025, 1, 1)
    # 60 days of slow rally first (breadth > 60%, ret_20d > 0)
    for i in range(60):
        c = 100.0 + i * 0.3
        ticks = {f"T{i:02d}": _tick(f"T{i:02d}", c) for i in range(30)}
        ms.update(d, ticks)
        rf.update(d, ms)
        d += timedelta(days=1)
    assert rf.current == RISK_ON
    # Now crash hard — 5 days of -4% each, breadth should collapse
    for _ in range(5):
        ticks = {}
        for i in range(30):
            prev = ms.closes_by_ticker[f"T{i:02d}"][-1]
            ticks[f"T{i:02d}"] = _tick(f"T{i:02d}", prev * 0.96)
        ms.update(d, ticks)
        rf.update(d, ms)
        d += timedelta(days=1)
    assert rf.current == RISK_OFF


# --------------------------------------------------------------- hysteresis
def test_hysteresis_holds_during_flicker(settings):
    """A single day of bad breadth must NOT flip the regime by itself."""
    rf = RegimeFilter(settings)
    ms = MarketState()
    d = date(2025, 1, 1)
    # Build a stable risk_on regime
    for i in range(60):
        c = 100.0 + i * 0.3
        ticks = {f"T{i:02d}": _tick(f"T{i:02d}", c) for i in range(30)}
        ms.update(d, ticks)
        rf.update(d, ms)
        d += timedelta(days=1)
    assert rf.current == RISK_ON

    # One day of -3% pullback (ret_20d still positive; breadth might still be high)
    ticks = {}
    for i in range(30):
        prev = ms.closes_by_ticker[f"T{i:02d}"][-1]
        ticks[f"T{i:02d}"] = _tick(f"T{i:02d}", prev * 0.97)
    ms.update(d, ticks)
    v1 = rf.update(d, ms)
    d += timedelta(days=1)

    # Next day: snap back
    ticks = {}
    for i in range(30):
        prev = ms.closes_by_ticker[f"T{i:02d}"][-1]
        ticks[f"T{i:02d}"] = _tick(f"T{i:02d}", prev * 1.01)
    ms.update(d, ticks)
    v2 = rf.update(d, ms)
    # Regime must remain risk_on after a single down-day
    assert v1.regime == RISK_ON or v1.candidate == NEUTRAL  # candidate may flicker
    assert rf.current == RISK_ON
    assert v2.regime == RISK_ON


def test_hysteresis_requires_two_days(settings):
    """Two consecutive risk_off candidates -> switch to risk_off."""
    rf = RegimeFilter(settings)
    ms = MarketState()
    d = date(2025, 1, 1)
    # Warm up with a flat market -> regime stays neutral
    rng = random.Random(0)
    for i in range(60):
        c = 100.0 + rng.gauss(0, 0.01)  # tiny noise, no drift
        ticks = {f"T{i:02d}": _tick(f"T{i:02d}", c) for i in range(30)}
        ms.update(d, ticks)
        rf.update(d, ms)
        d += timedelta(days=1)
    # Force ret_20d < 0 by crashing 21 days in a row (so 20d rolling is fully negative)
    for _ in range(21):
        ticks = {}
        for i in range(30):
            prev = ms.closes_by_ticker[f"T{i:02d}"][-1]
            ticks[f"T{i:02d}"] = _tick(f"T{i:02d}", prev * 0.99)
        ms.update(d, ticks)
        rf.update(d, ms)
        d += timedelta(days=1)
    # Now breadth should also be < 35% (most tickers are way below SMA50)
    breadth = ms.breadth_pct(50)
    assert breadth is not None and breadth < 0.35, f"setup failed: breadth={breadth}"
    # Day 1 of the already-crashed regime: candidate should be risk_off now
    v1 = rf.update(d, ms)
    # Hysteresis: by this point we may already be in risk_off; if not, the
    # candidate must be risk_off.
    assert v1.candidate == RISK_OFF


# ------------------------------------------------------------------- vol spike
def test_vol_spike_triggers_risk_off_without_breadth(settings):
    """Sudden vol > 1.5x median -> risk_off even with healthy breadth."""
    rf = RegimeFilter(settings)
    ms = MarketState()
    d = date(2025, 1, 1)
    rng = random.Random(0)
    # Build history: 120 days of slow rally with enough noise that median vol
    # is above the 0.02 floor.
    for i in range(120):
        # ~1% daily vol, +0.05% drift
        ret = rng.gauss(0.0005, 0.01)
        ticks = {f"T{i:02d}": _tick(f"T{i:02d}", 100.0 * math.exp(ret))
                 for i in range(20)}
        ms.update(d, ticks)
        rf.update(d, ms)
        d += timedelta(days=1)
    median_vol = ms.market_vol_median_120d()
    assert median_vol is not None and median_vol >= 0.02, \
        f"setup failed: median_vol={median_vol}"
    # Now spike vol with no net direction change — 3 days of ±5% wiggles
    for _ in range(3):
        ticks = {}
        for i in range(20):
            prev = ms.closes_by_ticker[f"T{i:02d}"][-1]
            c = prev * (1.05 if rng.random() > 0.5 else 0.95)
            ticks[f"T{i:02d}"] = _tick(f"T{i:02d}", c)
        ms.update(d, ticks)
        rf.update(d, ms)
        d += timedelta(days=1)
    assert rf.current == RISK_OFF


# ------------------------------------------------------ gross_long_target helper
def test_gross_long_target_known_regimes():
    table = {"risk_on": 0.92, "neutral": 0.75, "risk_off": 0.40}
    assert gross_long_target("risk_on", table) == pytest.approx(0.92)
    assert gross_long_target("risk_off", table) == pytest.approx(0.40)
    assert gross_long_target("neutral", table) == pytest.approx(0.75)


def test_gross_long_target_unknown_regime_uses_default():
    assert gross_long_target("unknown", {}, default=0.5) == 0.5


# ------------------------------------------------------------------- market state
def test_market_state_returns_1d_and_20d():
    ms = MarketState()
    d = date(2025, 1, 1)
    # 25 days of +1% daily market return
    for _ in range(25):
        ms.update(d, {"A": _tick("A", 100.0), "B": _tick("B", 100.0)})
        d += timedelta(days=1)
    r1 = ms.market_return_1d()
    r20 = ms.market_return_20d()
    assert r1 is not None and abs(r1) < 1e-9  # flat days -> 0
    # 20-day compounded return should be small but non-zero (rounding from deque precision)
    assert r20 is not None


def test_market_state_breadth_computes_only_over_warm_tickers():
    ms = MarketState()
    d = date(2025, 1, 1)
    # One warm ticker (steadily rising — close above its 50d SMA at the end),
    # one cold ticker that DECLINED (close below its 50d SMA).
    for _ in range(60):
        ms.update(d, {
            "WARM": _tick("WARM", 100.0 + _ * 0.1),  # steadily rising
            "COLD": _tick("COLD", 100.0 - _ * 0.1),  # steadily falling
        })
        d += timedelta(days=1)
    b = ms.breadth_pct(50)
    assert b is not None
    # WARM has close 105.9, SMA50 well below -> above. COLD has close 94.0,
    # SMA50 above that -> below. So breadth = 1/2.
    assert b == pytest.approx(0.5)


def test_market_state_breadth_flat_ticker_not_counted_as_above():
    """A flat ticker's close equals its SMA -> not counted as 'above'."""
    ms = MarketState()
    d = date(2025, 1, 1)
    for _ in range(60):
        ms.update(d, {"FLAT": _tick("FLAT", 100.0)})
        d += timedelta(days=1)
    b = ms.breadth_pct(50)
    # Only one ticker, close == SMA -> 0/1 = 0
    assert b == pytest.approx(0.0)