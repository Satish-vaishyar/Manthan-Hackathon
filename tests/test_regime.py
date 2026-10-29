"""tests/test_regime.py — coverage for bot.models.regime (M5)."""
from __future__ import annotations

import math
from datetime import date, datetime, timezone
from typing import List

import pytest

from bot.contracts import DayBundle, PriceTickData, PriceTickEvent
from bot.models.regime import (
    ALL_REGIMES,
    DEFAULT_GROSS_TARGET,
    NEUTRAL,
    RISK_OFF,
    RISK_ON,
    RegimeConfig,
    RegimeDetector,
)


def _bundle(tickers: List[str], closes: dict[str, float], sd: date) -> DayBundle:
    return DayBundle(
        simulated_date=sd,
        ticks={t: PriceTickData(ticker=t, open=p, high=p, low=p, close=p, volume=1000)
               for t, p in closes.items() if t in tickers},
        news=[],
        expected_tickers=set(tickers),
        assembled_at=datetime.now(timezone.utc),
        assembly_latency_ms=0.0,
        complete_by_count=True,
        complete_by_deadline=False,
    )


def test_initial_state_is_neutral_during_warmup():
    from datetime import timedelta
    rd = RegimeDetector(RegimeConfig(min_history_days=20))
    start = date(2024, 5, 1)
    # First 19 days: warmup -> neutral
    for d in range(19):
        b = _bundle(["AAA", "BBB"],
                    {"AAA": 100.0 + d, "BBB": 50.0 + d},
                    start + timedelta(days=d))
        assert rd.update(b) == NEUTRAL
    assert rd.days_observed == 19


def test_risk_on_after_strong_uptrend_with_low_vol():
    from datetime import timedelta
    rd = RegimeDetector(RegimeConfig(min_history_days=10, vol_high=0.30,
                                      trend_up=0.05, vol_low=0.10))
    start = date(2024, 5, 1)
    # 30 days of slow, steady uptrend (log-return ~+0.2%/day, vol ~0)
    px = {"AAA": 100.0, "BBB": 100.0, "CCC": 100.0}
    for d in range(30):
        for t in px:
            px[t] *= 1.002  # +0.2% / day, identical across tickers -> dispersion = 0
        b = _bundle(list(px.keys()), px, start + timedelta(days=d))
        regime = rd.update(b)
    # After 30 days of low-vol uptrend with cumulative return ~ +0.6%*30 = +6%
    assert regime == RISK_ON, f"expected RISK_ON, got {regime}; snap={rd.snapshot()}"


def test_risk_off_on_high_vol():
    from datetime import timedelta
    rd = RegimeDetector(RegimeConfig(min_history_days=10, vol_high=0.20,
                                      vol_low=0.10))
    import random
    rng = random.Random(42)
    start = date(2024, 5, 1)
    px = {"AAA": 100.0, "BBB": 100.0}
    for d in range(30):
        # High-vol daily returns (5% stdev)
        for t in px:
            r = rng.gauss(0, 0.05)
            px[t] *= math.exp(r)
        b = _bundle(list(px.keys()), px, start + timedelta(days=d))
        regime = rd.update(b)
    # 5% daily stdev -> ~79% annualised -> well above vol_high=0.20
    assert regime == RISK_OFF, f"expected RISK_OFF, got {regime}"


def test_risk_off_on_deep_drawdown():
    from datetime import timedelta
    rd = RegimeDetector(RegimeConfig(min_history_days=10, trend_dn=-0.10,
                                      vol_high=0.50, vol_low=0.50))
    start = date(2024, 5, 1)
    px = {"AAA": 100.0, "BBB": 100.0, "CCC": 100.0}
    # 30 days of identical -1% daily drop -> cumulative ~-26%
    for d in range(30):
        for t in px:
            px[t] *= 0.99
        b = _bundle(list(px.keys()), px, start + timedelta(days=d))
        regime = rd.update(b)
    # cumulative log return ~ -30 * log(0.99) = ~+0.30, below -10% threshold
    assert regime == RISK_OFF


def test_neutral_for_choppy_market():
    from datetime import timedelta
    rd = RegimeDetector(RegimeConfig(min_history_days=10, vol_high=0.50,
                                      vol_low=0.05, trend_up=0.20, trend_dn=-0.20))
    import random
    rng = random.Random(7)
    start = date(2024, 5, 1)
    px = {"AAA": 100.0, "BBB": 100.0}
    for d in range(30):
        for t in px:
            r = rng.gauss(0, 0.005)  # 0.5% daily stdev -> ~8% annualised
            px[t] *= math.exp(r)
        b = _bundle(list(px.keys()), px, start + timedelta(days=d))
        regime = rd.update(b)
    # Low vol, no trend -> neutral
    assert regime == NEUTRAL


def test_regime_change_logged_on_transition():
    from datetime import timedelta
    rd = RegimeDetector(RegimeConfig(min_history_days=10, vol_high=0.20,
                                      vol_low=0.10))
    import random
    rng = random.Random(42)
    start = date(2024, 5, 1)
    # First: low-vol uptrend
    px = {"AAA": 100.0, "BBB": 100.0}
    for d in range(30):
        for t in px:
            px[t] *= 1.002
        rd.update(_bundle(list(px.keys()), px, start + timedelta(days=d)))
    on_regime = rd.current
    # Then: high-vol crash
    for d in range(30, 60):
        for t in px:
            r = rng.gauss(-0.02, 0.05)
            px[t] *= math.exp(r)
        rd.update(_bundle(list(px.keys()), px, start + timedelta(days=d)))
    assert rd.current != on_regime
    assert rd.current == RISK_OFF


def test_default_gross_targets():
    # Per docs/04 §4
    assert DEFAULT_GROSS_TARGET[RISK_ON] == 1.00
    assert DEFAULT_GROSS_TARGET[NEUTRAL] == 0.80
    assert DEFAULT_GROSS_TARGET[RISK_OFF] == 0.50


def test_all_regimes_listed():
    assert set(ALL_REGIMES) == {RISK_ON, NEUTRAL, RISK_OFF}


def test_snapshot_has_diagnostics():
    rd = RegimeDetector(RegimeConfig(min_history_days=5))
    for d in range(10):
        b = _bundle(["AAA", "BBB"], {"AAA": 100.0 + d, "BBB": 50.0 - d},
                    date(2024, 5, d + 1))
        rd.update(b)
    snap = rd.snapshot()
    assert snap["days_observed"] == 10
    assert snap["regime"] in ALL_REGIMES
    assert snap["days_in_regime"] >= 1


def test_empty_bundle_returns_current_regime():
    rd = RegimeDetector()
    initial = rd.current
    empty = DayBundle(
        simulated_date=date(2024, 5, 1), ticks={}, news=[],
        expected_tickers=set(), assembled_at=datetime.now(timezone.utc),
        assembly_latency_ms=0.0, complete_by_count=True, complete_by_deadline=False,
    )
    assert rd.update(empty) == initial


def test_constructor_scaling_by_gross_target():
    """Constructor should scale weights by regime's gross_long_target."""
    from bot.config import StrategySettings
    from bot.portfolio.constructor import TickerScore, build_target_weights

    s = StrategySettings(top_k=2, max_weight_name=1.0)
    scores = [
        TickerScore(ticker="AAA", score=1.0, vol=0.02, price=100.0),
        TickerScore(ticker="BBB", score=1.0, vol=0.02, price=100.0),
    ]
    # With risk_off multiplier of 0.5, total weight should be 0.5 not 1.0
    w_risk_off = build_target_weights(
        scores, s, 1_000_000.0,
        regime="risk_off",
        gross_long_target={"risk_on": 1.0, "neutral": 0.8, "risk_off": 0.5},
    )
    w_risk_on = build_target_weights(
        scores, s, 1_000_000.0,
        regime="risk_on",
        gross_long_target={"risk_on": 1.0, "neutral": 0.8, "risk_off": 0.5},
    )
    assert sum(w_risk_off.values()) == pytest.approx(0.5, abs=1e-6)
    assert sum(w_risk_on.values()) == pytest.approx(1.0, abs=1e-6)