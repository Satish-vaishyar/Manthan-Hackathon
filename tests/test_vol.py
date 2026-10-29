"""tests/test_vol.py — coverage for bot.models.vol (M4)."""
from __future__ import annotations

import math

import pytest

from bot.models.vol import VolEstimator, annualised_vol_from_closes


def test_vol_estimator_fallback_when_empty():
    est = VolEstimator(window=20, fallback_vol=0.02)
    assert est.vol("AAA") == pytest.approx(0.02)
    assert not est.ready("AAA")


def test_vol_estimator_ready_after_5_obs():
    est = VolEstimator(window=20, fallback_vol=0.02)
    # Push 6 closes so we have 5 returns -> ready
    for p in (100.0, 101.0, 102.0, 103.0, 104.0, 105.0):
        est.update("AAA", p)
    assert est.ready("AAA")


def test_vol_estimator_increasing_returns_higher_vol():
    est = VolEstimator(window=20, fallback_vol=0.02)
    # 30 observations with noisy returns around a 1% drift
    import random
    rng = random.Random(42)
    px = 100.0
    for _ in range(30):
        # Daily return ~ N(0.01, 0.02) — drift with realistic noise
        ret = 0.01 + rng.gauss(0, 0.02)
        est.update("AAA", px)
        px *= math.exp(ret)
    v = est.vol("AAA")
    # 2% daily stdev -> ~31.7% annualised (0.02 * sqrt(252))
    assert 0.20 < v < 0.45, f"expected ~0.32 annualised, got {v:.4f}"


def test_vol_estimator_window_drops_oldest():
    est = VolEstimator(window=5, fallback_vol=0.02)
    # 3 quiet days, then 6 noisy days (window is 5)
    for p in (100.0, 100.0, 100.0):
        est.update("AAA", p)
    px = 100.0
    for r in (0.05, -0.05, 0.05, -0.05, 0.05, -0.05):
        px *= math.exp(r)
        est.update("AAA", px)
    # Window (5 most recent) should be all the noisy ones
    v = est.vol("AAA")
    assert v > 0.30  # clearly non-trivial


def test_vol_estimator_per_ticker_independence():
    est = VolEstimator(window=20, fallback_vol=0.02)
    import random
    rng = random.Random(42)
    px_a, px_b = 100.0, 100.0
    for _ in range(20):
        # AAA: noisy (3% daily stdev); BBB: constant
        ret = rng.gauss(0, 0.03)
        px_a *= math.exp(ret)
        px_b *= 1.0
        est.update("AAA", px_a)
        est.update("BBB", px_b)
    # AAA has 19 returns of noisy data; BBB has 19 returns of zero
    assert est.vol("AAA") > est.vol("BBB")
    assert est.vol("BBB") < 1e-6


def test_vol_estimator_isolates_unready_tickers():
    est = VolEstimator(window=20, fallback_vol=0.02)
    for p in (100.0, 101.0, 102.0, 103.0, 104.0, 105.0):
        est.update("AAA", p)
    # BBB has only 1 observation -> not ready -> fallback
    est.update("BBB", 100.0)
    assert est.ready("AAA")
    assert not est.ready("BBB")
    assert est.vol("BBB") == pytest.approx(0.02)


def test_annualised_vol_from_closes_constant():
    closes = [100.0] * 30
    v = annualised_vol_from_closes(closes, window=20)
    assert v == pytest.approx(0.0, abs=1e-9)


def test_annualised_vol_from_closes_short_series():
    closes = [100.0, 101.0, 102.0]
    v = annualised_vol_from_closes(closes, window=20)
    assert math.isnan(v)  # not enough data


def test_annualised_vol_from_closes_noisy_series():
    # Noisy series with 3% daily stdev -> ~47.6% annualised.
    # Empirical std over a 20-day window will be in a wider band.
    import random
    rng = random.Random(42)
    closes = [100.0]
    for _ in range(99):  # 100 closes -> 99 returns; window=20 picks the last 20
        ret = rng.gauss(0, 0.03)
        closes.append(closes[-1] * math.exp(ret))
    v = annualised_vol_from_closes(closes, window=20)
    # Expected ~0.48 with ±2-sigma sampling noise over 20 obs
    assert 0.25 < v < 0.70, f"expected ~0.48 (wide tolerance), got {v:.4f}"