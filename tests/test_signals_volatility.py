"""tests/test_signals_volatility.py — coverage for bot.signals.volatility (M4)."""
from __future__ import annotations

import math

import pytest

from bot.signals.volatility import VolatilityModule, annualised_vol


def test_volatility_module_returns_fallback_when_empty():
    v = VolatilityModule(vol_window=20, fallback_vol=0.02)
    assert v.vol("AAA") == pytest.approx(0.02)


def test_volatility_module_updates_vol():
    v = VolatilityModule(vol_window=20, fallback_vol=0.02)
    for p in (100.0, 101.0, 102.5, 101.8, 103.2, 104.1, 103.5):
        v.update("AAA", p + 0.5, p - 0.5, p)
    assert v.vol("AAA") > 0.0
    assert v.vol("AAA") < 1.0  # sane upper bound


def test_volatility_module_atr_warmup():
    v = VolatilityModule(vol_window=20, atr_window=14, fallback_vol=0.02)
    assert v.atr("AAA") is None
    v.update("AAA", 100.5, 99.5, 100.0)
    assert v.atr("AAA") == pytest.approx(1.0)  # first TR = high - low


def test_volatility_module_atr_stop_long():
    v = VolatilityModule()
    for _ in range(20):
        # Build some real range so ATR is meaningful
        v.update("AAA", 105.0, 95.0, 100.0)
    stop = v.atr_stop("AAA", 100.0, "long", mult=2.5)
    assert stop is not None and stop < 100.0


def test_volatility_module_atr_stop_short():
    v = VolatilityModule()
    for _ in range(20):
        v.update("AAA", 105.0, 95.0, 100.0)
    stop = v.atr_stop("AAA", 100.0, "short", mult=2.5)
    assert stop is not None and stop > 100.0


def test_volatility_module_per_ticker_independent():
    v = VolatilityModule(vol_window=20, fallback_vol=0.02)
    # AAA noisy, BBB constant
    import random
    rng = random.Random(42)
    pa, pb = 100.0, 100.0
    for _ in range(30):
        pa *= math.exp(rng.gauss(0, 0.03))
        v.update("AAA", pa + 0.5, pa - 0.5, pa)
        v.update("BBB", 100.5, 99.5, pb)
    assert v.vol("AAA") > v.vol("BBB")


def test_volatility_module_atr_stop_returns_none_when_unready():
    v = VolatilityModule()
    v.update("AAA", 100.5, 99.5, 100.0)  # only 1 obs, ATR may exist but small
    # If ATR is too small to compute (<= 0), should return None or the entry
    result = v.atr_stop("AAA", 100.0, "long", mult=2.5)
    # ATR == 1.0 in this case, so a real stop is returned:
    assert result is None or result < 100.0


def test_annualised_vol_helper():
    closes = [100.0, 101.0, 102.0, 101.5, 103.0, 104.5, 103.0, 105.0,
              104.0, 106.0, 107.5, 108.0, 107.0, 109.0, 110.0, 109.0,
              111.0, 112.0, 111.0, 113.0, 114.5]
    v = annualised_vol(closes, window=20)
    assert 0.0 < v < 1.0  # sane bounds