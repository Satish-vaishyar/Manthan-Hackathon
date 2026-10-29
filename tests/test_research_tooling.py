"""Tests for the offline research tooling (scripts/).

Covers the pure helpers factored out of:
  - scripts/optimize_weights.py   (regime bucketing)
  - scripts/tune_stops.py         (ATR proxy, stop simulation, FIFO matching)
  - scripts/tune_thresholds.py    (walk-forward split, inverse-vol weights)

All fixtures are small and synthetic — no live server, no run journals.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import optimize_weights as ow  # noqa: E402
import tune_stops as ts  # noqa: E402
import tune_thresholds as tt  # noqa: E402


# ----------------------------------------------------------------------
# regime bucketing
# ----------------------------------------------------------------------
def test_regime_day_buckets_basic() -> None:
    dates = ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]
    rmap = {"2026-01-01": "risk_on", "2026-01-02": "risk_off",
            "2026-01-03": None}  # missing -> neutral; 4th absent -> neutral
    buckets = ow.regime_day_buckets(dates, {k: v for k, v in rmap.items() if v})
    assert buckets["risk_on"] == [0]
    assert buckets["risk_off"] == [1]
    assert buckets["neutral"] == [2, 3]


def test_regime_day_buckets_unknown_label_is_neutral() -> None:
    buckets = ow.regime_day_buckets(["d1"], {"d1": "weird"})
    assert buckets["neutral"] == [0]
    assert buckets["risk_on"] == [] and buckets["risk_off"] == []


def test_optimize_by_regime_falls_back_for_small_buckets() -> None:
    rng = np.random.default_rng(0)
    D, T = 40, 6
    dates = [f"2026-01-{d + 1:02d}" for d in range(D)]
    # 35 risk_on days, 5 neutral (below min) -> neutral falls back to pooled.
    rmap = {d: ("risk_on" if i < 35 else "neutral") for i, d in enumerate(dates)}
    S = {s: rng.normal(size=(D, T)) for s in ow.SIGNALS}
    P = 100 * np.cumprod(1 + rng.normal(0, 0.01, size=(D, T)), axis=0)
    F = ow.forward_returns(P)
    pooled = ow.optimize(S, F, n_boot=50)
    out = ow.optimize_by_regime(dates, S, F, rmap, pooled, n_boot=50, min_days=30)
    assert out["risk_on"]["fallback"] is False
    assert out["neutral"]["fallback"] is True
    assert out["risk_off"]["fallback"] is True
    assert out["neutral"]["estimate"]["weights"] == pooled.weights


# ----------------------------------------------------------------------
# ATR proxy
# ----------------------------------------------------------------------
def test_atr_constant_moves() -> None:
    """Prices stepping by exactly 2/day -> ATR converges to 2."""
    P = np.tile(100 + 2.0 * np.arange(30), (2, 1)).T  # (30, 2)
    atr = ts.atr_from_prices(P, period=14)
    assert np.all(np.isnan(atr[:14]))
    assert atr[14, 0] == pytest.approx(2.0)
    assert atr[-1, 0] == pytest.approx(2.0)


def test_atr_too_short_panel_is_all_nan() -> None:
    P = np.ones((5, 2))
    assert np.isnan(ts.atr_from_prices(P, period=14)).all()


def test_atr_handles_nan_prices() -> None:
    P = 100 + np.arange(20, dtype=float).reshape(20, 1)
    P[10, 0] = np.nan
    atr = ts.atr_from_prices(P, period=5)
    assert np.isfinite(atr[-1, 0])  # carries previous ATR through the gap


# ----------------------------------------------------------------------
# stop simulation
# ----------------------------------------------------------------------
def test_simulate_stop_triggers() -> None:
    closes = np.array([100.0, 99.0, 90.0, 95.0])   # entry day 0, exit day 3
    atrs = np.array([4.0] * 4)                      # stop = 100 - 2*4 = 92
    px, idx, stopped = ts.simulate_stop_exit(closes, atrs, 0, 3, 100.0, 2.0)
    assert stopped and idx == 2 and px == pytest.approx(92.0)


def test_simulate_stop_never_triggers() -> None:
    closes = np.array([100.0, 99.0, 98.0, 101.0])
    atrs = np.array([1.0] * 4)                      # stop = 97
    px, idx, stopped = ts.simulate_stop_exit(closes, atrs, 0, 3, 100.0, 3.0)
    assert not stopped and idx == 3 and np.isnan(px)


def test_simulate_stop_nan_atr_no_trade() -> None:
    px, idx, stopped = ts.simulate_stop_exit(
        np.array([1.0, 1.0]), np.array([np.nan, np.nan]), 0, 1, 1.0, 3.0)
    assert not stopped and np.isnan(px)


# ----------------------------------------------------------------------
# FIFO round-trip matching
# ----------------------------------------------------------------------
def _order(action, qty, px, date, charges=20.0):
    return {"status": "FILLED", "action": action, "ticker": "AAA",
            "quantity": qty, "execution_price": px, "simulated_date": date,
            "executed_at": date + "T00:00:00Z",
            "total_charges_and_taxes": charges}


def test_match_round_trips_simple() -> None:
    orders = [_order("BUY", 10, 100.0, "2026-01-01"),
              _order("SELL", 10, 110.0, "2026-01-05")]
    trips = ts.match_round_trips(orders)
    assert len(trips) == 1
    t = trips[0]
    assert t.pnl == pytest.approx(10 * 10 - 40.0)  # gross 100 - buy/sell charges
    assert t.hold_days == 4


def test_match_round_trips_partial_and_rejected_ignored() -> None:
    orders = [_order("BUY", 10, 100.0, "2026-01-01"),
              _order("SELL", 4, 110.0, "2026-01-02"),
              _order("SELL", 6, 90.0, "2026-01-03"),
              {"status": "REJECTED", "action": "BUY", "ticker": "AAA",
               "quantity": 5, "execution_price": 100.0,
               "simulated_date": "2026-01-04"}]
    trips = ts.match_round_trips(orders)
    assert len(trips) == 2
    assert trips[0].qty == 4 and trips[1].qty == 6
    # total gross = 4*10 + 6*(-10) = -20; charges = 20/sell leg + 20/buy lot
    # prorated by filled fraction (4/10 then 6/6 of the remaining lot).
    assert sum(t.pnl for t in trips) == pytest.approx(-20 - (20 * 2 + 20 * 0.4 + 20 * 1.0))


def test_match_round_trips_shorts_ignored() -> None:
    orders = [_order("SELL", 5, 100.0, "2026-01-01"),
              {"status": "FILLED", "action": "COVER", "ticker": "AAA",
               "quantity": 5, "execution_price": 90.0,
               "simulated_date": "2026-01-02", "executed_at": "2026-01-02"}]
    assert ts.match_round_trips(orders) == []


def test_hold_band() -> None:
    assert ts.hold_band(0) == "<3d scalp"
    assert ts.hold_band(2) == "<3d scalp"
    assert ts.hold_band(5) == "3-7d"
    assert ts.hold_band(15) == "8-30d"
    assert ts.hold_band(100) == ">30d"


# ----------------------------------------------------------------------
# walk-forward + inverse-vol
# ----------------------------------------------------------------------
def test_walk_forward_split_contiguous_and_complete() -> None:
    fit, val = tt.walk_forward_split(100, 0.6)
    assert len(fit) == 60 and len(val) == 40
    assert fit[-1] + 1 == val[0]
    assert np.array_equal(np.concatenate([fit, val]), np.arange(100))


def test_walk_forward_split_small_rounding() -> None:
    fit, val = tt.walk_forward_split(3, 0.6)
    assert len(fit) == 2 and len(val) == 1
    with pytest.raises(ValueError):
        tt.walk_forward_split(1, 0.6)
    with pytest.raises(ValueError):
        tt.walk_forward_split(10, 1.5)


def test_rolling_inv_vol_prefers_low_vol() -> None:
    rng = np.random.default_rng(1)
    rets = np.column_stack([rng.normal(0, 0.01, 40),   # low vol
                            rng.normal(0, 0.05, 40)])  # high vol
    w = tt.rolling_inv_vol(rets, d=40, win=20)
    assert w.sum() == pytest.approx(1.0)
    assert w[0] > 0.7 > w[1]


def test_rolling_inv_vol_insufficient_history() -> None:
    assert not tt.rolling_inv_vol(np.zeros((3, 2)), d=3).any()


def test_simulate_portfolio_no_signal_no_trades() -> None:
    D, T = 30, 4
    ens = np.full((D, T), 0.1)  # below any threshold
    P = np.tile(100.0, (D, T))
    r = tt.simulate_portfolio(ens, P, 0.5, 0.05, 4, np.arange(D - 1))
    assert r["net_return"] == pytest.approx(0.0)
    assert r["n_trades"] == 0


def test_simulate_portfolio_gains_when_prices_rise() -> None:
    D, T = 40, 3
    ens = np.full((D, T), 0.9)
    P = np.tile((100 * np.cumprod(np.full(D, 1.01)))[:, None], (1, T))
    r = tt.simulate_portfolio(ens, P, 0.5, 0.05, 2, np.arange(20, D - 1))
    assert r["net_return"] > 0.10  # ~1%/day compounding for 19 days
    assert r["n_active_days"] > 0
