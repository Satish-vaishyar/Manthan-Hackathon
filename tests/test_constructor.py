"""tests/test_constructor.py — inverse-vol sizing, caps, threshold, min notional."""
from __future__ import annotations

from datetime import date

import pytest

from bot.config import StrategySettings
from bot.portfolio.constructor import (
    TickerScore,
    build_target_weights,
    build_trade_deltas,
    inverse_vol_weights,
)


def test_inverse_vol_weights_normalises_and_caps():
    scores = [
        TickerScore(ticker="AAA", score=1.0, vol=0.01),
        TickerScore(ticker="BBB", score=0.5, vol=0.02),
        TickerScore(ticker="CCC", score=0.1, vol=0.05),
    ]
    w = inverse_vol_weights(scores, top_k=3, max_weight=0.5, equity=1_000_000.0)
    assert abs(sum(w.values()) - 1.0) < 1e-6
    assert max(w.values()) <= 0.5 + 1e-9


def test_inverse_vol_top_k():
    scores = [TickerScore(ticker=f"T{i}", score=1.0 - 0.1 * i, vol=0.02) for i in range(5)]
    w = inverse_vol_weights(scores, top_k=3, max_weight=1.0, equity=1_000_000.0)
    assert set(w.keys()) == {"T0", "T1", "T2"}


def test_build_target_weights_with_sector_caps():
    s = StrategySettings(top_k=4, max_weight_name=1.0)
    scores = [
        TickerScore(ticker="AAA", score=1.0, vol=0.02, sector="Tech"),
        TickerScore(ticker="BBB", score=1.0, vol=0.02, sector="Tech"),
        TickerScore(ticker="CCC", score=1.0, vol=0.02, sector="Finance"),
        TickerScore(ticker="DDD", score=1.0, vol=0.02, sector="Finance"),
    ]
    w = build_target_weights(
        scores, s, equity=1_000_000.0,
        sector_caps={"Tech": 0.3, "Finance": 0.99},
        sector_of={t.ticker: t.sector for t in scores},
    )
    # Tech weight should be <= 30%
    tech_w = sum(w.get(t, 0.0) for t in ("AAA", "BBB"))
    assert tech_w <= 0.3 + 1e-6


def test_build_trade_deltas_respects_threshold():
    s = StrategySettings(top_k=2, max_weight_name=0.5, rebalance_threshold=0.05,
                          min_trade_notional=10_000.0)
    last_close = {"AAA": 100.0, "BBB": 200.0}
    target = {"AAA": 0.5, "BBB": 0.5}
    # Current holdings match target exactly -> no trades
    holdings = {"AAA": 5000, "BBB": 2500}  # both = 50% of 1M
    out = build_trade_deltas(
        holdings, {}, last_close, target, 1_000_000.0,
        settings=s, max_orders_remaining=4,
    )
    assert out.trades == []


def test_build_trade_deltas_generates_buy():
    s = StrategySettings(top_k=2, max_weight_name=0.5, rebalance_threshold=0.02,
                          min_trade_notional=10_000.0)
    last_close = {"AAA": 100.0}
    target = {"AAA": 0.5}
    holdings: dict = {}
    out = build_trade_deltas(
        holdings, {}, last_close, target, 1_000_000.0,
        settings=s, max_orders_remaining=4,
    )
    assert len(out.trades) == 1
    t = out.trades[0]
    assert t.action == "BUY"
    assert t.notional >= s.min_trade_notional


def test_build_trade_deltas_exits_position_not_in_target():
    s = StrategySettings(top_k=2, max_weight_name=0.5, rebalance_threshold=0.01,
                          min_trade_notional=10_000.0)
    last_close = {"AAA": 100.0, "BBB": 100.0}
    holdings = {"AAA": 1000, "BBB": 5000}  # BBB big position, no longer in target
    target = {"AAA": 0.5}
    out = build_trade_deltas(
        holdings, {}, last_close, target, 1_000_000.0,
        settings=s, max_orders_remaining=4,
    )
    actions = {t.ticker: t.action for t in out.trades}
    assert actions.get("BBB") == "SELL"


def test_build_trade_deltas_respects_max_orders():
    s = StrategySettings(top_k=5, max_weight_name=0.2, rebalance_threshold=0.01,
                          min_trade_notional=10_000.0)
    last_close = {f"T{i}": 100.0 for i in range(5)}
    target = {f"T{i}": 0.2 for i in range(5)}
    out = build_trade_deltas(
        {}, {}, last_close, target, 1_000_000.0,
        settings=s, max_orders_remaining=2,
    )
    assert len(out.trades) <= 2