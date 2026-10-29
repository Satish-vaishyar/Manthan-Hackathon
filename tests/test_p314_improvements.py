"""Tests for the P3-14 / P3-15 improvements (cash utilization, cap, rl-tuner wiring)."""
from __future__ import annotations

import pytest

from bot.config import load_settings
from bot.portfolio.constructor import TickerScore, build_target_weights, build_trade_deltas
from bot.portfolio.risk import RiskManager


def _settings(monkeypatch) -> "Settings":  # type: ignore[name-defined]
    monkeypatch.setenv("BOT_ACCESS_TOKEN", "dummy")
    monkeypatch.setenv("TEAM_USERNAME", "mn2632")
    return load_settings(env="sandbox")


def test_cash_buffer_reduced(monkeypatch) -> None:
    s = _settings(monkeypatch)
    assert s.risk.cash_buffer == pytest.approx(0.02)


def test_min_notional_raised(monkeypatch) -> None:
    s = _settings(monkeypatch)
    assert s.strategy.min_trade_notional == pytest.approx(75_000.0)


def test_target_cash_utilization_default(monkeypatch) -> None:
    s = _settings(monkeypatch)
    assert s.risk.target_cash_utilization == pytest.approx(0.95)  # raised 2026-10-30


def test_max_cash_per_trade_default(monkeypatch) -> None:
    s = _settings(monkeypatch)
    assert s.risk.max_cash_per_trade == pytest.approx(0.90)


def test_risk_seed_equity_no_false_dd() -> None:
    """RiskManager must not flag drawdown until equity is seeded."""
    rm = RiskManager.__new__(RiskManager)
    from bot.config import RiskSettings
    rm.settings = RiskSettings()
    rm.calendar = None
    rm.dd = None  # type: ignore[assignment]
    rm.__init__(rm.settings, None)  # reset
    # Before seed, peak_equity == 0 -> drawdown == 0 even at zero equity
    assert rm.dd.drawdown == 0.0
    # After seed at 1M, drawdown stays at 0
    rm.seed_equity(1_000_000.0)
    assert rm.dd.drawdown == 0.0
    assert rm.dd.peak_equity == 1_000_000.0
    # Drop to 900k -> 10% drawdown
    rm.update_equity(900_000.0)
    assert rm.dd.drawdown == pytest.approx(0.10)


def test_cash_utilization_inflation(monkeypatch) -> None:
    """When current utilisation < target, weights are scaled up."""
    from bot.portfolio.constructor import inverse_vol_weights
    s = _settings(monkeypatch)
    scores = [
        TickerScore(ticker="HAVN", score=1.0, vol=0.02),
        TickerScore(ticker="PLNX", score=0.8, vol=0.02),
    ]
    # Current utilisation = 50% of equity; target 85% -> scale by 1.7 (capped at 1.5)
    holdings = {"HAVN": int(1_000_000 * 0.5 / 100.0)}  # ~5k shares @ 100 -> 500k
    last_close = {"HAVN": 100.0, "PLNX": 100.0}
    # Baseline (no scale): each capped at max_weight_name=0.15 -> sum=0.30
    pre = inverse_vol_weights(scores, top_k=s.strategy.top_k,
                              max_weight=s.strategy.max_weight_name, equity=1_000_000.0)
    pre_total = sum(pre.values())
    # With scale-up to target 0.85: scale = 0.85/0.5 = 1.7 -> clamped 1.5
    w = build_target_weights(
        scores, s.strategy, equity=1_000_000.0,
        current_holdings=holdings, current_shorts={}, last_close=last_close,
        target_cash_utilization=0.85,
    )
    post_total = sum(w.values())
    # 2026-10-30 semantics: utilization scale-up never exceeds the HARD
    # per-name cap (no concentration). With two names already at the cap,
    # gross stays at 2 * max_weight_name; the invariant is: never decrease,
    # never breach the cap.
    assert post_total >= pre_total, (
        f"scale-up must not shrink gross: pre={pre_total} post={post_total}"
    )
    cap = s.strategy.max_weight_name
    assert all(x <= cap + 1e-9 for x in w.values()), (
        f"hard per-name cap breached: {w}"
    )
    # The 1.5x clamp applies only until the hard per-name cap binds; with
    # both names already at the cap, post == pre == 2 * cap.
    assert abs(post_total - min(pre_total * 1.5, 2 * cap)) < 1e-6


def test_trade_drops_marginal_trades(monkeypatch) -> None:
    """Trades whose notional is below min_worthwhile should be filtered."""
    s = _settings(monkeypatch)
    # Small trade: 100 shares @ 100 = 10k (way below 50% of 75k = 37.5k min)
    deltas = build_trade_deltas(
        current_holdings={"HAVN": 0},
        current_shorts={},
        last_close={"HAVN": 100.0, "PLNX": 100.0},
        target_weights={"HAVN": 0.10},  # 10% of 1M = 100k; qty=1000 shares
        equity=1_000_000.0,
        settings=s.strategy,
        max_orders_remaining=4,
        short_allowed=True,
    )
    # Notional = 1000 * 100 = 100k > 37.5k min_worthwhile, so trade should be kept
    assert any(t.ticker == "HAVN" for t in deltas.trades)

    # Now make the target tiny: 1 share @ 100 = 100 notional
    deltas2 = build_trade_deltas(
        current_holdings={"HAVN": 0},
        current_shorts={},
        last_close={"HAVN": 100.0},
        target_weights={"HAVN": 0.001},
        equity=1_000_000.0,
        settings=s.strategy,
        max_orders_remaining=4,
        short_allowed=True,
    )
    # Should be skipped (rounded to zero or below worthwhile threshold)
    assert all(t.ticker != "HAVN" for t in deltas2.trades) or len(deltas2.trades) == 0


def test_rl_tuner_block_parsed(monkeypatch) -> None:
    s = _settings(monkeypatch)
    assert s.strategy.rl_tuner is not None
    assert s.strategy.rl_tuner.enabled is True
    assert len(s.strategy.rl_tuner.arms) == 5