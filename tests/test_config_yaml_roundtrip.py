"""Regression: ensure configs/default.yaml values actually reach Settings.

Catches the bug where StrategySettings and SignalsSettings silently fell back
to dataclass defaults because `weights` was popped from the dict before being
passed to the constructor.
"""
from __future__ import annotations

import pytest

from bot.config import load_settings


def _make_settings(monkeypatch, env: str = "sandbox") -> "Settings":  # type: ignore[name-defined]
    monkeypatch.setenv("BOT_ACCESS_TOKEN", "dummy")
    monkeypatch.setenv("TEAM_USERNAME", "mn2632")
    return load_settings(env=env)


def test_strategy_weights_from_yaml(monkeypatch) -> None:
    s = _make_settings(monkeypatch)
    # Default.yaml fine-tune (2026-10-29) sets these.
    assert s.strategy.weights == {"rank": 0.55, "news": 0.20, "mom": 0.25}
    assert s.strategy.top_k == 8  # raised 2026-10-30: 6 under-filled the 10-name book
    assert s.strategy.event_long_threshold == pytest.approx(0.75)
    assert s.strategy.event_short_threshold == pytest.approx(-0.75)
    assert s.strategy.rebalance_threshold == pytest.approx(0.05)
    assert s.strategy.exclude_tickers == ["BLUX"]


def test_risk_settings_from_yaml(monkeypatch) -> None:
    s = _make_settings(monkeypatch)
    assert s.risk.stop_atr_mult == pytest.approx(3.5)
    assert s.risk.dd_soft == pytest.approx(0.12)
    assert s.risk.max_consecutive_loss_pause == 0  # disabled by default


def test_excluded_ticker_dropped_from_target_weights(monkeypatch) -> None:
    """BLUX is on the blacklist and must not appear in the target weights."""
    from bot.portfolio.constructor import TickerScore, build_target_weights
    s = _make_settings(monkeypatch)
    scores = [
        TickerScore(ticker="BLUX", score=1.0, vol=0.02),
        TickerScore(ticker="HAVN", score=0.9, vol=0.02),
        TickerScore(ticker="PLNX", score=0.8, vol=0.02),
    ]
    w = build_target_weights(scores, s.strategy, equity=1_000_000)
    assert "BLUX" not in w
    # HAVN and PLNX should both be picked up.
    assert set(w.keys()) == {"HAVN", "PLNX"}