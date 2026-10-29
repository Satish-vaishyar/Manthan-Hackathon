"""Tests for the per-ticker RL bandit (P3-22) + take-profit knob (P3-23)."""
from __future__ import annotations

import pytest

from bot.config import Settings, load_settings
from bot.portfolio.rl_tuner import TickerBandit, build_bandit_from_settings


def _bandit(**kw) -> TickerBandit:
    kw.setdefault("ema_alpha", 0.5)
    kw.setdefault("min_obs", 2)
    return TickerBandit(**kw)


def test_no_opinion_before_min_obs() -> None:
    b = _bandit()
    b.observe("AAA", -50_000)
    assert b.multiplier("AAA") == 1.0  # only 1 obs < min_obs=2


def test_loser_penalised_toward_floor() -> None:
    b = _bandit()
    for _ in range(6):
        b.observe("AAA", -10_000)
    mult = b.multiplier("AAA")
    assert mult == pytest.approx(0.5, abs=0.05)  # z -> -1 -> 1 - 0.5 = floor


def test_winner_reward_capped() -> None:
    b = _bandit()
    for _ in range(6):
        b.observe("AAA", +10_000)
    mult = b.multiplier("AAA")
    assert mult == pytest.approx(1.15, abs=0.05)  # capped, never more


def test_multiplier_bounded_mixed() -> None:
    b = _bandit()
    for pnl in (+10_000, -8_000, +12_000, -9_000):
        b.observe("AAA", pnl)
    mult = b.multiplier("AAA")
    assert 0.5 <= mult <= 1.15  # always bounded — no gambling either way


def test_disabled_bandit_is_neutral() -> None:
    b = _bandit(enabled=False)
    for _ in range(5):
        b.observe("AAA", -100_000)
    assert b.multiplier("AAA") == 1.0


def test_config_block_parsed(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("BOT_ACCESS_TOKEN", "x")
    monkeypatch.setenv("TEAM_USERNAME", "t")
    s = load_settings("configs/default.yaml")
    tb = s.strategy.ticker_bandit
    assert tb is not None and tb.enabled is True
    assert tb.floor == pytest.approx(0.5)
    assert tb.cap == pytest.approx(1.15)
    bandit = build_bandit_from_settings(s.strategy)
    assert bandit is not None and bandit.enabled


def test_take_profit_knob_default_off() -> None:
    s = Settings(env="sandbox", server_ip="127.0.0.1", team_username="t",
                 bot_access_token="x")
    assert s.strategy.take_profit_pct == 0.0  # code default; YAML sets 0.10


def test_take_profit_knob_from_yaml() -> None:
    s = load_settings("configs/default.yaml")
    assert s.strategy.take_profit_pct == pytest.approx(0.10)
