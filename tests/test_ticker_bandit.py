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
    # Disabled in YAML: superseded by trailing_sl (P3-24).
    assert s.strategy.take_profit_pct == pytest.approx(0.0)


def test_trailing_sl_block_parsed() -> None:
    s = load_settings("configs/default.yaml")
    tsl = s.strategy.trailing_sl
    assert tsl is not None and tsl.enabled is True
    assert tsl.initial_pct == pytest.approx(0.10)
    assert tsl.trail_pct == pytest.approx(0.05)


def test_trailing_sl_ratchet_math() -> None:
    """stop = max(entry*(1-init), peak*(1-trail)) with trail above entry only."""
    entry, init, trail = 100.0, 0.10, 0.05

    def _stop(peak: float) -> float:
        s = entry * (1 - init)
        if peak > entry:
            s = max(s, peak * (1 - trail))
        return s

    # At entry: stop is the initial -10% (trail leg must NOT tighten it).
    assert _stop(entry) == pytest.approx(90.0)
    # Peak 130: trail stop = 123.5 -> ratcheted up, locks +23.5%.
    assert _stop(130.0) == pytest.approx(123.5)
    # Peak 105 (just above entry): trail stop 99.75 still below initial? No:
    # 105*0.95 = 99.75 > 90 -> ratchets up but stays under entry (loss < 10%).
    assert _stop(105.0) == pytest.approx(99.75)
    # Once ratcheted, a lower peak never lowers the stop.
    ratcheted = _stop(130.0)
    assert max(ratcheted, _stop(110.0)) == ratcheted
