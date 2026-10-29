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


def test_veto_after_consistent_losses() -> None:
    import datetime as dt
    b = _bandit()
    day = dt.date(2026, 1, 1)
    for i in range(6):
        b.observe("AAA", -10_000, day=day + dt.timedelta(days=i))
    assert b.is_vetoed("AAA", today=day + dt.timedelta(days=6)) is True
    # Not enough history -> no veto.
    assert b.is_vetoed("BBB", today=day) is False


def test_veto_self_heals_with_quiet_days() -> None:
    import datetime as dt
    b = _bandit(day_decay=0.5)  # fast decay for the test
    day = dt.date(2026, 1, 1)
    for i in range(6):
        b.observe("AAA", -10_000, day=day + dt.timedelta(days=i))
    # Vetoed on the last fill day (evidence fresh).
    assert b.is_vetoed("AAA", today=day + dt.timedelta(days=5)) is True
    # Quiet days decay the evidence (0.5^9 ~ 0.002): the veto lifts, unlike
    # a static blacklist — the ticker re-earns eligibility by going quiet.
    assert b.is_vetoed("AAA", today=day + dt.timedelta(days=14)) is False


def test_veto_not_triggered_by_winners() -> None:
    import datetime as dt
    b = _bandit()
    day = dt.date(2026, 1, 1)
    for i in range(8):
        b.observe("AAA", +5_000, day=day + dt.timedelta(days=i))
    assert b.is_vetoed("AAA", today=day + dt.timedelta(days=8)) is False


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
    # Disabled after the exit tournament (runs/_exit_tournament.md): every
    # mechanical exit overlay lost money vs native score-decay exits.
    assert tsl is not None and tsl.enabled is False
    assert tsl.initial_pct == pytest.approx(0.10)
    assert tsl.breakeven_trigger_pct == pytest.approx(0.03)
    assert tsl.lock_start_pct == pytest.approx(0.05)
    assert tsl.first_lock_pct == pytest.approx(0.03)
    assert tsl.step_pct == pytest.approx(0.02)


def test_trailing_sl_stepped_math() -> None:
    """User spec: -10% initial; +3% gain -> breakeven; +5% -> +3%; +2% steps."""
    import math

    def _stop(entry: float, peak: float) -> float:
        gain = (peak - entry) / entry
        if gain >= 0.05:
            locked = 0.03 + 0.02 * math.floor((gain - 0.05) / 0.02)
            return entry * (1 + locked)
        if gain >= 0.03:
            return entry
        return entry * 0.90

    e = 100.0
    assert _stop(e, 100.0) == pytest.approx(90.0)   # at entry: -10%
    assert _stop(e, 102.9) == pytest.approx(90.0)   # +2.9%: still -10%
    assert _stop(e, 103.0) == pytest.approx(100.0)  # +3%: breakeven
    assert _stop(e, 104.9) == pytest.approx(100.0)  # +4.9%: still breakeven
    assert _stop(e, 105.0) == pytest.approx(103.0)  # +5%: lock +3%
    assert _stop(e, 106.9) == pytest.approx(103.0)  # +6.9%: still +3%
    assert _stop(e, 107.0) == pytest.approx(105.0)  # +7%: lock +5%
    assert _stop(e, 130.0) == pytest.approx(127.0)  # +30%: lock +27%
    # Ratchet: peak stored, stop computed off peak — pullback doesn't lower it.
    assert _stop(e, 130.0) > _stop(e, 110.0)
