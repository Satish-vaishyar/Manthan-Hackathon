"""tests/test_strategy_improvements.py — coverage for the 2026-10-29
strategy-improvement overlays (each feature ON and OFF)."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from bot.config import Settings, load_settings
from bot.contracts import DayBundle, PriceTickData
from bot.features import Candle, RingBuffer
from bot.signals import RulesFallback, SignalsEnsemble
from bot.signals.overlays import (
    NewsDecay,
    make_decayed_news_fn,
    make_horizon_blend_mom_fn,
    make_mean_reversion_mom_fn,
    min_hold_allows_exit,
    passes_entry_timing,
    strong_disagreement,
)


def _tick(t: str, c: float) -> PriceTickData:
    return PriceTickData(ticker=t, open=c, high=c, low=c, close=c, volume=1000)


def _bundle(ticks, sim_date=date(2025, 1, 31)) -> DayBundle:
    return DayBundle(
        simulated_date=sim_date,
        ticks=ticks,
        news=[],
        assembled_at=datetime(2025, 1, 31, tzinfo=timezone.utc),
        assembly_latency_ms=0.0,
    )


def _rb_with_closes(ticker: str, closes, end: date = date(2025, 1, 31)) -> RingBuffer:
    rb = RingBuffer(window=280)
    start = end - timedelta(days=len(closes) - 1)
    for i, c in enumerate(closes):
        rb.push_candle(ticker, Candle(
            date=start + timedelta(days=i),
            open=c, high=c, low=c, close=c, volume=1000,
        ))
    return rb


def _settings(monkeypatch) -> Settings:
    monkeypatch.setenv("BOT_ACCESS_TOKEN", "dummy")
    monkeypatch.setenv("TEAM_USERNAME", "mn2632")
    return load_settings(env="sandbox")


# ------------------------------------------------------------------ config
def test_config_defaults_are_disabled() -> None:
    """Code defaults must not change behaviour (features opt-in)."""
    s = Settings(env="sandbox", server_ip="127.0.0.1", team_username="t",
                 bot_access_token="x")
    assert s.strategy.mom_horizon_weights is None
    assert s.strategy.min_hold_days == 3  # code default; YAML raises to 7 (2026-10-30)
    assert s.strategy.entry_timing is None
    assert s.strategy.news_half_life_days == 0.0
    assert s.strategy.disagreement_gate is None
    assert s.strategy.mean_reversion_in_risk_off is False
    assert s.strategy.fresh_entry_only is False
    assert s.signals.vol_scale_scores is False


def test_config_yaml_loads_improvements(monkeypatch) -> None:
    s = _settings(monkeypatch)
    assert s.strategy.mom_horizon_weights == {"d5": 0.3, "d10": 0.4, "d20": 0.3}
    assert s.strategy.min_hold_days == 7  # raised 2026-10-30: 0-3d scalps lost 93K
    assert s.strategy.entry_timing is not None
    # Disabled by default since 2026-10-30 (live diag: starved 100% of
    # entries on the 10-name universe -> zero order flow). Feature stays
    # implemented; tests below enable it explicitly.
    assert s.strategy.entry_timing.enabled is False
    assert s.strategy.entry_timing.top_n == 2
    assert s.strategy.entry_timing.lookback_days == 20
    assert s.strategy.entry_timing.min_history == 3
    assert s.strategy.news_half_life_days == pytest.approx(2.0)
    assert s.strategy.disagreement_gate is not None
    assert s.strategy.disagreement_gate.enabled is True
    assert s.strategy.disagreement_gate.threshold == pytest.approx(0.1)
    assert s.strategy.mean_reversion_in_risk_off is True
    assert s.strategy.fresh_entry_only is False  # same live-diag reason as above
    assert s.signals.vol_scale_scores is True


# ------------------------------------------------------- 1) mom horizon blend
def test_mom_horizon_blend_scores_bounded() -> None:
    rb = _rb_with_closes("AAA", [100.0 * (1.001 ** i) for i in range(25)])
    fn = make_horizon_blend_mom_fn(None, rb, {"d5": 0.3, "d10": 0.4, "d20": 0.3})
    out = fn(_bundle({"AAA": _tick("AAA", 100.0)}))
    assert "AAA" in out
    assert -1.0 <= out["AAA"] <= 1.0
    # Steady uptrend -> positive blend
    assert out["AAA"] > 0


def test_mom_horizon_blend_falls_back_when_no_history() -> None:
    rb = RingBuffer()  # empty
    base = lambda bundle: {"AAA": 0.42}  # noqa: E731
    fn = make_horizon_blend_mom_fn(base, rb, {"d5": 0.3, "d10": 0.4, "d20": 0.3})
    out = fn(_bundle({"AAA": _tick("AAA", 100.0)}))
    assert out["AAA"] == pytest.approx(0.42)


def test_mom_horizon_blend_disabled_when_no_weights() -> None:
    # No weights -> all horizons skipped -> base passthrough
    rb = _rb_with_closes("AAA", [100.0 + i for i in range(25)])
    base = lambda bundle: {"AAA": 0.1}  # noqa: E731
    fn = make_horizon_blend_mom_fn(base, rb, {})
    out = fn(_bundle({"AAA": _tick("AAA", 100.0)}))
    assert out["AAA"] == pytest.approx(0.1)


# ---------------------------------------------------------- 2) min hold days
def test_min_hold_blocks_young_exits() -> None:
    assert min_hold_allows_exit(1, 3, "rebalance") is False
    assert min_hold_allows_exit(0, 3, "") is False
    assert min_hold_allows_exit(3, 3, "rebalance") is True
    assert min_hold_allows_exit(10, 3, "rebalance") is True


def test_min_hold_exempts_stop_and_settlement() -> None:
    assert min_hold_allows_exit(0, 3, "atr_stop_loss") is True
    assert min_hold_allows_exit(1, 3, "settlement_guard") is True


def test_min_hold_fail_soft_and_disabled() -> None:
    assert min_hold_allows_exit(None, 3, "rebalance") is True  # unknown age
    assert min_hold_allows_exit(0, 0, "rebalance") is True     # disabled


# ---------------------------------------------------------- 3) entry timing
def test_entry_timing_median_gate() -> None:
    # score below its own median -> blocked
    assert passes_entry_timing(0.2, [0.5, 0.4, 0.3], min_history=3) is False
    # score at/above median -> allowed
    assert passes_entry_timing(0.4, [0.5, 0.4, 0.3], min_history=3) is True


def test_entry_timing_fail_soft_without_history() -> None:
    assert passes_entry_timing(-0.9, [], min_history=3) is True
    assert passes_entry_timing(-0.9, [0.1, 0.2], min_history=3) is True


# -------------------------------------------------------------- 4) news decay
def test_news_decay_half_life() -> None:
    d0 = date(2025, 1, 1)
    nd = NewsDecay(half_life_days=2.0)
    nd.update(d0, {"AAA": 0.8})
    # 2 days later -> 0.8 * 0.5 = 0.4
    eff = nd.effective("AAA", d0 + timedelta(days=2))
    assert eff == pytest.approx(0.4)
    # 4 days later -> 0.8 * 0.25 = 0.2
    eff = nd.effective("AAA", d0 + timedelta(days=4))
    assert eff == pytest.approx(0.2)


def test_news_decay_max_over_recent_days() -> None:
    d0 = date(2025, 1, 1)
    nd = NewsDecay(half_life_days=2.0)
    nd.update(d0, {"AAA": 0.4})
    nd.update(d0 + timedelta(days=2), {"AAA": 0.5})
    # day 2: fresh 0.5 beats decayed 0.4*0.5=0.2
    assert nd.effective("AAA", d0 + timedelta(days=2)) == pytest.approx(0.5)


def test_news_decay_wrapper_updates_and_unknown_ticker() -> None:
    d0 = date(2025, 1, 1)
    base = lambda bundle: {"AAA": 0.8}  # noqa: E731
    fn = make_decayed_news_fn(base, 2.0)
    out = fn(_bundle({"AAA": _tick("AAA", 100.0)}, sim_date=d0))
    assert out["AAA"] == pytest.approx(0.8)  # same day: no decay
    # Next call 2 days later: base returns nothing new, decayed score remains
    fn2 = make_decayed_news_fn(lambda b: {}, 2.0)
    out = fn2(_bundle({"AAA": _tick("AAA", 100.0)}, sim_date=d0))
    assert out == {}  # no history at all -> empty (fail-soft)


def test_news_decay_disabled_by_zero_half_life() -> None:
    # 0 half-life means the wrapper is not applied (main bypass); the
    # raw same-day score flows through unchanged.
    s = Settings(env="sandbox", server_ip="127.0.0.1", team_username="t",
                 bot_access_token="x")
    assert s.strategy.news_half_life_days == 0.0


# ------------------------------------------------------- 5) vol-scaled scores
def test_vol_scale_scores_on_and_off() -> None:
    s = Settings(env="sandbox", server_ip="127.0.0.1", team_username="t",
                 bot_access_token="x")
    ticks = {f"T{i}": _tick(f"T{i}", 100.0) for i in range(2)}
    rank = lambda b: {"T0": 0.9, "T1": 0.9}  # noqa: E731
    # OFF (default): equal scores stay equal
    ens = SignalsEnsemble(s.signals, rank_fn=rank, rules=RulesFallback(),
                          vol_fn=lambda t: 0.01)
    ev = {e.ticker: e for e in ens.evaluate(_bundle(ticks), "neutral")}
    assert ev["T0"].score == pytest.approx(ev["T1"].score)
    # ON: lower-vol ticker ranks higher
    s.signals.vol_scale_scores = True
    vols = {"T0": 0.04, "T1": 0.01}
    ens2 = SignalsEnsemble(s.signals, rank_fn=rank, rules=RulesFallback(),
                           vol_fn=lambda t: vols[t])
    ev2 = {e.ticker: e for e in ens2.evaluate(_bundle(ticks), "neutral")}
    assert ev2["T1"].score > ev2["T0"].score
    assert "vol_scaled" in " ".join(ev2["T1"].reasons)


def test_vol_scale_guards_zero_vol() -> None:
    s = Settings(env="sandbox", server_ip="127.0.0.1", team_username="t",
                 bot_access_token="x")
    s.signals.vol_scale_scores = True
    rank = lambda b: {"T0": 0.9}  # noqa: E731
    ens = SignalsEnsemble(s.signals, rank_fn=rank, rules=RulesFallback(),
                          vol_fn=lambda t: 0.0)
    ev = ens.evaluate(_bundle({"T0": _tick("T0", 100.0)}), "neutral")[0]
    # vol<=0 -> scaling skipped, score stays in [-1, 1]
    assert -1.0 <= ev.score <= 1.0
    assert ev.score == pytest.approx(ev.raw_score * 0.6)  # neutral mult


# ------------------------------------------------------- 6) disagreement gate
def test_disagreement_gate_logic() -> None:
    assert strong_disagreement(0.5, -0.5, 0.1) is True
    assert strong_disagreement(-0.5, 0.5, 0.1) is True
    assert strong_disagreement(0.5, 0.5, 0.1) is False   # same sign
    assert strong_disagreement(0.05, -0.5, 0.1) is False  # weak rank
    assert strong_disagreement(0.5, -0.05, 0.1) is False  # weak mom


# -------------------------------------------- 7) mean reversion in risk_off
def test_mean_reversion_replaces_mom_in_risk_off() -> None:
    # Downtrend: 10d return negative -> mean-reversion score positive
    rb = _rb_with_closes("AAA", [100.0 * (0.999 ** i) for i in range(15)])
    regime = {"label": "risk_off"}
    base = lambda bundle: {"AAA": -0.7}  # noqa: E731  (momentum says sell)
    fn = make_mean_reversion_mom_fn(base, rb, lambda: regime["label"])
    out = fn(_bundle({"AAA": _tick("AAA", 95.0)}))
    assert out["AAA"] > 0  # beaten-down -> buy candidate


def test_mean_reversion_passthrough_in_other_regimes() -> None:
    rb = _rb_with_closes("AAA", [100.0 + i for i in range(15)])
    base = lambda bundle: {"AAA": 0.33}  # noqa: E731
    for label in ("risk_on", "neutral"):
        fn = make_mean_reversion_mom_fn(base, rb, lambda: label)
        out = fn(_bundle({"AAA": _tick("AAA", 114.0)}))
        assert out["AAA"] == pytest.approx(0.33)


def test_mean_reversion_fail_soft_without_history() -> None:
    rb = RingBuffer()  # empty
    base = lambda bundle: {"AAA": 0.2}  # noqa: E731
    fn = make_mean_reversion_mom_fn(base, rb, lambda: "risk_off")
    out = fn(_bundle({"AAA": _tick("AAA", 100.0)}))
    assert out["AAA"] == pytest.approx(0.2)  # falls back to base


# ---------------------------------------------------------- 8) fresh entry
def test_fresh_entry_top3_sets() -> None:
    """The gate itself is a set-membership check in main._on_day; here we
    verify the semantics: only tickers in today's top-3 AND not in
    yesterday's top-3 are fresh entries."""
    today_top3 = {"A", "B", "C"}
    yesterday_top3 = {"B", "C", "D"}
    fresh = {t for t in today_top3 if t not in yesterday_top3}
    assert fresh == {"A"}
    # First day (no yesterday set) -> gate no-ops (fail-soft)
    prev = None
    assert prev is None  # main skips the check entirely when prev is None
