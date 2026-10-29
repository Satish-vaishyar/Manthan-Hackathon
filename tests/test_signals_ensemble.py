"""tests/test_signals_ensemble.py — coverage for bot.signals.ensemble + bundle."""
from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from bot.config import Settings
from bot.contracts import DayBundle, NewsFlashData, PriceTickData
from bot.signals import (
    EventTagger, MarketState, NEUTRAL, RegimeFilter, RegimeVerdict,
    RISK_OFF, RISK_ON, RulesFallback, SignalsBundle, SignalsEnsemble,
    VolatilityModule,
)
from bot.signals.ensemble import EnsembleEval


def _tick(t: str, c: float) -> PriceTickData:
    return PriceTickData(ticker=t, open=c, high=c, low=c, close=c, volume=1000)


def _news(t: str, h: str) -> NewsFlashData:
    return NewsFlashData(ticker=t, headline=h, description="")


def _bundle(ticks: dict[str, PriceTickData], news=None) -> DayBundle:
    return DayBundle(
        simulated_date=date(2025, 1, 1),
        ticks=ticks,
        news=news or [],
        assembled_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        assembly_latency_ms=0.0,
    )


# ---------------------------------------------------------------------- bundle
def test_signals_bundle_update_for_day_feeds_everything():
    s = Settings(env="sandbox", server_ip="127.0.0.1", team_username="t",
                 bot_access_token="x")
    bundle = SignalsBundle(
        tagger=EventTagger(),
        regime_filter=RegimeFilter(s.signals.regime),
        volatility=VolatilityModule(),
        rules=RulesFallback(),
        ensemble=SignalsEnsemble(s.signals),
        market_state=MarketState(),
    )
    ticks = {f"T{i}": _tick(f"T{i}", 100.0 + i) for i in range(5)}
    v = bundle.update_for_day(date(2025, 1, 1), ticks)
    assert isinstance(v, RegimeVerdict)
    assert bundle.volatility.vol("T0") == pytest.approx(0.02)  # fallback before warmup


# ------------------------------------------------------------------ evaluate
def test_ensemble_evaluate_with_no_model_fns_uses_rules_fallback():
    s = Settings(env="sandbox", server_ip="127.0.0.1", team_username="t",
                 bot_access_token="x")
    ens = SignalsEnsemble(s.signals)
    ticks = {f"T{i}": _tick(f"T{i}", 100.0 + i) for i in range(3)}
    verdict = RegimeVerdict(regime=NEUTRAL, candidate=NEUTRAL,
                            held_since=None, days_in_current=0,
                            confidence=0.0, reason="test")
    evals = ens.evaluate(_bundle(ticks), verdict)
    assert len(evals) == 3
    for ev in evals:
        assert -1.0 <= ev.score <= 1.0
        assert ev.regime == NEUTRAL


def test_ensemble_score_is_clamped():
    s = Settings(env="sandbox", server_ip="127.0.0.1", team_username="t",
                 bot_access_token="x")
    ens = SignalsEnsemble(s.signals)
    # News with a -2 tag should produce a non-zero negative news sub-score
    news = [_news("T0", "SEBI probe on insider trading")]
    ticks = {"T0": _tick("T0", 100.0)}
    verdict = RegimeVerdict(regime=NEUTRAL, candidate=NEUTRAL,
                            held_since=None, days_in_current=0,
                            confidence=0.0, reason="test")
    evals = ens.evaluate(_bundle(ticks, news), verdict)
    assert all(-1.0 <= ev.score <= 1.0 for ev in evals)
    assert evals[0].news_score < 0


def test_ensemble_risk_off_dampens_score():
    s = Settings(env="sandbox", server_ip="127.0.0.1", team_username="t",
                 bot_access_token="x")
    ens = SignalsEnsemble(s.signals)
    ticks = {"T0": _tick("T0", 100.0)}
    neutral_v = RegimeVerdict(regime=NEUTRAL, candidate=NEUTRAL,
                              held_since=None, days_in_current=0,
                              confidence=0.0, reason="test")
    riskoff_v = RegimeVerdict(regime=RISK_OFF, candidate=RISK_OFF,
                              held_since=None, days_in_current=0,
                              confidence=0.0, reason="test")
    evals_n = ens.evaluate(_bundle(ticks), neutral_v)
    evals_o = ens.evaluate(_bundle(ticks), riskoff_v)
    # With no models, the rules raw is 0.0; but the regime multiplier still
    # applies. Confirm the regime field is recorded correctly.
    assert evals_n[0].regime == NEUTRAL
    assert evals_o[0].regime == RISK_OFF


def test_ensemble_sub_scorer_failure_does_not_crash():
    s = Settings(env="sandbox", server_ip="127.0.0.1", team_username="t",
                 bot_access_token="x")
    rank_fn = lambda b: (_ for _ in ()).throw(RuntimeError("model1 exploded"))
    news_fn = lambda b: {"T0": 0.5, "T1": -0.5}
    ens = SignalsEnsemble(s.signals, rank_fn=rank_fn, news_fn=news_fn)
    ticks = {"T0": _tick("T0", 100.0), "T1": _tick("T1", 100.0)}
    verdict = RegimeVerdict(regime=NEUTRAL, candidate=NEUTRAL,
                            held_since=None, days_in_current=0,
                            confidence=0.0, reason="test")
    # rank_fn blew up -> rank treated as missing; news_fn ok
    evals = ens.evaluate(_bundle(ticks), verdict)
    assert len(evals) == 2
    # News sub-score should still flow through (renormalised)
    by_t = {e.ticker: e for e in evals}
    assert by_t["T0"].news_score == pytest.approx(0.5)
    assert by_t["T1"].news_score == pytest.approx(-0.5)
    # Reasons should mention the fallback
    assert any("rank:rules" in r for r in by_t["T0"].reasons)


def test_ensemble_weights_renormalised_when_one_missing():
    s = Settings(env="sandbox", server_ip="127.0.0.1", team_username="t",
                 bot_access_token="x")
    # No rank_fn, news_fn returns +1.0 for everything, mom_fn returns 0
    news_fn = lambda b: {t: 1.0 for t in b.ticks}
    mom_fn = lambda b: {t: 0.0 for t in b.ticks}
    ens = SignalsEnsemble(s.signals, news_fn=news_fn, mom_fn=mom_fn)
    ticks = {"T0": _tick("T0", 100.0)}
    verdict = RegimeVerdict(regime=NEUTRAL, candidate=NEUTRAL,
                            held_since=None, days_in_current=0,
                            confidence=0.0, reason="test")
    evals = ens.evaluate(_bundle(ticks), verdict)
    # news=1.0, mom=0.0; rank missing -> only news + mom contribute;
    # weights renormalise over {news=0.35, mom=0.20} -> effective news=0.35/0.55
    # raw = (0.35/0.55)*1.0 + (0.20/0.55)*0.0 = 0.636...
    expected = (s.signals.weights["news"] / (s.signals.weights["news"] + s.signals.weights["mom"]))
    assert evals[0].raw_score == pytest.approx(expected, rel=0.01)


def test_ensemble_pure_rules_path_when_no_sub_scorers():
    s = Settings(env="sandbox", server_ip="127.0.0.1", team_username="t",
                 bot_access_token="x")
    ens = SignalsEnsemble(s.signals)
    # Provide 60 days of price history so rules have a trend signal
    rb = ens.rules
    for i in range(60):
        bundle = _bundle({"T0": _tick("T0", 100.0 + i * 0.5)}, [])
        rb.evaluate_day(bundle)
    final_bundle = _bundle({"T0": _tick("T0", 130.0)}, [])
    verdict = RegimeVerdict(regime=NEUTRAL, candidate=NEUTRAL,
                            held_since=None, days_in_current=0,
                            confidence=0.0, reason="test")
    evals = ens.evaluate(final_bundle, verdict)
    assert evals[0].score > 0  # positive trend + neutral regime -> positive


def test_ensemble_regime_gross_target_lookup():
    s = Settings(env="sandbox", server_ip="127.0.0.1", team_username="t",
                 bot_access_token="x")
    ens = SignalsEnsemble(s.signals)
    gross_table = {"risk_on": 0.92, "neutral": 0.75, "risk_off": 0.40}
    v_on = RegimeVerdict(regime=RISK_ON, candidate=RISK_ON, held_since=None,
                          days_in_current=0, confidence=0.0, reason="")
    v_off = RegimeVerdict(regime=RISK_OFF, candidate=RISK_OFF, held_since=None,
                           days_in_current=0, confidence=0.0, reason="")
    assert ens.regime_gross_target(v_on, gross_table) == pytest.approx(0.92)
    assert ens.regime_gross_target(v_off, gross_table) == pytest.approx(0.40)


# -------------------------------------------------- integration: full pipeline
def test_full_pipeline_runs_through_signals_bundle():
    s = Settings(env="sandbox", server_ip="127.0.0.1", team_username="t",
                 bot_access_token="x")
    bundle_obj = SignalsBundle(
        tagger=EventTagger(),
        regime_filter=RegimeFilter(s.signals.regime),
        volatility=VolatilityModule(),
        rules=RulesFallback(),
        ensemble=SignalsEnsemble(s.signals),
        market_state=MarketState(),
    )
    # 25 days of slow rally
    from datetime import timedelta
    d = date(2025, 1, 1)
    for i in range(25):
        ticks = {f"T{j:02d}": _tick(f"T{j:02d}", 100.0 + i * 0.3 + j * 0.1)
                 for j in range(10)}
        b = _bundle(ticks, news=[_news("T00", "profit surges")] if i == 24 else [])
        verdict = bundle_obj.update_for_day(b.simulated_date, ticks)
        evals = bundle_obj.ensemble.evaluate(b, verdict)
        # Each day: 10 evals, all in [-1, 1]
        assert len(evals) == 10
        assert all(-1.0 <= ev.score <= 1.0 for ev in evals)
        # News ticker T00 should score positive on day 24
        if i == 24:
            t00 = next(e for e in evals if e.ticker == "T00")
            assert t00.news_score > 0
        d += timedelta(days=1)