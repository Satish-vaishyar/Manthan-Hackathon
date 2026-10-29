"""tests/test_signals_rules_fallback.py — coverage for bot.signals.rules_fallback (R)."""
from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from bot.contracts import DayBundle, NewsFlashData, PriceTickData
from bot.signals.event_tagger import EventTagger
from bot.signals.rules_fallback import RulesFallback, TickerEval


def _tick(t: str, c: float) -> PriceTickData:
    return PriceTickData(ticker=t, open=c, high=c, low=c, close=c, volume=1000)


def _news(t: str, h: str) -> NewsFlashData:
    return NewsFlashData(ticker=t, headline=h, description="")


def _bundle(ticks: dict[str, PriceTickData], news: list[NewsFlashData]) -> DayBundle:
    return DayBundle(
        simulated_date=date(2025, 1, 1),
        ticks=ticks,
        news=news,
        assembled_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        assembly_latency_ms=0.0,
    )


# ---------------------------------------------------------------- core shape
def test_rules_fallback_returns_one_eval_per_ticker():
    rb = RulesFallback()
    ticks = {f"T{i}": _tick(f"T{i}", 100.0 + i) for i in range(5)}
    evals = rb.evaluate_day(_bundle(ticks, []))
    assert len(evals) == 5
    for t, ev in evals.items():
        assert isinstance(ev, TickerEval)
        assert -1.0 <= ev.score <= 1.0
        assert "trend" in ev.sub_scores
        assert "rsi" in ev.sub_scores
        assert "news" in ev.sub_scores


def test_rules_fallback_score_zero_with_no_history():
    rb = RulesFallback()
    evals = rb.evaluate_day(_bundle({"AAA": _tick("AAA", 100.0)}, []))
    # Only 1 close -> SMA50/SMA20/RSI undefined -> defaults -> score ~ 0
    assert abs(evals["AAA"].score) < 0.5


def test_rules_fallback_trend_picks_up_after_history():
    rb = RulesFallback()
    bundle = _bundle({"AAA": _tick("AAA", 100.0)}, [])
    # 60 days of flat prices
    for i in range(60):
        bundle.ticks = {"AAA": _tick("AAA", 100.0)}
        rb.evaluate_day(bundle)
    # Now a strong rally day
    bundle.ticks = {"AAA": _tick("AAA", 115.0)}
    evals = rb.evaluate_day(bundle)
    assert evals["AAA"].score > 0  # trend +0, RSI still neutral/overbought


# ------------------------------------------------------------------- news path
def test_rules_fallback_news_uses_tagger():
    rb = RulesFallback(tagger=EventTagger())
    news = [_news("AAA", "Company profit surges 20%")]
    ticks = {"AAA": _tick("AAA", 100.0)}
    evals = rb.evaluate_day(_bundle(ticks, news))
    assert evals["AAA"].sub_scores["news"] > 0.5
    assert any("positive" in r for r in evals["AAA"].reasons)


def test_rules_fallback_negative_news_dominates():
    rb = RulesFallback(tagger=EventTagger())
    news = [_news("AAA", "SEBI probe on insider trading")]
    ticks = {"AAA": _tick("AAA", 100.0)}
    evals = rb.evaluate_day(_bundle(ticks, news))
    assert evals["AAA"].sub_scores["news"] < -0.5


def test_rules_fallback_news_only_for_matching_ticker():
    rb = RulesFallback(tagger=EventTagger())
    news = [_news("BBB", "Profit surges 20%")]  # wrong ticker
    ticks = {"AAA": _tick("AAA", 100.0)}
    evals = rb.evaluate_day(_bundle(ticks, news))
    assert evals["AAA"].sub_scores["news"] == 0.0


def test_rules_fallback_state_accumulates_across_days():
    rb = RulesFallback(tagger=EventTagger())
    # Three days of positive news for AAA
    for i in range(3):
        bundle = _bundle({"AAA": _tick("AAA", 100.0)}, [_news("AAA", "profit surges")])
        evals = rb.evaluate_day(bundle)
    assert rb.state["AAA"].news_count == 3
    assert rb.state["AAA"].news_score_sum > 0