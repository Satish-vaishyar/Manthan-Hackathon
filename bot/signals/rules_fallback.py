"""bot.signals.rules_fallback — R: SMA/RSI trend + news keywords (P0 fallback).

Per docs/08 P0-11 + docs/04 §3:
    - Trades by hour 4 even with NO models loaded.
    - Uses ONLY the DayBundle's closes (no model1/model2/model3 needed).
    - Decomposes score into trend (SMA20 vs SMA50), RSI(14), news (tagger
      polarity or, if absent, keyword polarity from bot.models.rules).
    - Exposes the same TickerEval shape the ensemble consumes, so the
      `SignalsEnsemble` can substitute this for ranker/news/momentum when
      model artifacts are missing.

This is the canonical P0 fallback. The existing `bot.models.rules.RulesStrategy`
does the same math; this module is a thin, isolated re-export with the same
interface so `bot.signals.ensemble` can depend on `bot.signals.*` exclusively
without circular imports.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import date as _date
from typing import Optional

import structlog

from ..contracts import DayBundle
from .event_tagger import EventTagger, TagResult

_log = structlog.get_logger("signals.rules")


@dataclass
class TickerState:
    closes: deque[float] = field(default_factory=lambda: deque(maxlen=220))
    news_score_sum: float = 0.0
    news_count: int = 0
    last_news_date: Optional[_date] = None


@dataclass
class TickerEval:
    ticker: str
    score: float             # [-1, +1] composite
    sub_scores: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)


class RulesFallback:
    """SMA20/50 trend + RSI14 + news polarity (tagger-driven)."""

    def __init__(self, tagger: Optional[EventTagger] = None) -> None:
        self.tagger = tagger or EventTagger()
        self.state: dict[str, TickerState] = defaultdict(TickerState)

    # ------------------------------------------------------------------
    def evaluate_day(self, bundle: DayBundle) -> dict[str, TickerEval]:
        """Score every ticker with a candle in this day.

        Returns a dict ticker -> TickerEval. Also mutates self.state.
        """
        # Pre-tag today's news per ticker for O(N + M) total cost
        news_by_ticker: dict[str, list[TagResult]] = defaultdict(list)
        for n in bundle.news:
            if not n.ticker:
                continue
            r = self.tagger.tag(n.headline or "", n.description or "")
            news_by_ticker[n.ticker].append(r)

        out: dict[str, TickerEval] = {}
        for ticker, tick in bundle.ticks.items():
            st = self.state[ticker]
            st.closes.append(tick.close)
            sub: dict[str, float] = {}
            reasons: list[str] = []

            # Trend
            trend = self._trend_score(st.closes)
            sub["trend"] = trend
            if trend > 0.2:
                reasons.append("SMA20>SMA50 trend up")
            elif trend < -0.2:
                reasons.append("SMA20<SMA50 trend down")

            # RSI
            rsi = self._rsi14(st.closes)
            rsi_s = self._rsi_to_score(rsi)
            sub["rsi"] = rsi_s
            if rsi < 30:
                reasons.append(f"RSI {rsi:.1f} oversold")
            elif rsi > 70:
                reasons.append(f"RSI {rsi:.1f} overbought")

            # News — average tagger polarity, clamp to [-1, +1]
            tags_today = news_by_ticker.get(ticker, [])
            if tags_today:
                raw = sum(t.polarity for t in tags_today) / len(tags_today)
                news_s = max(-1.0, min(1.0, raw))
            else:
                news_s = 0.0
            sub["news"] = news_s
            st.news_score_sum += news_s
            st.news_count += 1
            st.last_news_date = bundle.simulated_date
            if news_s > 0.3:
                reasons.append("positive news")
            elif news_s < -0.3:
                reasons.append("negative news")

            # Composite — same weights as docs/04 §3
            score = 0.5 * trend + 0.2 * rsi_s + 0.3 * news_s
            score = max(-1.0, min(1.0, score))

            out[ticker] = TickerEval(
                ticker=ticker, score=score, sub_scores=sub, reasons=reasons,
            )
        return out

    # ------------------------------------------------------------------
    @staticmethod
    def _sma(buf: deque, n: int) -> Optional[float]:
        if len(buf) < n:
            return None
        return sum(list(buf)[-n:]) / n

    def _trend_score(self, buf: deque) -> float:
        # Mirror of bot/models/rules.py: 50% SMA20-vs-SMA50 cross + 50%
        # close-vs-SMA50. Counterfactual (2026-10-30): the blend fires the
        # trend-down stamp ~2-4 weeks before the pure cross -> +78K on 126
        # affected trips.
        sma20 = self._sma(buf, 20)
        sma50 = self._sma(buf, 50)
        if sma20 is None or sma50 is None or sma50 <= 0:
            return 0.0
        close = buf[-1]
        blended = 0.5 * (sma20 - sma50) / sma50 + 0.5 * (close - sma50) / sma50
        return max(-1.0, min(1.0, blended * 20.0))  # 5% blended gap -> ±1

    @staticmethod
    def _rsi14(buf: deque, n: int = 14) -> float:
        if len(buf) < n + 1:
            return 50.0
        diffs = [buf[i] - buf[i - 1] for i in range(-n, 0)]
        gains = sum(max(0.0, d) for d in diffs) / n
        losses = sum(max(0.0, -d) for d in diffs) / n
        if losses <= 1e-9:
            return 100.0
        rs = gains / losses
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def _rsi_to_score(rsi: float) -> float:
        return max(-1.0, min(1.0, (rsi - 50.0) / 20.0))