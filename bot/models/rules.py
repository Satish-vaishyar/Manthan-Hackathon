"""bot.models.rules — P0 fallback strategy: SMA trend + RSI + news keywords.

Per docs/04 + docs/08 P0 — this is what runs at hour 4 if no ML model is
plugged in. Once model1 is exported to ONNX, it can be substituted in
`bot.models.ensemble.py` without touching the rest of the system.

Output: per-ticker ensemble score in [-1, +1] plus sub-scores for the journal.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import date as _date
from typing import Optional

import structlog

from ..contracts import DayBundle

_log = structlog.get_logger("rules")

# News keyword polarity — docs/04 §2 H1 (news drives next-day return)
_POSITIVE_KEYWORDS = {
    "beat", "beats", "surge", "surges", "rally", "rallies", "upgrade", "upgrades",
    "record", "high", "growth", "profit", "expansion", "wins", "win", "approval",
    "approved", "breakthrough", "partnership", "deal", "acquisition",
}
_NEGATIVE_KEYWORDS = {
    "miss", "misses", "plunge", "plunges", "crash", "downgrade", "downgrades",
    "loss", "losses", "decline", "fall", "falls", "weak", "warning", "probe",
    "investigation", "fraud", "lawsuit", "recall", "bankruptcy", "default",
}


@dataclass
class TickerState:
    """Per-ticker incremental state for the rules strategy."""
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


class RulesStrategy:
    """SMA20/50 trend + RSI14 + news-keyword polarity, cross-sectionally ranked."""

    def __init__(self) -> None:
        self.state: dict[str, TickerState] = defaultdict(TickerState)

    def evaluate_day(self, bundle: DayBundle) -> dict[str, TickerEval]:
        """Score every ticker with a candle in this day.

        Returns a dict ticker -> TickerEval. Also mutates self.state.
        """
        out: dict[str, TickerEval] = {}
        for ticker, tick in bundle.ticks.items():
            st = self.state[ticker]
            st.closes.append(tick.close)
            sub: dict[str, float] = {}
            reasons: list[str] = []

            # ---- Trend: SMA20 vs SMA50 ----
            sma_score = self._trend_score(st.closes)
            sub["trend"] = sma_score
            if sma_score > 0.2:
                reasons.append(f"SMA20>SMA50 trend up")
            elif sma_score < -0.2:
                reasons.append(f"SMA20<SMA50 trend down")

            # ---- RSI14 ----
            rsi = self._rsi14(st.closes)
            rsi_score = self._rsi_to_score(rsi)
            sub["rsi"] = rsi_score
            if rsi < 30:
                reasons.append(f"RSI {rsi:.1f} oversold")
            elif rsi > 70:
                reasons.append(f"RSI {rsi:.1f} overbought")

            # ---- News polarity for this ticker today ----
            news_score = self._news_score(bundle, ticker)
            st.news_score_sum += news_score
            st.news_count += 1
            st.last_news_date = bundle.simulated_date
            sub["news"] = news_score
            if news_score > 0.3:
                reasons.append(f"positive news")
            elif news_score < -0.3:
                reasons.append(f"negative news")

            # ---- Composite: trend 0.5 + RSI 0.2 + news 0.3 (mirrors docs/04 §3) ----
            score = 0.5 * sma_score + 0.2 * rsi_score + 0.3 * news_score
            score = max(-1.0, min(1.0, score))

            out[ticker] = TickerEval(
                ticker=ticker, score=score, sub_scores=sub, reasons=reasons,
            )

        return out

    # ------------------------------------------------------------------
    # Indicators
    # ------------------------------------------------------------------
    @staticmethod
    def _sma(buf: deque, n: int) -> Optional[float]:
        if len(buf) < n:
            return None
        return sum(list(buf)[-n:]) / n

    def _trend_score(self, buf: deque) -> float:
        """SMA20-vs-SMA50 relative spread, scaled so a 5% gap saturates to ±1."""
        sma20 = self._sma(buf, 20)
        sma50 = self._sma(buf, 50)
        if sma20 is None or sma50 is None or sma50 <= 0:
            return 0.0
        diff = (sma20 - sma50) / sma50
        return max(-1.0, min(1.0, diff * 20.0))  # 5% spread -> +1

    @staticmethod
    def _rsi14(buf: deque, n: int = 14) -> float:
        """Classic 14-period RSI; 50.0 (neutral) until there's enough history."""
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
        # Map RSI 30->70 to [0,1] linearly, 50->0.5
        return max(-1.0, min(1.0, (rsi - 50.0) / 20.0))

    def _news_score(self, bundle: DayBundle, ticker: str) -> float:
        """Keyword polarity over today's headlines for `ticker`, in [-1, +1];
        3+ matching items saturate the score."""
        pos = neg = 0
        for n in bundle.news:
            if n.ticker != ticker:
                continue
            text = (n.headline or "").lower()
            for kw in _POSITIVE_KEYWORDS:
                if kw in text:
                    pos += 1
            for kw in _NEGATIVE_KEYWORDS:
                if kw in text:
                    neg += 1
        total = pos + neg
        if total == 0:
            return 0.0
        # Scale so 3+ matching items saturate
        return max(-1.0, min(1.0, (pos - neg) / max(3, total)))