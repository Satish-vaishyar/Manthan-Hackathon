"""bot.models.ensemble — combiner for ranker + news + momentum (docs/04 §3).

Wraps the rules strategy by default; once the ML ranker (model1 ONNX) is
shipped, replace `_rank_scores` to call it instead. The rest of the bot sees
the same TickerEval interface.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..contracts import DayBundle
from ..config import StrategySettings
from .rules import RulesStrategy, TickerEval


@dataclass
class EnsembleEval:
    ticker: str
    score: float                  # combined
    rank_score: float             # ranker sub-score (ML model)
    news_score: float
    momentum_score: float
    reasons: list[str]


class Ensemble:
    """Weighted blend of ranker + news + momentum.

    Default `rank_fn` returns the rules strategy's `trend + RSI` sub-score.
    Swap in `model1.predict_day(...)` once the model is exported.
    """

    def __init__(self, settings: StrategySettings,
                 rank_fn=None,
                 news_fn=None,
                 mom_fn=None,
                 vol_fn=None,
                 vol_scale: bool = False) -> None:
        self.settings = settings
        self.rules = RulesStrategy()
        self.rank_fn = rank_fn or self._default_rank
        self.news_fn = news_fn or self._default_news
        self.mom_fn = mom_fn or self._default_mom
        # Volatility-scaled conviction (opt-in): divide the final blended
        # score by realized vol before ranking. vol<=0 -> skip scaling.
        self.vol_fn = vol_fn
        self.vol_scale = vol_scale

    def evaluate(self, bundle: DayBundle) -> list[EnsembleEval]:
        """Score every ticker in the bundle: rules base evaluation, then
        blend rank/news/mom sub-scores with strategy weights (clipped to
        [-1, +1]). NOTE: TradingBot actually uses signals.SignalsEnsemble;
        this class is the lighter-weight legacy path."""
        rule_evals = self.rules.evaluate_day(bundle)
        out: list[EnsembleEval] = []
        for t, ev in rule_evals.items():
            r = self.rank_fn(ev)
            n = self.news_fn(ev)
            m = self.mom_fn(ev)
            w = self.settings.weights
            score = w.get("rank", 0.45) * r + w.get("news", 0.35) * n + w.get("mom", 0.20) * m
            score = max(-1.0, min(1.0, score))
            if self.vol_scale and self.vol_fn is not None:
                try:
                    v = float(self.vol_fn(t) or 0.0)
                except Exception:  # noqa: BLE001
                    v = 0.0
                if v > 0:
                    score = score / v  # intentionally not re-clipped
            out.append(EnsembleEval(
                ticker=t, score=score,
                rank_score=r, news_score=n, momentum_score=m,
                reasons=list(ev.reasons),
            ))
        return out

    # Default sub-scorers (rules strategy decomposition) ------------------
    @staticmethod
    def _default_rank(ev: TickerEval) -> float:
        # Trend (0.7) + RSI (0.3)
        return 0.7 * ev.sub_scores.get("trend", 0.0) + 0.3 * ev.sub_scores.get("rsi", 0.0)

    @staticmethod
    def _default_news(ev: TickerEval) -> float:
        return ev.sub_scores.get("news", 0.0)

    @staticmethod
    def _default_mom(ev: TickerEval) -> float:
        # Same as trend for the rules-only fallback
        return ev.sub_scores.get("trend", 0.0)