"""bot.signals.ensemble — SignalsEnsemble: rank + news + mom + regime gating.

Per docs/04 §4 + docs/05 §5 + docs/08 P1-08:
    - Combines rank_fn (model1), news_fn (model2 + tagger fallback), momentum
      with configurable weights from `SignalsSettings.weights`.
    - Multiplies the combined score by a regime multiplier derived from the
      regime's `gross_long_target` (risk_off = reduce, risk_on = full).
    - Returns one `EnsembleEval` per ticker, ready for the constructor.

Graceful degradation (P1-08 acceptance criterion):
    - Each sub-scorer is a soft dependency. If `rank_fn` raises or returns
      `None`, we treat it as missing and renormalise the weights over the
      remaining signals.
    - `news_fn` falls back to `EventTagger` polarity automatically if model2
      isn't loaded.
    - `mom_fn` falls back to the trend sub-score from `RulesFallback`.

The "missing signal" path is the P0 path: rules + tagger + vol, no model
artifacts, and the bot still trades. The full path (model1+2+3 artifacts
present) replaces each sub-scorer without changing call signatures.

Regime input contract (post-M5):
    - `evaluate(bundle, regime)` receives a regime LABEL string
      ('risk_on' / 'neutral' / 'risk_off'). The label comes from
      `bot.models.regime.RegimeDetector.update(bundle)`. This is the same
      three-label vocabulary as before — only the call shape is narrower.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import structlog

from ..config import SignalsSettings
from ..contracts import DayBundle
from ..models.regime import NEUTRAL
from .rules_fallback import RulesFallback, TickerEval

_log = structlog.get_logger("signals.ensemble")


# Type aliases for the optional sub-scorers
RankFn = Callable[[DayBundle], dict[str, float]]
NewsFn = Callable[[DayBundle], dict[str, float]]
MomFn = Callable[[DayBundle], dict[str, float]]


@dataclass
class EnsembleEval:
    ticker: str
    score: float            # regime-adjusted, in [-1, +1]
    raw_score: float        # pre-regime, in [-1, +1]
    rank_score: float
    news_score: float
    momentum_score: float
    regime: str
    reasons: list[str] = field(default_factory=list)


# Per-regime multiplier on the raw score. Positive scores stay positive;
# negative scores get MORE negative in risk_off (the bot wants to short more
# when things are bad, but the constructor enforces gross-long targets).
_REGIME_SIGN = {
    "risk_on": +1.0,
    "neutral": +0.6,   # dampen exposure
    "risk_off": +0.25, # heavily dampen
}


class SignalsEnsemble:
    """Combines all signals + regime gating.

    Use:
        ens = SignalsEnsemble(settings, rank_fn=..., news_fn=..., mom_fn=...)
        evals = ens.evaluate(bundle, regime_label)
        # evals: list[EnsembleEval], one per ticker in bundle.ticks
    """

    def __init__(self,
                 settings: SignalsSettings,
                 *,
                 rank_fn: Optional[RankFn] = None,
                 news_fn: Optional[NewsFn] = None,
                 mom_fn: Optional[MomFn] = None,
                 rules: Optional[RulesFallback] = None) -> None:
        self.settings = settings
        self.rules = rules or RulesFallback()
        self.rank_fn = rank_fn
        self.news_fn = news_fn
        self.mom_fn = mom_fn

    # ------------------------------------------------------------------
    def _safe(self, fn_name: str, fn, bundle: DayBundle) -> dict[str, float]:
        """Run a sub-scorer; on failure, log + return empty dict (= missing)."""
        if fn is None:
            return {}
        try:
            return dict(fn(bundle) or {})
        except Exception as e:  # noqa: BLE001
            _log.warning("ensemble_subscorer_failed",
                         fn=fn_name, err=repr(e), component="ensemble")
            return {}

    def evaluate(self, bundle: DayBundle,
                 regime: str = NEUTRAL) -> list[EnsembleEval]:
        """Score the day, returning one EnsembleEval per ticker.

        `regime` is the day's label ('risk_on' / 'neutral' / 'risk_off'),
        normally produced by `bot.models.regime.RegimeDetector.update(bundle)`.
        """
        # 1) Sub-scores (missing signal -> {})
        rank_scores = self._safe("rank", self.rank_fn, bundle)
        news_scores = self._safe("news", self.news_fn, bundle)
        mom_scores = self._safe("mom", self.mom_fn, bundle)

        # 2) Rules sub-scores (always available — P0 fallback)
        rule_evals: dict[str, TickerEval] = self.rules.evaluate_day(bundle)
        tickers = list(bundle.ticks.keys())

        # 3) Per-ticker weighted blend
        weights = self.settings.weights
        out: list[EnsembleEval] = []
        for t in tickers:
            r = rank_scores.get(t)
            n = news_scores.get(t)
            m = mom_scores.get(t)
            # If a sub-scorer is missing, fall back to rules for that axis
            rule_eval = rule_evals.get(t)
            if r is None:
                r = 0.7 * rule_eval.sub_scores.get("trend", 0.0) + 0.3 * rule_eval.sub_scores.get("rsi", 0.0) if rule_eval else 0.0
            if n is None:
                n = rule_eval.sub_scores.get("news", 0.0) if rule_eval else 0.0
            if m is None:
                m = rule_eval.sub_scores.get("trend", 0.0) if rule_eval else 0.0

            # Renormalise weights over the signals we actually have
            w_rank = weights.get("rank", 0.45) if rank_scores else 0.0
            w_news = weights.get("news", 0.35) if news_scores else 0.0
            w_mom = weights.get("mom", 0.20) if mom_scores else 0.0
            w_sum = w_rank + w_news + w_mom
            if w_sum <= 0:
                # Pure-rules fallback: use rule composite
                raw = rule_eval.score if rule_eval else 0.0
                w_rank = w_news = w_mom = 0.0
            else:
                raw = (w_rank * r + w_news * n + w_mom * m) / w_sum

            # Regime gating — multiply raw by regime multiplier, preserve sign
            reg_mult = _REGIME_SIGN.get(regime, _REGIME_SIGN[NEUTRAL])
            score = max(-1.0, min(1.0, raw * reg_mult))

            reasons: list[str] = []
            if rule_eval is not None:
                reasons.extend(rule_eval.reasons)
            if regime != NEUTRAL:
                reasons.append(f"regime={regime} (mult={reg_mult:.2f})")
            if not rank_scores:
                reasons.append("rank:rules")
            if not news_scores:
                reasons.append("news:tagger")

            out.append(EnsembleEval(
                ticker=t, score=score, raw_score=max(-1.0, min(1.0, raw)),
                rank_score=r, news_score=n, momentum_score=m,
                regime=regime, reasons=reasons,
            ))
        return out

    # ------------------------------------------------------------------
    def regime_gross_target(self, regime: str,
                            gross_table: dict[str, float]) -> float:
        """Convenience: regime -> gross long target from the risk settings."""
        if regime not in gross_table:
            return 0.75
        return max(0.0, min(1.0, float(gross_table[regime])))