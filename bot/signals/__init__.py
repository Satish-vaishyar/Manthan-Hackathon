"""bot.signals — rule-based signal layer (M1a, M3 rules, M4, R) + ensemble.

Per docs/05 §3 (M1a), §5 (M3), §6 (M4) and docs/08 P0-11 + P1-08:

    event_tagger    M1a   keyword + regex tagger with signed polarity
    regime          M3    rule-based risk-on / neutral / risk-off (HMM optional)
    volatility      M4    EWMA realised vol (delegates to bot.models.vol) + ATR helper
    sector_cluster  P2-08 TF-IDF + KMeans over /stocks company descriptions
    rules_fallback  R     SMA/RSI + news keyword fallback (P0 — hour 4)
    ensemble        -     combines the above with weights + regime gating

This package is the P0 fallback path: even before model1-3 artifacts ship,
the bot trades with `rules_fallback + volatility + regime + sector_cluster`
alone. Each model file is a soft dependency; missing artifact -> skip + log.
"""
from __future__ import annotations

from dataclasses import dataclass

from .ensemble import EnsembleEval, SignalsEnsemble
from .event_tagger import EventTagger, TagResult
from .regime import (
    ALL_REGIMES, MarketState, NEUTRAL, RegimeFilter, RegimeVerdict,
    RISK_OFF, RISK_ON, gross_long_target,
)
from .rules_fallback import RulesFallback, TickerEval
from .sector_cluster import SectorCluster, cluster_sectors
from .volatility import VolatilityModule, annualised_vol


@dataclass
class SignalsBundle:
    """One-stop bundle the bot instantiates at startup.

    Holds every signal component + the cross-cutting state (market_state) the
    regime filter consumes. `bot.main.TradingBot` wires these together; tests
    can instantiate a bare bundle to drive the pipeline end-to-end.
    """

    tagger: EventTagger
    regime_filter: RegimeFilter
    volatility: VolatilityModule
    rules: RulesFallback
    ensemble: SignalsEnsemble
    market_state: MarketState
    sector_cluster: SectorCluster | None = None
    sector_of: dict[str, str] | None = None

    # ------------------------------------------------------------------
    def update_for_day(self, sim_date, ticks) -> RegimeVerdict:
        """Push today's ticks into vol + market state + regime filter.

        Returns the regime verdict the ensemble should use.
        """
        for ticker, tick in ticks.items():
            self.volatility.update_from_tick(
                ticker, tick.open, tick.high, tick.low, tick.close,
            )
        self.market_state.update(sim_date, ticks)
        return self.regime_filter.update(sim_date, self.market_state)


__all__ = [
    "ALL_REGIMES",
    "EnsembleEval",
    "EventTagger",
    "MarketState",
    "NEUTRAL",
    "RegimeFilter",
    "RegimeVerdict",
    "RISK_OFF",
    "RISK_ON",
    "RulesFallback",
    "SectorCluster",
    "SignalsBundle",
    "SignalsEnsemble",
    "TagResult",
    "TickerEval",
    "VolatilityModule",
    "annualised_vol",
    "cluster_sectors",
    "gross_long_target",
]