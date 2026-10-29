"""bot.signals — rule-based signal layer (M1a, M4, R, sector clustering) + ensemble.

Per docs/05 §3 (M1a), §6 (M4) and docs/08 P0-11 + P1-08:

    event_tagger    M1a   keyword + regex tagger with signed polarity
    volatility      M4    EWMA realised vol (delegates to bot.models.vol) + ATR helper
    sector_cluster  P2-08 TF-IDF + KMeans over /stocks company descriptions
    rules_fallback  R     SMA/RSI + news keyword fallback (P0 — hour 4)
    ensemble        -     combines the above with weights + regime gating

Regime detection (M5) lives in `bot.models.regime` and is wired directly by
`bot.main.TradingBot`. This package re-exports the regime labels + the
gross-target lookup so downstream code can import a single namespace.

This package is the P0 fallback path: even before model1-3 artifacts ship,
the bot trades with `rules_fallback + volatility + sector_cluster` alone.
Each model file is a soft dependency; missing artifact -> skip + log.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..models.regime import (
    ALL_REGIMES, NEUTRAL, RISK_OFF, RISK_ON,
    RegimeConfig, RegimeDetector, RegimeState,
)
from .ensemble import EnsembleEval, SignalsEnsemble
from .event_tagger import EventTagger, TagResult
from .rules_fallback import RulesFallback, TickerEval
from .sector_cluster import SectorCluster, cluster_sectors
from .volatility import VolatilityModule, annualised_vol


# Local lookup helper. The remote M5 doesn't export `gross_long_target` from
# `bot.models.regime` (its `build_target_weights` does the lookup internally
# given `gross_long_target=`), so we keep the helper here for callers that
# want to read the table themselves.
def gross_long_target(regime: str, gross_table: dict[str, float],
                      default: float = 0.75) -> float:
    """Look up the regime's gross long target with a sane fallback."""
    if regime not in gross_table:
        return default
    return max(0.0, min(1.0, float(gross_table[regime])))


@dataclass
class SignalsBundle:
    """One-stop bundle the bot instantiates at startup.

    Holds every signal component; `bot.main.TradingBot` wires these together
    and uses `bot.models.regime.RegimeDetector` directly for regime detection.
    """

    tagger: EventTagger
    volatility: VolatilityModule
    rules: RulesFallback
    ensemble: SignalsEnsemble
    sector_cluster: SectorCluster | None = None
    sector_of: dict[str, str] | None = None

    # ------------------------------------------------------------------
    def update_volatility(self, sim_date, ticks: Iterable) -> None:
        """Push today's ticks into the vol module (M4).

        M5 regime detection is performed by the bot's `RegimeDetector`
        directly against the day's bundle, so this method only feeds the
        per-ticker vol state.
        """
        for ticker, tick in ticks.items():
            self.volatility.update_from_tick(
                ticker, tick.open, tick.high, tick.low, tick.close,
            )


__all__ = [
    "ALL_REGIMES",
    "EnsembleEval",
    "EventTagger",
    "NEUTRAL",
    "RISK_OFF",
    "RISK_ON",
    "RegimeConfig",
    "RegimeDetector",
    "RegimeState",
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