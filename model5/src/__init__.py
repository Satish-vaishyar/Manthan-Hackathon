"""model5 — Market Regime Detector (M5).

Once-per-sim-day classifier that emits one of `risk_on` / `neutral` /
`risk_off` and gates the bot's long-side gross exposure. CPU-only,
parameter-free; canonical home for M5.

`bot.models.regime` re-exports everything from this package for backwards
compatibility.
"""
from .regime import (
    ALL_REGIMES,
    DEFAULT_GROSS_TARGET,
    NEUTRAL,
    RISK_OFF,
    RISK_ON,
    RegimeConfig,
    RegimeDetector,
    RegimeState,
)

__all__ = [
    "ALL_REGIMES",
    "DEFAULT_GROSS_TARGET",
    "NEUTRAL",
    "RISK_OFF",
    "RISK_ON",
    "RegimeConfig",
    "RegimeDetector",
    "RegimeState",
]