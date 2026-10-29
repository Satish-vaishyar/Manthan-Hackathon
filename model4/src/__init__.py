"""model4 — Realised Volatility (M4).

Per-ticker rolling-window annualised volatility estimator. CPU-only,
parameter-free; canonical home for M4.

`bot.models.vol` re-exports everything from this package for backwards
compatibility.
"""
from .vol import VolEstimator, annualised_vol_from_closes, _Window, EPS

__all__ = ["VolEstimator", "annualised_vol_from_closes", "_Window", "EPS"]