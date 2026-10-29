"""Next-day excess-return labels from Kafka candles (same definition as model2)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def next_day_excess(prices: pd.DataFrame, max_gap: int, clip: float) -> pd.DataFrame:
    p = prices[["date", "ticker", "close"]].copy()
    p["date"] = pd.to_datetime(p["date"])
    p = p.sort_values(["ticker", "date"])
    g = p.groupby("ticker")
    next_c, next_d = g["close"].shift(-1), g["date"].shift(-1)
    ok = (next_d - p["date"]).dt.days <= max_gap        # no label across sandbox-loop jumps
    r = np.where(ok, np.log(next_c / p["close"]), np.nan)
    p["r_next"] = np.clip(r, -clip, clip)
    p["y_bps"] = (p["r_next"] - p.groupby("date")["r_next"].transform("mean")) * 1e4
    p["date"] = p["date"].dt.strftime("%Y-%m-%d")
    return p[["date", "ticker", "y_bps"]].dropna()
