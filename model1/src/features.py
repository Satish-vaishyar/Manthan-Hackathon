"""Ticker-agnostic daily features from OHLCV.

Every feature is scale-free (log returns, ratios, z-scores), so the model can be trained on the
40 REST tickers and applied to whatever tickers the live Kafka stream sends. The same function must
be used at inference time inside the bot; FEATURE_NAMES fixes the column order.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-9

FEATURE_NAMES = [
    "r1", "gap", "intraday", "hl_range", "clv", "upper_wick", "lower_wick",
    "vol_z20", "vol_ratio_5_20",
    "r5", "r10", "r20", "r60",
    "rv20", "rv60", "rv_ratio",
    "dist_sma20", "dist_sma50", "dist_sma200",
    "rsi14", "bb_pctb20", "macd_hist_n", "atr14_pct",
    "dist_252_high", "dist_252_low",
    "mkt_r1", "excess_r1", "cs_rank_r20",
]


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / (dn + EPS))


def ticker_features(g: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker features. `g` is one ticker's OHLCV sorted by date."""
    o, h, l, c, v = (g[k].astype(float) for k in ("open", "high", "low", "close", "volume"))
    lc = np.log(c)
    r1 = lc.diff()
    f = pd.DataFrame(index=g.index)
    f["r1"] = r1
    f["gap"] = np.log(o / c.shift(1))
    f["intraday"] = np.log(c / o)
    rng = (h - l).clip(lower=EPS)
    f["hl_range"] = np.log(h / l)
    f["clv"] = (2 * c - h - l) / rng
    f["upper_wick"] = (h - np.maximum(o, c)) / rng
    f["lower_wick"] = (np.minimum(o, c) - l) / rng
    lv = np.log1p(v)
    f["vol_z20"] = (lv - lv.rolling(20).mean()) / (lv.rolling(20).std() + EPS)
    f["vol_ratio_5_20"] = np.log((v.rolling(5).mean() + 1) / (v.rolling(20).mean() + 1))
    for n in (5, 10, 20, 60):
        f[f"r{n}"] = lc.diff(n)
    f["rv20"] = r1.rolling(20).std()
    f["rv60"] = r1.rolling(60).std()
    f["rv_ratio"] = np.log((f["rv20"] + EPS) / (f["rv60"] + EPS))
    for n in (20, 50, 200):
        f[f"dist_sma{n}"] = lc - np.log(c.rolling(n, min_periods=max(10, n // 2)).mean())
    f["rsi14"] = _rsi(c) / 100.0 - 0.5
    m20, s20 = c.rolling(20).mean(), c.rolling(20).std()
    f["bb_pctb20"] = (c - (m20 - 2 * s20)) / (4 * s20 + EPS) - 0.5
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    hist = macd - macd.ewm(span=9, adjust=False).mean()
    f["macd_hist_n"] = hist / (atr + EPS)
    f["atr14_pct"] = atr / c
    f["dist_252_high"] = lc - np.log(h.rolling(252, min_periods=60).max())
    f["dist_252_low"] = lc - np.log(l.rolling(252, min_periods=60).min())
    return f


def build_feature_frame(prices: pd.DataFrame) -> pd.DataFrame:
    """Long frame (date, ticker, FEATURE_NAMES...) from long OHLCV prices."""
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
    parts = []
    for _, g in prices.groupby("ticker", sort=False):
        parts.append(ticker_features(g))
    feats = pd.concat(parts).loc[prices.index]
    out = pd.concat([prices[["date", "ticker"]], feats], axis=1)
    # Cross-sectional (same-day) context: uses only day-D information, no lookahead.
    out["mkt_r1"] = out.groupby("date")["r1"].transform("mean")
    out["excess_r1"] = out["r1"] - out["mkt_r1"]
    out["cs_rank_r20"] = out.groupby("date")["r20"].rank(pct=True) - 0.5
    return out[["date", "ticker"] + FEATURE_NAMES]
