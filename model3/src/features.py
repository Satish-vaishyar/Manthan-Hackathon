"""Momentum / trend features (ticker-agnostic, volatility-normalised).

Every value is a return divided by volatility, a ratio or a same-day market statistic, so the model can be
trained on the 40 REST tickers and used on any live ticker. The bot must use this same file at inference.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-9
RET_HORIZONS = (1, 5, 21, 63, 126, 252)
MACD_PAIRS = ((8, 24), (16, 48), (32, 96))

FEATURE_NAMES = (
    [f"nret_{k}" for k in RET_HORIZONS]                       # r(t-k, t) / (sigma_t * sqrt(k))
    + [f"macd_{s}_{l}" for s, l in MACD_PAIRS]                 # Baz et al. normalised MACD
    + ["cpd_mean", "cpd_vol", "dd_63", "du_63"]                # regime-change (changepoint-lite) features
    + ["mkt_nret_21", "mkt_nret_63", "breadth_63"]             # same-day market context
)


def ticker_features(close: pd.Series, vol_span: int) -> tuple[pd.DataFrame, pd.Series]:
    """Features for one ticker (close sorted by date). Returns (features, sigma_t daily)."""
    lc = np.log(close.astype(float))
    r = lc.diff()
    sigma = r.ewm(span=vol_span, min_periods=20).std()
    f = pd.DataFrame(index=close.index)
    for k in RET_HORIZONS:
        f[f"nret_{k}"] = lc.diff(k) / (sigma * np.sqrt(k) + EPS)
    for s, l in MACD_PAIRS:
        m = close.ewm(span=s, adjust=False).mean() - close.ewm(span=l, adjust=False).mean()
        q = m / (close.rolling(63, min_periods=20).std() + EPS)
        f[f"macd_{s}_{l}"] = q / (q.rolling(252, min_periods=63).std() + EPS)
    # Changepoint-lite: is the recent regime different from the longer one?
    m21, m126 = r.rolling(21).mean(), r.rolling(126, min_periods=63).mean()
    s126 = r.rolling(126, min_periods=63).std()
    f["cpd_mean"] = (m21 - m126) / (s126 / np.sqrt(21) + EPS)
    f["cpd_vol"] = np.log((r.rolling(21).std() + EPS) / (s126 + EPS))
    f["dd_63"] = (lc - lc.rolling(63, min_periods=20).max()) / (sigma * np.sqrt(63) + EPS)
    f["du_63"] = (lc - lc.rolling(63, min_periods=20).min()) / (sigma * np.sqrt(63) + EPS)
    return f, sigma


def build_feature_frame(prices: pd.DataFrame, vol_span: int = 60) -> pd.DataFrame:
    """Long frame (date, ticker, sigma, FEATURE_NAMES...) from long OHLCV prices."""
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
    parts = []
    for _, g in prices.groupby("ticker", sort=False):
        f, sigma = ticker_features(g["close"], vol_span)
        f["sigma"] = sigma
        parts.append(f)
    out = pd.concat([prices[["date", "ticker"]], pd.concat(parts).loc[prices.index]], axis=1)
    # Same-day market context (equal-weight across tickers present that day).
    for k in (21, 63):
        out[f"mkt_nret_{k}"] = out.groupby("date")[f"nret_{k}"].transform("mean")
    out["breadth_63"] = out.groupby("date")["nret_63"].transform(lambda s: (s > 0).mean()) - 0.5
    out[FEATURE_NAMES] = out[FEATURE_NAMES].clip(-5, 5)
    return out[["date", "ticker", "sigma"] + FEATURE_NAMES]
