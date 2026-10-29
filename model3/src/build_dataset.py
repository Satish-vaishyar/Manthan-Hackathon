"""Build the dense model3 tensors from data/raw/prices.csv.

Output data/processed/dataset.npz:
  X      [D, N, F] float32  momentum features (0 where missing)
  valid  [D, N]    bool     candle present and all features defined
  r_next [D, N]    float32  next-day log return (close D -> close D+1), winsorised
  sigma  [D, N]    float32  ex-ante daily volatility (EWMA)
  y_trend, m_trend [D, N]   1 if the next `trend_horizon`-day return > 0, and its availability mask
  dates, tickers, feature_names, split [D] (0 train, 1 val, 2 test, -1 embargo)

  python src/build_dataset.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from features import FEATURE_NAMES, build_feature_frame

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "model3.yaml"))
    cfg = yaml.safe_load(open(ap.parse_args().config))["data"]

    prices = pd.read_csv(ROOT / cfg["raw_prices"], parse_dates=["date"])[["date", "ticker", "close"]]
    feats = build_feature_frame(prices, cfg["vol_span"])

    prices = prices.sort_values(["ticker", "date"])
    lc = np.log(prices["close"])
    g = lc.groupby(prices["ticker"])
    lab = prices[["date", "ticker"]].copy()
    lab["r_next"] = (g.shift(-1) - lc).clip(-cfg["label_clip"], cfg["label_clip"])
    lab["r_trend"] = g.shift(-cfg["trend_horizon"]) - lc

    dates = np.array(sorted(feats["date"].unique()))
    tickers = np.array(sorted(feats["ticker"].unique()))
    D, N, F = len(dates), len(tickers), len(FEATURE_NAMES)
    di, ti = pd.Series(np.arange(D), index=dates), pd.Series(np.arange(N), index=tickers)

    def dense(frame: pd.DataFrame, col: str) -> np.ndarray:
        a = np.full((D, N), np.nan)
        a[di[frame["date"]].values, ti[frame["ticker"]].values] = frame[col].values
        return a

    X = np.full((D, N, F), np.nan)
    X[di[feats["date"]].values, ti[feats["ticker"]].values] = feats[FEATURE_NAMES].values
    sigma = dense(feats, "sigma")
    r_next, r_trend = dense(lab, "r_next"), dense(lab, "r_trend")
    # A day is usable once the slow features exist (nret_252 needs a year of history).
    valid = ~np.isnan(X).any(-1) & ~np.isnan(sigma) & (sigma > 0)

    d = pd.to_datetime(dates)
    split = np.full(D, -1, dtype=np.int8)
    tr_end, va_end, te_end = (pd.Timestamp(cfg[k]) for k in ("train_end", "val_end", "test_end"))
    split[d <= tr_end] = 0
    split[(d > tr_end) & (d <= va_end)] = 1
    split[(d > va_end) & (d <= te_end)] = 2
    for boundary in (tr_end, va_end):
        first = int(np.argmax(d > boundary))
        split[first:first + cfg["embargo_days"]] = -1

    out = ROOT / cfg["processed"]
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, X=np.nan_to_num(X).astype(np.float32), valid=valid,
        r_next=np.nan_to_num(r_next).astype(np.float32), m_next=~np.isnan(r_next) & valid,
        sigma=np.nan_to_num(sigma, nan=1.0).astype(np.float32),
        y_trend=(np.nan_to_num(r_trend) > 0).astype(np.float32), m_trend=~np.isnan(r_trend) & valid,
        dates=np.array([str(x)[:10] for x in dates]), tickers=tickers, feature_names=np.array(FEATURE_NAMES),
        split=split,
    )
    print(f"dataset: D={D} days, N={N} tickers, F={F} features -> {out}")
    for s, name in ((0, "train"), (1, "val"), (2, "test")):
        m = split == s
        print(f"  {name:5s}: {m.sum():4d} days {d[m].min().date()} -> {d[m].max().date()}, "
              f"usable ticker-days={int(valid[m].sum())}")
    print(f"  first usable day per ticker needs ~252 days of history; usable share overall: {valid.mean():.1%}")


if __name__ == "__main__":
    main()
