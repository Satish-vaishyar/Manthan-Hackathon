"""Build the dense training tensor for model1 from data/raw/prices.csv.

Output: data/processed/dataset.npz with
  X        [D, N, F] float32  robust-scaled features (0 where missing)
  valid    [D, N]    bool     ticker has a candle that day (and r1 is defined)
  y1, y5   [D, N]    float32  forward excess log return over 1 / 5 days (close D -> close D+h)
  yvol     [D, N]    float32  log mean |r| over the next 5 days (volatility target)
  ymask1, ymask5 [D, N] bool  label available
  ymkt     [D]       float32  1 if equal-weight market return over next 5 days > 0
  dates [D], tickers [N], feature_names [F], split [D] (0 train, 1 val, 2 test, -1 embargo/unused)
and data/processed/scaler.json (median / IQR per feature, fitted on train dates only).

Usage:  python src/build_dataset.py --config configs/model1.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from features import FEATURE_NAMES, build_feature_frame

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "model1.yaml"))
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))["data"]

    prices = pd.read_csv(ROOT / cfg["raw_prices"], parse_dates=["date"])
    prices = prices[["date", "ticker", "open", "high", "low", "close", "volume"]]
    feats = build_feature_frame(prices)

    # Forward labels per ticker (log returns from the close we would fill at).
    prices = prices.sort_values(["ticker", "date"])
    lc = np.log(prices["close"])
    r1 = lc.groupby(prices["ticker"]).diff()
    clip = cfg["label_clip"]
    r1c = r1.clip(-clip, clip)
    grp = r1c.groupby(prices["ticker"])
    lab = prices[["date", "ticker"]].copy()
    lab["f1"] = grp.shift(-1)
    lab["f5"] = sum(grp.shift(-k) for k in range(1, 6))
    lab["fvol"] = np.log(sum(grp.shift(-k).abs() for k in range(1, 6)) / 5 + 1e-4)

    dates = np.array(sorted(feats["date"].unique()))
    tickers = np.array(sorted(feats["ticker"].unique()))
    D, N, F = len(dates), len(tickers), len(FEATURE_NAMES)
    di = pd.Series(np.arange(D), index=dates)
    ti = pd.Series(np.arange(N), index=tickers)

    def dense(frame: pd.DataFrame, col: str) -> np.ndarray:
        a = np.full((D, N), np.nan, dtype=np.float64)
        a[di[frame["date"]].values, ti[frame["ticker"]].values] = frame[col].values
        return a

    X = np.full((D, N, F), np.nan, dtype=np.float64)
    rows, cols = di[feats["date"]].values, ti[feats["ticker"]].values
    X[rows, cols, :] = feats[FEATURE_NAMES].values
    valid = ~np.isnan(X[:, :, FEATURE_NAMES.index("r1")])

    f1, f5, fvol = dense(lab, "f1"), dense(lab, "f5"), dense(lab, "fvol")
    ymask1 = valid & ~np.isnan(f1)
    ymask5 = valid & ~np.isnan(f5)
    # Excess returns: subtract the same-day cross-sectional mean of labelled tickers.
    y1 = f1 - np.nanmean(np.where(ymask1, f1, np.nan), axis=1, keepdims=True)
    y5 = f5 - np.nanmean(np.where(ymask5, f5, np.nan), axis=1, keepdims=True)
    mkt5 = np.nanmean(np.where(ymask5, f5, np.nan), axis=1)
    ymkt = (mkt5 > 0).astype(np.float32)

    # Splits with embargo.
    d = pd.to_datetime(dates)
    split = np.full(D, -1, dtype=np.int8)
    tr_end, va_end, te_end = (pd.Timestamp(cfg[k]) for k in ("train_end", "val_end", "test_end"))
    emb = cfg["embargo_days"]
    split[d <= tr_end] = 0
    split[(d > tr_end) & (d <= va_end)] = 1
    split[(d > va_end) & (d <= te_end)] = 2
    for boundary in (tr_end, va_end):
        first = np.argmax(d > boundary)
        split[first:first + emb] = -1

    # Robust scaler fitted on train rows only.
    tr = X[split == 0][valid[split == 0]]
    med = np.nanmedian(tr, axis=0)
    iqr = np.nanpercentile(tr, 75, axis=0) - np.nanpercentile(tr, 25, axis=0)
    iqr = np.where(iqr < 1e-8, 1.0, iqr)
    Xs = np.clip((X - med) / iqr, -5, 5)
    Xs = np.nan_to_num(Xs, nan=0.0).astype(np.float32)
    Xs[~valid] = 0.0

    out = ROOT / cfg["processed"]
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, X=Xs, valid=valid,
        y1=np.nan_to_num(y1).astype(np.float32), y5=np.nan_to_num(y5).astype(np.float32),
        yvol=np.nan_to_num(fvol).astype(np.float32), ymask1=ymask1, ymask5=ymask5, ymkt=ymkt,
        dates=np.array([str(x)[:10] for x in dates]), tickers=tickers,
        feature_names=np.array(FEATURE_NAMES), split=split,
    )
    json.dump({"features": FEATURE_NAMES, "median": med.tolist(), "iqr": iqr.tolist(), "clip": 5.0},
              open(out.parent / "scaler.json", "w"), indent=2)

    print(f"dataset: D={D} days, N={N} tickers, F={F} features -> {out}")
    for s, name in ((0, "train"), (1, "val"), (2, "test")):
        m = split == s
        print(f"  {name:5s}: {m.sum():4d} days {str(d[m].min().date()) if m.any() else '-'} -> "
              f"{str(d[m].max().date()) if m.any() else '-'}, labelled ticker-days={int(ymask1[m].sum())}")
    print(f"  NaN feature share among valid rows: {np.isnan(X[valid]).mean():.3%}")


if __name__ == "__main__":
    main()
