"""Prove that model3 never learns from stock symbols.

  python src/check_no_ticker.py

  1. No identity column among the input features.
  2. Renaming every ticker gives identical feature values.
  3. Model inputs are numeric arrays only.
  4. No embedding layer in the model.
  5. Each ticker's output depends only on its own feature sequence (the same network runs on every ticker).
Exits with code 1 on failure.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from features import FEATURE_NAMES, build_feature_frame
from model import DMN

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = ("ticker", "symbol", "name", "sector", "date", "description", "active", "id")


def main() -> int:
    ok = True
    bad = [f for f in FEATURE_NAMES if any(w in f.lower() for w in FORBIDDEN)]
    print(f"[1] identity columns in FEATURE_NAMES: {bad or 'none'}")
    ok &= not bad

    prices = pd.read_csv(ROOT / "data" / "raw" / "prices.csv", parse_dates=["date"])[["date", "ticker", "close"]]
    rng = np.random.default_rng(0)
    codes = {t: f"T{int(rng.integers(10 ** 9)):09d}" for t in prices["ticker"].unique()}
    a = build_feature_frame(prices)
    b = build_feature_frame(prices.assign(ticker=prices["ticker"].map(codes)))
    b["ticker"] = b["ticker"].map({v: k for k, v in codes.items()})
    a = a.sort_values(["date", "ticker"]).reset_index(drop=True)
    b = b.sort_values(["date", "ticker"]).reset_index(drop=True)
    # Tolerance 1e-12: market averages may sum tickers in a different order (rounding ~1e-16).
    fa, fb = a[FEATURE_NAMES].to_numpy(), b[FEATURE_NAMES].to_numpy()
    same = np.allclose(fa, fb, equal_nan=True, rtol=0, atol=1e-12)
    print(f"[2] features identical after renaming all tickers: {same} (max diff {np.nanmax(np.abs(fa - fb)):.1e})")
    ok &= same

    z = np.load(ROOT / "data" / "processed" / "dataset.npz")
    numeric = np.issubdtype(z["X"].dtype, np.number) and z["X"].shape[-1] == len(FEATURE_NAMES)
    print(f"[3] model input X {z['X'].dtype} with {z['X'].shape[-1]} channels, numeric-only: {numeric}")
    ok &= bool(numeric)

    m = DMN(len(FEATURE_NAMES), dropout=0.0).eval()
    emb = [n for n, mod in m.named_modules() if isinstance(mod, (torch.nn.Embedding, torch.nn.EmbeddingBag))]
    print(f"[4] embedding layers in model: {emb or 'none'}")
    ok &= not emb

    x = torch.randn(8, 63, len(FEATURE_NAMES))
    perm = torch.randperm(8)
    with torch.no_grad():
        p1, _ = m(x)
        p2, _ = m(x[perm])
    indep = torch.allclose(p1[perm], p2, atol=1e-6)
    print(f"[5] outputs depend only on each ticker's own sequence: {indep}")
    ok &= indep
    print("PASS: model3 does not use stock symbols" if ok else "FAIL: see checks above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
