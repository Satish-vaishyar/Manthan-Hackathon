"""Prove that model1 never learns from stock symbols.

  python src/check_no_ticker.py

Checks:
  1. No identity column (ticker/symbol/name/sector/date) is among the model's input features.
  2. Renaming every ticker to a random code produces bit-identical feature values.
  3. The processed dataset fed to the model contains only numeric feature/label arrays; symbols are kept
     only as a lookup for reports.
  4. The model has no embedding layer (nothing that could store a per-ticker identity).
  5. The model is permutation-equivariant across tickers: shuffling ticker order shuffles outputs the same way
     and leaves each ticker's prediction unchanged, so slot position carries no identity either.
Exits with code 1 if any check fails.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from features import FEATURE_NAMES, build_feature_frame
from model import CSTT

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = ("ticker", "symbol", "name", "sector", "date", "description", "active", "id")


def main() -> int:
    ok = True

    bad = [f for f in FEATURE_NAMES if any(w in f.lower() for w in FORBIDDEN)]
    print(f"[1] identity columns in FEATURE_NAMES: {bad or 'none'}")
    ok &= not bad

    prices = pd.read_csv(ROOT / "data" / "raw" / "prices.csv", parse_dates=["date"])
    prices = prices[["date", "ticker", "open", "high", "low", "close", "volume"]]
    rng = np.random.default_rng(0)
    codes = {t: f"T{rng.integers(1e9):09d}" for t in prices["ticker"].unique()}
    a = build_feature_frame(prices)
    renamed = prices.assign(ticker=prices["ticker"].map(codes))
    b = build_feature_frame(renamed)
    b["ticker"] = b["ticker"].map({v: k for k, v in codes.items()})
    a = a.sort_values(["date", "ticker"]).reset_index(drop=True)
    b = b.sort_values(["date", "ticker"]).reset_index(drop=True)
    same = np.allclose(a[FEATURE_NAMES].to_numpy(), b[FEATURE_NAMES].to_numpy(), equal_nan=True, rtol=0, atol=0)
    print(f"[2] features identical after renaming all tickers: {same}")
    ok &= same

    z = np.load(ROOT / "data" / "processed" / "dataset.npz")
    model_inputs = {"X": z["X"].dtype, "valid": z["valid"].dtype}
    numeric = all(np.issubdtype(dt, np.number) or dt == bool for dt in model_inputs.values())
    print(f"[3] model input arrays {model_inputs} numeric-only: {numeric}; "
          f"X has {z['X'].shape[-1]} channels = {len(FEATURE_NAMES)} features")
    ok &= numeric and z["X"].shape[-1] == len(FEATURE_NAMES)

    m = CSTT(n_features=len(FEATURE_NAMES), window=64, dropout=0.0).eval()
    emb = [n for n, mod in m.named_modules() if isinstance(mod, (torch.nn.Embedding, torch.nn.EmbeddingBag))]
    print(f"[4] embedding layers in model: {emb or 'none'}")
    ok &= not emb

    torch.manual_seed(0)
    x = torch.randn(2, 12, 64, len(FEATURE_NAMES))
    tm = torch.ones(2, 12, 64, dtype=torch.bool)
    nm = torch.ones(2, 12, dtype=torch.bool)
    perm = torch.randperm(12)
    with torch.no_grad():
        o1 = m(x, tm, nm)["ret1"]
        o2 = m(x[:, perm], tm[:, perm], nm[:, perm])["ret1"]
    equi = torch.allclose(o1[:, perm], o2, atol=1e-5)
    print(f"[5] permutation-equivariant across tickers: {equi} "
          f"(max diff {float((o1[:, perm] - o2).abs().max()):.2e})")
    ok &= equi

    print("PASS: model1 does not use stock symbols" if ok else "FAIL: see checks above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
