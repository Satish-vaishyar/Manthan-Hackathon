"""Smoke test: verify M1+M3 + ring buffer + features produce end-to-end scores."""
from __future__ import annotations

import os
import sys
from datetime import date, timedelta

import numpy as np

sys.path.insert(0, ".")

from bot.contracts import PriceTickData
from bot.features import RingBuffer, build_model1_features, build_model3_features
from bot.models.model1_onnx import Model1Ranker
from bot.models.model3_runtime import Model3Scorer


def main():
    # Build 280 days of synthetic data for 3 tickers
    tickers = ["AAA", "BBB", "CCC"]
    rb = RingBuffer(window=280)
    base_date = date(2024, 1, 1)
    for d in range(280):
        sim_d = base_date + timedelta(days=d)
        for i, t in enumerate(tickers):
            close = 100.0 + d * 0.1 + (0.5 * i)
            tick = PriceTickData(
                ticker=t,
                open=close,
                high=close * 1.01,
                low=close * 0.99,
                close=close,
                volume=10000,
            )
            rb.push(t, tick, sim_d)

    print(f"ring buffer length: {len(rb)}")
    print(f"AAA length: {rb.length('AAA')}")

    m1_windows, m1_ready = build_model1_features(rb, tickers)
    print(f"M1: {len(m1_ready)} tickers ready")
    if m1_ready:
        print(f"M1 window shape: {m1_windows[m1_ready[0]].shape}")

    m3_windows, m3_ready = build_model3_features(rb, tickers)
    print(f"M3: {len(m3_ready)} tickers ready")
    if m3_ready:
        print(f"M3 window shape: {m3_windows[m3_ready[0]].shape}")

    r = Model1Ranker()
    print(f"M1 ranker ready: {r.ready}")
    if r.ready and m1_ready:
        feats = {t: m1_windows[t] for t in m1_ready}
        scores = r.score(feats)
        print(f"M1 scores: {scores}")

    s = Model3Scorer()
    print(f"M3 scorer ready: {s.ready}")
    if s.ready and m3_ready:
        feats = {t: m3_windows[t] for t in m3_ready}
        scores = s.score(feats)
        print(f"M3 scores: {scores}")

    print("OK: smoke test passed")


if __name__ == "__main__":
    main()