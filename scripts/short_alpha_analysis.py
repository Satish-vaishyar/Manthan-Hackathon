#!/usr/bin/env python
"""scripts/short_alpha_analysis.py — Short Alpha diagnostics (P4-01).

Answers the two gating questions from the short-engine spec BEFORE any
short config is enabled:

  Step 5: does the ensemble score (both tails) predict forward returns?
          -> bucket table for S_long ranges (+0.75.. and -0.75..).
  Step 2: does a bearish signal predict NEGATIVE forward returns net of
          costs, and at which entry threshold (0.75/0.80/0.85/0.90)?
          -> directional accuracy, avg short return, profit factor, max
             drawdown, Sharpe, net P&L per threshold.

Data: server REST — /api/v1/stocks (universe) + /stocks/{t}/history (OHLCV).
Signal reconstruction is price/volume-only (rank & momentum proxies);
historical news is NOT available from the API, so the news axis is dropped
and the remaining weights renormalised — the same graceful-degradation rule
the live engine uses. Components are cross-sectionally demeaned each day
(market-adjusted) and clipped to [-1, 1] (z/3), matching the live [-1,1]
sub-score scale so the thresholds are comparable.

Run:
    set BOT_ACCESS_TOKEN=... (or jwt.txt) && python scripts/short_alpha_analysis.py
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
import urllib.request

import numpy as np

BASE = f"http://{os.environ.get('SERVER_IP', '192.168.65.157')}:3002"
TOKEN = os.environ.get("BOT_ACCESS_TOKEN", "")
if not TOKEN:
    for p in (r"C:\Users\Samarth\AppData\Local\Temp\opencode\jwt.txt", "jwt.txt"):
        if os.path.exists(p):
            TOKEN = open(p, encoding="utf-8-sig").read().strip()
            break

HORIZONS = (1, 3, 5, 10)          # days — the bot's actual holding scale
THRESHOLDS = (0.75, 0.80, 0.85, 0.90)
COST_BPS = 25.0                    # round-trip brokerage+slippage estimate
NOTIONAL = 100_000.0               # P&L per trade at this size
W_RANK, W_MOM = 0.5, 0.5           # news unavailable historically -> renorm


def get(path: str) -> dict:
    req = urllib.request.Request(BASE + path,
                                 headers={"Authorization": "Bearer " + TOKEN})
    return json.load(urllib.request.urlopen(req, timeout=30))


def fetch_closes() -> tuple[list[str], dict[str, np.ndarray], dict[str, np.ndarray]]:
    stocks = get("/api/v1/stocks")
    items = stocks.get("stocks", stocks) if isinstance(stocks, dict) else stocks
    tickers = [it["ticker"] for it in items if isinstance(it, dict) and it.get("ticker")]
    closes, vols = {}, {}
    for t in sorted(tickers):
        try:
            raw = get(f"/api/v1/stocks/{t}/history?days=280")
        except Exception:
            continue
        rows = raw if isinstance(raw, list) else next(
            (raw[k] for k in ("prices", "history", "data", "candles")
             if isinstance(raw, dict) and isinstance(raw.get(k), list)), [])
        dates, c, v = [], [], []
        for r in rows:
            if not isinstance(r, dict):
                continue
            try:
                px = float(r.get("close") or r.get("c") or 0)
                vv = float(r.get("volume") or r.get("v") or 0)
            except (TypeError, ValueError):
                continue
            if px > 0:
                dates.append(r.get("date") or r.get("simulated_date"))
                c.append(px)
                v.append(vv)
        if len(c) >= 60:
            closes[t] = np.array(c)
            vols[t] = np.array(v)
        time.sleep(0.05)  # stay under 100 req/min
    return sorted(closes), closes, vols


def zclip(x: np.ndarray) -> np.ndarray:
    """Cross-sectional z-score clipped to [-1, 1] (z/3) -> live score scale."""
    mu, sd = np.nanmean(x), np.nanstd(x)
    if sd <= 1e-12:
        return np.zeros_like(x)
    return np.clip((x - mu) / sd / 3.0, -1.0, 1.0)


def main() -> None:
    tickers, closes, vols = fetch_closes()
    print(f"universe: {len(tickers)} tickers with >=60d history")
    n_days = min(len(c) for c in closes.values())
    P = np.array([closes[t][:n_days] for t in tickers])   # T x N
    V = np.array([vols[t][:n_days] for t in tickers])
    ret1 = np.diff(P, axis=1) / P[:, :-1]                  # T-1 x N daily rets

    # --- reconstruct components (market-adjusted, [-1,1]) ---------------
    def roll_ret(k: int) -> np.ndarray:
        out = np.full_like(P, np.nan)
        out[:, k:] = P[:, k:] / P[:, :-k] - 1.0
        return out

    mom5, mom10, mom20 = roll_ret(5), roll_ret(10), roll_ret(20)
    M_raw = (mom5 + mom10 + mom20) / 3.0                    # momentum proxy
    vtrend = np.full_like(P, np.nan)                        # volume trend proxy
    w = 10
    for i in range(w, P.shape[1]):
        vtrend[:, i] = np.nanmean(V[:, i - w:i], axis=1) / (np.nanmean(V[:, :i], axis=1) + 1e-9)
    R_raw = 0.6 * mom20 + 0.4 * (vtrend - 1.0)              # rank proxy
    # cross-sectionally demean + clip per day
    M = np.apply_along_axis(lambda r: zclip(np.where(np.isfinite(r), r, 0.0)), 0, M_raw)
    R = np.apply_along_axis(lambda r: zclip(np.where(np.isfinite(r), r, 0.0)), 0, R_raw)
    S_long = W_RANK * R + W_MOM * M                          # in [-1, 1]
    S_short = -S_long                                        # bearish mirror

    # --- forward returns per horizon ------------------------------------
    fwd = {h: np.full_like(P, np.nan) for h in HORIZONS}
    for h in HORIZONS:
        fwd[h][:, :-h] = P[:, h:] / P[:, :-h] - 1.0

    valid = np.isfinite(S_long) & (np.arange(P.shape[1]) >= 21)[None, :]

    # --- Step 5: bucket table (both tails), 5d horizon ------------------
    print("\n== Step 5: ensemble score vs 5d forward return (bucket means) ==")
    print(f"{'bucket':>16} {'n':>6} {'dir_long_ok':>11} {'avg_fwd':>9} {'med_fwd':>9}")
    buckets = [(-1.01, -0.95), (-0.95, -0.85), (-0.85, -0.75),
               (-0.75, -0.5), (-0.5, 0.5),
               (0.5, 0.75), (0.75, 0.85), (0.85, 0.95), (0.95, 1.01)]
    f5 = fwd[5]
    for lo, hi in buckets:
        mask = valid & (S_long >= lo) & (S_long < hi) & np.isfinite(f5)
        n = int(mask.sum())
        if n == 0:
            print(f"{f'[{lo:+.2f},{hi:+.2f})':>16} {0:>6}")
            continue
        r = f5[mask]
        print(f"{f'[{lo:+.2f},{hi:+.2f})':>16} {n:>6} {(r > 0).mean():>11.1%} "
              f"{np.mean(r):>9.2%} {np.median(r):>9.2%}")

    # --- Step 2: short-signal quality per threshold ----------------------
    print("\n== Step 2: SHORT signal quality (entry when S_short >= thr) ==")
    print(f"costs assumed: {COST_BPS:.0f} bps round-trip | P&L at Rs {NOTIONAL:,.0f}/trade")
    hdr = f"{'thr':>5} {'h':>3} {'n':>6} {'dir_acc':>8} {'avg_net':>8} {'PF':>6} {'maxDD':>7} {'Sharpe':>7} {'net_P&L':>10}"
    print(hdr)
    best = None
    for thr in THRESHOLDS:
        for h in HORIZONS:
            f_h = fwd[h]
            mask = valid & (S_short >= thr) & np.isfinite(f_h)
            n = int(mask.sum())
            if n < 30:
                continue
            gross = -f_h[mask]                                # short gross
            net = gross - COST_BPS / 1e4
            dir_acc = float((net > 0).mean())
            avg_net = float(net.mean())
            wins, losses = net[net > 0].sum(), -net[net <= 0].sum()
            pf = float(wins / losses) if losses > 0 else float("inf")
            # per-signal equity curve (chronological, equal weight per day)
            daily = np.full(P.shape[1], np.nan)
            for d in range(P.shape[1]):
                m = mask[:, d]
                if m.any():
                    daily[d] = net[m].mean()
            daily = daily[np.isfinite(daily)]
            eq = np.cumprod(1.0 + daily / max(h, 1))
            peak = np.maximum.accumulate(eq)
            max_dd = float(np.max(1.0 - eq / peak)) if len(eq) else 0.0
            sharpe = float(np.nanmean(daily) / (np.nanstd(daily) + 1e-12)
                           * math.sqrt(252 / h)) if len(daily) > 5 else 0.0
            pnl = avg_net * NOTIONAL * n
            print(f"{thr:>5.2f} {h:>3} {n:>6} {dir_acc:>8.1%} {avg_net:>8.2%} "
                  f"{pf:>6.2f} {max_dd:>7.1%} {sharpe:>7.2f} {pnl:>10,.0f}")
            if best is None or (pnl > best[0] and max_dd < 0.25):
                best = (pnl, thr, h, dir_acc, avg_net, pf, max_dd, sharpe, n)

    print("\n== VERDICT ==")
    if best and best[4] > 0 and best[5] > 1.0:
        _, thr, h, da, an, pf, dd, sh, n = best
        print(f"bearish tail IS predictive: best thr={thr:.2f} h={h}d "
              f"(n={n}, dir {da:.1%}, avg net {an:.2%}, PF {pf:.2f}, "
              f"maxDD {dd:.1%}, Sharpe {sh:.2f})")
        print(f"recommend: enable short_engine (mode: hedge first), "
              f"entry_threshold: {thr:.2f}")
    else:
        print("bearish tail does NOT predict declines net of costs.")
        print("recommend: keep short_engine disabled (mode: off) — "
              "a convincing-looking negative score is not a trade.")


if __name__ == "__main__":
    sys.exit(main())
