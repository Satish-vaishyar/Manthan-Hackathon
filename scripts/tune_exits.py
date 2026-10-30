"""scripts/tune_exits.py — exit-strategy tournament over the live tradebook.

Replays every completed BUY->SELL round trip (server orderbook, FIFO-matched
by tune_stops.match_round_trips) against the daily close panel and asks:
"what would this trade have done under exit strategy X?"

Walk-forward convention (matches the live bot): the strategy is evaluated on
each day's CLOSE after entry, up to and including the recorded exit day; if
a rule triggers, the trade exits AT THAT CLOSE (not at an optimistic stop
price). If nothing triggers, the recorded outcome stands. This is the same
close-based execution the live stepped trailing SL uses, so results map 1:1
to production behaviour.

Strategies tested:
  recorded            — the bot's actual exits (baseline)
  sl5 / sl10          — fixed stop at -5% / -10% from entry
  atr2.0 / atr3.5     — stop at entry - mult*ATR14(entry)
  tp10 / tp15 / tp20  — flat take-profit at +10/15/20%
  tp10_sl10           — both flat rules, whichever hits first
  trail5 / trail10    — continuous trail: stop = peak*(1-x) (min entry*0.9)
  stepped             — user spec: -10%; +3% gain -> BE; +5% -> +3%; 2% steps

Usage:
    python scripts/tune_exits.py [--orderbook runs/orderbook_full_....json]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import optimize_weights as ow  # noqa: E402
from tune_stops import match_round_trips, atr_from_prices, ATR_PERIOD  # noqa: E402


# ----------------------------------------------------------------------
# Exit simulators: walk closes[entry+1 .. exit], return (exit_px, exit_idx)
# or None (no trigger -> recorded outcome stands).
# ----------------------------------------------------------------------
def sim_flat_stop(closes, i0, i1, entry_px, pct):
    stop = entry_px * (1 - pct)
    for d in range(i0 + 1, i1 + 1):
        c = closes[d]
        if np.isfinite(c) and c <= stop:
            return c, d
    return None


def sim_atr_stop(closes, atrs, i0, i1, entry_px, mult):
    a = atrs[i0]
    if not np.isfinite(a):
        return None
    stop = entry_px - mult * a
    for d in range(i0 + 1, i1 + 1):
        c = closes[d]
        if np.isfinite(c) and c <= stop:
            return c, d
    return None


def sim_flat_tp(closes, i0, i1, entry_px, pct):
    tgt = entry_px * (1 + pct)
    for d in range(i0 + 1, i1 + 1):
        c = closes[d]
        if np.isfinite(c) and c >= tgt:
            return c, d
    return None


def sim_tp_sl(closes, i0, i1, entry_px, tp, sl):
    tgt, stop = entry_px * (1 + tp), entry_px * (1 - sl)
    for d in range(i0 + 1, i1 + 1):
        c = closes[d]
        if not np.isfinite(c):
            continue
        if c >= tgt or c <= stop:
            return c, d
    return None


def sim_trail(closes, i0, i1, entry_px, trail_pct, floor_pct=0.10):
    peak = entry_px
    floor_stop = entry_px * (1 - floor_pct)
    for d in range(i0 + 1, i1 + 1):
        c = closes[d]
        if not np.isfinite(c):
            continue
        peak = max(peak, c)
        stop = floor_stop
        if peak > entry_px:
            stop = max(stop, peak * (1 - trail_pct))
        if c <= stop:
            return c, d
    return None


def sim_stepped(closes, i0, i1, entry_px,
                init=0.10, be=0.03, lock_start=0.05, first_lock=0.03, step=0.02):
    peak = entry_px
    for d in range(i0 + 1, i1 + 1):
        c = closes[d]
        if not np.isfinite(c):
            continue
        peak = max(peak, c)
        gain = (peak - entry_px) / entry_px
        if gain >= lock_start:
            locked = first_lock + step * math.floor((gain - lock_start) / step)
            stop = entry_px * (1 + locked)
        elif gain >= be:
            stop = entry_px
        else:
            stop = entry_px * (1 - init)
        if c <= stop:
            return c, d
    return None


# ----------------------------------------------------------------------
# Tournament
# ----------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--orderbook", default=None)
    ap.add_argument("--report", default=os.path.join("runs", "_exit_tournament.md"))
    args = ap.parse_args()

    ob = args.orderbook
    if ob is None:
        cands = sorted(glob.glob(os.path.join(args.runs, "orderbook_full_*.json")))
        if not cands:
            # fall back to a fresh pull if no snapshot exists
            print("no orderbook_full_*.json; pull one first"); return 1
        ob = cands[-1]
    with open(ob, "r", encoding="utf-8-sig") as fh:
        book = json.load(fh)
    orders = book["orders"] if isinstance(book, dict) else book
    trips = match_round_trips(orders)
    print(f"{len(orders)} orders -> {len(trips)} round trips")

    dates, tickers, S, P, _ = ow.load_panels(args.runs)
    di = {d: i for i, d in enumerate(dates)}
    ti = {t: j for j, t in enumerate(tickers)}
    ATR = atr_from_prices(P, ATR_PERIOD)

    usable = []
    for tr in trips:
        i0, i1 = di.get(tr.entry_date), di.get(tr.exit_date)
        j = ti.get(tr.ticker)
        if i0 is None or i1 is None or j is None or i1 <= i0:
            continue
        usable.append((tr, i0, i1, j))
    print(f"{len(usable)} trips mappable to the price panel")
    if not usable:
        return 1

    strategies = {
        # Partial profit booking: sell `frac` of the position at +pct (exit at
        # that close), remainder follows the recorded exit. Blends "book
        # profits" with "let winners run" — tested per user request.
        "p50@10":   ("partial", 0.50, 0.10),
        "p50@15":   ("partial", 0.50, 0.15),
        "p33@10":   ("partial", 0.33, 0.10),
        "p25@8":    ("partial", 0.25, 0.08),
        "sl5":      lambda c, a, i0, i1, e: sim_flat_stop(c, i0, i1, e, 0.05),
        "sl10":     lambda c, a, i0, i1, e: sim_flat_stop(c, i0, i1, e, 0.10),
        # Disaster stops (user request 2026-10-30): only cap the -19K-class
        # tail outliers, leave normal score-decay exits untouched.
        "sl12":     lambda c, a, i0, i1, e: sim_flat_stop(c, i0, i1, e, 0.12),
        "sl15":     lambda c, a, i0, i1, e: sim_flat_stop(c, i0, i1, e, 0.15),
        "sl20":     lambda c, a, i0, i1, e: sim_flat_stop(c, i0, i1, e, 0.20),
        "atr2.0":   lambda c, a, i0, i1, e: sim_atr_stop(c, a, i0, i1, e, 2.0),
        "atr3.5":   lambda c, a, i0, i1, e: sim_atr_stop(c, a, i0, i1, e, 3.5),
        "tp10":     lambda c, a, i0, i1, e: sim_flat_tp(c, i0, i1, e, 0.10),
        "tp15":     lambda c, a, i0, i1, e: sim_flat_tp(c, i0, i1, e, 0.15),
        "tp20":     lambda c, a, i0, i1, e: sim_flat_tp(c, i0, i1, e, 0.20),
        "tp10_sl10": lambda c, a, i0, i1, e: sim_tp_sl(c, i0, i1, e, 0.10, 0.10),
        "trail5":   lambda c, a, i0, i1, e: sim_trail(c, i0, i1, e, 0.05),
        "trail10":  lambda c, a, i0, i1, e: sim_trail(c, i0, i1, e, 0.10),
        "stepped":  lambda c, a, i0, i1, e: sim_stepped(c, i0, i1, e),
    }

    def stats(pnls, n_trig, sim_days):
        n = len(pnls)
        wins = [x for x in pnls if x > 0]
        losses = [x for x in pnls if x <= 0]
        return {
            "total": sum(pnls),
            "win": len(wins) / n * 100 if n else 0.0,
            "avg_win": sum(wins) / len(wins) if wins else 0.0,
            "avg_loss": sum(losses) / len(losses) if losses else 0.0,
            "worst": min(pnls) if pnls else 0.0,
            "triggered": n_trig,
            "avg_days": (sum(sim_days) / n) if n else 0.0,
        }

    rows = {}
    # Baseline: recorded exits.
    base_pnls = [tr.pnl for tr, *_ in usable]
    base_days = [tr.hold_days for tr, *_ in usable]
    rows["recorded"] = stats(base_pnls, 0, base_days)

    for name, fn in strategies.items():
        pnls, trig, days = [], 0, []
        if isinstance(fn, tuple) and fn[0] == "partial":
            _, frac, pct = fn
            for tr, i0, i1, j in usable:
                closes = P[:, j]
                hit = sim_flat_tp(closes, i0, i1, tr.entry_price, pct)
                if hit is not None and np.isfinite(hit[0]):
                    tp_px, _d = hit
                    # frac sold at the TP close; rest at the recorded exit.
                    pnl = (frac * tr.qty * (tp_px - tr.entry_price)
                           + (1 - frac) * tr.qty * (tr.exit_price - tr.entry_price)
                           - tr.charges)
                    pnls.append(pnl)
                    trig += 1
                else:
                    pnls.append(tr.pnl)
                days.append(tr.hold_days)
            rows[name] = stats(pnls, trig, days)
            continue
        for tr, i0, i1, j in usable:
            closes, atrs = P[:, j], ATR[:, j]
            hit = fn(closes, atrs, i0, i1, tr.entry_price)
            if hit is not None and np.isfinite(hit[0]):
                px, d = hit
                pnls.append((px - tr.entry_price) * tr.qty - tr.charges)
                trig += 1
                import datetime as dt
                days.append(max((dt.date.fromisoformat(dates[d])
                                 - dt.date.fromisoformat(tr.entry_date)).days, 0))
            else:
                pnls.append(tr.pnl)
                days.append(tr.hold_days)
        rows[name] = stats(pnls, trig, days)

    # ---- report ----
    lines = ["# Exit-strategy tournament (close-based replay of the live tradebook)",
             "",
             f"trips: {len(usable)}   panel: {len(dates)} days x {len(tickers)} tickers",
             "",
             "| strategy | total P&L (Rs) | vs recorded | win % | avg win | avg loss | worst | triggered | avg hold d |",
             "|---|---|---|---|---|---|---|---|---|"]
    base_total = rows["recorded"]["total"]
    for name in ["recorded"] + list(strategies):
        r = rows[name]
        lines.append(
            f"| {name} | {r['total']:,.0f} | {r['total']-base_total:+,.0f} | "
            f"{r['win']:.1f} | {r['avg_win']:,.0f} | {r['avg_loss']:,.0f} | "
            f"{r['worst']:,.0f} | {r['triggered']} | {r['avg_days']:.1f} |")
    best = max((k for k in rows if k != "recorded"),
               key=lambda k: rows[k]["total"])
    lines += ["",
              f"**Best by total P&L: `{best}`** "
              f"({rows[best]['total'] - base_total:+,.0f} vs recorded)."]
    report = "\n".join(lines)
    with open(args.report, "w", encoding="utf-8") as fh:
        fh.write(report + "\n")
    print(report)
    print(f"\nwrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
