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


def sim_underwater_ts(closes, i0, i1, entry_px, days, band=0.0):
    """Underwater time-stop: on day `days` after entry, if the close is below
    entry*(1-band), exit at that close. Caps PATIENCE on dead money — winners
    at day N are already positive and untouched (unlike flat stops, which cap
    recovery). Tradebook 2026-10-30: >10%-loss bucket holds 34d avg, twice
    the time winners need to reach +10% (24d)."""
    d = i0 + int(days)
    if d > i1:
        return None
    c = closes[d]
    if np.isfinite(c) and c < entry_px * (1 - band):
        return c, d
    return None


def sim_armed_stop(closes, i0, i1, entry_px, arm_pct, stop_pct):
    """Giveback guard: once the close reaches entry*(1+arm_pct) (i.e. a profit
    rung would have fired), arm a stop at entry*(1+stop_pct). Protects the
    REMAINDER after a rung books 25% — the PLNX 2025-11 case: +9.5% peak,
    rung sold 25%, then the 75% remainder round-tripped to -7.5% because no
    guard existed between the rung and the lagging trend exit."""
    armed = False
    stop = entry_px * (1 + stop_pct)
    for d in range(i0 + 1, i1 + 1):
        c = closes[d]
        if not np.isfinite(c):
            continue
        if not armed and c >= entry_px * (1 + arm_pct):
            armed = True
        if armed and c <= stop:
            return c, d
    return None


def sim_first(closes, i0, i1, entry_px, *fns):
    """Combine exit simulators: return the exit that fires EARLIEST."""
    best = None
    for fn in fns:
        hit = fn()
        if hit is not None and (best is None or hit[1] < best[1]):
            best = hit
    return best


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
        # Underwater time-stops (tradebook evidence 2026-10-30): big losers
        # hold 34d avg — twice what winners need. Exit dead money at day N.
        "uw15":     lambda c, a, i0, i1, e: sim_underwater_ts(c, i0, i1, e, 15),
        "uw20":     lambda c, a, i0, i1, e: sim_underwater_ts(c, i0, i1, e, 20),
        "uw25":     lambda c, a, i0, i1, e: sim_underwater_ts(c, i0, i1, e, 25),
        "uw30":     lambda c, a, i0, i1, e: sim_underwater_ts(c, i0, i1, e, 30),
        # 3% tolerance band: only exit if losing by MORE than 3% at day N
        # (ignores flat/noise positions, keeps marginal drifters).
        "uw25b3":   lambda c, a, i0, i1, e: sim_underwater_ts(c, i0, i1, e, 25, 0.03),
        "uw30b3":   lambda c, a, i0, i1, e: sim_underwater_ts(c, i0, i1, e, 30, 0.03),
        # Underwater stop + disaster floor combo: time-stop at 25d AND hard
        # -15% floor at any time — exit at WHICHEVER fires first (the floor
        # catches the crash-before-day-25 case the time-stop misses).
        "uw25_sl15": lambda c, a, i0, i1, e: sim_first(
            c, i0, i1, e,
            lambda: sim_underwater_ts(c, i0, i1, e, 25),
            lambda: sim_flat_stop(c, i0, i1, e, 0.15)),
        "uw30_sl15": lambda c, a, i0, i1, e: sim_first(
            c, i0, i1, e,
            lambda: sim_underwater_ts(c, i0, i1, e, 30),
            lambda: sim_flat_stop(c, i0, i1, e, 0.15)),
        # Giveback guards (user complaint 2026-10-30: PLNX +9.5% -> -7.5%
        # on the 75% remainder after rung 1 booked 25%). Stop arms only AFTER
        # the close reaches the first rung; until then the position is
        # untouched — unlike global trailing stops, which LOSE (trail5/10).
        "be@8":     lambda c, a, i0, i1, e: sim_armed_stop(c, i0, i1, e, 0.08, 0.00),
        "lock2@8":  lambda c, a, i0, i1, e: sim_armed_stop(c, i0, i1, e, 0.08, 0.02),
        "lock4@8":  lambda c, a, i0, i1, e: sim_armed_stop(c, i0, i1, e, 0.08, 0.04),
        # Armed at rung 2 (+14%): looser, only protects bigger winners.
        "lock4@14": lambda c, a, i0, i1, e: sim_armed_stop(c, i0, i1, e, 0.14, 0.04),
        # Armed + stepped trailing floor for the remainder.
        "trail6@8": lambda c, a, i0, i1, e: sim_armed_stop(c, i0, i1, e, 0.08, -0.02),
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
