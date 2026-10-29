"""scripts/tune_stops.py — offline stop-loss distance tuning from the tradebook.

Loads the full order book (runs/orderbook_full_*.json) and the daily price
panel from the journals (via scripts/optimize_weights.load_panels), then for
each completed BUY -> SELL round trip simulates what the exit would have been
under stop_atr_mult in {2.0, ..., 4.5}:

    stop_price = entry_price - stop_atr_mult * ATR14(ticker, entry_day)

The trade is marked stopped on the first day after entry whose close breaches
the stop; a stopped trade exits at the stop price (optimistic fill assumption)
on that day instead of the recorded exit. If the recorded exit comes first,
the recorded outcome stands.

Note: journals only record one price per ticker/day (no OHLC), so the "true
range" is proxied by |close_d - close_{d-1}| — an ATR built on close-to-close
moves. That understates intraday range slightly, uniformly across candidates.

Ranks candidates by total net P&L and expectancy per trade, splits results by
holding-period band, and writes runs/_stop_tuning.md. Pure offline analysis —
no bot changes.

Usage:
    python scripts/tune_stops.py [--orderbook runs/orderbook_full_....json]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import optimize_weights as ow  # noqa: E402

STOP_MULTS = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
BASELINE_MULT = 3.5
ATR_PERIOD = 14
HOLD_BANDS = [("<3d scalp", 0, 2), ("3-7d", 3, 7), ("8-30d", 8, 30), (">30d", 31, 10**9)]


# ----------------------------------------------------------------------
# Pure helpers (unit-tested)
# ----------------------------------------------------------------------
def atr_from_prices(P: np.ndarray, period: int = ATR_PERIOD) -> np.ndarray:
    """Per-ticker ATR proxy from a (D, T) close-price panel.

    TR_d = |P_d - P_{d-1}| (close-to-close; no OHLC available). ATR is the
    Wilder-style running mean seeded with the simple mean of the first
    `period` TRs. Returns a (D, T) array, NaN before seed.
    """
    P = np.asarray(P, dtype=float)
    D, T = P.shape
    tr = np.abs(np.diff(P, axis=0))  # (D-1, T); tr[d] = |P[d+1]-P[d]|
    out = np.full((D, T), np.nan)
    if D - 1 < period:
        return out
    seed = np.nanmean(tr[:period], axis=0)
    out[period] = seed  # ATR aligned to day `period` (uses TRs up to it)
    for d in range(period, D - 1):
        prev = out[d]
        nxt = np.where(np.isfinite(tr[d]), (prev * (period - 1) + tr[d]) / period, prev)
        out[d + 1] = nxt
    return out


def simulate_stop_exit(
    closes: np.ndarray,
    atrs: np.ndarray,
    entry_idx: int,
    exit_idx: int,
    entry_price: float,
    mult: float,
) -> tuple[float, int, bool]:
    """Simulate one long round trip under an ATR stop.

    closes/atrs: per-ticker daily series (NaN-safe). Walks days after entry
    up to the recorded exit day; if close <= entry - mult*ATR(entry) the trade
    exits at the stop price that day. Returns (exit_price, exit_idx, stopped).
    """
    stop = entry_price - mult * atrs[entry_idx]
    if not np.isfinite(stop):
        return float("nan"), exit_idx, False
    for d in range(entry_idx + 1, min(exit_idx, closes.size - 1) + 1):
        c = closes[d]
        if np.isfinite(c) and c <= stop:
            return stop, d, True
    return float("nan"), exit_idx, False


@dataclass
class RoundTrip:
    ticker: str
    qty: float
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    pnl: float          # recorded net pnl (after charges)
    charges: float

    @property
    def hold_days(self) -> int:
        try:
            import datetime as dt
            d0 = dt.date.fromisoformat(self.entry_date)
            d1 = dt.date.fromisoformat(self.exit_date)
            return max((d1 - d0).days, 0)
        except (ValueError, TypeError):
            return 0


def match_round_trips(orders: list[dict]) -> list[RoundTrip]:
    """FIFO-match FILLED BUYs with subsequent FILLED SELLs per ticker.

    Partial overlaps match min(qty). Shorts (SELL first / COVER) are ignored —
    stop tuning here is for the long side only.
    """
    trips: list[RoundTrip] = []
    by_ticker: dict[str, list[dict]] = {}
    for o in orders:
        if o.get("status") != "FILLED":
            continue
        by_ticker.setdefault(o.get("ticker", ""), []).append(o)
    for t, ords in by_ticker.items():
        ords.sort(key=lambda o: (o.get("simulated_date") or "",
                                 o.get("executed_at") or ""))
        open_lots: list[dict] = []
        for o in ords:
            act = o.get("action")
            qty = float(o.get("quantity") or 0)
            px = float(o.get("execution_price") or 0)
            if qty <= 0 or px <= 0:
                continue
            if act == "BUY":
                open_lots.append({"qty": qty, "price": px,
                                  "date": o.get("simulated_date")})
            elif act == "SELL":
                remaining = qty
                sell_charges = float(o.get("total_charges_and_taxes") or 0)
                while remaining > 1e-9 and open_lots:
                    lot = open_lots[0]
                    take = min(lot["qty"], remaining)
                    frac = take / qty
                    gross = (px - lot["price"]) * take
                    chg = sell_charges * frac + 20.0 * (take / lot["qty"])
                    trips.append(RoundTrip(
                        ticker=t, qty=take,
                        entry_date=lot["date"], exit_date=o.get("simulated_date"),
                        entry_price=lot["price"], exit_price=px,
                        pnl=gross - chg, charges=chg))
                    lot["qty"] -= take
                    remaining -= take
                    if lot["qty"] <= 1e-9:
                        open_lots.pop(0)
    return trips


def hold_band(days: int) -> str:
    """Holding-period band label."""
    for label, lo, hi in HOLD_BANDS:
        if lo <= days <= hi:
            return label
    return HOLD_BANDS[-1][0]


# ----------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------
def main() -> int:
    """CLI: tradebook + price panel -> stop-mult sweep -> runs/_stop_tuning.md."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--orderbook", default=None,
                    help="default: newest runs/orderbook_full_*.json")
    ap.add_argument("--report", default=os.path.join("runs", "_stop_tuning.md"))
    args = ap.parse_args()

    ob_path = args.orderbook
    if ob_path is None:
        cands = sorted(glob.glob(os.path.join(args.runs, "orderbook_full_*.json")))
        if not cands:
            print("no orderbook_full_*.json found"); return 1
        ob_path = cands[-1]
    with open(ob_path, "r", encoding="utf-8-sig") as fh:
        book = json.load(fh)
    orders = book["orders"] if isinstance(book, dict) else book
    trips = match_round_trips(orders)
    print(f"loaded {len(orders)} orders -> {len(trips)} completed BUY->SELL round trips")
    if not trips:
        print("no round trips; aborting"); return 1

    dates, tickers, S, P, _ = ow.load_panels(args.runs)
    di = {d: i for i, d in enumerate(dates)}
    ti = {t: j for j, t in enumerate(tickers)}
    ATR = atr_from_prices(P, ATR_PERIOD)

    # Attach panel indices; drop trips outside the panel or without ATR.
    usable = []
    for tr in trips:
        i0, i1 = di.get(tr.entry_date), di.get(tr.exit_date)
        j = ti.get(tr.ticker)
        if i0 is None or i1 is None or j is None or i1 <= i0:
            continue
        if not np.isfinite(ATR[i0, j]):
            continue
        usable.append((tr, i0, i1, j))
    print(f"{len(usable)} round trips mappable to the price panel with ATR")
    if not usable:
        print("nothing usable; aborting"); return 1

    def recorded_pnl() -> float:
        return sum(tr.pnl for tr, *_ in usable)

    results = {}  # mult -> dict(total, expectancy, n_stopped, per_band{...})
    for mult in STOP_MULTS:
        per_band = {b[0]: {"n": 0, "pnl": 0.0, "stopped": 0} for b in HOLD_BANDS}
        total, stopped_n = 0.0, 0
        for tr, i0, i1, j in usable:
            closes, atrs = P[:, j], ATR[:, j]
            sim_px, sim_idx, stopped = simulate_stop_exit(
                closes, atrs, i0, i1, tr.entry_price, mult)
            if stopped and np.isfinite(sim_px):
                # Exit at stop price instead of recorded exit; same charges.
                pnl = (sim_px - tr.entry_price) * tr.qty - tr.charges
                stopped_n += 1
            else:
                pnl = tr.pnl
            total += pnl
            band = per_band[hold_band(tr.hold_days)]
            band["n"] += 1
            band["pnl"] += pnl
            band["stopped"] += int(stopped)
        results[mult] = {
            "total": total,
            "expectancy": total / len(usable),
            "n_stopped": stopped_n,
            "per_band": per_band,
        }

    rec_total = recorded_pnl()
    best = max(STOP_MULTS, key=lambda m: results[m]["total"])
    base = results[BASELINE_MULT]

    lines = [
        "# Stop-loss distance tuning (offline, from tradebook + journal price panel)",
        "",
        f"- round trips analysed: **{len(usable)}** (of {len(trips)} matched)",
        f"- ATR: Wilder {ATR_PERIOD}-day on close-to-close TR proxy",
        f"- recorded (as-traded) net P&L: **₹{rec_total:,.0f}**",
        "",
        "## Total net P&L by stop_atr_mult",
        "",
        "| stop_atr_mult | net P&L (₹) | vs recorded | vs 3.5 baseline | expectancy/trade (₹) | trades stopped |",
        "|---|---|---|---|---|---|",
    ]
    for m in STOP_MULTS:
        r = results[m]
        lines.append(
            f"| {m:.1f} | {r['total']:,.0f} | {r['total'] - rec_total:+,.0f} "
            f"| {r['total'] - base['total']:+,.0f} | {r['expectancy']:,.0f} "
            f"| {r['n_stopped']} |"
        )
    lines += ["", "## Split by holding-period band (net P&L ₹ / trades stopped)", ""]
    header = "| band | n | " + " | ".join(f"{m:.1f}" for m in STOP_MULTS) + " |"
    lines += [header, "|---|---|" + "---|" * len(STOP_MULTS)]
    for label, _, _ in HOLD_BANDS:
        row = [f"| {label} | {results[STOP_MULTS[0]]['per_band'][label]['n']} "]
        for m in STOP_MULTS:
            b = results[m]["per_band"][label]
            row.append(f"{b['pnl']:,.0f} ({b['stopped']}) ")
        lines.append("| " + " | ".join(c.strip() for c in row) + "|")
    lines += [
        "",
        "## Recommendation",
        "",
        f"**stop_atr_mult = {best:.1f}** maximises simulated net P&L "
        f"(₹{results[best]['total']:,.0f}; {results[best]['total'] - base['total']:+,.0f} "
        f"vs the current 3.5 baseline, {results[best]['total'] - rec_total:+,.0f} vs "
        f"as-traded). Expectancy ₹{results[best]['expectancy']:,.0f}/trade over "
        f"{len(usable)} round trips; {results[best]['n_stopped']} trades would have "
        "been stopped out. Caveats: close-to-close ATR proxy, optimistic stop fills,",
        "and no re-entry modelling — treat as directional evidence, not gospel.",
    ]
    with open(args.report, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {args.report}")
    print(f"best stop_atr_mult={best:.1f}  pnl={results[best]['total']:,.0f} "
          f"(vs 3.5: {results[best]['total'] - base['total']:+,.0f}, "
          f"vs recorded: {results[best]['total'] - rec_total:+,.0f})")
    for m in STOP_MULTS:
        r = results[m]
        print(f"  {m:.1f}: total={r['total']:>12,.0f}  exp={r['expectancy']:>9,.0f}  "
              f"stopped={r['n_stopped']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
