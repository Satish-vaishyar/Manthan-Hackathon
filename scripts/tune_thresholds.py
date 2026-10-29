"""scripts/tune_thresholds.py — walk-forward grid search of strategy thresholds.

Uses the journal score/price panels as a proxy backtest. Grid:
  - event_long_threshold in {0.5, 0.6, 0.7, 0.75, 0.8}
  - rebalance_threshold  in {0.025, 0.05, 0.08}
  - top_k                in {4, 6, 8}

Proxy strategy: each sim day, among tickers whose `ensemble` score clears
event_long_threshold, hold the top_k by score, weighted by inverse
20-day return volatility. A position is only adjusted when its target weight
differs from the current weight by more than rebalance_threshold. Cost model:
₹35 per round trip, charged as ₹17.5 per entry/exit side against a ₹10L
notional. NaN price moves count as 0 return (position carried flat).

Walk-forward: fit on the first 60% of sim days, validate on the last 40%
(--fit-frac). Writes runs/_threshold_tuning.md with the in-sample vs
out-of-sample table and a recommendation. Pure offline analysis; numpy-only.

Usage:
    python scripts/tune_thresholds.py [--fit-frac 0.6]
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import optimize_weights as ow  # noqa: E402

EVENT_LONG = [0.5, 0.6, 0.7, 0.75, 0.8]
REBALANCE = [0.025, 0.05, 0.08]
TOP_K = [4, 6, 8]
NOTIONAL = 1_000_000.0
ROUND_TRIP_COST = 35.0
VOL_WIN = 20


# ----------------------------------------------------------------------
# Pure helpers (unit-tested)
# ----------------------------------------------------------------------
def walk_forward_split(n_days: int, fit_frac: float = 0.6) -> tuple[np.ndarray, np.ndarray]:
    """(fit_idx, val_idx): contiguous index arrays, fit first fit_frac of days."""
    if not 0.0 < fit_frac < 1.0:
        raise ValueError("fit_frac must be in (0, 1)")
    if n_days < 2:
        raise ValueError("need at least 2 days")
    cut = min(max(int(round(n_days * fit_frac)), 1), n_days - 1)
    return np.arange(cut), np.arange(cut, n_days)


def rolling_inv_vol(rets: np.ndarray, d: int, win: int = VOL_WIN) -> np.ndarray:
    """Inverse-volatility weights over tickers using rets[:d] (last win days).

    rets: (D-1, T) daily returns. Returns (T,) weights summing to 1 (0 for
    tickers with <5 observations or zero variance); all-zero if unusable.
    """
    lo = max(0, d - win)
    hist = rets[lo:d]  # (<=win, T)
    T = rets.shape[1]
    w = np.zeros(T)
    if hist.shape[0] < 5:
        return w
    valid = np.isfinite(hist).sum(axis=0) >= 5
    sd = np.nanstd(np.where(np.isfinite(hist), hist, np.nan), axis=0, ddof=1)
    ok = valid & np.isfinite(sd) & (sd > 0)
    if not ok.any():
        return w
    iv = np.where(ok, 1.0 / np.where(sd > 0, sd, 1.0), 0.0)
    s = iv.sum()
    return iv / s if s > 0 else w


def cross_sectional_percentile(ens: np.ndarray) -> np.ndarray:
    """Per-day cross-sectional percentile rank (0..1) of scores, NaN-safe.

    The journals' raw ensemble scores almost never exceed 0.5 (99th pct
    ~= 0.24), so absolute thresholds would keep the proxy portfolio flat.
    Ranking each day's cross-section maps scores onto the threshold grid.
    """
    from bot.portfolio.weight_optimizer import _rankdata
    ens = np.asarray(ens, dtype=float)
    out = np.full_like(ens, np.nan)
    for d in range(ens.shape[0]):
        row = ens[d]
        m = np.isfinite(row)
        n = int(m.sum())
        if n:
            out[d, m] = (_rankdata(row[m]) - 1.0) / max(n - 1.0, 1.0)
    return out


def simulate_portfolio(
    ens: np.ndarray,
    P: np.ndarray,
    event_thr: float,
    reb_thr: float,
    top_k: int,
    day_idx: np.ndarray,
) -> dict:
    """Daily-rebalanced inverse-vol top_k portfolio over the given days.

    Returns dict(net_return, n_trades, n_active_days). Costs: ₹17.5 per
    entry/exit side (₹35 round trip) against NOTIONAL.
    """
    D, T = P.shape
    rets = np.full((max(D - 1, 0), T), np.nan)
    if D > 1:
        rets[:] = P[1:] / P[:-1] - 1.0
    equity = 1.0
    w = np.zeros(T)
    n_sides = 0
    n_active = 0
    for d in day_idx:
        if d >= D - 1:
            break
        scores = ens[d]
        elig = np.isfinite(scores) & (scores >= event_thr)
        target = np.zeros(T)
        if elig.any():
            cand = np.where(elig)[0]
            order = cand[np.argsort(-scores[cand], kind="mergesort")][:top_k]
            iv = rolling_inv_vol(rets, d)
            sub = iv[order]
            if sub.sum() > 0:
                target[order] = sub / sub.sum()
        new_w = np.where(np.abs(target - w) > reb_thr, target, w)
        entries = (w <= 0) & (new_w > 0)
        exits = (w > 0) & (new_w <= 0)
        n_sides += int(entries.sum() + exits.sum())
        r = np.where(np.isfinite(rets[d]), rets[d], 0.0)
        day_ret = float(new_w @ r)
        day_cost = (entries.sum() + exits.sum()) * (ROUND_TRIP_COST / 2.0) / NOTIONAL
        equity *= max(1.0 + day_ret - day_cost, 1e-9)
        if new_w.sum() > 0:
            n_active += 1
            grown = new_w * (1.0 + r)
            s = grown.sum()
            w = grown / s if s > 0 else np.zeros(T)
        else:
            w = np.zeros(T)
    return {"net_return": equity - 1.0, "n_trades": n_sides // 2,
            "n_active_days": n_active}


# ----------------------------------------------------------------------
# Grid search
# ----------------------------------------------------------------------
def main() -> int:
    """CLI: grid-search thresholds with walk-forward split -> report."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--fit-frac", type=float, default=0.6)
    ap.add_argument("--report", default=os.path.join("runs", "_threshold_tuning.md"))
    ap.add_argument("--raw", action="store_true",
                    help="threshold raw ensemble scores instead of per-day percentiles")
    args = ap.parse_args()

    dates, tickers, S, P, n_lines = ow.load_panels(args.runs, signals=["ensemble"])
    ens_raw = S["ensemble"]
    print(f"loaded {n_lines} records | {len(dates)} days x {len(tickers)} tickers "
          f"| ensemble coverage {np.isfinite(ens_raw).mean():.1%}")
    ens = ens_raw if args.raw else cross_sectional_percentile(ens_raw)
    if not args.raw:
        print("thresholds applied to per-day cross-sectional percentiles of the "
              "ensemble score (raw scores rarely exceed 0.5 — 99th pct "
              f"{np.nanpercentile(ens_raw, 99):.2f})")
    fit_idx, val_idx = walk_forward_split(len(dates), args.fit_frac)
    # Skip a volatility warm-up at the very start of the fit window.
    fit_idx = fit_idx[fit_idx >= VOL_WIN]
    print(f"walk-forward: fit days {fit_idx.size} [{dates[fit_idx[0]]}..{dates[fit_idx[-1]]}], "
          f"validate {val_idx.size} [{dates[val_idx[0]]}..{dates[val_idx[-1]]}]")

    rows = []
    for thr, reb, k in itertools.product(EVENT_LONG, REBALANCE, TOP_K):
        is_r = simulate_portfolio(ens, P, thr, reb, k, fit_idx)
        oos_r = simulate_portfolio(ens, P, thr, reb, k, val_idx)
        rows.append({"event": thr, "reb": reb, "top_k": k,
                     "is_ret": is_r["net_return"], "oos_ret": oos_r["net_return"],
                     "is_trades": is_r["n_trades"], "oos_trades": oos_r["n_trades"],
                     "oos_active": oos_r["n_active_days"]})
        print(f"thr={thr:.2f} reb={reb:.3f} k={k}: IS {is_r['net_return']:+.2%}  "
              f"OOS {oos_r['net_return']:+.2%}  (trades {is_r['n_trades']}/{oos_r['n_trades']})")

    best_is = max(rows, key=lambda r: r["is_ret"])
    best_oos = max(rows, key=lambda r: r["oos_ret"])
    rows_sorted = sorted(rows, key=lambda r: r["oos_ret"], reverse=True)

    lines = [
        "# Threshold walk-forward re-optimization (proxy backtest)",
        "",
        f"- proxy: top_k inverse-vol portfolio, daily rebalance, ₹35 round-trip "
        f"per trade on ₹{NOTIONAL:,.0f} notional",
        "- thresholds applied to per-day cross-sectional percentiles of the "
        "journal ensemble score (raw scores 99th pct "
        f"{np.nanpercentile(ens_raw, 99):.2f} would leave the book flat)" if not args.raw else
        "- thresholds applied to raw journal ensemble scores (--raw)",
        f"- fit: {fit_idx.size} days ({dates[fit_idx[0]]}..{dates[fit_idx[-1]]}); "
        f"validate: {val_idx.size} days ({dates[val_idx[0]]}..{dates[val_idx[-1]]})",
        "",
        "## Grid results (sorted by out-of-sample net return)",
        "",
        "| event_long_thr | rebalance_thr | top_k | IS net ret | OOS net ret | IS trades | OOS trades | OOS active days |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows_sorted:
        lines.append(
            f"| {r['event']:.2f} | {r['reb']:.3f} | {r['top_k']} "
            f"| {r['is_ret']:+.2%} | {r['oos_ret']:+.2%} "
            f"| {r['is_trades']} | {r['oos_trades']} | {r['oos_active']} |"
        )
    lines += [
        "",
        "## Recommendation",
        "",
        f"- Best in-sample: event_long_threshold={best_is['event']:.2f}, "
        f"rebalance_threshold={best_is['reb']:.3f}, top_k={best_is['top_k']} "
        f"(IS {best_is['is_ret']:+.2%}, OOS {best_is['oos_ret']:+.2%})",
        f"- Best out-of-sample: event_long_threshold={best_oos['event']:.2f}, "
        f"rebalance_threshold={best_oos['reb']:.3f}, top_k={best_oos['top_k']} "
        f"(IS {best_oos['is_ret']:+.2%}, OOS {best_oos['oos_ret']:+.2%})",
        "",
        "Prefer the best-OOS triple when it is also strong IS; large IS→OOS "
        "decay for the best-IS triple indicates overfitting. This is a proxy "
        "backtest (no event mechanics, fills or gap risk), so confirm any "
        "config change in the sandbox before adopting.",
    ]
    with open(args.report, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {args.report}")
    print(f"best IS : {best_is}")
    print(f"best OOS: {best_oos}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
