"""scripts/optimize_weights.py — estimate optimal ensemble weights offline.

Reads every runs/*/journal.jsonl, builds:
  - score matrices  S_k[d, t]  for k in {rank, news, mom}
  - price matrix    P[d, t]    (any recorded price that day)
  - forward returns F[d, t] = P[d+1, t]/P[d, t] - 1  (per-ticker, over the
    union sim-calendar)

Then runs bot.portfolio.weight_optimizer.optimize() and writes:
  - configs/optimized_weights.json  (consumed by the bot at boot when
    strategy.weights_mode == "optimized")
  - runs/_weight_optimization.md    (human-readable report)

Usage:
    python scripts/optimize_weights.py [--boot 1000] [--min-days 5]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot.portfolio.weight_optimizer import optimize  # noqa: E402

SIGNALS = ["rank", "news", "mom"]


def load_panels(runs_dir: str):
    """Return (dates, tickers, scores{sig: (D,T)}, prices(D,T))."""
    score: dict[str, dict] = {s: {} for s in SIGNALS}  # (date,ticker)->val
    price: dict = {}
    files = sorted(glob.glob(os.path.join(runs_dir, "*", "journal.jsonl")))
    n_lines = 0
    for path in files:
        try:
            with open(path, "r", encoding="utf-8-sig") as fh:
                for line in fh:
                    line = line.strip()
                    if not line or not line.startswith("{"):
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    d = rec.get("sim_date")
                    t = rec.get("ticker")
                    if not d or not t:
                        continue
                    sc = rec.get("scores") or {}
                    for s in SIGNALS:
                        v = sc.get(s)
                        if isinstance(v, (int, float)) and (d, t) not in score[s]:
                            score[s][(d, t)] = float(v)
                    p = rec.get("price")
                    if isinstance(p, (int, float)) and p > 0 and (d, t) not in price:
                        price[(d, t)] = float(p)
                    n_lines += 1
        except OSError:
            continue
    dates = sorted({d for (d, _) in price})
    tickers = sorted({t for (_, t) in price})
    D, T = len(dates), len(tickers)
    di = {d: i for i, d in enumerate(dates)}
    ti = {t: j for j, t in enumerate(tickers)}
    P = np.full((D, T), np.nan)
    for (d, t), v in price.items():
        P[di[d], ti[t]] = v
    S = {}
    for s in SIGNALS:
        M = np.full((D, T), np.nan)
        for (d, t), v in score[s].items():
            if d in di and t in ti:
                M[di[d], ti[t]] = v
        S[s] = M
    return dates, tickers, S, P, n_lines


def forward_returns(P: np.ndarray) -> np.ndarray:
    """F[d, t] = P[d+1, t]/P[d, t] - 1 (NaN where either side missing)."""
    F = np.full_like(P, np.nan)
    F[:-1] = P[1:] / P[:-1] - 1.0
    return F


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--min-days", type=int, default=5)
    ap.add_argument("--gamma", type=float, default=1.0,
                    help="risk-aversion for the mean-variance estimator")
    ap.add_argument("--out", default=os.path.join("configs", "optimized_weights.json"))
    ap.add_argument("--report", default=os.path.join("runs", "_weight_optimization.md"))
    args = ap.parse_args()

    dates, tickers, S, P, n_lines = load_panels(args.runs)
    print(f"loaded {n_lines} journal records | {len(dates)} sim days x {len(tickers)} tickers")
    if len(dates) < args.min_days:
        print(f"not enough sim days ({len(dates)} < {args.min_days}); aborting")
        return 1
    F = forward_returns(P)
    est = optimize(S, F, n_boot=args.boot, gamma=args.gamma)

    payload = {
        "weights": est.weights,
        "n_days": est.n_days,
        "ic_mean": est.ic_mean,
        "ic_ir": est.ic_ir,
        "ci95": {k: list(v) for k, v in est.ci95.items()},
        "p_nonpositive": est.p_nonpositive,
        "estimators": est.raw,
        "signals": est.labels,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"wrote {args.out}")
    print("final weights:", {k: round(v, 4) for k, v in est.weights.items()})
    print("IC means:    ", {k: round(v, 4) for k, v in est.ic_mean.items()})
    print("IC IRs:      ", {k: round(v, 4) for k, v in est.ic_ir.items()})

    lines = [
        "# Ensemble weight optimization (offline, Monte Carlo + Markowitz-on-IC)",
        "",
        f"- sim days used: **{est.n_days}**, tickers: {len(tickers)}, bootstraps: {args.boot}",
        "",
        "## Final (shrunk) weights",
        "",
        "| signal | weight | 95% CI (MC) | P(w<=0) | IC mean | IC IR |",
        "|---|---|---|---|---|---|",
    ]
    for l in est.labels:
        lo, hi = est.ci95.get(l, (float("nan"),) * 2)
        lines.append(
            f"| {l} | **{est.weights[l]:.3f}** | [{lo:.3f}, {hi:.3f}] "
            f"| {est.p_nonpositive.get(l, float('nan')):.2f} "
            f"| {est.ic_mean.get(l, float('nan')):.4f} "
            f"| {est.ic_ir.get(l, float('nan')):.3f} |"
        )
    lines += ["", "## Raw estimators", "",
              "| estimator | " + " | ".join(est.labels) + " |",
              "|---|" + "---|" * len(est.labels)]
    for name, w in est.raw.items():
        lines.append(f"| {name} | " + " | ".join(f"{w[l]:.3f}" for l in est.labels) + " |")
    lines += ["",
              "Method: per-day cross-sectional Spearman IC of each signal vs next-day",
              "return; weights = bootstrap-mean of simplex-projected Markowitz-on-IC",
              "(w ∝ Σ⁻¹μ), shrunk toward equal weights by the empirical P(w≤0).",
              "See bot/portfolio/weight_optimizer.py docstring for the full math."]
    with open(args.report, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
