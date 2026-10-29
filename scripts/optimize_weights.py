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
    python scripts/optimize_weights.py [--boot 1000] [--min-days 5] [--by-regime]

With --by-regime the sim days are bucketed by the journal `regime` field
(risk_on / neutral / risk_off; null and unknown values count as neutral) and
one WeightEstimate per regime is written to
configs/optimized_weights_by_regime.json. Buckets with fewer than
--min-regime-days usable days fall back to the pooled estimate (noted in the
report) — P3-xx: regime labels are sparsely populated in older journals.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot.portfolio.weight_optimizer import WeightEstimate, optimize  # noqa: E402

SIGNALS = ["rank", "news", "mom"]
REGIMES = ["risk_on", "neutral", "risk_off"]
MIN_REGIME_DAYS = 30


def load_panels(runs_dir: str, signals: list[str] | None = None):
    """Return (dates, tickers, scores{sig: (D,T)}, prices(D,T), n_lines).

    `signals` defaults to SIGNALS; pass e.g. ["ensemble"] to load other
    per-ticker score fields from the journals.
    """
    sigs = signals or SIGNALS
    score: dict[str, dict] = {s: {} for s in sigs}  # (date,ticker)->val
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
                    for s in sigs:
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
    for s in sigs:
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


def load_regimes(runs_dir: str) -> dict[str, str]:
    """Return {sim_date: regime} using the modal non-null regime per day.

    null / missing / unknown values count as 'neutral' (P3-xx: older journals
    often have regime == null).
    """
    per_day: dict[str, Counter] = defaultdict(Counter)
    files = sorted(glob.glob(os.path.join(runs_dir, "*", "journal.jsonl")))
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
                    if not d:
                        continue
                    r = rec.get("regime")
                    if r not in REGIMES:
                        r = "neutral"
                    per_day[d][r] += 1
        except OSError:
            continue
    return {d: c.most_common(1)[0][0] for d, c in per_day.items()}


def regime_day_buckets(
    dates: list[str], regime_map: dict[str, str]
) -> dict[str, list[int]]:
    """Bucket day indices by regime; days without a label -> neutral."""
    buckets: dict[str, list[int]] = {r: [] for r in REGIMES}
    for i, d in enumerate(dates):
        r = regime_map.get(d, "neutral")
        if r not in buckets:
            r = "neutral"
        buckets[r].append(i)
    return buckets


def estimate_payload(est: WeightEstimate) -> dict:
    """JSON-serialisable payload for one WeightEstimate."""
    return {
        "weights": est.weights,
        "n_days": est.n_days,
        "ic_mean": est.ic_mean,
        "ic_ir": est.ic_ir,
        "ci95": {k: list(v) for k, v in est.ci95.items()},
        "p_nonpositive": est.p_nonpositive,
        "estimators": est.raw,
        "signals": est.labels,
    }


def _report_lines(est: WeightEstimate, n_tickers: int, n_boot: int) -> list[str]:
    """Markdown lines describing one WeightEstimate."""
    lines = [
        f"- sim days used: **{est.n_days}**, tickers: {n_tickers}, bootstraps: {n_boot}",
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
    if est.raw:
        lines += ["", "| estimator | " + " | ".join(est.labels) + " |",
                  "|---|" + "---|" * len(est.labels)]
        for name, w in est.raw.items():
            lines.append(f"| {name} | " + " | ".join(f"{w[l]:.3f}" for l in est.labels) + " |")
    return lines


def optimize_by_regime(
    dates: list[str],
    S: dict[str, np.ndarray],
    F: np.ndarray,
    regime_map: dict[str, str],
    pooled: WeightEstimate,
    *,
    n_boot: int = 1000,
    gamma: float = 1.0,
    min_days: int = MIN_REGIME_DAYS,
) -> dict[str, dict]:
    """One estimate per regime bucket; small buckets fall back to pooled."""
    buckets = regime_day_buckets(dates, regime_map)
    out: dict[str, dict] = {}
    for r in REGIMES:
        idx = buckets[r]
        est = None
        note = "bucket too small"
        if len(idx) >= min_days:
            sub_S = {k: v[idx] for k, v in S.items()}
            est = optimize(sub_S, F[idx], n_boot=n_boot, gamma=gamma)
            if est.n_days < 5 or not est.raw:
                # Optimizer fell back internally (e.g. a signal is flat for
                # the whole bucket, so its IC is NaN every day — P3-xx: mom
                # is constant on risk_off days in these journals).
                est = None
                note = "fewer than 5 days with ICs for all signals (flat signal?)"
        if est is not None:
            out[r] = {"estimate": estimate_payload(est), "fallback": False,
                      "bucket_days": len(idx)}
        else:
            fb = estimate_payload(pooled)
            fb["fallback_reason"] = note
            out[r] = {"estimate": fb, "fallback": True,
                      "bucket_days": len(idx)}
    return out


def main() -> int:
    """CLI: load journal panels -> forward returns -> optimize -> write
    configs/optimized_weights.json + a markdown report. Exit 1 if too few days."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--min-days", type=int, default=5)
    ap.add_argument("--gamma", type=float, default=1.0,
                    help="risk-aversion for the mean-variance estimator")
    ap.add_argument("--out", default=os.path.join("configs", "optimized_weights.json"))
    ap.add_argument("--report", default=os.path.join("runs", "_weight_optimization.md"))
    ap.add_argument("--by-regime", action="store_true",
                    help="also emit configs/optimized_weights_by_regime.json")
    ap.add_argument("--min-regime-days", type=int, default=MIN_REGIME_DAYS,
                    help="regime buckets smaller than this fall back to pooled")
    ap.add_argument("--regime-out",
                    default=os.path.join("configs", "optimized_weights_by_regime.json"))
    args = ap.parse_args()

    dates, tickers, S, P, n_lines = load_panels(args.runs)
    print(f"loaded {n_lines} journal records | {len(dates)} sim days x {len(tickers)} tickers")
    if len(dates) < args.min_days:
        print(f"not enough sim days ({len(dates)} < {args.min_days}); aborting")
        return 1
    F = forward_returns(P)
    est = optimize(S, F, n_boot=args.boot, gamma=args.gamma)

    payload = estimate_payload(est)
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
        "## Pooled estimate — final (shrunk) weights",
        "",
    ]
    lines += _report_lines(est, len(tickers), args.boot)
    lines += ["",
              "Method: per-day cross-sectional Spearman IC of each signal vs next-day",
              "return; weights = bootstrap-mean of simplex-projected Markowitz-on-IC",
              "(w ∝ Σ⁻¹μ), shrunk toward equal weights by the empirical P(w≤0).",
              "See bot/portfolio/weight_optimizer.py docstring for the full math."]

    if args.by_regime:
        regime_map = load_regimes(args.runs)
        by_reg = optimize_by_regime(
            dates, S, F, regime_map, est,
            n_boot=args.boot, gamma=args.gamma,
            min_days=args.min_regime_days,
        )
        os.makedirs(os.path.dirname(args.regime_out), exist_ok=True)
        with open(args.regime_out, "w", encoding="utf-8") as fh:
            json.dump({r: v["estimate"] | {"fallback": v["fallback"],
                                           "bucket_days": v["bucket_days"]}
                       for r, v in by_reg.items()}, fh, indent=2)
        print(f"wrote {args.regime_out}")
        lines += ["", "## Regime-conditional weights", ""]
        for r in REGIMES:
            v = by_reg[r]
            w = v["estimate"]["weights"]
            tag = ("**fallback to pooled** (bucket too small)"
                   if v["fallback"] else "own estimate")
            lines += [
                f"### {r} — {v['bucket_days']} days ({tag})",
                "",
                "| signal | weight | IC mean | IC IR |",
                "|---|---|---|---|",
            ]
            for l in SIGNALS:
                lines.append(
                    f"| {l} | **{w[l]:.3f}** "
                    f"| {v['estimate']['ic_mean'].get(l, float('nan')):.4f} "
                    f"| {v['estimate']['ic_ir'].get(l, float('nan')):.3f} |"
                )
            lines.append("")
            print(f"regime {r}: {v['bucket_days']} days, "
                  f"fallback={v['fallback']}, "
                  f"weights={{ {', '.join(f'{k}: {val:.3f}' for k, val in w.items())} }}")
    with open(args.report, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
