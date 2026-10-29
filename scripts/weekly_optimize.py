"""scripts/weekly_optimize.py — one-command weekly weight re-optimization.

Runs (in-process, reusing scripts/optimize_weights.py):
  a) the pooled optimization            -> configs/optimized_weights.json
  b) the by-regime optimization         -> configs/optimized_weights_by_regime.json
  c) the markdown report                -> runs/_weight_optimization.md
  d) appends a dated section to runs/_weight_history.md tracking weights,
     n_days and IC IRs over time.

Idempotent: re-running on the same day replaces that day's history section
instead of duplicating it.

Usage:
    python scripts/weekly_optimize.py [--boot 1000] [--date YYYY-MM-DD]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import optimize_weights as ow  # noqa: E402

HISTORY_PATH = os.path.join("runs", "_weight_history.md")


def history_section(
    date_str: str,
    pooled,
    by_reg: dict[str, dict],
) -> str:
    """Markdown for one dated history entry."""
    lines = [
        f"## {date_str}",
        "",
        f"- n_days (pooled): **{pooled.n_days}**",
        "",
        "| bucket | signal | weight | IC IR |",
        "|---|---|---|---|",
    ]
    for l in pooled.labels:
        lines.append(f"| pooled | {l} | **{pooled.weights[l]:.3f}** "
                     f"| {pooled.ic_ir.get(l, float('nan')):.3f} |")
    for r in ow.REGIMES:
        v = by_reg[r]
        e = v["estimate"]
        tag = " (pooled fallback)" if v["fallback"] else ""
        for l in e["signals"]:
            lines.append(f"| {r}{tag} | {l} | **{e['weights'][l]:.3f}** "
                         f"| {e['ic_ir'].get(l, float('nan')):.3f} |")
    lines.append("")
    return "\n".join(lines)


def upsert_history(path: str, date_str: str, section: str) -> None:
    """Replace the `## <date_str>` section in the history file, or append it."""
    existing = ""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            existing = fh.read()
    header = "# Weight optimization history\n"
    body = existing[len(header):] if existing.startswith(header) else existing
    # Remove any previous section for this date (up to the next ## or EOF).
    pat = re.compile(rf"^## {re.escape(date_str)}\n.*?(?=^## |\Z)",
                     re.M | re.S)
    body = pat.sub("", body).rstrip() + "\n\n" if body.strip() else ""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(header + "\n" + body + section)


def main() -> int:
    """CLI: pooled + by-regime optimization, JSONs, report, history upsert."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--min-days", type=int, default=5)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--date", default=dt.date.today().isoformat(),
                    help="history entry date (default: today)")
    ap.add_argument("--history", default=HISTORY_PATH)
    args = ap.parse_args()

    dates, tickers, S, P, n_lines = ow.load_panels(args.runs)
    print(f"loaded {n_lines} journal records | {len(dates)} sim days x {len(tickers)} tickers")
    if len(dates) < args.min_days:
        print(f"not enough sim days ({len(dates)} < {args.min_days}); aborting")
        return 1
    F = ow.forward_returns(P)

    # (a) pooled
    pooled = ow.optimize(S, F, n_boot=args.boot, gamma=args.gamma)
    pooled_payload = ow.estimate_payload(pooled)
    out_pooled = os.path.join("configs", "optimized_weights.json")
    with open(out_pooled, "w", encoding="utf-8") as fh:
        json.dump(pooled_payload, fh, indent=2)
    print(f"wrote {out_pooled}")
    print("pooled weights:", {k: round(v, 4) for k, v in pooled.weights.items()})

    # (b) by-regime
    regime_map = ow.load_regimes(args.runs)
    by_reg = ow.optimize_by_regime(dates, S, F, regime_map, pooled,
                                   n_boot=args.boot, gamma=args.gamma)
    out_reg = os.path.join("configs", "optimized_weights_by_regime.json")
    with open(out_reg, "w", encoding="utf-8") as fh:
        json.dump({r: v["estimate"] | {"fallback": v["fallback"],
                                       "bucket_days": v["bucket_days"]}
                   for r, v in by_reg.items()}, fh, indent=2)
    print(f"wrote {out_reg}")

    # (c) refresh the main report with both sections (reuse ow.main's writer
    # by invoking its code path via argv would reload panels; instead we do it
    # inline, cheaply).
    report = os.path.join("runs", "_weight_optimization.md")
    lines = ["# Ensemble weight optimization (offline, Monte Carlo + Markowitz-on-IC)",
             "", f"_weekly run {args.date}_", "",
             "## Pooled estimate — final (shrunk) weights", ""]
    lines += ow._report_lines(pooled, len(tickers), args.boot)
    lines += ["", "## Regime-conditional weights", ""]
    for r in ow.REGIMES:
        v = by_reg[r]
        e = v["estimate"]
        tag = "**fallback to pooled**" if v["fallback"] else "own estimate"
        lines += [f"### {r} — {v['bucket_days']} days ({tag})", "",
                  "| signal | weight | IC mean | IC IR |", "|---|---|---|---|"]
        for l in e["signals"]:
            lines.append(f"| {l} | **{e['weights'][l]:.3f}** "
                         f"| {e['ic_mean'].get(l, float('nan')):.4f} "
                         f"| {e['ic_ir'].get(l, float('nan')):.3f} |")
        lines.append("")
    with open(report, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {report}")

    # (d) history upsert (idempotent per date)
    upsert_history(args.history, args.date, history_section(args.date, pooled, by_reg))
    print(f"updated {args.history} (section {args.date})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
