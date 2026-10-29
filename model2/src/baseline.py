"""Baselines model2 must beat, on the same val/test ticker-days.

  python src/baseline.py

  zero_shot_all       mean over all items of FinBERT (p_pos - p_neg)
  zero_shot_relevant  same, using only items that mention the company (falls back to all items)
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from train import evaluate

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    cfg = yaml.safe_load(open(ROOT / "configs" / "model2.yaml"))
    df = pd.read_parquet(ROOT / cfg["data"]["dataset"])
    if not df["has_teacher"].any():
        raise SystemExit("No teacher sentiment in dataset — run src/prepare.py without --no-teacher first.")
    df["s"] = df["t_pos"] - df["t_neg"]
    res = {}
    for split, name in ((1, "val"), (2, "test")):
        d = df[df["split"] == split]
        rows = []
        for (date, t), g in d.groupby(["date", "ticker"]):
            rel = g[g["relevant"] == 1]
            rows.append({"date": date, "ticker": t, "y_ret": g["y_ret"].iloc[0],
                         "all": g["s"].mean(), "relevant": (rel if len(rel) else g)["s"].mean()})
        r = pd.DataFrame(rows)
        res[name] = {"zero_shot_all": evaluate(r.assign(pred=r["all"])),
                     "zero_shot_relevant": evaluate(r.assign(pred=r["relevant"]))}
    out = ROOT / cfg["export"]["out_dir"] / "baseline_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
