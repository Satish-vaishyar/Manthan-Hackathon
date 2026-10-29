"""Classic momentum baselines on the same backtest (the bar model3 must beat).

  python src/baselines.py

  buy_and_hold      pos = 1 for every ticker (volatility-targeted)
  tsmom_12m         pos = sign(12-month return)            (Moskowitz, Ooi & Pedersen)
  tsmom_12m_long    pos = max(sign(12-month return), 0)
  macd_combo        pos = mean of phi(Y) over 3 MACD speeds  (Baz et al.), phi(y) = y*exp(-y^2/4)/0.89
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

import backtest
from dataset import load

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    cfg = yaml.safe_load(open(ROOT / "configs" / "model3.yaml"))
    tc = cfg["train"]
    z = load(str(ROOT / cfg["data"]["processed"]))
    names = list(z["feature_names"])
    X = z["X"]
    signals = {
        "buy_and_hold": np.ones(X.shape[:2]),
        "tsmom_12m": np.sign(X[..., names.index("nret_252")]),
        "tsmom_12m_long": np.clip(np.sign(X[..., names.index("nret_252")]), 0, None),
        "macd_combo": np.mean([y * np.exp(-y ** 2 / 4) / 0.89 for y in
                               (X[..., names.index(c)] for c in ("macd_8_24", "macd_16_48", "macd_32_96"))], axis=0),
    }
    res = {}
    for split, name in ((1, "val"), (2, "test")):
        days = np.where(z["split"] == split)[0]
        mask = z["valid"][days] & z["m_next"][days]
        res[name] = {k: backtest.run(s[days], z["r_next"][days], z["sigma"][days], mask,
                                     sigma_target=tc["sigma_target"], max_leverage=tc["max_leverage"],
                                     cost_bps=tc["cost_bps"]) for k, s in signals.items()}
    out = ROOT / cfg["export"]["out_dir"] / "baseline_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"), indent=2)
    for name, r in res.items():
        print(name, {k: round(v["sharpe"], 2) for k, v in r.items()})


if __name__ == "__main__":
    main()
