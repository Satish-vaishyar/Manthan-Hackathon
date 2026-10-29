"""Baseline to beat: LightGBM on day-D features only (no sequence), same splits and metrics.

  python src/baseline_lgbm.py --config configs/model1.yaml

model1 is shipped only if its test IC beats this baseline (see TRAINING_PROMPT.md, acceptance criteria).
Uses the GPU build of LightGBM if available (device="gpu"), otherwise CPU.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml

from train import evaluate

ROOT = Path(__file__).resolve().parents[1]


def frame(z, split: int) -> pd.DataFrame:
    d_idx, t_idx = np.where((z["split"][:, None] == split) & z["valid"] & z["ymask1"])
    df = pd.DataFrame(z["X"][d_idx, t_idx], columns=list(z["feature_names"]))
    df["date"], df["ticker"] = z["dates"][d_idx], z["tickers"][t_idx]
    df["y1"], df["y5"] = z["y1"][d_idx, t_idx], z["y5"][d_idx, t_idx]
    df["target"] = df.groupby("date")["y1"].rank(pct=True) - 0.5
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "model1.yaml"))
    cfg = yaml.safe_load(open(ap.parse_args().config))
    z = np.load(ROOT / cfg["data"]["processed"])
    feats = list(z["feature_names"])
    tr, va, te = frame(z, 0), frame(z, 1), frame(z, 2)
    params = dict(objective="regression", learning_rate=0.03, num_leaves=15, max_depth=4,
                  min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                  reg_lambda=1.0, n_estimators=2000, verbose=-1)
    try:
        model = lgb.LGBMRegressor(device="gpu", **params)
        model.fit(tr[feats].iloc[:200], tr["target"].iloc[:200])
        print("LightGBM: GPU")
    except Exception:
        model = lgb.LGBMRegressor(**params)
        print("LightGBM: CPU")
    model.fit(tr[feats], tr["target"], eval_set=[(va[feats], va["target"])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
    res = {}
    for name, df in (("val", va), ("test", te)):
        df["pred1"] = model.predict(df[feats])
        res[name] = evaluate(df, "pred1", "y1", cfg["eval"]["top_k"])
    out = ROOT / cfg["export"]["out_dir"] / "baseline_lgbm_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"), indent=2)
    imp = sorted(zip(feats, model.feature_importances_), key=lambda t: -t[1])[:10]
    print(json.dumps(res, indent=2)); print("top features:", imp)


if __name__ == "__main__":
    main()
