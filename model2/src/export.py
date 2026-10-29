"""Package trained seeds for the bot and measure CPU latency.

  python src/export.py

Writes artifacts/model2_seed<k>.pt (fine-tuned layers + heads; the frozen FinBERT layers are re-loaded from the
Hugging Face cache) and artifacts/model2_meta.json. Copy the FinBERT cache folder to the bot laptop too
(see TRAINING_PROMPT.md) because the venue network may have no internet.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import pandas as pd
import torch
import yaml

from infer import FEATURES, NewsScorer

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    cfg = yaml.safe_load(open(ROOT / "configs" / "model2.yaml"))
    art = ROOT / cfg["export"]["out_dir"]
    members = []
    for ck in sorted(art.glob("seed_*/best.pt")):
        st = torch.load(ck, map_location="cpu", weights_only=True)
        name = f"model2_seed{st['seed']}.pt"
        shutil.copy(ck, art / name)
        members.append({"seed": st["seed"], "pt": name, "val": st["val"]})
    if not members:
        raise SystemExit("no artifacts/seed_*/best.pt found — run src/train.py first")
    meta = {"model": "model2-NRIM", "encoder": cfg["model"]["encoder"], "max_items": cfg["text"]["max_items_per_day"],
            "max_len": cfg["text"]["max_len"], "desc_chars": cfg["text"]["desc_chars"], "features": FEATURES,
            "members": members}
    json.dump(meta, open(art / "model2_meta.json", "w"), indent=2)

    # Latency check on a realistic day: 10 tickers with real headlines.
    news = pd.read_csv(ROOT / cfg["data"]["news_csv"])
    day = news[news["date"] == news["date"].value_counts().index[min(1, news["date"].nunique() - 1)]]
    items = day[["ticker", "headline", "description"]].to_dict("records")
    tickers = sorted(news["ticker"].unique())
    from text import load_aliases
    aliases = load_aliases(ROOT / cfg["data"]["aliases_auto"], ROOT / cfg["data"]["aliases_manual"])
    results = {}
    for q in (False, True):
        scorer = NewsScorer(art, quantize=q)
        scorer.score_day(items, {}, tickers, aliases)          # warm-up
        t0 = time.perf_counter()
        results[q] = scorer.score_day(items, {}, tickers, aliases)
        meta[f"cpu_ms_day_{'int8' if q else 'fp32'}"] = round((time.perf_counter() - t0) * 1000, 1)
    diff = max(abs(a - b) for t in results[False] for a, b in zip(results[False][t], results[True][t]))
    meta["int8_max_abs_diff"] = round(diff, 4)
    meta["cpu_ms_day_items"] = len(items)
    json.dump(meta, open(art / "model2_meta.json", "w"), indent=2)
    print(f"{len(members)} member(s); CPU latency for a day with {len(items)} items: "
          f"fp32 {meta['cpu_ms_day_fp32']:.0f} ms, int8 {meta['cpu_ms_day_int8']:.0f} ms; "
          f"int8 max abs diff {diff:.3f} (use int8 in the bot only if this is < 0.05)")
    print("example output:", {t: [round(x, 3) for x in v] for t, v in list(results[True].items())[:3]})


if __name__ == "__main__":
    main()
