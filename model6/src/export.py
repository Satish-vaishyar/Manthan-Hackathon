"""Package model6 for the bot and check its speed.

  python src/export.py

Writes artifacts/model6_meta.json (taxonomy, calibrated impacts, feature names, latency). The bot loads the
tagger with EventTagger.from_files("model6") — it needs configs/ + artifacts/calibration.json + src/.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import pandas as pd
import yaml

from tagger import EventTagger
from text import load_aliases

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    cfg = yaml.safe_load(open(ROOT / "configs" / "model6.yaml"))
    dc = cfg["data"]
    art = ROOT / cfg["export"]["out_dir"]
    tagger = EventTagger.from_files(ROOT)
    news = pd.read_csv(ROOT / dc["news_csv"])
    news["description"] = news["description"].fillna("")
    tickers = sorted(news["ticker"].unique())
    aliases = load_aliases(ROOT / dc["aliases_auto"], ROOT / dc["aliases_manual"])
    busiest = news["date"].value_counts().index[0]
    day = news[news["date"] == busiest].to_dict("records")
    tagger.score_day(day, tickers, aliases)
    t0 = time.perf_counter()
    for _ in range(20):
        feats = tagger.score_day(day, tickers, aliases, with_evidence=True)
    ms = (time.perf_counter() - t0) / 20 * 1000
    example = next(((t, f) for t, f in feats.items() if f["has_event"]), (None, None))
    meta = {
        "model": "model6-EventTagger",
        "calibrated": tagger.calibrated,
        "taxonomy_sha1": hashlib.sha1(open(ROOT / "configs" / "events.yaml", "rb").read()).hexdigest()[:12],
        "events": {k: {"polarity": v["polarity"], "impact_bps": tagger.impacts[k]} for k, v in tagger.events.items()},
        "features": list(next(iter(feats.values())).keys() - {"evidence"}),
        "cpu_ms_busiest_day": round(ms, 2), "busiest_day_items": len(day),
    }
    json.dump(meta, open(art / "model6_meta.json", "w"), indent=2)
    print(f"calibrated={tagger.calibrated}; {len(meta['events'])} event types; "
          f"CPU {ms:.1f} ms for the busiest day ({len(day)} items)")
    if example[0]:
        print(f"example {example[0]}: score {example[1]['event_score']:+.3f}; evidence {example[1]['evidence'][:3]}")


if __name__ == "__main__":
    main()
