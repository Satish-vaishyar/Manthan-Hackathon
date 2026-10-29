"""Suggest new event patterns from recorded headlines.

  python src/mine_keywords.py

Writes artifacts/keyword_suggestions.csv with two kinds of rows:
  coverage  frequent 1-3-word phrases in headlines that mention the company but got NO event tag
  lift      phrases whose next-day excess return is far from zero (min 3 occurrences), with mean and t-stat
A human decides what goes into configs/events.yaml; nothing is added automatically.
"""
from __future__ import annotations

import collections
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from labels import next_day_excess
from tagger import EventTagger
from text import clean, compile_aliases, load_aliases, mask

ROOT = Path(__file__).resolve().parents[1]
STOP = set("the a an of to in on for and or with at by from as is are was be its it this that after over new "
           "says said will has have more than how why what who into up out about company".split())


def phrases(text: str) -> set[str]:
    words = [w for w in re.findall(r"[a-z][a-z'\-]+", text.lower())]
    out = set()
    for n in (1, 2, 3):
        for i in range(len(words) - n + 1):
            g = words[i:i + n]
            if g[0] in STOP or g[-1] in STOP:
                continue
            out.add(" ".join(g))
    return out


def main() -> None:
    cfg = yaml.safe_load(open(ROOT / "configs" / "model6.yaml"))
    dc = cfg["data"]
    news = pd.read_csv(ROOT / dc["news_csv"])
    news["description"] = news["description"].fillna("")
    prices = pd.read_csv(ROOT / dc["prices_csv"])
    labels = next_day_excess(prices, dc["max_gap_days"], dc["label_clip"])
    tickers = sorted(set(prices["ticker"]) | set(news["ticker"]))
    pats = compile_aliases(load_aliases(ROOT / dc["aliases_auto"], ROOT / dc["aliases_manual"]))
    tagger = EventTagger.from_files(ROOT, use_calibration=False)

    df = news.merge(labels, on=["date", "ticker"], how="left")
    cov, lift = collections.Counter(), collections.defaultdict(list)
    for r in df.itertuples(index=False):
        it = tagger.tag_item(r.headline, r.description, r.ticker, tickers, pats)
        ph = phrases(mask(clean(r.headline), r.ticker, tickers, pats))
        if it.relevant and not it.events:
            cov.update(ph)
        if not np.isnan(r.y_bps):
            for p in ph:
                lift[p].append(r.y_bps)
    rows = [{"kind": "coverage", "phrase": p, "count": c} for p, c in cov.most_common(60) if c >= 2]
    for p, ys in lift.items():
        if len(ys) >= 3:
            y = np.array(ys)
            t = y.mean() / (y.std(ddof=1) / np.sqrt(len(y)) + 1e-9)
            rows.append({"kind": "lift", "phrase": p, "count": len(y), "mean_y_bps": round(float(y.mean()), 1),
                         "t_stat": round(float(t), 2)})
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["kind", "count"], ascending=[True, False])
        lifted = out[out["kind"] == "lift"].reindex(
            out[out["kind"] == "lift"]["t_stat"].abs().sort_values(ascending=False).index)
        out = pd.concat([out[out["kind"] == "coverage"], lifted])
    path = ROOT / cfg["export"]["out_dir"] / "keyword_suggestions.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(f"{(out['kind'] == 'coverage').sum() if len(out) else 0} coverage and "
          f"{(out['kind'] == 'lift').sum() if len(out) else 0} lift suggestions -> {path}")
    if len(out):
        print(out.head(25).to_string(index=False))


if __name__ == "__main__":
    main()
