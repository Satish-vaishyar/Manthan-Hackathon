"""Merge every capture in data/raw/captures/ into clean news.csv and prices.csv, then mine company aliases.

  python src/consolidate.py

The sandbox replays the same dataset in loops, so the same simulated date can be captured several times:
prices are de-duplicated on (date, ticker) and news on (date, ticker, headline).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from text import clean, mine_aliases

ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def main() -> None:
    cfg = yaml.safe_load(open(ROOT / "configs" / "model2.yaml"))["data"]
    cap = ROOT / cfg["captures_dir"]
    ticks, news = [], []
    for f in sorted(cap.glob("*/*.jsonl")):
        for r in read_jsonl(f):
            p = r.get("payload") or {}
            et = p.get("event_type") or p.get("event")
            d = p.get("data") or {}
            sb = p.get("sandbox") or {}
            if et == "price_tick":
                ticks.append({"date": p.get("simulated_date"), "ticker": d.get("ticker"),
                              **{k: d.get(k) for k in ("open", "high", "low", "close", "volume")},
                              "loop": sb.get("loop_count"), "capture": f.parent.name})
            elif et == "news_flash":
                news.append({"date": p.get("simulated_date"), "ticker": d.get("ticker"),
                             "headline": clean(d.get("headline", "")), "description": clean(d.get("description", "")),
                             "loop": sb.get("loop_count"), "kafka_ts": p.get("timestamp"), "capture": f.parent.name})
    prices = pd.DataFrame(ticks).dropna(subset=["date", "ticker", "close"])
    prices = prices.sort_values(["date", "ticker", "loop"]).drop_duplicates(["date", "ticker"])
    nws = pd.DataFrame(news).dropna(subset=["date", "ticker"])
    raw_n = len(nws)
    nws = nws[nws["headline"].str.len() > 0]
    nws = nws.sort_values(["date", "ticker", "kafka_ts"]).drop_duplicates(["date", "ticker", "headline"])

    out_n, out_p = ROOT / cfg["news_csv"], ROOT / cfg["prices_csv"]
    out_n.parent.mkdir(parents=True, exist_ok=True)
    nws.to_csv(out_n, index=False)
    prices.to_csv(out_p, index=False)
    aliases = mine_aliases(nws)
    json.dump(aliases, open(ROOT / cfg["aliases_auto"], "w"), indent=2)

    print(f"captures: {sorted({p.name for p in cap.iterdir() if p.is_dir()})}")
    print(f"prices: {len(prices)} rows, {prices['ticker'].nunique()} tickers, {prices['date'].nunique()} dates "
          f"({prices['date'].min()} -> {prices['date'].max()})")
    print(f"news:   {raw_n} raw -> {len(nws)} unique, {nws['ticker'].nunique()} tickers, {nws['date'].nunique()} dates")
    print(f"aliases (auto, review them and edit configs/aliases_manual.json): {aliases}")


if __name__ == "__main__":
    main()
