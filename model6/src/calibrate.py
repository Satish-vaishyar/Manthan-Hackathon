"""Event study + impact calibration + honest evaluation.

  python src/calibrate.py

1. Tags every recorded news item with the expert-prior tagger (configs/events.yaml).
2. Event study on the first 70% of dates: weighted mean next-day excess return per event type.
3. Shrinks each mean toward its prior:  posterior = (n_eff * mean + k * prior) / (n_eff + k).
4. Evaluates prior-only vs calibrated scores on the last 30% of dates (never seen in step 2).
5. Refits on ALL dates for deployment -> artifacts/calibration.json.

Outputs in artifacts/: calibration.json (deploy), calibration_holdout.json, event_study.csv, metrics.json,
tagged_news.csv (every item with its tags and evidence, for human review).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from labels import next_day_excess
from tagger import EventTagger
from text import compile_aliases, load_aliases

ROOT = Path(__file__).resolve().parents[1]


def item_events(tagger: EventTagger, news: pd.DataFrame, tickers: list[str], pats: dict) -> pd.DataFrame:
    rows = []
    for r in news.itertuples(index=False):
        it = tagger.tag_item(r.headline, r.description if isinstance(r.description, str) else "",
                             r.ticker, tickers, pats)
        for e in it.events:
            rows.append({"date": r.date, "ticker": r.ticker, "headline": r.headline, "relevant": it.relevant,
                         "event": e.event, "field": e.field, "evidence": e.evidence, "negated": e.negated,
                         "hedged": e.hedged, "magnitude": e.magnitude, "weight": e.weight,
                         "contribution_bps": e.contribution_bps})
        if not it.events:
            rows.append({"date": r.date, "ticker": r.ticker, "headline": r.headline, "relevant": it.relevant,
                         "event": None})
    return pd.DataFrame(rows)


def fit(ev: pd.DataFrame, tagger: EventTagger, k: float) -> dict:
    out = {}
    for name, e in tagger.events.items():
        d = ev[(ev["event"] == name) & (~ev["negated"].astype(bool)) & ev["y_bps"].notna()]
        w = d["weight"].abs().to_numpy()
        y = d["y_bps"].to_numpy()
        n_eff = float(w.sum())
        mean = float((w * y).sum() / n_eff) if n_eff > 0 else 0.0
        sd = float(y.std(ddof=1)) if len(y) > 1 else float("nan")
        t = mean / (sd / np.sqrt(len(y))) if len(y) > 1 and sd > 0 else float("nan")
        post = (n_eff * mean + k * e["prior_bps"]) / (n_eff + k)
        out[name] = {"prior_bps": e["prior_bps"], "count": int(len(y)), "n_eff": round(n_eff, 2),
                     "mean_bps": round(mean, 1), "t_stat": None if np.isnan(t) else round(float(t), 2),
                     "posterior_bps": round(float(post), 1)}
    return out


def evaluate(tagger: EventTagger, news: pd.DataFrame, labels: pd.DataFrame, tickers: list[str],
             aliases: dict) -> dict:
    rows = []
    for date, g in news.groupby("date"):
        feats = tagger.score_day(g.to_dict("records"), tickers, aliases)
        for t, f in feats.items():
            rows.append({"date": date, "ticker": t, **f})
    s = pd.DataFrame(rows).merge(labels, on=["date", "ticker"], how="inner")
    ev = s[s["has_event"] > 0]
    res = {"ticker_days": int(len(s)), "with_event": int(len(ev)),
           "coverage": float(len(ev) / max(len(s), 1))}
    for col in ("event_score", "rule_polarity"):
        if len(ev) >= 5 and ev[col].nunique() > 1:
            res[col] = {
                "ic_spearman": float(ev[col].rank().corr(ev["y_bps"].rank())),
                "hit_rate": float((np.sign(ev[col]) == np.sign(ev["y_bps"]))[ev[col] != 0].mean()),
                "mean_y_bps_when_positive": float(ev.loc[ev[col] > 0, "y_bps"].mean()),
                "mean_y_bps_when_negative": float(ev.loc[ev[col] < 0, "y_bps"].mean()),
                "n_positive": int((ev[col] > 0).sum()), "n_negative": int((ev[col] < 0).sum()),
            }
    return res


def main() -> None:
    cfg = yaml.safe_load(open(ROOT / "configs" / "model6.yaml"))
    events_cfg = yaml.safe_load(open(ROOT / "configs" / "events.yaml"))
    dc = cfg["data"]
    news = pd.read_csv(ROOT / dc["news_csv"])
    news["description"] = news["description"].fillna("")
    prices = pd.read_csv(ROOT / dc["prices_csv"])
    labels = next_day_excess(prices, dc["max_gap_days"], dc["label_clip"])
    tickers = sorted(set(prices["ticker"]) | set(news["ticker"]))
    aliases = load_aliases(ROOT / dc["aliases_auto"], ROOT / dc["aliases_manual"])
    pats = compile_aliases(aliases)

    prior = EventTagger(events_cfg, cfg)
    ev = item_events(prior, news, tickers, pats).merge(labels, on=["date", "ticker"], how="left")
    dates = sorted(news["date"].unique())
    cut = dates[int(len(dates) * dc["calib_fraction"])] if len(dates) > 1 else dates[-1]
    calib_ev = ev[ev["date"] < cut]
    k = cfg["calibration"]["shrinkage_k"]

    holdout_cal = {"fitted_on": f"dates < {cut}", "shrinkage_k": k, "events": fit(calib_ev, prior, k)}
    full_cal = {"fitted_on": "all dates", "shrinkage_k": k, "events": fit(ev, prior, k)}
    art = ROOT / cfg["export"]["out_dir"]
    art.mkdir(parents=True, exist_ok=True)
    json.dump(holdout_cal, open(art / "calibration_holdout.json", "w"), indent=2)
    json.dump(full_cal, open(art / "calibration.json", "w"), indent=2)

    test_news, test_labels = news[news["date"] >= cut], labels[labels["date"] >= cut]
    metrics = {
        "data": {"items": int(len(news)), "dates": len(dates), "eval_from": cut,
                 "items_with_any_event": int(ev.dropna(subset=["event"]).drop_duplicates(
                     ["date", "ticker", "headline"]).shape[0]),
                 "relevant_items": int(ev.drop_duplicates(["date", "ticker", "headline"])["relevant"].sum())},
        "holdout_prior_only": evaluate(prior, test_news, test_labels, tickers, aliases),
        "holdout_calibrated": evaluate(EventTagger(events_cfg, cfg, holdout_cal), test_news, test_labels,
                                       tickers, aliases),
    }
    json.dump(metrics, open(art / "metrics.json", "w"), indent=2)

    study = pd.DataFrame([{"event": e, **v} for e, v in full_cal["events"].items()]).sort_values("count",
                                                                                               ascending=False)
    study.to_csv(art / "event_study.csv", index=False)
    ev.to_csv(art / "tagged_news.csv", index=False)

    print(json.dumps(metrics, indent=2))
    shown = study[study["count"] >= cfg["calibration"]["min_count_report"]]
    print("\nevent study (all dates, events with >= "
          f"{cfg['calibration']['min_count_report']} labelled items):")
    print(shown.to_string(index=False) if len(shown) else "  none yet — record more news")


if __name__ == "__main__":
    main()
