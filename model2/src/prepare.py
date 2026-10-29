"""Build the model2 training table: one row per news item, joined with ticker-day labels and price context.

  python src/prepare.py                 # uses FinBERT on GPU (if present) for teacher sentiment
  python src/prepare.py --no-teacher    # skip the teacher (distillation loss is then disabled)

Columns: date, ticker, text (company-masked), relevant (weak label), t_pos/t_neg/t_neu (teacher probs),
ctx_* (day-D candle context), y_ret (next-day excess log return), y_abs (|next-day return|), has_label, split.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from text import compile_aliases, load_aliases, mask, mentions

ROOT = Path(__file__).resolve().parents[1]
CTX_FEATURES = ["ctx_ret1", "ctx_gap", "ctx_intraday", "ctx_hl_range", "ctx_clv"]


def price_frame(prices: pd.DataFrame, max_gap: int, clip: float) -> pd.DataFrame:
    p = prices.copy()
    p["date"] = pd.to_datetime(p["date"])
    p = p.sort_values(["ticker", "date"])
    g = p.groupby("ticker")
    prev_c, prev_d = g["close"].shift(1), g["date"].shift(1)
    next_c, next_d = g["close"].shift(-1), g["date"].shift(-1)
    ok_prev = (p["date"] - prev_d).dt.days <= max_gap
    ok_next = (next_d - p["date"]).dt.days <= max_gap
    rng = (p["high"] - p["low"]).clip(lower=1e-9)
    p["ctx_ret1"] = np.where(ok_prev, np.log(p["close"] / prev_c), 0.0)
    p["ctx_gap"] = np.where(ok_prev, np.log(p["open"] / prev_c), 0.0)
    p["ctx_intraday"] = np.log(p["close"] / p["open"])
    p["ctx_hl_range"] = np.log(p["high"] / p["low"])
    p["ctx_clv"] = (2 * p["close"] - p["high"] - p["low"]) / rng
    r = np.where(ok_next, np.log(next_c / p["close"]), np.nan)
    p["r_next"] = np.clip(r, -clip, clip)
    # Excess vs all tickers that have a next-day return on the same date (not only tickers with news).
    p["y_ret"] = p["r_next"] - p.groupby("date")["r_next"].transform("mean")
    p["y_abs"] = p["r_next"].abs()
    p["has_label"] = p["r_next"].notna()
    p[CTX_FEATURES] = p[CTX_FEATURES].fillna(0.0)
    return p[["date", "ticker", *CTX_FEATURES, "y_ret", "y_abs", "has_label"]]


@torch.no_grad()
def teacher_probs(texts: list[str], name: str, batch: int = 64) -> np.ndarray:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(name)
    mdl = AutoModelForSequenceClassification.from_pretrained(name).to(dev).eval()
    labels = {v.lower(): int(k) for k, v in mdl.config.id2label.items()}
    order = [labels["positive"], labels["negative"], labels["neutral"]]
    out = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], truncation=True, max_length=128, padding=True, return_tensors="pt").to(dev)
        out.append(torch.softmax(mdl(**enc).logits.float(), -1)[:, order].cpu().numpy())
        print(f"  teacher {min(i + batch, len(texts))}/{len(texts)}", end="\r")
    print()
    return np.concatenate(out) if out else np.zeros((0, 3))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-teacher", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / "configs" / "model2.yaml"))
    dc, tc = cfg["data"], cfg["text"]

    news = pd.read_csv(ROOT / dc["news_csv"])
    prices = pd.read_csv(ROOT / dc["prices_csv"])
    tickers = sorted(set(prices["ticker"]) | set(news["ticker"]))
    pats = compile_aliases(load_aliases(ROOT / dc["aliases_auto"], ROOT / dc["aliases_manual"]))

    raw = (news["headline"].fillna("") + ". " + news["description"].fillna("").str[:tc["desc_chars"]]).tolist()
    news["relevant"] = [int(mentions(t, k, pats)) for t, k in zip(raw, news["ticker"])]
    news["text"] = [mask(t, k, tickers, pats) for t, k in zip(raw, news["ticker"])]
    news["date"] = pd.to_datetime(news["date"])

    pf = price_frame(prices, dc["max_gap_days"], dc["label_clip"])
    df = news.merge(pf, on=["date", "ticker"], how="left")
    df["has_label"] = df["has_label"].fillna(False).astype(bool)
    df[CTX_FEATURES] = df[CTX_FEATURES].fillna(0.0)

    if args.no_teacher:
        df[["t_pos", "t_neg", "t_neu"]] = 0.0
        df["has_teacher"] = False
    else:
        print(f"computing teacher sentiment with {cfg['model']['teacher']} ...")
        df[["t_pos", "t_neg", "t_neu"]] = teacher_probs(df["text"].tolist(), cfg["model"]["teacher"])
        df["has_teacher"] = True

    # Date-ordered split with an embargo between parts.
    dates = np.array(sorted(df.loc[df["has_label"], "date"].unique()))
    n = len(dates)
    a, b = int(n * dc["split"][0]), int(n * (dc["split"][0] + dc["split"][1]))
    emb = dc["embargo_days"]
    split_of = {d: 0 for d in dates[:a]}
    split_of.update({d: 1 for d in dates[a + emb:b]})
    split_of.update({d: 2 for d in dates[b + emb:]})
    df["split"] = df["date"].map(split_of).fillna(-1).astype(int)
    df.loc[~df["has_label"], "split"] = -1

    out = ROOT / dc["dataset"]
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    cols = ["date", "ticker", "headline", "text", "relevant", "t_pos", "t_neg", "t_neu", "has_teacher",
            *CTX_FEATURES, "y_ret", "y_abs", "has_label", "split"]
    df[cols].to_parquet(out, index=False)

    td = df[df["has_label"]].groupby(["date", "ticker"]).size()
    print(f"items: {len(df)}  relevant (weak label): {df['relevant'].mean():.1%}  labelled items: {df['has_label'].sum()}")
    print(f"labelled ticker-days: {len(td)}  (train/val/test items: "
          f"{[(df['split'] == s).sum() for s in (0, 1, 2)]})")
    leaks = [t for t in tickers if df["text"].str.contains(rf"\b{t}\b").any()]
    print(f"ticker symbols left in model text: {leaks or 'none'}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
