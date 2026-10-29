# bot/features — incremental feature engineering

This package is reserved for the live incremental-feature stack:

- `ring_buffer.py` — per-ticker rolling window (e.g. 220 days)
- `indicators.py` — incremental SMA, EMA, RSI, MACD, ATR, Bollinger
- `cross_section.py` — cross-sectional rank, z-score, sector means
- `news_features.py` — news-keyword tagger, sentiment scoring, polarity decay

At inference time, these produce the 28 ticker-agnostic features consumed by
`model1` (see `model1/src/features.py` for the catalogue).

The current P0 fallback strategy (`bot/models/rules.py`) computes its own
minimal indicators (SMA20/50, RSI14) inline. Once the ML ranker is wired
in via `bot/models/ensemble.py`, this folder becomes the home of the
incremental-feature code that feeds it.

Per `docs/05 §3`: features must be incremental (no pandas-history per tick)
to stay within the 3-4 s per-day budget.