# model2 — Data Report

Checked on 2026-10-29 against the captures in `data/data/kafka/` and `data/data/news/` (copied into
`model2/data/raw/captures/`).

## 1. What is available now

| Item | Value |
|---|---|
| Raw news messages | 734 (689 from the 5-minute capture + 45 from the news probe) |
| Unique news after de-duplication | **410** items, 10 tickers, 41 simulated dates |
| Kafka candles | 370 rows = 10 tickers × 37 dates (2024-10-25 → 2025-07-11), from sandbox loops 1 and 6 |
| Labelled ticker-days (news + next-day price) | **173** (train 114 / val 17 / test 22) |
| Items mentioning their company (weak "relevant" label) | **14.9%** |

## 2. Schemas

**Raw capture line (`captures/<run>/news.jsonl`, `ticks.jsonl`)**
```json
{"received_at": "...", "topic": "market.news", "partition": 7, "offset": 26709,
 "payload": {"event_type": "news_flash", "timestamp": "...", "simulated_date": "2025-06-13",
             "data": {"ticker": "PLNX", "headline": "...", "description": "...", "date": "2025-06-13"},
             "sandbox": {"is_sandbox": true, "stage": "development", "loop_count": 6, "cycle_day": 158,
                         "total_days": 500, "interval_seconds": 5}}}
```
`price_tick` payloads carry `data: {ticker, open, high, low, close, volume}`. All captures parse without errors.

**`data/processed/news.csv`** (from `consolidate.py`): `date, ticker, headline, description, loop, kafka_ts, capture`.
**`data/processed/prices.csv`**: `date, ticker, open, high, low, close, volume, loop, capture`.
**`data/processed/dataset.parquet`** (from `prepare.py`): one row per item with `text` (company-masked),
`relevant`, teacher `t_pos/t_neg/t_neu`, `ctx_*` (5 candle features), `y_ret`, `y_abs`, `has_label`, `split`.

## 3. Problems found and how they are handled

| # | Problem | Handling |
|---|---|---|
| P1 | **Far too little data**: 114 training ticker-days | `train.py` refuses a full run below 300. **Record ≥ 1 full sandbox loop (500 days ≈ 42 min at 5 s/day)** with `src/record_kafka.py`. Expect ≈ 5,000+ labelled ticker-days. |
| P2 | **Most headlines are unrelated** to the company (sports, politics, film reviews); only ~15% mention it | Relevance head trained on alias-based weak labels; relevance-gated pooling; baseline compares all vs relevant-only |
| P3 | The venue masking swaps real names for brand aliases (DreamScape, RouteWave, Polestar, Haven Depot, Sonata) and sometimes swaps ordinary words for ticker symbols ("could CSTC" = "could cost") | Aliases mined automatically (CamelCase brands kept) + `configs/aliases_manual.json`; every ticker symbol is removed from text; own symbol/aliases → "the company" |
| P4 | 324 duplicate (ticker, headline) rows, including a 335-item burst on 2025-06-23 | De-duplicated on (date, ticker, headline) |
| P5 | Same simulated dates repeat across sandbox loops | Prices de-duplicated on (date, ticker); labels only when the next candle is ≤ 5 calendar days later, so loop jumps never create fake returns |
| P6 | Mojibake (`�`) and smart quotes | `text.clean()` normalises them |
| P7 | Kafka tickers are rejected by REST `/history` | Labels and price context come from the Kafka candles themselves |

Check: after masking, **no ticker symbol remains in any model input text** (`prepare.py` prints this).

## 4. Pipeline check on this laptop (CPU, 1 epoch, NOT a real result)

- `consolidate.py`, `prepare.py` (FinBERT teacher), `baseline.py`, `train.py --smoke`, `export.py`, `infer.NewsScorer`
  all ran without errors. The smoke outputs were deleted afterwards.
- Zero-shot FinBERT baseline on 17–22 ticker-days: pooled IC +0.17 to +0.24, which is too few samples to mean anything.
- CPU inference for a day with 17 items: **≈ 360 ms per ensemble member with int8**, ≈ 890 ms with fp32.
  Re-check the int8 difference after real training (`export.py` reports it).
