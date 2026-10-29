# `data/` — MANTHAN 2026 hackathon bundle

> Sources: REST gateway `http://192.168.65.157:3002/api/v1` and Kafka broker
> `192.168.65.157:9094` (topics `market.ticks`, `market.news`). All timestamps
> and dates in the data are **simulated** by the venue.

---

## ⚠️ Critical finding — REST universe ≠ Kafka universe

The order gateway's REST API advertises a **40-ticker** universe, but Kafka is
currently pushing a **different 10-ticker** universe. They do **not overlap at
all**. The REST endpoint rejects the Kafka tickers as `INVALID_TICKER`.

| Source | Tickers | Tradable? |
|---|---|---|
| REST `/api/v1/stocks` | 40 tickers (AETX, AQUA, BAMR, …, WLFG) | ✅ via REST history/orders |
| Kafka `market.ticks` / `market.news` | 10 tickers (AERX, BLUX, DSNY, HAVN, HMDP, MOTR, ORCX, PLNX, ROUT, SONA) | ❌ rejected by REST as INVALID_TICKER |

**Implication for the live run**: the active trading universe during the
official 1-hour simulation will almost certainly be **whichever set Kafka is
pushing at that moment**, not the REST list. Your bot must:

1. Subscribe to Kafka `market.ticks` at startup and build the ticker universe
   from the first batch of `price_tick` events.
2. NOT hard-code ticker symbols (the PS says this; it's also enforced by the
   `INVALID_TICKER` error you'd otherwise get).
3. Use Kafka as the source of truth for prices and news; use REST for orders,
   portfolio queries, heartbeat, and stage polling only.

---

## Folder layout

```
data/
├── prices.csv                      # REST-derived OHLCV, long format (49,654 rows)
├── prices_wide.csv                 # wide pivot for charting  (1374 dates x 40 tickers)
├── prices/<TICKER>.csv             # 40 per-ticker REST CSVs
├── tickers.csv                     # 40 tickers with name + auto-guessed sector
├── stopped_tickers.csv             # 23 REST tickers that stop sending after 2025-08-29
├── split_crash_flags.csv           # 3 single-day >25% moves; verdict: real crashes
├── training_calendar.txt           # simulated-date reminders
├── meta.json                       # REST capture provenance
├── news/                           # one-shot 30s news probes
│   └── news_<timestamp>.jsonl
└── kafka/                          # raw Kafka captures (NDJSON per topic)
    └── 20261029T075240Z_f09695/
        ├── ticks.jsonl             # 370 raw market.ticks messages
        ├── news.jsonl              # 689 raw market.news messages
        ├── summary.json            # capture metadata
        └── derived/
            ├── prices_kafka.csv    # 370 OHLCV rows, 10 tickers, 37 dates
            ├── prices_kafka_wide.csv
            ├── news.csv            # 689 clean news rows, 10 tickers
            ├── kafka_tickers.csv
            └── kafka_summary.json
```

---

## How to load

```python
import pandas as pd
import json

# REST universe + history (the 40 tickers the gateway knows about)
prices  = pd.read_csv("data/prices.csv", parse_dates=["date"])
tickers = pd.read_csv("data/tickers.csv")
stopped = pd.read_csv("data/stopped_tickers.csv")
flags   = pd.read_csv("data/split_crash_flags.csv")
meta    = json.loads(open("data/meta.json").read())

# Kafka universe + history (the 10 tickers actually streaming)
prices_k = pd.read_csv("data/kafka/20261029T075240Z_f09695/derived/prices_kafka.csv")
news     = pd.read_csv("data/kafka/20261029T075240Z_f09695/derived/news.csv")
```

---

## Counts

### REST (40-ticker universe)
- Rows: **49,654** (cut at training cutoff **2026-07-27**)
- Date range: 2021-02-04 -> 2026-07-27
- Expected active in live: **17**
- Expected stopped (after 2025-08-29): **23**
- Suspected crash rows flagged: **3** (FLXN, SYNM, CYNT) — likely real crashes, not splits
  (volume spikes 5–31×, no clean integer price ratio)

### Kafka (10-ticker universe)
- Price rows: **370**
- Date range: 2024-10-25 -> 2025-07-11
- News rows: **689** for the same 10 tickers
- Only ~37 trading days captured (single partial sandbox loop). For a richer
  history you'd need to record for ~40+ minutes (one full 500-day loop) or
  rely on REST `/api/v1/stocks/<ticker>/history` if/when those 10 tickers
  become tradable there.

---

## Sandbox-vs-PS discrepancies

| Setting | PS says | Sandbox currently shows |
|---|---|---|
| `interval_seconds` | 3–4 s per day | **5** s per day |
| `total_days` | not given | **500** |
| Dataset name | "40 Stocks, 2016–2025" | actual dates run **2021–2026** |
| Date labels | n/a | simulated; **do not map to wall-clock calendar** |

The live run may use different values again. Don't hard-code the cadence.

---

## Bot checklist (informed by what we learned)

1. ☐ Build ticker universe from the **first** `price_tick` batch from Kafka
2. ☐ Subscribe to **both** `market.ticks` and `market.news`
3. ☐ Pull historical warm-start from `/api/v1/stocks/<ticker>/history?days=2000`
   for every ticker you discover (anti-lookahead window expands with sim date)
4. ☐ Detect tickers that **stop** ticking (no `price_tick` for 3+ consecutive
   days) — don't trade them; value any open holding at last known close
5. ☐ Filter the 3 suspected crash days (`is_anomaly_crash==True`) before
   training return-based features
6. ☐ Use **simulated dates** for tax-horizon / settlement-Friday logic
7. ☐ Poll `GET /api/v1/stage` every 2–3 s during pre-flight; arm trading only
   when `stage=='simulation'` AND `status=='RUNNING'`
8. ☐ ACK pre-flight ticks with `POST /api/v1/bot/heartbeat` at least every 15 s
9. ☐ Cap order rate at ~120 req/min; respect 30 s penalty box on HTTP 429
10. ☐ Pre-compute `data/prices.csv` indicators as a backtest source; use
    `data/kafka/.../derived/news.csv` to label news sentiment in development

---

## File generation scripts

- `C:\rev\download_history.py` — pulls REST history (idempotent, 8 workers)
- `C:\rev\build_data.py`       — produces `data/` from the raw REST history
- `C:\rev\record_news.py`      — short news-only Kafka recorder
- `C:\rev\record_kafka.py`     — combined ticks+news Kafka recorder
- `C:\rev\consolidate_kafka.py`— turns raw Kafka NDJSON into clean CSVs
- `C:\rev\investigate_anomalies.py` — crash/split forensics
- `C:\rev\finalize_data.py`    — this README
