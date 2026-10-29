# Manthan 2026 — Autonomous Trading Bot

A single-process Python 3.11 asyncio bot that:

1. Subscribes to the platform's Kafka stream (`market.ticks`, `market.news`)
2. ACKs every `preflight_tick` to flip the portal's readiness panel to **Ready**
3. Groups events into per-day **DayBundles** (complete-by-count or hard-deadline)
4. Scores tickers with the strategy stack (rules → ensemble → model1 when trained)
5. Constructs target weights, sizes trades, validates, rate-limits, submits
6. Applies fills to a local ledger, writes a JSONL decision journal
7. Tracks DD breakers, settlement guard, and Indian tax engine (STCG/LTCG, ₹1L exemption)

The architecture follows `docs/02` and `docs/10` exactly. **Trained model artifacts are plugged in later** — the bot runs on the P0 rules fallback until model1 is exported.

## Quick start

```bash
# 1. Install deps
pip install -r requirements.txt

# 2. Set environment
export SERVER_IP=192.168.65.157
export TEAM_USERNAME=MN2632
export BOT_ACCESS_TOKEN="eyJhbGciOi..."   # from the portal login

# 3. Just connect (readiness only — what we have running today)
python scripts/connect.py

# 4. Full bot (when ready to trade)
python -m bot.main --env sandbox
python -m bot.main --env sandbox --dry-run     # score but don't submit
```

## Repo layout (per `docs/10`)

```
Manthan-Hackathon/
├── bot/
│   ├── main.py              entrypoint
│   ├── config.py            typed Settings from yaml + env
│   ├── contracts.py         pydantic v2 models (Kafka + REST)
│   ├── lifecycle.py         BOOT→DISCOVER→PREFLIGHT→TRADING state machine
│   ├── ingest/
│   │   ├── kafka_consumer.py    aiokafka + backoff + queue
│   │   ├── day_assembler.py     DayBundle builder (400ms / 800ms deadlines)
│   │   ├── heartbeat.py         preflight ACK + watchdog
│   │   └── stage_watcher.py     polls /api/v1/stage
│   ├── execution/
│   │   ├── api_client.py        httpx async wrapper (only network in execution/)
│   │   ├── rate_limiter.py      token bucket + 429 penalty
│   │   ├── validator.py         pure pre-submission checks
│   │   └── order_manager.py     validate → risk → rate-limit → POST → ledger
│   ├── portfolio/
│   │   ├── ledger.py            FIFO lots, SHORTs, FY tracking
│   │   ├── tax_engine.py        STCG/LTCG + ₹1L exemption + harvesting
│   │   ├── settlement.py        every-10-trading-days calendar
│   │   ├── risk.py              DD breakers, name/sector caps, cash buffer
│   │   └── constructor.py       scores → weights → trade deltas
│   ├── models/
│   │   ├── rules.py             P0 fallback: SMA20/50 + RSI14 + news keywords
│   │   └── ensemble.py          ranker + news + momentum blend
│   ├── journal/
│   │   ├── journal.py           JSONL writer (one record per decision)
│   │   └── metrics.py           latency, reject-rate, utilisation
│   └── features/                (placeholder; will be filled in later)
├── recorder/                    (placeholder; Kafka → parquet recorder)
├── research/                    (placeholder; offline scripts)
├── dashboard/                   (placeholder; Streamlit app)
├── mocks/                       (placeholder; mock Kafka + gateway)
├── configs/
│   ├── default.yaml
│   ├── sandbox.yaml
│   └── live.yaml
├── models/                      trained artifacts (gitignored except .gitkeep)
├── data/                        gitignored
├── runs/                        per-run journals, logs (gitignored)
├── tests/                       unit tests
├── wheels/                      offline pip wheels (gitignored)
├── scripts/
│   ├── connect.py               readiness-only Kafka consumer (live now)
│   ├── requirements-connect.txt
│   └── README.md
├── docs/                        the team's 12-doc spec
├── model1/                      pre-existing PyTorch Transformer
├── data/data/                   bundled prices + Kafka captures
├── requirements.txt
├── NOTES.md
└── README.md
```

## What's running today

The `scripts/connect.py` is the readiness-only consumer that was smoke-tested
live on `192.168.65.157:9094` (Oct 29 2026). It:

- Connects to Kafka with `group_id=group-MN2632`
- Subscribes to both topics
- Logs every price_tick + news_flash it sees
- ACKs every `preflight_tick` via `POST /api/v1/bot/heartbeat`
- Re-sends the last seq every 10 s (watchdog)
- Polls `GET /api/v1/bot/status` every 15 s and prints what the portal sees

See `scripts/README.md` for the smoke-test transcript.

## What's ready but not wired

The `bot/` package is complete up to the same spec the docs require. To
go from "connect only" to "trade", run:

```bash
python -m bot.main --env sandbox --dry-run     # score, don't submit
python -m bot.main --env sandbox                # full live trading
```

The bot will:

1. Call `GET /api/v1/stocks` to populate the universe (no hardcoded tickers).
2. Call `GET /api/v1/portfolio` to reconcile against the platform.
3. Subscribe to Kafka, ACK heartbeats, drive the lifecycle.
4. For each DayBundle, run the ensemble, build weights, submit up to 4 orders.

The P0 rules strategy (`bot/models/rules.py`) is in place. When model1 is
trained and exported (`model1/src/export.py`), plug it into `bot/models/ensemble.py`
by replacing `_default_rank` with a call to the ONNX runtime. No other
component changes.

## Tests

```bash
pytest tests/ -v
```

Covers:
- Ledger: BUY / SELL FIFO / SHORT / COVER / BUY-covers-short / fees / DP
- Validator: every rejection code in `docs/03 §3`
- Tax engine: ST/LT boundary (365 vs 366 days), ₹1L exemption, set-off order, FY
- Rate limiter: sustained, burst, 429 penalty
- Constructor: inverse-vol sizing, name cap, threshold, min notional
- Day assembler: complete-by-count, deadline, loop, out-of-order stash
- Risk: DD breaker, settlement guard, sector cap, cash buffer

See `docs/11_TESTING_AND_RUNBOOK.md` for the canonical test list.

## Team

Team **MN2632 — Unseen Geeks**. Plan & roles: `docs/09_TEAM_PLAN_AND_TIMELINE.md`.
Venue facts discovered live: `NOTES.md`.

## License

Internal — hackathon use only.