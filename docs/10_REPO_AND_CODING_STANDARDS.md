# 10 — Repository Layout & Coding Standards

## 1. Layout

```
Manthan/
├── docs/                       # these documents
├── bot/
│   ├── main.py                 # entrypoint: python -m bot.main --env sandbox|live [--dry-run]
│   ├── config.py               # loads configs/*.yaml + env vars into a typed Settings object
│   ├── contracts.py            # pydantic models: ticks, news, preflight, API responses
│   ├── lifecycle.py            # state machine BOOT→DISCOVER→PREFLIGHT→TRADING→PAUSED/COMPLETED/SAFE_MODE
│   ├── ingest/
│   │   ├── kafka_consumer.py
│   │   ├── day_assembler.py
│   │   ├── heartbeat.py
│   │   └── stage_watcher.py
│   ├── features/
│   │   ├── ring_buffer.py
│   │   ├── indicators.py       # incremental indicators
│   │   ├── cross_section.py
│   │   └── news_features.py
│   ├── models/
│   │   ├── event_tagger.py
│   │   ├── finbert_onnx.py
│   │   ├── news_scorer.py
│   │   ├── ranker.py
│   │   ├── regime.py
│   │   ├── vol.py
│   │   ├── rules.py            # fallback strategy
│   │   └── ensemble.py
│   ├── portfolio/
│   │   ├── ledger.py
│   │   ├── constructor.py
│   │   ├── risk.py
│   │   ├── tax_engine.py
│   │   └── settlement.py
│   ├── execution/
│   │   ├── api_client.py
│   │   ├── rate_limiter.py
│   │   ├── validator.py
│   │   └── order_manager.py
│   └── journal/
│       ├── journal.py
│       └── metrics.py
├── recorder/recorder.py
├── research/                   # offline scripts (doc 05 §7), backtest.py, sim_broker.py
├── dashboard/app.py            # Streamlit Mission Control
├── mocks/                      # mock Kafka producer + mock FastAPI gateway for offline dev
├── configs/
│   ├── default.yaml
│   ├── sandbox.yaml
│   └── live.yaml
├── models/                     # trained artifacts + *.meta.json (git-lfs or gitignored + shared drive)
├── data/                       # gitignored: raw/, history/, features.parquet
├── runs/                       # gitignored: per-run journal, logs, snapshots
├── tests/
├── wheels/                     # offline pip wheels (gitignored)
├── requirements.txt
├── NOTES.md                    # venue facts discovered during the event
└── README.md
```

## 2. Core dependencies

```
python==3.11.*
aiokafka  httpx  orjson  pydantic>=2  pyyaml  structlog
numpy  pandas  pyarrow  polars
lightgbm  scikit-learn  hmmlearn  arch  shap  joblib
onnxruntime  transformers  optimum  tokenizers
streamlit  plotly
pytest  pytest-asyncio  fastapi  uvicorn   # mocks/tests
```

Pin exact versions in `requirements.txt` after the first working install; build `wheels/` from it.

## 3. Coding standards

- **Type hints everywhere**; dataclasses or pydantic for every structure crossing a module boundary.
- **Pure logic is pure**: ledger, validator, tax engine, constructor, risk → no I/O, fully unit-tested.
- **No network calls outside `execution/api_client.py` and `ingest/kafka_consumer.py`.**
- **No magic numbers**: every threshold lives in `configs/*.yaml`.
- **Money**: use `float` with rounding to 2 decimals on display; quantities are always `int`.
- **Dates**: simulated dates are `datetime.date`; never mix with wall-clock time. Wall-clock latency uses `time.perf_counter_ns()`.
- **Exceptions**: never swallow silently. Catch at component boundary → log with context → fallback.
- **Async**: never `time.sleep` or blocking I/O in the event loop; use `asyncio.to_thread` / queues.
- **Logging**: `structlog` JSON, one event per line, always include `sim_date`, `component`, `event`.

## 4. Config example (`configs/default.yaml`)

```yaml
env: sandbox
server_ip: ${SERVER_IP}
team_username: ${TEAM_USERNAME}
kafka: { port: 9094, consumers: 4, topics: [market.ticks, market.news] }
api:   { port: 3002, timeout_s: 1.5, rate_per_min: 100 }
timing: { news_window_ms: 400, hard_deadline_ms: 800 }
strategy:
  weights: { rank: 0.45, news: 0.35, mom: 0.20 }
  top_k: 8
  max_weight_name: 0.15
  rebalance_threshold: 0.025
  min_trade_notional: 50000
  max_orders_per_day: 4
  event_long_threshold: 0.6
  event_short_threshold: -0.6
risk: { stop_atr_mult: 2.5, dd_soft: 0.08, dd_hard: 0.15, cash_buffer: 0.05, max_short_gross: 0.20 }
tax:  { ltcg_wait_days: 30, harvest_window_days: 10, min_tax_saving: 70 }
```

## 5. Git workflow

- `main` always runnable. Short-lived branches `feat/<id>-<name>` (IDs from doc 08).
- Pull request (or pair review) for anything in `execution/` or `portfolio/`.
- Commit messages: `P1-09: add rebalance threshold to constructor`.
- Tags: `v0-fallback`, `v1-ensemble`, `v2-final`.

## 6. Journal record schema (one JSON line)

```json
{
  "ts_wall": "2026-10-19T10:00:00.612Z", "sim_date": "2021-04-15", "ticker": "ALPHACORP",
  "decision": "BUY", "qty": 120, "price": 1450.25, "source": "core|event|risk|tax|settlement",
  "scores": {"rank": 0.71, "news": 0.82, "mom": 0.35, "ensemble": 0.66},
  "regime": "risk_on", "target_w": 0.081, "current_w": 0.0,
  "reasons": ["news EARNINGS_BEAT +24% YoY", "ranker 2/40", "close in top 20% of range"],
  "shap_top3": [["news_mem", 0.012], ["ret_20", 0.006], ["vol_z_20", 0.004]],
  "latency_ms": {"assemble": 402, "decide": 31, "ack": 118},
  "order_id": "…", "status": "FILLED|REJECTED|HOLD", "reject_code": null
}
```
