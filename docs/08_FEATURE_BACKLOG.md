# 08 — Feature Backlog

Owner codes: **A** = Infra/Execution, **B** = ML, **C** = Strategy/Risk/Tax, **D** = Backtest/Dashboard.
Priorities: **P0** = must work before hour 4 (fallback bot); **P1** = core competitive bot by hour 14;
**P2** = differentiators by hour 18. Nothing new after code freeze (hour 18).

## P0 — Fallback bot (target: hour 4, git tag `v0-fallback`)

| ID | Feature | Owner | Acceptance criteria |
|----|---------|-------|---------------------|
| P0-01 | Config loader (`config.yaml` + env `BOT_ACCESS_TOKEN`, `SERVER_IP`, `TEAM_USERNAME`) | A | Bot starts with missing optional values defaulted; fails fast on missing token |
| P0-02 | Contracts (pydantic models for all messages & API responses) | A | Parses all PS samples; ignores unknown fields |
| P0-03 | Kafka consumer (N tasks, group `group-<user>`, latest, no autocommit, reconnect) | A | Receives ticks & news in sandbox; survives broker restart |
| P0-04 | Heartbeat responder + watchdog | A | Portal shows **Ready**; ACK interval < 15 s always |
| P0-05 | Stage watcher + trading gate | A | Gate opens only on RUNNING; closes on PAUSED/COMPLETED |
| P0-06 | Universe discovery + history warm-up | A | Tickers from `/stocks`; ≥ 200 bars per ticker loaded (spread requests) |
| P0-07 | Day Assembler (completion rule, news window, loop detection) | A | Logs one `DayBundle` per day with all tickers; detects sandbox loop |
| P0-08 | Ledger with platform-exact fill rules | C | Unit tests pass for BUY/SELL/SHORT/COVER/BUY-covers-short |
| P0-09 | Order validator (pure function) | C | 100% branch coverage; rejects all invalid cases in doc 06 §4.2 |
| P0-10 | Rate limiter + order sender + rejection handlers | A | 0 × 429 in a 30-min sandbox run |
| P0-11 | Rules strategy (SMA20/50 trend + RSI14 + news keywords) | C | Trades in sandbox, ≤ 4 orders/day, 0 rejections |
| P0-12 | Structured logs + decision journal (JSONL) | D | One record per ticker per day incl. HOLD and reason |
| P0-13 | Recorder process (Kafka → Parquet) | D | Running from hour 0.5 continuously; files rotate hourly |

## P1 — Competitive bot (target: hour 14)

| ID | Feature | Owner | Acceptance criteria |
|----|---------|-------|---------------------|
| P1-01 | Incremental Feature Engine (all features in doc 05 §6) | B | Matches pandas reference implementation to 1e-9 on test data; < 10 ms/day |
| P1-02 | History puller + dataset builder (loop dedupe, labels) | B | `features.parquet` reproducible from raw |
| P1-03 | Event tagger (M1a) | B | Tags ≥ 70% of recorded headlines with ≥ 1 tag |
| P1-04 | FinBERT ONNX int8 scorer (M1b) | B | ≤ 30 ms/headline on team laptop, offline |
| P1-05 | News calibration head (M1c) + decayed memory | B | Top-decile minus bottom-decile next-day return > 0.3% on holdout |
| P1-06 | LightGBM ranker (M2) + walk-forward eval | B | IC report generated; acceptance rule in doc 05 §4 applied |
| P1-07 | Regime model (rules) (M3) + vol model (M4) | B/C | Regime timeline plot; hysteresis works |
| P1-08 | Ensemble + model fallbacks | B | Killing any model file → bot still trades with remaining signals |
| P1-09 | Portfolio Constructor (core top-K, inverse-vol, caps, rebalance threshold, cost gate) | C | Backtest turnover < agreed cap; no trade < ₹50k except exits |
| P1-10 | Event satellite book (long/short news trades, time stops) | C | Trades appear with `source=event` in journal |
| P1-11 | Risk Manager (limits, ATR stops, DD breaker, settlement guard, cash buffer) | C | Unit tests; sandbox run with no ACCOUNT_BLOCKED |
| P1-12 | Settlement calendar + verification logger | C | Detects first forced square-off in sandbox and records anchor |
| P1-13 | Tax Engine (lots, ST/LT, set-off, FY, LTCG-wait, harvesting) | C | Hand-computed test cases pass; per-FY tax report |
| P1-14 | Reconciliation (`GET /portfolio` schedule, drift handling) | A | Drift metric shown; auto-heals |
| P1-15 | Timeout-safe order retry via `GET /orders` | A | Simulated timeout never produces duplicate fills |
| P1-16 | Backtester with SimBroker (doc 07) | D | Parity test within 0.5% |
| P1-17 | Latency instrumentation (tick→decision→ack) | A | p50/p95 logged per day |

## P2 — Differentiators (target: hour 18)

| ID | Feature | Owner | Acceptance criteria |
|----|---------|-------|---------------------|
| P2-01 | Streamlit Mission Control: equity, exposure, regime, positions, orders, rejections, latency histogram, request budget gauge, per-FY tax | D | Refreshes ≤ 2 s; runs on second laptop reading shared journal |
| P2-02 | Trade reason cards with top-3 SHAP features + news headline | D/B | Click any trade → full explanation |
| P2-03 | Backtest HTML report (doc 07 §4) | D | Generated in one command |
| P2-04 | HMM regime (if it beats rules) | B | Walk-forward comparison report |
| P2-05 | Kill switch (file + dashboard button) and SAFE_MODE | A | Tested drill: enters SAFE_MODE < 1 day |
| P2-06 | Hot-standby dry-run instance on 2nd laptop | A | Produces identical decisions; can be switched live in < 1 min |
| P2-07 | Chaos test suite (Kafka drop, 429 storm, malformed msg, API timeout) | A/D | All drills pass (doc 11) |
| P2-08 | Sector clustering from descriptions | B | Sector caps active |
| P2-09 | Architecture diagram + 5-slide jury deck | All | Ready by hour 19 |

## Stretch (only if everything above is done and stable)

- Multi-horizon ranker blend (h=1, h=3, h=5).
- Pairs/stat-arb within sector clusters (cointegration on 120-d windows) for market-neutral exposure.
- Online re-weighting of ensemble by trailing 60-day signal performance (bounded ±20%).
