# 02 — System Architecture

## 1. Design principles

1. **One process, one source of truth.** All trading state (ledger, features, risk) lives in one Python
   asyncio process. No distributed state, no locks across processes.
2. **Event-driven, deadline-bound.** Each simulated day is a pipeline with a hard deadline.
3. **Fail safe, not fail open.** Any component failure degrades to HOLD or the rules fallback, never to
   random or unvalidated orders.
4. **Local-first state.** Minimise REST calls; the platform is the reconciliation target, not the hot path.
5. **Explainability by construction.** Every decision produces a journal record.
6. **Offline.** No internet dependency at runtime.

## 2. High-level diagram

```
                        ┌──────────────────────────── TRADING BOT (Python 3.11, asyncio) ────────────────────────────┐
  Kafka :9094           │                                                                                             │
  market.ticks ─┐       │ ┌─────────────┐   ┌───────────────┐   ┌────────────────┐   ┌──────────────────────────┐     │
  market.news  ─┼──────►│ │ Ingestion   │──►│ Day Assembler │──►│ Feature Engine │──►│ Signal Layer             │     │
  (10 parts)    │       │ │ N consumers │   │ per sim_date  │   │ ring buffers   │   │ news · ranker · regime · │     │
                │       │ │ 1 group     │   │ + news window │   │ incremental    │   │ rules → ensemble score   │     │
                │       │ └─────┬───────┘   └───────────────┘   └────────────────┘   └────────────┬─────────────┘     │
                │       │       │ preflight_tick                                                  ▼                   │
                │       │ ┌─────▼───────┐   ┌───────────────┐   ┌────────────────┐   ┌──────────────────────────┐     │
                │       │ │ Heartbeat   │   │ Stage Watcher │   │ Tax Engine     │◄─►│ Portfolio Constructor    │     │
                │       │ └─────────────┘   └───────┬───────┘   └────────────────┘   │ target weights, costs    │     │
                │       │                           │ gates trading                   └────────────┬─────────────┘     │
                │       │ ┌───────────────────┐     │           ┌────────────────┐                ▼                   │
                │       │ │ Portfolio Ledger  │◄────┴───fills───│ Order Manager  │◄──── Risk Manager (pre-trade)    │
                │       │ │ cash·lots·shorts  │                 │ validate·limit │                                    │
                │       │ └─────────┬─────────┘                 │ send·reconcile │──── HTTP keep-alive ───┐           │
                │       │           ▼                           └────────────────┘                        │           │
                │       │ ┌───────────────────┐                                                           │           │
                │       │ │ Decision Journal  │ (JSONL + SQLite)                                          │           │
                │       │ └─────────┬─────────┘                                                           │           │
                │       └───────────┼─────────────────────────────────────────────────────────────────────┼───────────┘
                │                   ▼                                                                     ▼
                │         Streamlit Mission Control                                         Order Gateway :3002
                └──────► Recorder process → Parquet → Backtester + Training (offline research)
```

## 3. Components

| Component | Responsibility | Key outputs |
|-----------|----------------|-------------|
| **Ingestion** | aiokafka consumers (1–10 tasks, same group), parse with `orjson` + pydantic, push to `asyncio.Queue` | `TickEvent`, `NewsEvent`, `PreflightEvent` |
| **Heartbeat** | ACK every `preflight_tick.seq`; watchdog re-sends last seq every ≤10 s | Ready status |
| **Stage Watcher** | Poll `/stage` (pre-flight: every 2.5 s; simulation: every ~30 s or on anomaly) | `TradingGate` (OPEN/CLOSED), `total_days`, `days_completed` |
| **Day Assembler** | Group events by `simulated_date`; detect day completion; detect sandbox loop (date goes backwards) | `DayBundle(date, candles{}, news[])` |
| **Feature Engine** | Incremental indicators per ticker in numpy ring buffers; cross-sectional features | `FeatureFrame (tickers × features)` |
| **Signal Layer** | News scorer, LightGBM ranker, regime model, vol model, rules; ensemble | `score[ticker]`, `regime`, `vol[ticker]` |
| **Portfolio Constructor** | Scores → target weights → share deltas; turnover & cost gate | `ProposedOrders` |
| **Tax Engine** | Lot tracking, STCG/LTCG estimate, LTCG-wait, loss/gain harvesting suggestions | tax-adjusted edge, harvest orders |
| **Risk Manager** | Hard limits, drawdown breaker, settlement calendar, cash buffer, final veto | `ApprovedOrders` |
| **Order Manager** | Client-side validation, priority queue, token-bucket rate limiter, async send, 429/timeout handling, reconciliation | fills, rejections |
| **Portfolio Ledger** | Local cash/lots/margin/short state; applies fills; mirrors platform rules | equity, exposure |
| **Decision Journal** | One record per ticker-decision (incl. HOLD), with features, scores, reasons, latency | JSONL/SQLite |
| **Recorder** | Separate process; writes every Kafka message to Parquet | training data |
| **Mission Control** | Streamlit dashboard reading the journal + ledger snapshots | live monitoring, jury demo |

## 4. Per-day pipeline and timing budget

```
T+0ms        first price_tick of new simulated_date arrives
T+0..50ms    all candles collected (count == universe size)        ── Day Assembler
T+150ms      news_flash items arrive
T+400ms      NEWS WINDOW CLOSES (configurable)                     ── or earlier if "news_done" heuristic
T+400..450   features updated, models scored (target < 50 ms)      ── Feature Engine + Signal Layer
T+450..470   target weights, risk checks, validation                ── Constructor + Risk
T+470..700   orders sent concurrently (≤4), acks received           ── Order Manager
T+800ms      HARD DEADLINE: if not done, send only what is ready, skip rest
T+3000ms+    next day begins
```

**Latency targets:** p50 tick→order-ack < 700 ms, p95 < 1200 ms. Measure and display them.

**Day completion rule:** a day is complete when `len(candles) == len(universe)` **and**
(`now ≥ T0 + news_window_ms` **or** `now ≥ T0 + hard_deadline_ms`). If some candles are missing at the
deadline, trade only tickers with data.

## 5. Kafka consumption design

- One process, `N = min(10, partitions)` consumer **tasks** with the same `group_id = group-<username>`.
  Kafka assigns partitions across them; all tasks push into a single `asyncio.Queue`.
- This gives partition-level parallelism (jury pillar 3) without splitting state.
- `auto_offset_reset="latest"`, `enable_auto_commit=False`. No offset store needed.
- On disconnect: exponential backoff reconnect (0.5 s → 8 s); the Day Assembler tolerates gaps.
- Messages are parsed defensively: unknown `event_type` → logged and ignored; malformed → counted metric.

## 6. State machine (bot lifecycle)

```
BOOT ──► DISCOVER (GET /stocks, /history warm-up) ──► PREFLIGHT (heartbeat, poll /stage)
                                                          │ stage=simulation & RUNNING
                                                          ▼
                     ┌──────────── PAUSED ◄──────── TRADING ─────────► COMPLETED (stop orders,
                     │  (no orders)  │ RUNNING again      │                    flush journal,
                     └───────────────┴────────────────────┘                    final snapshot)
   Any state ──► SAFE_MODE (kill switch / repeated errors / ACCOUNT_BLOCKED): only risk-reducing orders
```

In the **development** stage the bot runs the same TRADING logic against the sandbox loop
(with `env=sandbox`, which enables loop-detection resets).

## 7. Concurrency model

- `asyncio` event loop; tasks: consumers ×N, heartbeat, stage watcher, day loop, order sender, journal
  writer, snapshot writer.
- CPU-bound model scoring is small (< 50 ms); run it inline. If FinBERT is slower than expected, run it
  in a `ProcessPoolExecutor` with a warm model, or precompute with ONNX int8.
- Journal writes go through a queue to a writer task (never block the decision path on disk I/O).

## 8. Failure modes and responses

| Failure | Detection | Response |
|---------|-----------|----------|
| Kafka disconnect | consumer error / no ticks for 2× interval | reconnect with backoff; no orders on stale data |
| HTTP timeout on order | > 1.5 s no response | **do not blindly retry**: `GET /orders`, resend only if absent |
| HTTP 429 | status code | pause all non-critical requests 31 s; journal it; lower limiter rate |
| Rejection code | response body | map to handler (see doc 06); increment error metric; never repeat same order |
| Model load/inference error | exception | fall back to rules strategy for that day |
| Ledger drift | reconciliation diff > tolerance | adopt platform values, log diff |
| ACCOUNT_BLOCKED | rejection / negative cash | SAFE_MODE: sell longs until cash > buffer |
| Sandbox loop | `simulated_date` decreases | reset feature buffers + regime; keep ledger |
| Stage PAUSED/COMPLETED | stage watcher / rejection | close TradingGate |

## 9. Deployment

- Run on one team laptop on wired/strong Wi-Fi during the live hour; second laptop runs the dashboard
  and a **hot-standby** bot instance in `dry_run` mode (computes decisions, sends nothing) for failover.
- Config via `config.yaml` + env vars (`BOT_ACCESS_TOKEN`, `SERVER_IP`, `TEAM_USERNAME`).
- Kill switch: creating file `KILL` in the run dir (or dashboard button) → SAFE_MODE.
