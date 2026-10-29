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
| **Signal Layer** | **Updated 2026-10-29 — all 6 trained models live:** M1 CSTT ranker (5 ONNX members, `bot/models/model1_onnx.py`), M2 FinBERT NRIM news scorer (3 members, offline from `model2/finbert_cache/` via `bot/model2_bridge.py`), M3 DMN momentum (5 members), M4 volatility (inverse-vol sizing), M5 regime detector (gross_long_target 0.92/0.75/0.40), M6 event tagger (±0.75 thresholds); rules fallback on any failure; ensemble | `score[ticker]`, `regime`, `vol[ticker]` |
| **Ensemble Weights** | **Updated 2026-10-29:** statistically estimated by `bot/portfolio/weight_optimizer.py` (Spearman IC + Markowitz-on-IC + Monte-Carlo bootstrap → `{rank 0.258, news 0.322, mom 0.420}` from `configs/optimized_weights.json`); **UCB1 RL tuner** (`bot/portfolio/rl_tuner.py`, 5 arms incl. optimized arm 0) overrides weights daily and learns from realized equity deltas | live weights |
| **Portfolio Constructor** | Scores → target weights → share deltas; turnover & cost gate; **Updated 2026-10-29:** cash-utilization targeting (85%), unconditional marginal-trade filter (notional < 50× brokerage skipped), exit floor ₹5K vs entry floor ₹75K | `ProposedOrders` |
| **Tax Engine** | Lot tracking, STCG/LTCG estimate, LTCG-wait, loss/gain harvesting suggestions | tax-adjusted edge, harvest orders |
| **Risk Manager** | Hard limits, drawdown breaker, settlement calendar, cash buffer, final veto | `ApprovedOrders` |
| **Order Manager** | Client-side validation, priority queue, token-bucket rate limiter, async send, 429/timeout handling, reconciliation. **Updated 2026-10-29 — zero-rejection architecture:** in-flight cash reservation, SELL/COVER clamped to server-verified holdings (negative holdings = shorts), BUY clamped to server cash × 0.90 × 0.98 − brokerage, POST retried ≤3× (1s/2s backoff; 429 → penalty-box + 5s/10s), give-up paths return `UNKNOWN` + immediate resync (never a phantom fill), server caches updated optimistically on FILLED | fills, rejections |
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

**Updated 2026-10-29 — observed live boot sequence:** M1(5)+M2(3)+M3(5) model members load →
universe discovery (10 tickers, retries 12×/5s — the platform is slow to open discovery right
after boot) → backfill ~1,920 candles → `kafka_connected` → preflight Ready (ACKs seq 1000+)
→ TRADING, orders FILLED on server. `SAFE_MODE → PREFLIGHT` was made a legal transition, and
`RiskManager.seed_equity()` seeds the drawdown peak from the first reconcile (fixes a boot-time
"drawdown 100%" false-positive that used to bounce the bot into SAFE_MODE on every restart).

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
| HTTP timeout on order | > 1.5 s no response | **Updated 2026-10-29:** retry POST ≤3× with 1s/2s backoff; on give-up return status `UNKNOWN` and fire `on_state_divergence` → immediate `_reconcile` (never synthesize a fill from an empty response — that was the root cause of an `INSUFFICIENT_MARGIN: Selling 1142 shares exceeds current holding (-1166)` divergence) |
| HTTP 429 | status code | pause all non-critical requests 31 s; journal it; lower limiter rate |
| Rejection code | response body | map to handler (see doc 06); increment error metric; never repeat same order |
| Model load/inference error | exception | fall back to rules strategy for that day |
| Ledger drift | reconciliation diff > tolerance | adopt platform values, log diff |
| ACCOUNT_BLOCKED | rejection / negative cash | SAFE_MODE: sell longs until cash > buffer |
| Sandbox loop | `simulated_date` decreases | reset feature buffers + regime; keep ledger |
| Stage PAUSED/COMPLETED | stage watcher / rejection | close TradingGate |

## 9. Deployment

**Updated 2026-10-29:**
- **One bot instance only** — never run two consumers of group `group-mn2632`.
- Entry points: `python -m bot.main --env sandbox` (non-interactive) or `python run_bot.py`
  (single-file launcher prompting `[1] KAFKA` / `[2] SIMULATION`; flags `--mode`, `--check`;
  token falls back to `jwt.txt`).
- Config via `configs/default.yaml` + env vars (`BOT_ACCESS_TOKEN`, `SERVER_IP`, `TEAM_USERNAME`,
  `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` — FinBERT runs fully offline from
  `model2/finbert_cache/`).
- Reconcile cadence 15 s (`timing.reconcile_interval_s`); stage poll 10 s in simulation.
- Kill switch: creating file `KILL` in the run dir (or dashboard button) → SAFE_MODE.
