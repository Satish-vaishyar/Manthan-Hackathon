# 03 — API & Kafka Contracts

All schemas below are copied from the PS. Fields marked **(verify)** are not documented and must be
confirmed against the portal's API tab (`http://<SERVER_IP>:3003` → APIs) on day one. Implement them
as pydantic models in `bot/contracts.py` with `extra="ignore"` so unknown fields don't crash us.

## 1. Kafka

| Setting | Value |
|---------|-------|
| Bootstrap | `<SERVER_IP>:9094` |
| Security | `PLAINTEXT` |
| Group id | `group-<team_username>` (exact prefix — the readiness tracker relies on it) |
| Topics | `market.ticks`, `market.news` (10 partitions each) |
| Offsets | `auto_offset_reset="latest"`, `enable_auto_commit=False` |
| Max useful consumers | 10 (11th is idle) |

### 1.1 `price_tick` (topic `market.ticks`, T+0)

```json
{
  "event_type": "price_tick",
  "timestamp": "2026-10-19T10:00:00.000Z",
  "simulated_date": "2021-04-15",
  "data": {
    "ticker": "ALPHACORP",
    "open": 1440.00, "high": 1465.50, "low": 1435.20, "close": 1450.25,
    "volume": 1285000
  }
}
```

- One message **per ticker per day** (verify: could also be a batch).
- Execution price for any order today = `data.close`.

### 1.2 `news_flash` (topic `market.news`, T+150 ms)

```json
{
  "event_type": "news_flash",
  "timestamp": "2026-10-19T10:00:00.150Z",
  "simulated_date": "2021-04-15",
  "data": {
    "ticker": "ALPHACORP",
    "headline": "ALPHACORP Q3 NET PROFIT SURGES 24% YOY",
    "description": "Company announces record exports and declares interim dividend...",
    "date": "2021-04-15"
  }
}
```

- 0..n items per day; may reference tickers not in the universe → ignore safely.
- Headlines contain the masked ticker name; strip it before NLP (`<TICKER>` token).

### 1.3 `preflight_tick` (topic `market.ticks`, pre-flight only)

```json
{
  "event": "preflight_tick",
  "event_type": "preflight_tick",
  "timestamp": "2026-10-20T10:00:00.000Z",
  "seq": 14,
  "data": { "status": "STANDBY", "seq": 14 }
}
```

→ respond with `POST /bot/heartbeat {"seq": 14}`. Requirement: ≥ 1 ACK per 15 s.

### 1.4 Dispatch rule

```python
et = msg.get("event_type") or msg.get("event")
if et == "price_tick": ...
elif et == "news_flash": ...
elif et == "preflight_tick": ...
else: metrics.unknown_event += 1
```

## 2. REST API

Base: `http://<SERVER_IP>:3002/api/v1`. Auth header: `Authorization: Bearer <BOT_ACCESS_TOKEN>`
(persistent token from Participant Guide step 1; **no login endpoint**).

| Method | Path | Auth | Use |
|--------|------|------|-----|
| POST | `/orders` | Bearer | Place market order |
| GET | `/orders` | Bearer | Order audit log, fills, rejection reasons (reconciliation, timeout recovery) |
| GET | `/portfolio` | Bearer | Cash, holdings, valuations, margin, P&L (reconciliation) |
| GET | `/portfolio/cash` | Bearer | Lightweight cash check |
| GET | `/stocks` | Bearer | Active universe (ticker, symbol, name, description) |
| GET | `/stocks/:ticker/history?days=N` | optional | Daily bars strictly before current sim date |
| GET | `/stocks/history?ticker=X&days=N` | optional | Alternative form; also `start_date`, `end_date` |
| GET | `/stage` | Public | Lifecycle stage and progress |
| GET | `/simulation/clock` | Public | Current day, total days, remaining days |
| POST | `/bot/heartbeat` | Bearer | Pre-flight ACK |

### 2.1 POST /orders

```json
{ "ticker": "ALPHACORP", "action": "BUY", "quantity": 100, "date": "2021-04-15" }
```

- `action ∈ {BUY, SELL, SHORT, COVER}`; `quantity` int in [1, 1_000_000]; `date` = current simulated date.
- Response body: **(verify)** order id, status, fill price, fees, tax. Store the raw response in the journal.

### 2.2 GET /stage

```json
{
  "stage": "simulation",          // development | ready | preflight | simulation
  "status": "RUNNING",            // IDLE | RUNNING | PAUSED | COMPLETED
  "has_run": true,
  "simulation_id": "sim_nse_2026",
  "interval_seconds": 4,
  "total_days": 30,
  "days_completed": 12,
  "current_date": "2021-01-16"
}
```

Trading gate OPEN iff `stage == "simulation" and status == "RUNNING"`
(in sandbox: `stage == "development" and status == "RUNNING"`, verify).

### 2.3 GET /stocks

```json
{ "status": "ok",
  "stocks": [ { "ticker": "ALPHACORP", "symbol": "ALPHACORP",
                "name": "Alpha Corporation Limited",
                "description": "Leading diversified manufacturing and industrial conglomerate..." } ] }
```

### 2.4 GET /stocks/:ticker/history

```json
{ "status": "ok", "ticker": "ALPHACORP", "current_simulation_date": "2021-04-15",
  "requested_days": 90, "date_range": {"start": "2021-01-15", "end": "2021-04-14"},
  "total_records": 61,
  "prices": [ {"date": "2021-01-15", "open": 1410.0, "high": 1432.5, "low": 1405.0,
               "close": 1422.75, "volume": 980000} ] }
```

`days` is calendar days (default 365). Anti-lookahead: only dates < current simulation date.

## 3. Rejection codes → handler

| Code | Root cause | Our prevention | Our reaction if it still happens |
|------|-----------|----------------|-----------------------------------|
| `ACCOUNT_NOT_VERIFIED` | Not verified at desk | Verify in first 30 min | Alert humans; stop orders |
| `SIMULATION_NOT_RUNNING` | Not in RUNNING | Trading gate | Close gate, force stage poll |
| `INSUFFICIENT_CAPITAL` | cost + ₹20 > cash | Ledger check incl. fees & buffer | Reconcile cash; shrink size next time |
| `INSUFFICIENT_MARGIN` | short notional + ₹20 > cash | Ledger check | Reconcile; skip short |
| `ACCOUNT_BLOCKED` | Negative cash | Cash buffer, settlement planning | SAFE_MODE: sell longs to restore cash |
| `NO_SHORT_POSITION` | COVER w/o short | Ledger check | Reconcile portfolio |
| `INVALID_TICKER` | Not in universe | Use `/stocks` only | Remove ticker, refresh universe |
| `INVALID_QUANTITY` | 0/neg/decimal/>1e6 | `int`, clamp | Log bug |
| `RATE_LIMITED` / HTTP 429 | > 120/min | Token bucket @ ~100/min | Freeze 31 s; reduce rate 20% |

## 4. Request budget

```
Per minute:        120 hard limit  →  we plan 100 (safety)
Days per minute:   15 (4 s/day)   –  20 (3 s/day)
Budget per day:    ~5–6 requests
  Orders:          ≤ 4 per day (priority 1)
  Reconciliation:  GET /portfolio once every ~5 days (priority 2)
  Stage/clock:     public; poll every 30 s during simulation (verify whether it counts)
  History:         only at startup / pre-flight (spread out, ≤ 1 per second)
Pre-flight:        heartbeat ≤ 1 per preflight_tick, dedupe identical seq
```
