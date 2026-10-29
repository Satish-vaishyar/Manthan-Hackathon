# 06 — Risk, Tax & Execution

## 1. Portfolio Ledger (local mirror of the platform)

```python
@dataclass
class Lot:            # one per BUY fill (FIFO assumed — verify with platform)
    ticker: str
    qty: int
    price: float      # fill price (close of buy day)
    buy_date: date    # simulated date
    fees: float       # ₹20 allocated to this lot

@dataclass
class ShortPos:
    ticker: str
    qty: int
    entry_price: float
    open_date: date
    margin_locked: float   # = qty * entry_price

@dataclass
class Ledger:
    cash: float
    lots: dict[str, deque[Lot]]
    shorts: dict[str, ShortPos]
    realised: dict[str, FYGains]   # key = FY label "FY2021-22"
    last_close: dict[str, float]
```

### Apply-fill rules (must match platform exactly)

| Action | Cash change | Position change |
|--------|-------------|-----------------|
| BUY q @p | `−(q·p + 20)` | If short open: cover `min(q, short_qty)` first (P&L settled, margin released), rest becomes a new Lot |
| SELL q @p | `+(q·p − 20 − 15.34)` | Consume lots FIFO; record realised gain per lot as ST/LT |
| SHORT q @p | `−(q·p + 20)` moved to margin (cash −q·p −20, margin +q·p) | New/increase ShortPos (weighted avg entry) |
| COVER q @p | `+margin_release + (entry − p)·q − 20` | Reduce ShortPos |
| Settlement | `+(entry − close)·qty` for all shorts, margin released | All shorts closed |

`equity = cash + margin_locked + Σ long_qty·close + Σ short_mtm` (matches the PS wealth formula;
verify sign convention of "Short Market Value" against `/portfolio`).

### Reconciliation
- **Updated 2026-10-29:** `GET /portfolio` on a **15 s cadence** (`timing.reconcile_interval_s`),
  at startup, after any rejection or `UNKNOWN` order status, and after each settlement day.
- If `|cash_local − cash_platform| > ₹100` or any quantity differs → adopt platform values, log `LEDGER_DRIFT`.
- **Server-holding cache:** each reconcile caches `_server_holdings` / `_server_shorts`
  (**negative holdings from the server are treated as shorts**). Every SELL/COVER is clamped to
  the server-verified quantity before submission and skipped entirely when nothing is sellable.
- **Startup holdings review (P3-19):** the boot reconcile loads ALL server holdings (legacy
  positions included) and logs `startup_holdings_loaded` + a per-name `holdings_snapshot`
  (qty/avg/last/unrealised P&L) at every 15 s reconcile. Legacy holdings the model no longer
  wants are sold on the first day-bundle (exit floor ₹5K).
- Server-side caches are updated **optimistically** on every FILLED order
  (`_apply_fill_to_server_cache`) so clamps stay accurate between reconciles.

## 2. Risk Manager

### 2.1 Hard limits (veto; never overridden)

| Limit | Value |
|-------|-------|
| Max weight per name (long) | 15% of equity |
| Max per sector cluster | 30% |
| Max gross long exposure | regime cap (92 / 75 / 40%) |
| Max gross short exposure | 20% of equity |
| Max single short | 5% of equity |
| Cash buffer | **Updated 2026-10-29:** 2% of equity (`risk.cash_buffer: 0.02`) + short margin; constructor targets 85% cash utilization |
| Max cash per trade | **Updated 2026-10-29:** `risk.max_cash_per_trade: 0.90` — validator rejects any BUY sized > 90% of available cash (root cause of 8 observed `INSUFFICIENT_CAPITAL` rejections) |
| Max orders/day | 4 (config) |
| Max quantity | 1,000,000 and integer ≥ 1 |
| No trading when | gate closed, SAFE_MODE (except risk-reducing), stale data (> 2 intervals old) |

### 2.2 Dynamic controls

| Control | Rule |
|---------|------|
| ATR stop (long) | exit if `close < max_close_since_entry − 3.5·ATR14` (**Updated 2026-10-29**, `stop_atr_mult: 3.5`) |
| ATR stop (short) | cover if `close > min_close_since_entry + 2.0·ATR14` |
| Time stop | event longs ≤ 5 days, event shorts ≤ 3 days |
| Drawdown breaker | **Updated 2026-10-29:** equity −12% (`dd_soft: 0.12`) from peak → halve all target weights; −15% (`dd_hard: 0.15`) → flat 5 days then ramp back. The breaker uses **server-reconciled equity** (`_last_server_equity`), not the stale local ledger mirror (a stale mirror once caused a false `RISK_DD_HARD 20%` → SAFE_MODE bounce with 0 trades), and `RiskManager.seed_equity()` seeds the peak from the first reconcile — fixing the boot-time "drawdown 100%" false-positive on every restart |
| Shock filter | |ret_1| > 4σ → size ×0.5 for that ticker for 3 days |
| Settlement guard | no new SHORT if `days_to_settlement < 3`; COVER all shorts on `days_to_settlement == 1` unless explicitly allowed |

### 2.3 Settlement calendar
- Settlement every 10 trading days on alternate Fridays. Build the calendar from simulated dates:
  count trading days since sim start; also detect Fridays from `simulated_date.weekday() == 4`.
- **Verify on day 1** which Friday is the first settlement (watch for forced square-off in sandbox and
  log it). Until verified, treat **every** Friday as a potential settlement (conservative).

## 3. Tax Engine

### 3.1 Rules implemented
- Holding period = sell_date − buy_date (calendar days). `> 365` → LTCG else STCG.
- STCG 15%, LTCG 10% above ₹1,00,000 exemption per FY.
- FY = 1 April → 31 March of the simulated calendar.
- Set-off order (Sec 70/71): STCL vs STCG first, remaining STCL vs LTCG; LTCL only vs LTCG.
- Short P&L treatment: **verify** (likely STCG/STCL).

### 3.2 Tax-aware decisions

| Technique | Trigger | Action |
|-----------|---------|--------|
| **LTCG wait** | A lot with unrealised gain is 335–365 days old and sell signal is weak (score > −0.5) | Delay the SELL until day 366 |
| **Loss harvesting** | Last 10 trading days of FY, realised net STCG > 0, a holding has unrealised loss > ₹5,000 | SELL the loser to offset; re-BUY next day if still in target (no wash-sale rule in India). Only if tax saved > ₹70 (2 orders + DP) |
| **LTCG gain harvesting** | Last 10 trading days of FY, LTCG realised this FY < ₹1L, a lot has LT unrealised gain | SELL up to the remaining exemption and re-BUY next day → step up cost basis tax-free |
| **Lot selection** | Partial sells | If platform uses FIFO we can't choose; otherwise sell highest-cost lots first |
| **Tax-adjusted edge** | Every SELL of a gain | `cost += est_tax(gain, ST/LT)` in the cost gate |

### 3.3 Reporting
Per FY: realised STCG, STCL, LTCG, LTCL, exemption used, estimated tax, tax saved by harvesting,
fees paid, DP paid. Shown on the dashboard and compared against the portal's Tax & Annual Filing view.

## 4. Order Manager

### 4.1 Pipeline

```
ProposedOrder → validate() → RiskManager.approve() → PriorityQueue → RateLimiter.acquire()
             → POST /orders (httpx.AsyncClient, keep-alive, timeout 1.5 s, retry ≤3×)
             → on 2xx: parse fill → Ledger.apply() → optimistic server-cache update → Journal
             → on rejection (4xx): handler table (doc 03 §3) → immediate resync → Journal
             → on timeout/give-up: status UNKNOWN → on_state_divergence → immediate _reconcile
```

### 4.1a Zero-rejection architecture (Updated 2026-10-29, P3-16…P3-20 — all live)

1. **Cash reservation for in-flight BUYs** (`validator.py`): `mark_in_flight(req, price)` reserves
   `qty·price + brokerage`; `_validate_buy` checks against `cash − reserved_cash`; released on
   `clear_in_flight`. A batch of same-day BUYs can no longer overdraw the wallet.
2. **Server-holding clamps:** every SELL/COVER clamped to server-verified holdings cached at each
   reconcile (negative holdings = shorts); skipped when nothing sellable.
3. **BUY clamps:** every BUY clamped to server cash × 0.90 (`max_cash_per_trade`) × 0.98
   (price-drift buffer) − brokerage.
4. **Order POST retried up to 3×** — network errors: 1s/2s backoff; HTTP 429: penalty-box + 5s/10s.
5. **No phantom fills, no blind REJECTEDs:** every give-up path (empty/unparseable body, retries
   exhausted) returns status `UNKNOWN` and triggers an immediate `_reconcile` via
   `on_state_divergence`. Root-cause example fixed: `INSUFFICIENT_MARGIN: Selling 1142 shares
   exceeds current holding (-1166)` was caused by phantom fills synthesised from empty order
   responses while the server was already short −1166.
6. **Optimistic cache updates** on every FILLED order (`_apply_fill_to_server_cache`).
7. **Marginal-trade filter unconditional:** notional < 50× brokerage (~₹1,000) is always skipped;
   risk-reducing exits are exempt.

Result on the tuned live run: **0 server rejections** (`INSUFFICIENT_*` count = 0).

### 4.2 Validation checklist (pure function, unit-tested)

```
ticker in universe (and not in strategy.exclude_tickers — Updated 2026-10-29: [BLUX])
action in {BUY, SELL, SHORT, COVER}
quantity is int and 1 ≤ q ≤ 1_000_000
gate open (or order is risk-reducing in SAFE_MODE)
BUY:   q·close + 20 ≤ (cash − reserved_cash) − buffer      # Updated 2026-10-29: in-flight reservation
       and q·close ≤ 0.90 × cash                           # max_cash_per_trade
SELL:  q ≤ long_qty(ticker) AND q ≤ server-verified holding # Updated 2026-10-29: server clamp
SHORT: q·close + 20 ≤ cash − buffer ; days_to_settlement ≥ 3
COVER: short_qty(ticker) ≥ q, clamped to server short qty   # Updated 2026-10-29
date = current simulated_date
no duplicate (same ticker+action) already in flight today
```

### 4.3 Rate limiter
- Token bucket, capacity 10, refill 100/60 tokens per second, shared by **all** authenticated calls.
- Priorities: 0 = heartbeat / settlement covers, 1 = risk exits, 2 = entries/rebalances, 3 = reconciliation, 4 = history.
- On 429: `penalty_until = now + 31 s`; drain nothing until then; reduce refill by 20% for 2 minutes.
- Metrics: requests/min (rolling), 429 count, queue depth.

### 4.4 HTTP client
- `httpx.AsyncClient(base_url, headers={"Authorization": f"Bearer {token}"}, http2=False, timeout=1.5,
  limits=Limits(max_keepalive_connections=10))`.
- Created once; reused; warmed up with `GET /stage` at boot.

## 5. SAFE_MODE

Entered on: kill switch, `ACCOUNT_BLOCKED`, ≥ 3 rejections in 20 days, ledger drift twice in a row,
unhandled exception in decision loop.

Behaviour: no new BUY/SHORT; cover shorts before settlement; if cash < 0 sell longs (smallest
unrealised gain first) until cash ≥ buffer; journal every action; exit SAFE_MODE only by human command
or automatically after 10 clean days (config).
