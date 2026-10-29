# WORK LOG — Live Bot Integration & Continuous Tuning

Chronological record of everything done on the live trading bot: integration,
bug fixes, tradebook analysis, tuning passes, and the RL weight tuner.
Kept up to date after every working session.

---

## 1. Mission

Integrate all 6 trained models + rule-based fallback into the live bot, trade
per the `docs/` spec on live Kafka data, and continuously improve trading
performance (analysis-driven tuning, **no full retraining**).

Platform constraints (hard rules):
- Pre-issued `BOT_ACCESS_TOKEN` (jwt.txt), `group_id=group-mn2632`
- Kafka at `192.168.65.157:9094`
- ACK every `preflight_tick` via `POST /api/v1/bot/heartbeat {"seq": N}`
  continuously (within 15 s)
- ONE bot instance only — never run two consumers

---

## 2. Models wired (all live, none retrained)

| # | Model | Role | Status |
|---|-------|------|--------|
| M1 | CSTT ranker (5 members) | cross-sectional ranking | live |
| M2 | FinBERT NRIM (3 members) | news sentiment, offline from `model2/finbert_cache/` | live |
| M3 | DMN (5 members) | momentum scorer | live |
| M4 | volatility model | inverse-vol sizing | live |
| M5 | regime detector | gross_long_target scaling | live |
| M6 | event tagger | event long/short thresholds | live |
| — | rules fallback | engages on any model failure | live |

Boot sequence observed: M1(5)+M2(3)+M3(5) load → universe(10) →
backfill ~1,920 candles → kafka_connected → preflight Ready (ACKs seq 1000+)
→ orders FILLED on server.

---

## 3. Launcher

`run_bot.py` (commit `bc78092`) — single-file launcher:
- Prompts `[1] KAFKA` (connect + ACK + trade on Kafka data only) /
  `[2] SIMULATION` (REST `/api/v1/stocks/{t}/history` polling, no Kafka)
- Flags: `--mode`, `--check`; token falls back to jwt.txt
- Sim-mode poller emits PriceTickEvents at ~1 req/s (≈72/min, under the
  100/min limit); no REST news endpoint exists → M2 neutral in sim mode
- **Orders are driven by Kafka data only** in KAFKA mode (no local/sim feeds)

---

## 4. Server-contract fixes (the "bot can't talk to the platform" class)

All in `bot/contracts.py` / `bot/main.py` / `bot/model2_bridge.py`:

1. **AliasChoices on response models** — server returns different key names
   than the spec:
   - `OrderResponse`: `order_id`, `execution_price`, `simulated_date`,
     `executed_at`, `dp_charges`
   - `PortfolioResponse`: `holdings→positions`,
     `available_cash/cash_balance→cash`, `total_wealth→equity`
   - `Position`: `avg_buy_price/average_buy_price→avg_price`
2. **sys.modules purge in model2_bridge** — fixes M1/M2 `model.py` filename
   collision (whichever loaded second used to silently shadow the first).
3. **Preflight router checks `event_type` first** — heartbeats were being
   misrouted.
4. **Heartbeat accepts wrapped `{_status, _body}`** responses.
5. **SAFE_MODE→PREFLIGHT made legal** in the lifecycle state machine.
6. **`_discover_universe` retries 12×/5s** — platform is slow to open
   discovery right after boot.

Platform quirks discovered:
- `/api/v1/orders` paginates at 50 — must pass `?limit=10000` for full pull.
- No `/api/v1/news` or `/api/v1/stocks/{t}` endpoints (404); history via
  `/api/v1/stocks/{t}/history?days=N`.
- Stage bounces development↔simulation; trades only fire in simulation
  windows; portal shows "Waiting for ACK" whenever preflight isn't actively
  broadcasting.

Commits: `d3ec439`, `9008a8f`, `4f2709a`.

---

## 5. Tradebook baseline & post-mortem (2026-10-29)

- Full orderbook: `runs/orderbook_full_20261029T202525Z.json` (400 orders)
- Analysis: `runs/_tradebook_analysis.md` + `runs/_analyze.py`
- Baseline: **+₹2.31L realized over ~168 sim-days**
- Bleeders identified:
  - **BLUX** — 22 trades, 45.5% win-rate, **−₹27,071**
  - mid-band churn — −₹24K
  - scalp churn — −₹13K
  - 8 `INSUFFICIENT_CAPITAL` rejections (orders sized at ~99% of cash)

### Tuning pass 1 (commit `e0fd26b`)
- weights `{rank 0.55, news 0.20, mom 0.25}`, top_k 6
- thresholds ±0.75, rebalance 0.05, stop_atr 3.5, dd_soft 0.12
- `exclude_tickers: [BLUX]`
- Result (40-min tuned run, `live_tuned_*.jsonl`): 207 orders, 0 BLUX,
  **+₹1,50,940 realized on new sells**

---

## 6. Six-point improvement suite + RL tuner (commit `74d9c19`, P3-14/P3-15)

User goals → implementation mapping:

### (1) Cash utilization > 80% majority of the time
- `risk.cash_buffer` 5% → **2%**
- New `risk.target_cash_utilization: 0.85`
- Constructor scales target weights up/down so long exposure tracks the
  target (scale clamped to [0.5, 1.5]); only active when caller explicitly
  passes the param (no silent change to legacy call sites)

### (2) Max profit in fewer trades
- `strategy.min_trade_notional` ₹50,000 → **₹75,000** (0.05%/side vs 0.07%)
- top_k 8 → 6 (concentration)

### (3) Less brokerage
- Constructor drops trades with notional < 50× brokerage (~₹1,000);
  later (P3-16) made **unconditional**
- Same 4 orders/day cap, but each order is now ≥₹75K

### (4) Less downtime
- **Boot DD-blip fixed**: `DrawdownState.peak_equity` used to seed at
  ₹1.0M while real equity was 0 → "drawdown 100%" false-positive →
  needless SAFE_MODE bounce on every restart. New
  `RiskManager.seed_equity()` seeds the peak from the first reconcile.
- `timing.reconcile_interval_s` 30 → **15 s** (was hard-coded; now config)
- `timing.stage_poll_simulation_s` 30 → **10 s**

### (5) Fewer wrong-trade errors (over-margin etc.)
- New `risk.max_cash_per_trade: 0.90` — validator rejects any BUY sized
  > 90% of available cash (root cause of the 8 INSUFFICIENT_CAPITAL
  rejections)

### (6) Maintain max total wealth
- `seed_equity()` calibrates the DD breaker from trade 1
- Trailing `_peak_equity` tracked on the bot for observability
- Cash-utilization targeting mechanically lifts realized wealth

### RL tuner (P3-15) — `bot/portfolio/rl_tuner.py`
- **UCB1 multi-armed bandit** over 5 hand-picked weight triples
  (rank-heavy / momentum-heavy / balanced / rank+mom-no-news / news-back)
- Opt-in via `strategy.rl_tuner.enabled` (enabled in `default.yaml`)
- Each day: tuner overrides `signals.weights` before `ensemble.evaluate()`;
  each reconcile: realized equity delta fed back as reward
- Round-robin warmup until every arm pulled once, then UCB1 with c=1.4
- Observability: `rl_tuner_ready`, `rl_tuner_commit`, `rl_tuner_switch`
  log events + `stats()` introspection

### Latent bug found & fixed
`StrategySettings.weights` was popped from the YAML dict before the
constructor → YAML weights silently replaced by dataclass defaults.
Fixed; regression test `test_config_yaml_roundtrip.py` added.

Tests: **140 → 154 passing** (14 new).

---

## 7. P3-16 — "keep rejected orders at 0" hardening

1. **Cash reservation for in-flight BUYs** (`validator.py`): batch of
   same-day BUYs each used to validate against the *full* wallet; later
   ones got server-rejected once earlier fills consumed cash. Now
   `mark_in_flight(req, price)` reserves `qty*price + brokerage` and
   `_validate_buy` checks against `cash - reserved_cash`; reservation
   released on `clear_in_flight`. OrderManager passes the last-close price
   at mark time.
2. **Marginal-trade filter made unconditional** (`constructor.py`):
   notional < 50× brokerage is always skipped, not just when the order
   budget overflows.

## 7b. P3-17 — idle-cash deployer (realised-profit redeployment)

Realised profits land as cash after fills, but the bot previously only
traded on day-bundles — freed capital sat idle between bundles. Now:

- After **every reconcile** (15 s cadence) `_maybe_deploy_idle_cash(p)`
  recomputes utilisation from the **server's** cash/equity (so freshly
  realised profit is included).
- If utilisation is > 5pp below `risk.target_cash_utilization` (0.85), it
  places ONE extra BUY for the highest-conviction ticker from the latest
  evaluation, sized to the utilisation gap, capped by
  `max_cash_per_trade × cash × 0.99` (1% price-drift buffer) minus
  brokerage.
- All guards still apply: daily order budget (4/day), min notional ₹75K,
  per-name 15% cap, validator incl. in-flight cash reservation.
- Config: `strategy.idle_cash_deploy {enabled, epsilon}` in
  `configs/default.yaml`; logged as `idle_cash_deploy` events.
- Caches `_last_scores` / `_last_regime` / `_last_sim_date` from `_on_day`
  so between-bundle orders reuse the freshest conviction ranking.

Tests: 154 → **157 passing**.

## 7c. P3-18 — statistically-estimated ensemble weights (commit `3475173`)

Hand-picked weights replaced by data-estimated ones
(`bot/portfolio/weight_optimizer.py`, pure numpy):
- Per-day cross-sectional **Spearman IC** of each signal vs next-day return
- Four estimators: IR-weighted (Grinold & Kahn), inverse-variance,
  **Markowitz-on-IC** (w = simplex(Σ⁻¹μ), Ledoit-Wolf-shrunk covariance),
  and a **1,000× Monte Carlo day-bootstrap** of the Markowitz estimator
- Final weights = bootstrap mean, shrunk toward equal weights by the
  empirical P(w≤0) per signal (data-driven James–Stein shrinkage)
- `scripts/optimize_weights.py` rebuilds panels from all journals
  (12,396 records, **500 sim days × 10 tickers**)

Result: `IC mean rank .017 / news .078 / mom .141` → optimal
**{rank .258, news .322, mom .420}** — the old hand-picked rank=0.55 was
overweighting the *weakest* signal. Boot loads
`configs/optimized_weights.json` when `strategy.weights_mode: optimized`
and injects it as RL-tuner arm 0.

Also fixed: DD breaker now uses **server-reconciled equity**
(`_last_server_equity`) — the stale local ledger mirror caused the false
`RISK_DD_HARD 20%` → SAFE_MODE bounce that produced 0 trades on the eval
run.

## 7d. P3-19 — startup holdings review + always-visible holdings

- Boot reconcile loads ALL current holdings from the server (legacy
  positions included) and logs `startup_holdings_loaded` +
  `holdings_snapshot` (per-name qty/avg/last/unrealised-P&L) at every
  15 s reconcile.
- Constructor exit path uses a new lower floor
  `strategy.exit_min_notional: 5000` (entries stay at 75K): any legacy
  holding the model no longer wants gets a SELL on the first day-bundle —
  exits are also exempt from the marginal-trade fee filter.

## 7e. P3-20 — zero server rejections

Root cause of `INSUFFICIENT_MARGIN: Selling 1142 shares exceeds current
holding (-1166)`: empty order responses were synthesised as phantom
fills → local ledger diverged from the server (server was already short
-1166 while the ledger believed long).

Fixes:
1. **No phantom fills**: empty/unparseable order responses now return
   status `UNKNOWN` and trigger an immediate reconcile
   (`on_state_divergence` callback → `_on_state_divergence`).
2. **Server-holding clamps**: reconcile caches `_server_holdings` /
   `_server_shorts` (negative holdings treated as shorts); every SELL /
   COVER is clamped to the server-verified quantity before submission,
   and skipped entirely when the server shows nothing sellable.
3. 4xx rejections also trigger an immediate resync.

Tests: **170 passing**.

---

## 8. Current live run (15-min evaluation)

- Bot: `runs/live_p315_20261029T222542Z.jsonl` (PID 26852)
- Booted: 2026-10-29 16:55:54 UTC
- Evaluation at T+15 min checks:
  - `reconciled` cash util vs 0.85 target
  - `order_submitted/filled` counts, notionals ≥ ₹75K
  - **0 BLUX trades**, **0 server rejections** (`INSUFFICIENT_*`)
  - `rl_tuner_ready` / `rl_tuner_commit` events
  - no boot `drawdown 100%` false-positive
  - reconcile cadence ~15 s

---

## 9. How to run

```powershell
Set-Location C:\rev\app\Manthan-Hackathon
$env:BOT_ACCESS_TOKEN = (Get-Content C:\Users\Samarth\AppData\Local\Temp\opencode\jwt.txt -Raw).Trim()
$env:TEAM_USERNAME    = "mn2632"
$env:SERVER_IP        = "192.168.65.157"
$env:HF_HUB_OFFLINE   = "1"
$env:TRANSFORMERS_OFFLINE = "1"
python run_bot.py                 # interactive [1] KAFKA / [2] SIMULATION
# or non-interactive:
python -m bot.main --env sandbox *> runs\live_$ts.jsonl
```

Tests: `python -m pytest tests/ -q` (154 passing as of P3-16).

---

## 10. Commit history

| Commit | Content |
|--------|---------|
| `d3ec439`–`4f2709a` | server-contract fixes, model wiring, lifecycle |
| `bc78092` | `run_bot.py` launcher (KAFKA/SIM prompt) |
| `e0fd26b` | tuning pass 1 (weights, BLUX blacklist, wider stops) |
| `74d9c19` | P3-14/P3-15: 6-point suite + RL tuner, 154 tests |
| pending | P3-16: rejection-0 hardening (cash reservation) + this doc |

**Push to origin still unverified** — `git push origin master` timed out
once; re-attempt and confirm.
