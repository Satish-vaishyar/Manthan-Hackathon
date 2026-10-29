# 11 — Testing & Runbook

> **Updated 2026-10-29 — current state:** `python -m pytest tests/ -q` → **170 tests passing**
> (grew 140 → 154 with P3-14/15, → 157 with P3-17, → 170 with P3-20; includes regression tests
> for the YAML weights round-trip, in-flight cash reservation, server-holding clamps, and UNKNOWN
> order status). New sections §7 (how to run) and §8 (PowerShell quirks) added below.

## 1. Unit tests (must pass before every tag)

| Area | Cases |
|------|-------|
| Ledger | BUY, SELL (FIFO, partial lots), SHORT, COVER, BUY-covers-short (partial and full), fees, DP charge, settlement profit/loss, liquidation order on settlement deficit |
| Validator | each rejection code in doc 03 §3 is prevented; integer/limits; duplicate-in-flight |
| Tax engine | ST vs LT boundary (365 vs 366 days), ₹1L exemption, set-off order (STCL→STCG→LTCG, LTCL→LTCG), FY boundary on 31 March / 1 April, harvesting trigger math |
| Rate limiter | 100/min sustained, burst cap, 429 penalty freezes 31 s, priority order |
| Day assembler | complete-by-count, complete-by-deadline, news after window, loop detection, out-of-order partitions |
| Features | incremental vs pandas reference equality; no lookahead (shifted-data test) |
| Constructor | caps, inverse-vol weights, rebalance threshold, min notional, max orders/day |
| Risk | stops, DD breaker states, settlement guard, cash buffer |

Example hand-computed tax case:
```
Buy 100 @ 1000 on 2021-04-01, sell 100 @ 1200 on 2022-04-02 (366 days) → LTCG 20,000 (−fees per platform rule)
Same FY: LTCG total 20,000 < 1,00,000 → tax 0
Buy 50 @ 2000 on 2021-06-01, sell @ 1800 on 2021-09-01 → STCL −10,000
Earlier STCG +30,000 same FY → net STCG 20,000 → tax 3,000
```

## 2. Integration tests (with mocks)

- `mocks/kafka_replay.py` replays recorded Parquet into a local Kafka (or an in-process fake queue) at
  configurable speed (1 day / 0.5 s for stress).
- `mocks/gateway.py` (FastAPI) implements `/orders`, `/portfolio`, `/stage`, `/stocks`, `/history`,
  `/bot/heartbeat` using the SimBroker, including 429 after 120 req/min.
- Test: full bot run over 300 days at 2× speed → 0 rejections, 0 × 429, ledger == mock portfolio.

## 3. Chaos drills (P2-07)

| Drill | How | Expected |
|-------|-----|----------|
| Kafka outage 20 s | stop broker / block port | Reconnect; no orders on stale data; resumes |
| API slow (2 s latency) | mock delay | Timeout path; no duplicate orders |
| 429 storm | mock returns 429 | Freeze 31 s; no further 429s |
| Malformed message | inject bad JSON | Counted and skipped; no crash |
| Model file missing | rename `models/ranker.txt` | Fallback weights; journal flag |
| Negative cash | force big settlement loss | SAFE_MODE; sells longs; recovers |
| Process crash | `kill -9` mid-run | Restart in < 30 s; ledger rebuilt from `/portfolio` + journal |
| Sandbox loop | dates jump back | Buffers reset; no panic trades |

## 4. Pre-flight checklist (hour 20)

- [ ] On venue Wi-Fi; `SERVER_IP` correct; `GET /stage` OK.
- [ ] Token in env; `GET /portfolio` OK; `ACCOUNT_NOT_VERIFIED` not returned.
- [ ] **Cash reset to ₹10,00,000 confirmed** (check whether the platform resets automatically before live).
- [ ] Running tag `v2-final`; `configs/live.yaml` loaded; `dry_run: false`.
- [ ] Models loaded; meta feature lists match; warm-up inference done.
- [ ] Universe loaded from `/stocks` (count matches portal); history warm-up done.
- [ ] Consumers in group `group-<username>`; heartbeat ACKs flowing; portal shows **Ready**.
- [ ] Dashboard live on laptop 2; hot-standby running in dry-run on laptop 2.
- [ ] Laptops plugged in; sleep/hibernate disabled; Windows Update paused; notifications off.
- [ ] Recorder running (we want the live-run data for the defense).
- [ ] Kill switch tested (create/delete `KILL` file in dry-run).

## 5. Live-hour runbook

**Updated 2026-10-29 — ops facts:**
- **One bot instance only** — never run two consumers of `group-mn2632`.
- Heartbeat ACK on every `preflight_tick`, always within 15 s; portal shows "Waiting for ACK"
  whenever preflight isn't actively broadcasting.
- Kafka `192.168.65.157:9094`, topics `market.ticks` + `market.news`.
- The platform stage bounces development↔simulation; **trades only fill during simulation**.
- Boot reconcile loads **ALL server holdings including legacy positions** and logs
  `startup_holdings_loaded` + a per-name `holdings_snapshot` (qty/avg/last/unrealised P&L) every
  15 s; legacy holdings the model no longer wants are sold on the first day-bundle (exit floor
  ₹5K). Check these events first when diagnosing "the bot is selling things I didn't expect".
- Reconcile cadence 15 s; stage poll 10 s in simulation.
- Logs are JSONL: `runs/live_*.jsonl` (stdout capture) + per-run journals
  `runs/<timestamp>/journal.jsonl`. Analysis artifacts: `runs/_tradebook_analysis.md`,
  `runs/_weight_optimization.md`, `runs/orderbook_full_*.json`.

| Signal on dashboard | Action |
|---------------------|--------|
| Everything green | **Do nothing.** Take screenshots every 10 min for the defense. |
| Any rejection | Read code in journal; if it repeats twice → captain decides SAFE_MODE |
| 429 | Should self-heal; if repeated → lower `rate_per_min` is NOT allowed live; enter SAFE_MODE instead |
| No ticks > 10 s while stage RUNNING | Check Wi-Fi; bot auto-reconnects; if bot process dead → restart (`python -m bot.main --env live`) |
| Drawdown breaker triggered | Expected behaviour; do not override |
| Bot crash | Restart on laptop 1; if laptop 1 broken → switch standby on laptop 2 to live (`--dry-run false`) |
| Stage COMPLETED | Let bot flush journal; export journal, equity, tax report for defense |

## 6. Post-run (hour 23)

- Export: `runs/live/journal.jsonl`, equity curve PNG, trades CSV, latency histogram, tax summary.
- Pick 3 trades to walk through in detail (one news long, one risk exit, one tax decision).

## 7. How to run (Updated 2026-10-29)

```powershell
Set-Location C:\rev\app\Manthan-Hackathon
$env:BOT_ACCESS_TOKEN = (Get-Content <path>\jwt.txt -Raw).Trim()   # token falls back to jwt.txt
$env:TEAM_USERNAME    = "mn2632"
$env:SERVER_IP        = "192.168.65.157"
$env:HF_HUB_OFFLINE   = "1"          # FinBERT served offline from model2/finbert_cache/
$env:TRANSFORMERS_OFFLINE = "1"

python run_bot.py                    # interactive launcher:
#   [1] KAFKA      — connect + ACK + trade on Kafka data only (no local/sim feeds)
#   [2] SIMULATION — REST /api/v1/stocks/{t}/history polling ~1 req/s (≈72/min, under the
#                    100/min budget), no Kafka; M2 neutral (no REST news endpoint)
# flags: --mode kafka|simulation (skip prompt), --check (preflight checks only)

# or non-interactive:
python -m bot.main --env sandbox *> runs\live_$ts.jsonl
```

Observed boot sequence: M1(5)+M2(3)+M3(5) members load → universe (10 tickers) → backfill
~1,920 candles → `kafka_connected` → preflight Ready (ACKs seq 1000+) → TRADING.

Tests: `python -m pytest tests/ -q` → **170 passing**.

## 8. PowerShell 5.1 quirks (Updated 2026-10-29)

- Use `Invoke-WebRequest -UseBasicParsing` (no IE engine available).
- **No `&&`** — use `;` or separate commands.
- **No heredocs** — write files with `Set-Content` / `Out-File` or the editor.
- Redirect both streams with `*> file` (used for the live JSONL logs above).
