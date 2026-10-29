# NOTES.md — venue facts discovered during the event

## Network & endpoints

| Item | Value | Source |
|---|---|---|
| Portal UI | `http://192.168.65.157:3003` | docs/01 |
| Order Gateway REST | `http://192.168.65.157:3002/api/v1` | docs/01 |
| Kafka broker | `192.168.65.157:9094` (PLAINTEXT, no auth) | docs/01 + smoke test |
| Stage | `GET /api/v1/stage` | docs/03 |
| Bot status | `GET /api/v1/bot/status` | docs/03 |
| Heartbeat | `POST /api/v1/bot/heartbeat` (body: `{seq: int}`) | docs/03 |
| Stocks (universe) | `GET /api/v1/stocks` | docs/03 |
| History | `GET /api/v1/stocks/history?ticker=X&days=N` | docs/05 |
| Orders | `POST /api/v1/orders`, `GET /api/v1/orders` | docs/03 |
| Portfolio | `GET /api/v1/portfolio`, `GET /api/v1/portfolio/cash` | docs/03 |
| Tax report | `GET /api/v1/tax/report` | docs/03 |
| Flags | `GET /api/v1/flags` | docs/03 |
| SSE stream | `GET /api/v1/events?token=<JWT>` | docs/03 |
| Simulation clock | `GET /api/v1/simulation/clock` | docs/03 |

## Stage observations (Oct 29 2026)

- Platform is currently in `development` stage.
- `interval_seconds=5` (not 3-4 s as the PS says — sandbox override).
- `total_days=500`, `active_dataset=independent_test_2yr` (10 tickers).
- Kafka is reachable but only emits `price_tick` + `news_flash` — no
  `preflight_tick` yet. Heartbeat returns `INVALID_SEQ` for null-seq ACKs.

## Universe divergence

| Source | Tickers |
|---|---|
| REST `/api/v1/stocks` | 40 tickers (AETX, AQUA, BAMR, …, WLFG) |
| Kafka `market.ticks` / `market.news` | 10 tickers (AERX, BLUX, DSNY, HAVN, HMDP, MOTR, ORCX, PLNX, ROUT, SONA) |
| Overlap | **none** — REST rejects Kafka tickers as `INVALID_TICKER` |

The bot must use the Kafka universe at runtime (not the REST list). The
`/stocks` endpoint is still used to populate sector metadata for the 40-ticker
universe (for backtests).

## Reconciliation starting state (after 8 smoke-test orders Oct 29)

- Cash: ₹9,99,805.49
- Realised P&L: −₹3.83
- Open positions: 0
- All 4 round-trips closed cleanly

## Smoke-test transcript (Oct 29 14:52)

```
[14:52:37] === Manthan bot connect ===
[14:52:37] server=192.168.65.157  kafka=192.168.65.157:9094  group=group-MN2632
[14:52:38] connected — subscribed to ['market.ticks', 'market.news']
[14:52:39] price_tick #1  2025-11-17 DSNY close=105.69
[14:52:39] news_flash #1  2025-11-17 SONA: Barcelona to play first game at renovated Camp Nou
[14:52:54] status: kafka_connected=True consumers=1 heartbeat_received=False last_seq=None
```

✅ Kafka connection first-try.
✅ Portal sees 1 active consumer.
⏳ Heartbeat ACK will start the moment the platform enters `preflight`.

## Known gotchas

1. **Repo is private** — `git ls-remote` works but anonymous `git fetch`
   fails with `RPC failed; curl 56 Recv failure`. Clone from your own shell
   then move into `C:\rev\app\`.
2. **Group ID is exact-prefix sensitive** — must be `group-<TEAM_USERNAME>`,
   the readiness tracker keys off it.
3. **Broker is :9094, not :9092** — the docs are explicit; the usual Kafka
   port is wrong here.
4. **`auto_offset_reset="latest"`** — any preflight_tick sent before the
   consumer joined is missed by design. Plan to be online before preflight
   starts.
5. **`BOT_ACCESS_TOKEN` is pre-issued** — paste from the portal DevTools
   network panel; do NOT call `/auth/login` from the bot.
6. **DD breakers hard-stop the bot** — `RISK_DD_HARD` triggers
   `SAFE_MODE` (only HOLD allowed). Tighten `dd_hard` in `live.yaml` if
   needed.

## Sandbox quirks observed

- Interval is 5 s, not 3-4 s.
- 10 active Kafka tickers, NOT 40. Treat the REST 40-ticker list as
  backtest/training universe only.
- Some tickers (HMDP, ORCX) show tiny price moves (₹0.01) which is normal.