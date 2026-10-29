# `scripts/connect.py` — readiness-only Kafka connector

The bot is in training mode, so this script does **only** what the portal needs to mark your team **Ready**:

1. Connect to Kafka broker `<SERVER_IP>:9094`
2. Subscribe to `market.ticks` + `market.news` with `group_id=group-<TEAM_USERNAME>`
3. ACK every `preflight_tick` via `POST /api/v1/bot/heartbeat`
4. Watchdog: re-send the last seq every 10 s (portal requires ≥1 ACK / 15 s)
5. Poll `GET /api/v1/bot/status` every 15 s and print what the portal sees

No orders, no strategy, no model calls — pure readiness.

## Setup

```bash
pip install -r scripts/requirements-connect.txt
```

## Run

```bash
export SERVER_IP=192.168.65.157
export KAFKA_PORT=9094
export TEAM_USERNAME=MN2632
export BOT_ACCESS_TOKEN="eyJhbGciOi..."
python scripts/connect.py
```

On Windows (PowerShell):
```powershell
$env:SERVER_IP = "192.168.65.157"
$env:KAFKA_PORT = "9094"
$env:TEAM_USERNAME = "MN2632"
$env:BOT_ACCESS_TOKEN = "eyJhbGciOi..."
python scripts/connect.py
```

## Expected output (when the platform is in `preflight` stage)

```
[12:34:01] === Manthan bot connect ===
[12:34:01] server=192.168.65.157  kafka=192.168.65.157:9094  group=group-MN2632
[12:34:01] gateway=http://192.168.65.157:3002
[12:34:01] token=Bearer **********
[12:34:01] connecting to Kafka 192.168.65.157:9094 (group=group-MN2632) …
[12:34:03] connected — subscribed to ['market.ticks', 'market.news']
[12:34:05] status: kafka_connected=True consumers=2 heartbeat_received=True last_seq=3 is_ready=True
[12:34:10] preflight_tick seq=4 -> ACK
…
```

When `is_ready=True`, the portal will show:
- ✅ **Kafka market data:** Connected
- ✅ **Pre-flight heartbeat:** Acknowledged

## Notes

- The Kafka broker is **`<SERVER_IP>:9094`** (per `docs/01_PROBLEM_SUMMARY.md` §2), not `9092`.
- `KAFKA_GROUP` is `group-<TEAM_USERNAME>` — exact prefix; the readiness tracker keys off it.
- `auto_offset_reset="latest"` and `enable_auto_commit=False` (per `docs/03_API_AND_KAFKA_CONTRACTS.md`).
- The token is the same JWT the portal UI uses; paste it from the browser DevTools network panel.

## Smoke-test result

Run on `192.168.65.157` against the live platform:

```
[14:52:37] === Manthan bot connect ===
[14:52:37] server=192.168.65.157  kafka=192.168.65.157:9094  group=group-MN2632
[14:52:37] connecting to Kafka 192.168.65.157:9094 (group=group-MN2632) …
[14:52:38] connected — subscribed to ['market.ticks', 'market.news']
[14:52:39] price_tick #1 2025-11-17 DSNY close=105.69
[14:52:39] news_flash #1 2025-11-17 SONA: Barcelona to play first game at renovated Camp Nou on Saturd
[14:52:54] status: kafka_connected=True consumers=1 heartbeat_received=False last_seq=None is_ready=False
```

✅ Kafka connection established on first try.
✅ Live ticks and news arriving.
✅ Portal sees 1 active consumer for our group.
⏳ Heartbeat ACK will fire as soon as the platform enters `preflight` stage (in `development` it returns `INVALID_SEQ` because there is no preflight_tick to echo — that is correct).

## Stage-dependent behaviour

| Platform stage | What we see | What we send |
|---|---|---|
| `development` (sandbox) | `price_tick`, `news_flash` | nothing (no preflight_tick) |
| `preflight` | `preflight_tick` with `seq` | `POST /heartbeat {seq}` |
| `simulation` | `price_tick`, `news_flash` (no preflight_tick) | nothing (trading window; this script does not trade) |
| `paused` / `completed` | nothing | nothing |

The watchdog re-sends the last seq every 10 s, so the portal stays green between preflight_ticks.

## When trading starts

This script does not submit orders. When the actual trading bot is wired up (`bot/main.py` per `docs/10`), it will inherit this same Kafka consumer and heartbeat logic and add:

- Day Assembler + per-day decision pipeline
- Portfolio Constructor + Risk Manager
- Order Manager with rate limiter and validator
- Decision Journal

For now, this script is the minimum that flips the readiness panel to **Connected**.