# 01 — Problem Summary (condensed from the official PS)

Source: `Hackathon_Problem_Statement.pdf` (14 pages). If anything here conflicts with the PDF or the
Participant Portal (`http://<SERVER_IP>:3003`), **the PDF/portal wins** — update this doc.

## 1. The task

Build an **autonomous trading bot** that:

1. **Ingests** streaming daily OHLCV candles and corporate news from Apache Kafka.
2. **Decides** BUY / SELL / SHORT / COVER / do nothing within a 3–4 second window per simulated day.
3. **Executes** market orders through a REST API with a Bearer token, handling confirmations and rejections.
4. **Manages** cash, holdings, short margin, transaction costs, and avoids negative-balance blocks.

Any language, any library and any AI tool is allowed. Stock symbols are **masked** (e.g. `ALPHACORP`)
and the replayed historical period is **confidential**.

## 2. Event facts

| Item | Value |
|------|-------|
| Team size | 2–4 |
| Starting capital | ₹10,00,000 |
| Kafka broker | `<SERVER_IP>:9094`, PLAINTEXT, no auth |
| Order API | `http://<SERVER_IP>:3002/api/v1` |
| Portal & docs | `http://<SERVER_IP>:3003` |
| Network | **Venue Wi-Fi only** (private subnet). Hotspots won't reach the servers. |
| Briefing video | https://youtu.be/QKgdReXq1-M (watch it during setup) |

## 3. Schedule

| Time | Stage |
|------|-------|
| 0:00–0:30 | Setup: Wi-Fi, register on portal, physical verification at desk, watch video |
| 0:30–20:00 | Development against the continuous sandbox loop; jury checkpoint reviews |
| 20:00–23:00 | Pre-flight readiness checks, then the **official 1-hour live simulation** |
| 23:00–24:00 | Top-5 final jury defense |

## 4. Market stream timing

- 1 simulated trading day ≈ **3–4 seconds** of wall-clock time (`interval_seconds` from `/stage`, e.g. 4).
- **T+0 ms**: `price_tick` candles for all stocks on `market.ticks`.
- **T+150 ms**: `news_flash` items on `market.news`.
- Remaining ~3 s: decide and send orders before the next day starts.
- Both topics have **10 partitions**; up to 10 consumers in group `group-<your_team_username>`.
- Use `auto_offset_reset='latest'`, `enable_auto_commit=False`.

## 5. Pre-flight

- Server emits `preflight_tick` (with `seq`) on `market.ticks`.
- Bot must `POST /api/v1/bot/heartbeat {"seq": n}` continuously, **at least once every 15 s**.
- When heartbeats arrive and Kafka is connected, the team shows **Ready** on the portal.
- During pre-flight, call `GET /stocks` (dynamic universe, **never hardcode tickers**) and optionally
  `GET /stocks/:ticker/history` to warm up indicators.
- Poll `GET /stage` every 2–3 s; start trading when `stage == "simulation"` and `status == "RUNNING"`.
  Stop sending orders on `PAUSED` / `COMPLETED`.

## 6. Order mechanics

| Action | Effect |
|--------|--------|
| BUY | Buy with cash. **If a short is open in that ticker, BUY covers the short first.** |
| SELL | Sell owned delivery shares. |
| SHORT | Borrow and sell; **100% of trade value locked as margin** from cash. |
| COVER | Buy back an open short, release margin, settle P&L. |

- Every order fills at the **close price of the current day's candle**.
- Quantity: positive integer, 1 to 1,000,000.
- Optional `date` field = current simulated date → used to measure our decision latency. **Always send it.**

## 7. Short settlement

- Every **10 trading days (alternate Fridays)**, all open shorts are force-closed at the close price.
- Profit is credited to cash; loss is debited from cash.
- If a loss exceeds cash, the platform liquidates our **most profitable longs first**.
- If cash is still negative → **ACCOUNT_BLOCKED** (no new BUY/SHORT until it is cleared).

## 8. Costs and tax (Income-tax Act, 1961)

| Item | Rule |
|------|------|
| Brokerage | ₹20 flat on **every** executed order |
| DP charge | ₹15.34 flat when **selling** delivery shares |
| STCG | 15% on net gains from shares held ≤ 365 days |
| LTCG | 10% on net gains from shares held > 365 days; first **₹1,00,000 per FY exempt** (Sec 112A) |
| Set-off | Sec 70 & 71: STCL offsets STCG then LTCG; LTCL offsets LTCG only; losses generate tax credits |

## 9. Rate limits

- **120 requests per minute** per token. Exceeding it → HTTP 429 + **30 s penalty box**.

## 10. Wealth formula

`Total Wealth = Available Cash + Margin Locked + Long Market Value + Short Market Value`
(positions valued at the latest Kafka close).

## 11. Sandbox

- During development the dataset loops continuously; at the end it **jumps back to Day 1 prices**
  (large fake P&L swings are expected).
- A **Reset Cash** button on the dashboard restores ₹10,00,000.
- The live simulation runs the dataset **once, no loop**.

## 12. How winners are chosen

1. After the live hour, an automated scorecard on: **Total Wealth, Cash Utilization, Decision Latency,
   Low Order Error Rate, Tax & Friction Efficiency** → shortlist.
2. Top-5 jury defense across five pillars:
   1. Technical implementation & reliability
   2. Strategy innovation & analytical depth
   3. Execution speed & capital efficiency
   4. Market frictions & tax optimisation
   5. Team coordination & explainability

> "A bot that produced moderate, steady gains with zero rejected orders, low decision latency, robust
> downside protection, and intelligent tax loss offset utilization will score significantly higher than
> a fragile bot that experienced massive drawdowns."

## 13. Implications we derived (not stated explicitly — verify early)

| # | Implication | Action |
|---|-------------|--------|
| I1 | We know the fill price (today's close) **before** ordering → we are betting on the **next** day's move. | Labels = forward returns from close. |
| I2 | News arrives after the candle but is executable at the same close → news-driven next-day drift is our main edge. | Wait for the news window before deciding. |
| I3 | 1 hour ÷ 3–4 s ≈ **900–1200 trading days ≈ 3.5–4.5 years** (if the run lasts the full hour). | LTCG, FY boundaries and harvesting become relevant. Read `total_days` from `/stage` to confirm. |
| I4 | 120 req/min ÷ 15–20 days/min ≈ **6–8 requests per day** in total. | Local ledger, rare reconciliation, ≤4 orders/day. |
| I5 | ₹55 round-trip cost → small trades are value-destroying. | Minimum trade notional ≈ ₹50,000. |
| I6 | Venue network is private → probably **no internet**. | Pre-download all models, wheels, data. |
| I7 | Live dataset period may differ from sandbox period. | Avoid overfitting; walk-forward validation; regime gating. |
| I8 | Matching masked prices to real NSE tickers would be look-ahead. | **Forbidden** by our team policy. |

## 14. Open questions to confirm at the venue / in the video

- [ ] Exact `total_days` and `interval_seconds` of the live run.
- [ ] Do public endpoints (`/stage`, `/simulation/clock`) count towards the 120/min limit?
- [ ] Lot-matching rule for tax (FIFO vs average cost)?
- [ ] Which calendar day is the FY boundary in the simulation (31 March of simulated date)?
- [ ] Is the settlement schedule anchored to simulation day 1 or to calendar Fridays?
- [ ] Number of tickers and typical number of news items per day.
- [ ] Does the order response return fill price, fees and tax? (Needed for the ledger.)
- [ ] Max lookback of `/history` (does `start_date` allow > 365 days?).
