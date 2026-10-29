# 12 — Jury Defense Preparation

> **Updated 2026-10-29 — with live results:** baseline **+₹2.31L realized over ~168 sim-days**;
> tuned run **+₹1.51L realized in 40 min** (207 orders, **0 BLUX trades, 0 server rejections**).
> Full 400-order orderbook pulled via `/api/v1/orders?limit=10000`
> (`runs/orderbook_full_20261029T202525Z.json`); post-mortem in `runs/_tradebook_analysis.md`.
> **170 pytest tests passing.** Six trained models + rules fallback all live on Kafka data.

## 1. Mapping our work to the five pillars

| Pillar | Our evidence |
|--------|--------------|
| 1. Technical implementation & reliability | Single-process asyncio architecture; pydantic contracts with AliasChoices against real server keys; **zero-rejection architecture** (in-flight cash reservation, server-holding clamps, UNKNOWN-status resync — no phantom fills); 0 `INSUFFICIENT_*` rejections on the tuned run; rate limiter with 0 × 429; SAFE_MODE and kill switch; boot DD false-positive fixed via `seed_equity()` |
| 2. Strategy innovation & analytical depth | News-impact model exploiting "news after candle, fill at close" (offline 3-member FinBERT NRIM); 5-member CSTT ONNX ranker + 5-member DMN momentum; regime gating; **statistically-estimated ensemble weights** (Spearman IC + Markowitz-on-IC + 1000× Monte-Carlo bootstrap → {rank .258, news .322, mom .420}, proving hand-picked rank=0.55 was overweighting the weakest signal); **UCB1 RL tuner** adapting weights live with the optimized triple as arm 0 |
| 3. Execution speed & capital efficiency | p50/p95 latency histogram; partition-parallel consumers; concurrent order dispatch; **85% target cash utilization** with constructor scaling + idle-cash deployer (tops up within 15 s of realized profits landing); ≤4 orders/day, largest-notional-first |
| 4. Market frictions & tax optimisation | ₹75K min entry notional + unconditional marginal-trade filter (notional < 50× brokerage skipped); BLUX blacklisted after post-mortem (−₹27K bleed); LTCG-wait; loss harvesting; ₹1L LTCG gain harvesting; per-FY tax report; friction ratio |
| 5. Team coordination & explainability | Clear module ownership; docs folder + WORK_LOG; decision journal with reasons for every trade (12,396-record corpus reused for weight estimation); every member presents one part |

## 2. Five-slide deck

1. **Problem → insight**: scored on balanced metrics; key insights (fill at close, news after candle, 120 req/min ≈ 6 per day, multi-year horizon → tax matters).
2. **Architecture** (diagram from doc 02) and per-day timing budget.
3. **Strategy & models**: core + satellite, ensemble, regime; backtest equity vs benchmarks; IC table.
4. **Risk, frictions & tax**: limits, DD breaker, settlement guard, tax techniques with ₹ saved.
5. **Live results**: wealth, utilisation, latency p50/p95, orders/rejections, fees & tax, 3 explained trades.

## 3. Speaking split

| Member | Covers |
|--------|--------|
| A | Architecture, reliability, latency, rate limiting, failure handling |
| B | Data, features, models, validation, leakage prevention |
| C | Strategy logic, risk controls, tax engine |
| D | Backtester parity, dashboard demo, live results & trade walkthroughs |

## 4. Expected questions and our answers (prepare and rehearse)

| Question | Answer outline |
|----------|----------------|
| Why did your bot buy X on day D? | Open journal record: scores, news headline, regime, size calculation, SHAP top features |
| How do you avoid look-ahead bias? | Labels from close(D); features only ≤ D; purged walk-forward; anti-lookahead history API; no unmasking |
| How did you validate the models? | Walk-forward IC, decile spreads, ship rule (only if composite score improves) |
| What if Kafka disconnects? | Reconnect with backoff; no trading on stale data; demonstrated in chaos drill |
| What if an order times out? | Check `/orders` before retry; never duplicate |
| How do you handle the rate limit? | Token bucket at 100/min, priorities, local ledger, measured 0 × 429 |
| Why not trade more? | Each round trip costs ₹55 plus tax; ₹75K minimum entry notional and a 50×-brokerage marginal-trade filter keep every trade material; cash utilisation still ~85% via constructor scaling + idle-cash deployer |
| How did you choose ensemble weights? | Not by hand: per-day cross-sectional Spearman IC of each signal vs next-day returns over 500 sim days × 10 tickers; four estimators (IR, inverse-variance, Markowitz-on-IC with Ledoit-Wolf shrinkage, 1000× day-bootstrap); bootstrap mean shrunk toward equal weights by P(w≤0). Then a UCB1 bandit adapts around that optimum live. |
| How do you guarantee zero rejections? | Validator reserves cash for in-flight BUYs; BUYs clamped to server cash × 0.90 × 0.98 − brokerage; SELL/COVER clamped to server-verified holdings (negative holdings = shorts); retries with backoff; empty responses become UNKNOWN + immediate resync, never phantom fills |
| Why did you exclude BLUX? | Data, not vibes: full-orderbook post-mortem showed 22 trades, 45.5% win rate, −₹27,071. Excluding it was the largest single tuning win (+₹1.51L in the 40-min tuned run with 0 BLUX trades) |
| How do shorts work in your bot? | Event-only, small, closed before settlement, cash buffer sized for adverse moves |
| Explain your tax optimisation. | LTCG-wait, loss harvesting before 31 March, ₹1L LTCG harvesting, per-FY numbers |
| What happens in a crash? | Drawdown breaker; regime cuts exposure; stops; show live/backtest drawdown |
| Why LightGBM and not deep learning? | Small daily dataset; trees are robust, fast (< 5 ms), explainable via SHAP |
| How does FinBERT handle masked names? | Replace ticker/company with a neutral token; calibration head learned on recorded outcomes |
| What would you do with more time? | Pairs/stat-arb in sector clusters, online ensemble reweighting, better event extraction |

## 5. Things to have open on screen during the defense

- Repo in IDE at `bot/` with the architecture diagram.
- Dashboard with the live run loaded.
- `runs/<timestamp>/journal.jsonl` filtered view (3 chosen trades).
- `runs/_tradebook_analysis.md` + `runs/_weight_optimization.md` (post-mortem + weight estimation).
- Backtest report HTML.
- Test run output (`python -m pytest tests/ -q` — **170 passing**).
