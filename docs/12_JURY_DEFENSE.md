# 12 — Jury Defense Preparation

## 1. Mapping our work to the five pillars

| Pillar | Our evidence |
|--------|--------------|
| 1. Technical implementation & reliability | Single-process asyncio architecture; pydantic contracts; validator with 0 rejections; rate limiter with 0 × 429; timeout-safe retries; chaos drill results; SAFE_MODE and kill switch; hot standby |
| 2. Strategy innovation & analytical depth | News-impact model exploiting "news after candle, fill at close"; walk-forward LightGBM ranker with IC stats; regime gating; ensemble with fallbacks; backtest vs benchmarks; SHAP explanations |
| 3. Execution speed & capital efficiency | p50/p95 latency histogram; partition-parallel consumers; concurrent order dispatch; avg cash utilisation chart; no lockouts; cash buffer logic |
| 4. Market frictions & tax optimisation | Min trade size and cost gate; turnover stats; LTCG-wait; loss harvesting; ₹1L LTCG gain harvesting; per-FY tax report; friction ratio |
| 5. Team coordination & explainability | Clear module ownership; docs folder; decision journal with reasons for every trade; every member presents one part |

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
| Why not trade more? | Each round trip costs ₹55 plus tax; cost gate requires 2× expected edge; cash utilisation still high through the core book |
| How do shorts work in your bot? | Event-only, small, closed before settlement, cash buffer sized for adverse moves |
| Explain your tax optimisation. | LTCG-wait, loss harvesting before 31 March, ₹1L LTCG harvesting, per-FY numbers |
| What happens in a crash? | Drawdown breaker; regime cuts exposure; stops; show live/backtest drawdown |
| Why LightGBM and not deep learning? | Small daily dataset; trees are robust, fast (< 5 ms), explainable via SHAP |
| How does FinBERT handle masked names? | Replace ticker/company with a neutral token; calibration head learned on recorded outcomes |
| What would you do with more time? | Pairs/stat-arb in sector clusters, online ensemble reweighting, better event extraction |

## 5. Things to have open on screen during the defense

- Repo in IDE at `bot/` with the architecture diagram.
- Dashboard with the live run loaded.
- `runs/live/journal.jsonl` filtered view (3 chosen trades).
- Backtest report HTML.
- Test run output (`pytest -q` all green).
