# 07 — Backtester Specification

The backtester is our main research tool **and** a key jury artifact. It must reproduce platform rules
so that backtest results predict live results.

## 1. Requirements

1. **Event-driven, day by day**, using the same `DayBundle` objects as the live bot.
2. **Reuses live code**: Feature Engine, Signal Layer, Portfolio Constructor, Risk Manager, Tax Engine
   and Ledger are imported from `bot/` — **no duplicate strategy logic**. Only the Order Manager is
   replaced by a `SimBroker`.
3. Deterministic: same data + config → identical output.
4. Fast: a 1000-day run on ~50 tickers in < 60 s (no FinBERT at backtest time — precompute news scores).

## 2. SimBroker rules (mirror of platform)

| Rule | Implementation |
|------|----------------|
| Fill price | Close of day D for orders decided on day D |
| Brokerage | ₹20 per order |
| DP charge | ₹15.34 per SELL |
| BUY with open short | Covers short first, remainder goes long |
| SHORT margin | 100% of notional locked from cash |
| Settlement | Every 10 trading days (alternate Fridays, configurable anchor) close all shorts at close |
| Settlement loss > cash | Liquidate longs with highest profit % first |
| Negative cash | Block BUY/SHORT (`ACCOUNT_BLOCKED`) |
| Rejections | Same codes as platform; counted in metrics |
| Rate limit | Optional: max N orders/day; reject extras as `RATE_LIMITED` |
| Tax | STCG 15%, LTCG 10% with ₹1L exemption per FY, set-off rules; tax computed at FY end and at end of run |

## 3. Inputs

```
python research/backtest.py --data data/features.parquet --news data/news_scores.parquet \
       --config configs/default.yaml --start 2020-04-01 --end 2021-03-31 --out reports/bt_default
```

- Optional `--loop-split` to evaluate on each sandbox loop separately.
- `--walk-forward` mode: retrain models on window k, test on window k+1, concatenate equity curves.

## 4. Outputs

- `equity.csv` (date, cash, margin, long_mv, short_mv, equity, exposure, regime)
- `trades.csv` (date, ticker, action, qty, price, fees, reason, scores)
- `tax.csv` per FY
- `report.html`: equity curve vs equal-weight buy-and-hold benchmark, drawdown chart, monthly returns,
  exposure/utilisation over time, turnover, fee & tax totals, hit-rate by signal source, rejection count.

## 5. Metrics (all reported)

| Metric | Definition |
|--------|------------|
| Total return, CAGR | vs ₹10L start |
| Sharpe, Sortino | daily, 252-day annualised |
| Max drawdown, longest DD | from equity curve |
| Avg cash utilisation | mean of (invested / equity) |
| Turnover | Σ traded notional / avg equity |
| Friction ratio | (fees + DP + tax) / gross P&L |
| Orders, rejections, orders/day max | — |
| Hit-rate & avg return by source | news events, core, shorts |
| Composite score | see doc 04 §1 |

## 6. Benchmarks to beat

1. Cash (0%).
2. Equal-weight buy-and-hold of all tickers (one BUY each at start).
3. Rules-only strategy (our fallback).

**Ship rule:** a component (news model, ranker, regime) enters the live ensemble only if adding it
improves the composite score in walk-forward tests **and** doesn't increase max drawdown by > 2 points.

## 7. Validation of the backtester itself

- Unit tests for every SimBroker rule with hand-computed cases (see doc 11).
- **Parity test**: run the live bot in the sandbox for ≥ 200 days, then backtest the same recorded days
  with the same config; ledger equity must match the portal's within 0.5% (differences → investigate rule mismatch).
