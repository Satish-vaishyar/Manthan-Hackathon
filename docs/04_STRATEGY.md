# 04 — Trading Strategy

## 1. Objective function (what we optimise)

We do **not** maximise raw profit. We maximise the scorecard:

| Metric | How our strategy addresses it |
|--------|-------------------------------|
| Total Wealth | Positive-expectancy signals, compounding, drawdown control |
| Cash Utilization | Core book keeps 60–90% of equity invested |
| Decision Latency | Decide within ~0.5 s of the candle; always send `date` |
| Low Order Error Rate | Client-side validation, ledger, rate limiter → target 0 rejections |
| Tax & Friction Efficiency | Low turnover, minimum trade size, LTCG-aware holding, loss/gain harvesting |

In backtests we track a **composite score**:
`0.40·CAGR_norm + 0.20·Sharpe_norm − 0.15·MaxDD_norm + 0.10·avg_utilization − 0.10·(fees+tax)/gross_pnl − 0.05·reject_rate`.

## 2. Core edge hypotheses

| # | Hypothesis | Signal | Confidence |
|---|-----------|--------|------------|
| H1 | Significant news on day D drives the return on day D+1 (and drifts for 2–5 days) | News-impact model | High (news arrives after candle but fills at same close) |
| H2 | Cross-sectional momentum (20–60 d) persists; very short-term (1–3 d) reverses | LightGBM ranker + rules | Medium |
| H3 | In high-volatility, falling-breadth regimes, long exposure loses | Regime model | Medium |
| H4 | Strong close location + volume surge confirms continuation | Rules overlay | Medium |

Each hypothesis must be **validated in the backtester** on recorded sandbox data before it gets weight.

## 3. Portfolio structure: Core + Satellite

### 3.1 Core book (target 55–75% of equity)
- Long-only, top-K tickers (K = 6–10, scaled with universe size) by ensemble score.
- Inverse-volatility weights, capped at 15% per name and 30% per sector cluster.
- **Rebalance only when** `|target_w − current_w| > 2.5%` of equity **and** trade notional ≥ ₹50,000.
- Holding period naturally long → most gains become LTCG eligible if the run is long enough.

### 3.2 Satellite event book (target 10–25% of equity)
- **Long event trade**: `news_score ≥ +0.6` and confirmation (close in top 30% of day range or volume
  z > 1) → BUY at today's close, size 3–6% of equity. Exit after 2–5 days, on trailing stop
  (2.5×ATR), or when `news_decay < 0.2`.
- **Short event trade**: `news_score ≤ −0.6`, price below 20-d SMA, regime not strongly risk-on, and
  **≥ 3 trading days before the next settlement** → SHORT 2–5% of equity. COVER after ≤ 3 days,
  on stop (2×ATR), or the day before settlement.
- Max simultaneous event trades: 4.

### 3.3 Cash
- Minimum cash buffer: `max(5% of equity, 15% × short notional)`.
- Regime sets max gross exposure: risk-on 92%, neutral 75%, risk-off 40% (+ shorts allowed).

## 4. Signal ensemble

For each ticker on day D:

```
s_rank   = cross-sectional percentile of LightGBM predicted fwd return     ∈ [0,1] → map to [-1,1]
s_news   = news-impact score (today's + decayed history)                     ∈ [-1,1]
s_mom    = rules: sign/strength of (20d ret, 50d SMA slope, RSI regime)      ∈ [-1,1]
score    = w_rank·s_rank + w_news·s_news + w_mom·s_mom
           (default 0.45 / 0.35 / 0.20 — tuned in backtester, then FROZEN)
score   *= regime_multiplier            (risk-on 1.0, neutral 0.8, risk-off 0.5 for longs)
```

If a model fails, its weight is redistributed to the others; if all models fail, use `s_mom` only.

## 5. From score to orders (daily decision loop)

```
1. Update ledger valuation with today's closes.
2. Check settlement calendar → shorts that must close today/tomorrow → COVER orders (priority 0).
3. Risk exits: stops, time-stops, drawdown breaker → SELL/COVER (priority 1).
4. Compute target weights:
     core   = top-K by score, inverse-vol, caps
     events = from news rules
     scale total to regime exposure; enforce cash buffer
5. Delta = target_shares − current_shares (integer, floor for buys).
6. Cost/tax gate per delta:
     expected_edge(₹) = |score| · k_edge · notional · horizon
     cost(₹)          = 20 (+15.34 if sell) + est_tax_if_realising_gain
     keep if expected_edge > 2 · cost  and notional ≥ ₹50,000 (exits exempt from min notional)
7. Tax engine adjustments: delay a SELL if lot hits LTCG within ≤ 30 trading days and the sell
   signal is weak (score > −0.5); add harvest orders near FY end (see doc 06).
8. Rank remaining orders by priority then |expected_edge − cost|; keep top N (N = budget, default 4).
9. Validate → send concurrently → apply fills to ledger → journal.
10. Journal a HOLD record for every ticker without an order.
```

Order of execution within a day: **sells/covers before buys** (frees cash), but send concurrently only
if the ledger confirms buys are affordable without the sell proceeds; otherwise send sells first, await
acks, then buys.

## 6. Position sizing details

```
w_i_raw   = (score_i^+) / vol_i                 for i in top-K
w_i       = w_i_raw / Σ w_raw · core_budget
w_i       = min(w_i, 0.15)
shares_i  = floor(w_i · equity / close_i)
```

`vol_i` = EWMA(λ=0.94) daily volatility, floored at 1% to avoid oversizing quiet names.

## 7. Special situations

| Situation | Rule |
|-----------|------|
| First days of simulation | Warm-start indicators from `/history`; still ramp exposure over first 3 days (avoid all-in on day 1) |
| Last 3% of `total_days` | No new event trades; close shorts; **keep longs** (selling costs fees + tax and wealth counts market value) |
| Ticker without news for long | Normal; news score decays to 0 |
| Gap/huge move (> 4σ) | Halve size for 3 days for that ticker |
| Sandbox loop reset | Reset indicators; mark journal; don't panic-sell |
| Missing candle for a held ticker | Hold; value at last known close |

## 8. Parameters (all in `config.yaml`, frozen at code freeze)

| Param | Default | Tuned range |
|-------|---------|-------------|
| `news_window_ms` | 400 | 250–600 |
| `hard_deadline_ms` | 800 | — |
| `top_k` | 8 | 5–12 |
| `max_weight_name` | 0.15 | 0.10–0.20 |
| `rebalance_threshold` | 0.025 | 0.015–0.05 |
| `min_trade_notional` | 50000 | 25k–100k |
| `max_orders_per_day` | 4 | 3–5 |
| `event_long_threshold` | 0.6 | 0.4–0.8 |
| `event_short_threshold` | −0.6 | −0.8 – −0.4 |
| `stop_atr_mult` | 2.5 | 2–3.5 |
| `dd_soft`, `dd_hard` | 8%, 15% | — |
| `ltcg_wait_days` | 30 | 15–45 |

## 9. What we explicitly do NOT do

- No unmasking tickers via external price data (look-ahead).
- No martingale/averaging down on losers.
- No leverage beyond platform margin; no shorts held through settlement by default.
- No intraday assumptions (we only have daily candles; fills are at close).
- No re-optimising parameters during the live hour.
