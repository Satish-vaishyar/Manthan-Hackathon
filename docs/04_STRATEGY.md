# 04 — Trading Strategy

## 1. Objective function (what we optimise)

We do **not** maximise raw profit. We maximise the scorecard:

| Metric | How our strategy addresses it |
|--------|-------------------------------|
| Total Wealth | Positive-expectancy signals, compounding, drawdown control |
| Cash Utilization | **Updated 2026-10-29:** `risk.target_cash_utilization: 0.85` — the constructor scales target weights toward 85% invested; an idle-cash deployer tops up after every 15 s reconcile when utilisation is >5pp below target |
| Decision Latency | Decide within ~0.5 s of the candle; always send `date` |
| Low Order Error Rate | Client-side validation, ledger, rate limiter → target 0 rejections |
| Tax & Friction Efficiency | Low turnover, minimum trade size, LTCG-aware holding, loss/gain harvesting |

In backtests we track a **composite score**:
`0.40·CAGR_norm + 0.20·Sharpe_norm − 0.15·MaxDD_norm + 0.10·avg_utilization − 0.10·(fees+tax)/gross_pnl − 0.05·reject_rate`.

## 2. Core edge hypotheses

| # | Hypothesis | Signal | Confidence |
|---|-----------|--------|------------|
| H1 | Significant news on day D drives the return on day D+1 (and drifts for 2–5 days) | News-impact model | High (news arrives after candle but fills at same close) |
| H2 | Cross-sectional momentum (20–60 d) persists; very short-term (1–3 d) reverses | M1 CSTT ranker (Updated 2026-10-29 — ONNX ensemble, not LightGBM) + M3 DMN + rules | Medium |
| H3 | In high-volatility, falling-breadth regimes, long exposure loses | Regime model | Medium |
| H4 | Strong close location + volume surge confirms continuation | Rules overlay | Medium |

Each hypothesis must be **validated in the backtester** on recorded sandbox data before it gets weight.

## 3. Portfolio structure: Core + Satellite

### 3.1 Core book (target ~85% of equity — Updated 2026-10-29)
- Long-only, top-K tickers (**K = 6**, tuned down from 8 for concentration) by ensemble score.
- Inverse-volatility weights, capped at 15% per name and 30% per sector cluster.
- **Rebalance only when** `|target_w − current_w| > 5%` of equity **and** trade notional ≥ ₹75,000
  (entries; exits use a lower ₹5,000 floor so legacy positions stay sellable).
- **`exclude_tickers: [BLUX]`** — tradebook post-mortem (runs/_tradebook_analysis.md): 22 trades,
  45.5% win-rate, −₹27,071. Blacklisting BLUX was the single biggest tuning win.
- Holding period naturally long → most gains become LTCG eligible if the run is long enough.

### 3.2 Satellite event book (target 10–25% of equity)
- **Long event trade**: `news_score ≥ +0.75` (Updated 2026-10-29, raised from 0.6) and confirmation
  (close in top 30% of day range or volume z > 1) → BUY at today's close, size 3–6% of equity.
  Exit after 2–5 days, on trailing stop (3.5×ATR, Updated 2026-10-29), or when `news_decay < 0.2`.
- **Short event trade**: `news_score ≤ −0.75` (Updated 2026-10-29), price below 20-d SMA, regime not strongly risk-on, and
  **≥ 3 trading days before the next settlement** → SHORT 2–5% of equity. COVER after ≤ 3 days,
  on stop (2×ATR), or the day before settlement.
- Max simultaneous event trades: 4.

### 3.3 Cash
- Minimum cash buffer: **Updated 2026-10-29: 2%** of equity (`risk.cash_buffer: 0.02`, lowered from
  5%) plus short-notional margin; the constructor scales weights toward
  `risk.target_cash_utilization: 0.85` (scale clamped to [0.5, 1.5]).
- Regime (M5) sets max gross exposure: risk-on 92%, neutral 75%, risk-off 40% (+ shorts allowed).
- **Idle-cash deployer** (`strategy.idle_cash_deploy {enabled, epsilon: 0.05}`): after every 15 s
  reconcile, if utilisation from the *server's* cash/equity is >5pp below target, place ONE extra
  BUY in the highest-conviction ticker, sized to the utilisation gap and capped by
  `max_cash_per_trade × cash × 0.99` − brokerage (all normal guards still apply).

## 4. Signal ensemble

For each ticker on day D:

```
s_rank   = M1 CSTT ranker score (5 ONNX members)                               ∈ [-1,1]
s_news   = M2 FinBERT NRIM news score (3 members, offline) + M6 event tagger   ∈ [-1,1]
s_mom    = M3 DMN momentum (5 members) + rules                                 ∈ [-1,1]
score    = w_rank·s_rank + w_news·s_news + w_mom·s_mom
score   *= regime_multiplier            (M5: risk-on 1.0, neutral 0.8, risk-off 0.5 for longs)
```

If a model fails, its weight is redistributed to the others; if all models fail, use the rules
fallback only.

**Updated 2026-10-29 — weights are ESTIMATED, not hand-picked**
(`bot/portfolio/weight_optimizer.py`, pure numpy; built by `scripts/optimize_weights.py` from
12,396 journal records = 500 sim days × 10 tickers in `runs/*/journal.jsonl`):
- Per-day cross-sectional **Spearman IC** of each signal vs next-day returns.
- Four estimators: IR-weighted (Grinold & Kahn), inverse-variance, **Markowitz-on-IC**
  (w = simplex(Σ⁻¹μ), Ledoit-Wolf-shrunk covariance), and a **1,000× Monte-Carlo day-bootstrap**
  of the Markowitz estimator.
- Final = bootstrap mean, shrunk toward equal weights by the empirical P(w≤0) per signal.
- Measured IC means: rank **.017** / news **.078** / mom **.141** → optimal
  **{rank 0.258, news 0.322, mom 0.420}** (the old hand-picked rank=0.55 was overweighting the
  *weakest* signal). Written to `configs/optimized_weights.json` +
  `runs/_weight_optimization.md`; loaded at boot when `strategy.weights_mode: optimized`.
  Static YAML weights are fallback only.

**Updated 2026-10-29 — RL tuner** (`bot/portfolio/rl_tuner.py`, config `strategy.rl_tuner`):
- **UCB1 multi-armed bandit** over 5 weight-triple arms; the statistically-optimized triple is
  injected as **arm 0**.
- Before each `ensemble.evaluate()` the tuner overrides `signals.weights`; the realized equity
  delta per reconcile is fed back as reward.
- Round-robin warmup until every arm is pulled once, then UCB1 (c=1.4).
- Observability: `rl_tuner_ready` / `rl_tuner_commit` / `rl_tuner_switch` log events + `stats()`.

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
     keep if expected_edge > 2 · cost  and notional ≥ ₹75,000 entries / ₹5,000 exits (Updated 2026-10-29);
     marginal trades (notional < 50× brokerage) are ALWAYS skipped, risk-reducing exits exempt
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

## 8. Parameters (all in `configs/default.yaml`)

**Updated 2026-10-29 — live values after tuning passes 1 + P3-14…P3-20** (rationale in WORK_LOG §5–§7):

| Param | Live value | Note |
|-------|-----------|------|
| `news_window_ms` / `hard_deadline_ms` | 400 / 800 | unchanged |
| `top_k` | **6** | was 8; concentration |
| `max_weight_name` | 0.15 | per-name cap |
| sector cap | 0.30 | per sector cluster |
| `rebalance_threshold` | **0.05** | was 0.025 |
| `min_trade_notional` | **₹75,000** | entries; was ₹50,000 |
| `exit_min_notional` | **₹5,000** | exits — legacy positions stay sellable |
| `max_orders_per_day` | 4 | largest-notional first |
| `exclude_tickers` | **[BLUX]** | −₹27K post-mortem |
| `event_long_threshold` / `event_short_threshold` | **+0.75 / −0.75** | was ±0.6 |
| `stop_atr_mult` | **3.5** | was 2.5 |
| `dd_soft`, `dd_hard` | **0.12, 0.15** | was 0.08/0.15 |
| `risk.cash_buffer` | **0.02** | was 0.05 |
| `risk.target_cash_utilization` | **0.85** | constructor scales weights toward 85% invested |
| `risk.max_cash_per_trade` | **0.90** | validator rejects BUYs > 90% of cash |
| `strategy.idle_cash_deploy` | enabled, epsilon 0.05 | deploys idle cash each 15 s reconcile |
| `strategy.weights_mode` | `optimized` | loads `configs/optimized_weights.json` |
| `strategy.rl_tuner` | enabled | UCB1 over 5 arms, optimized triple = arm 0 |
| `timing.reconcile_interval_s` / `stage_poll_simulation_s` | **15 / 10** | was 30/30 |
| `ltcg_wait_days` | 30 | unchanged |

**Live results (Updated 2026-10-29):** baseline **+₹2.31L realized over ~168 sim-days**; tuned run
**+₹1.51L in 40 min** (207 orders, **0 BLUX trades, 0 server rejections**). Full 400-order
orderbook: `runs/orderbook_full_20261029T202525Z.json` (pulled via `/api/v1/orders?limit=10000`).

## 9. What we explicitly do NOT do

- No unmasking tickers via external price data (look-ahead).
- No martingale/averaging down on losers.
- No leverage beyond platform margin; no shorts held through settlement by default.
- No intraday assumptions (we only have daily candles; fills are at close).
- No re-optimising parameters during the live hour.
