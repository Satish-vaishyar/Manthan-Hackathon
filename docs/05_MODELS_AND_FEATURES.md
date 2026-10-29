# 05 — Models & Features

## 1. Data sources

| Source | When | Content | Use |
|--------|------|---------|-----|
| `GET /stocks/:ticker/history` | Hour 0.5–1, and at each bot start | ≤365 days (maybe more via `start_date`) daily OHLCV | Training prices, indicator warm-up |
| Recorder (Kafka → Parquet) | From hour 0.5, continuous | Every candle + news of the sandbox loop | **Only source of labelled news** |
| `GET /stocks` | Startup | Names + business descriptions | Sector clustering |
| Pre-downloaded public data (optional) | Before event | NSE daily bars (e.g. NIFTY-100), Financial PhraseBank, FiQA | Pre-training priors only |

**Data hygiene**
- Store raw messages as-is (`raw/ticks/date=.../*.parquet`, `raw/news/...`).
- Deduplicate by `(simulated_date, ticker)`; on sandbox loop, tag `loop_id` so the same date isn't double counted.
- Every label uses **close(D) → close(D+h)** returns because we fill at close(D).

## 2. Model inventory

| ID | Model | Type | Input | Output | Latency budget | Fallback |
|----|-------|------|-------|--------|----------------|----------|
| M1a | Event tagger | Regex/keyword rules | headline + description | event tags + polarity | < 1 ms | — |
| M1b | FinBERT (`ProsusAI/finbert`) | Pre-trained transformer, ONNX int8 | headline (+ first 200 chars desc) | p(pos), p(neg), p(neu) | ≤ 30 ms/headline | M1a only |
| M1c | News calibration head | Logistic regression / small LGBM | M1a + M1b + parsed numbers + price context | `news_score ∈ [-1,1]` | < 1 ms | `p_pos − p_neg` |
| M2 | Return ranker | LightGBM (`LGBMRegressor` on rank or `LGBMRanker`) | ~60 features | predicted fwd return rank | < 5 ms for all tickers | momentum rules |
| M3 | Regime | Gaussian HMM (2–3 states) or rule set | market return, market vol, breadth | regime ∈ {risk-on, neutral, risk-off} | < 1 ms | rules |
| M4 | Volatility | EWMA (λ=0.94); optional GARCH(1,1) | returns | σ per ticker | < 1 ms | 20-d stdev |
| R | Rules strategy | SMA/RSI/breakout/news keywords | indicators | [-1,1] | < 1 ms | — |

## 3. M1 — News impact model

### 3.1 Event tagger (M1a) — starter dictionary

| Tag | Polarity | Keyword patterns (case-insensitive) |
|-----|----------|-------------------------------------|
| EARNINGS_BEAT | + | profit (surges\|jumps\|rises\|soars), record (revenue\|profit\|exports), beats estimates |
| EARNINGS_MISS | − | profit (falls\|plunges\|declines\|slumps), loss widens, misses estimates |
| DIVIDEND_BUYBACK | + | dividend, buyback, bonus issue |
| ORDER_WIN | + | wins order, bags contract, secures deal, L1 bidder |
| MNA | ± | acquire, merger, stake sale, takeover |
| RATING_UP / RATING_DOWN | + / − | upgrade / downgrade, outlook (positive\|negative) |
| REGULATORY_BAD | −− | probe, raid, SEBI, penalty, fraud, investigation, ban, show-cause |
| MGMT_EXIT | − | resigns, steps down, CFO quits |
| DEBT_STRESS | −− | default, downgrade to junk, insolvency, NCLT |
| GUIDANCE_UP / DOWN | + / − | raises guidance / cuts guidance |
| CAPACITY_EXPANSION | + | expansion, new plant, capex |

Also parse numbers: `(\d+(\.\d+)?)\s?%` near (YoY|QoQ|growth|surge|fall) → signed magnitude feature.
Extend the dictionary by reading the recorded headlines in the first hours (top 200 frequent n-grams).

### 3.2 FinBERT (M1b)
- Pre-download weights before the event; export to ONNX and quantise to int8 (`optimum`).
- Replace the masked ticker and company name with `the company` to avoid token noise.
- Batch all headlines of a day in one forward pass.

### 3.3 Calibration head (M1c)
- **Label**: `y = sign(r_{D→D+1})` (classification) or `r_{D→D+3}` winsorised (regression),
  using ticker's return minus market average return (to remove market moves).
- **Features**: FinBERT probs, event tag one-hots, parsed % magnitude, headline length,
  number of news items today for this ticker, ticker's 5-d return, 20-d vol, distance from 20-d SMA.
- **Model**: L2 logistic regression first (explainable); try LightGBM if data > ~1,000 news rows.
- **Validation**: time-ordered split (first 70% train, last 30% test); report AUC, hit-rate, and
  average next-day excess return by score decile. Ship only if top-decile return − bottom-decile > costs.
- **Output**: `news_score = 2·p(up) − 1`; per-ticker memory `m_t = 0.8·m_{t−1} + news_score_t`
  (half-life ≈ 3 days), clipped to [−1, 1].

## 4. M2 — Cross-sectional return ranker

- **Target**: next-day (h=1) and 3-day (h=3) close-to-close excess return; train on cross-sectional
  rank (0..1) per date to remove market-wide effects. Choose h by backtest.
- **Training set**: all tickers × all dates from history + recorded stream (dedupe loops).
- **Model**: LightGBM, `num_leaves=15`, `max_depth=4`, `n_estimators≈300`, `learning_rate=0.03`,
  `min_child_samples=50`, `subsample=0.8`, `colsample_bytree=0.8`, early stopping on validation.
- **Validation**: purged walk-forward (e.g. 5 folds, each test window 40 days, 5-day embargo).
  Metrics: daily Spearman IC (mean, t-stat), top-K vs bottom-K spread, turnover.
- **Acceptance**: mean IC > 0.02 with t-stat > 2 across folds, else weight → 0 and rely on rules + news.
- **Explainability**: `shap.TreeExplainer` top-3 features stored per decision.

## 5. M3 — Regime model

Inputs (daily, market = equal-weight average of all tickers):
- market 1-d and 20-d return, market 20-d realised vol, breadth = % tickers above 50-d SMA,
  dispersion = cross-sectional stdev of returns.

Option A (default, robust): rules
```
risk-off  if breadth < 35% and market_20d_ret < 0  or  market_vol > 1.5 × its 120-d median
risk-on   if breadth > 60% and market_20d_ret > 0
neutral   otherwise
```
Option B: `hmmlearn.GaussianHMM(n_components=3)` on the same inputs, states labelled by mean return.
Use B only if it beats A in the backtester. Add 2-day hysteresis to avoid flip-flopping.

## 6. Feature catalogue (Feature Engine)

All features computed **incrementally** from ring buffers (length 260) per ticker; cross-sectional
features computed once per day across tickers. Names are the canonical column names.

| Group | Features |
|-------|----------|
| Returns | `ret_1, ret_3, ret_5, ret_10, ret_20, ret_60`, `gap = open/prev_close − 1` |
| Candle | `clv = (2C − H − L)/(H − L)`, `range_pct = (H − L)/C`, `body_pct = (C − O)/C`, `upper_wick`, `lower_wick` |
| Trend | `dist_sma_10/20/50/200 = C/SMA − 1`, `sma_20_slope`, `ema_12_26_macd`, `macd_hist`, `adx_14` |
| Mean reversion | `rsi_2`, `rsi_14`, `bb_pctb_20`, `bb_width_20`, `zscore_20` |
| Volatility | `atr_14_pct`, `rv_20`, `rv_60`, `parkinson_20`, `vol_ratio = rv_20/rv_60` |
| Volume | `vol_z_20`, `vol_ratio_5_20`, `obv_slope_20`, `ret_x_volz = ret_1 · vol_z_20` |
| Breakout | `high_20_break`, `low_20_break`, `dist_52w_high`, `dist_52w_low` |
| Cross-sectional | `cs_rank_<feature>` for ret_5, ret_20, rv_20, vol_z; `mkt_ret_1`, `mkt_ret_20`, `beta_60`, `resid_ret_5` |
| Sector | `sector_id` (KMeans on TF-IDF of `/stocks` descriptions, k≈5–8), `sector_rel_ret_20` |
| News | `news_score_today`, `news_mem`, `news_count_5d`, `days_since_news`, `tag_<EVENT>` one-hots |
| Calendar/state | `days_to_settlement`, `days_to_fy_end`, `sim_progress` (not fed to M2 — used by rules/tax only) |

**Leakage checklist** (code review must tick all):
- [ ] No feature uses data from D+1 or later.
- [ ] Cross-sectional ranks computed only within day D.
- [ ] Scalers/encoders fitted on training folds only.
- [ ] News of day D is allowed (it arrives before our day-D decision).
- [ ] Sandbox loop duplicates removed from training set.

## 7. Training workflow

```
research/
  01_pull_history.py      → data/history/*.parquet
  02_build_dataset.py     → data/features.parquet (features + labels, loop-deduped)
  03_train_news.py        → models/news_head.joblib, reports/news_eval.md
  04_train_ranker.py      → models/ranker.txt (LightGBM), reports/ranker_eval.md
  05_regime.py            → models/regime.joblib
  06_backtest.py          → reports/backtest_<config>.html
```

- Re-train at hours ~8, ~13 and ~17 as more sandbox data accumulates. **Final model frozen at hour 18.**
- Every model file has a sidecar `*.meta.json`: training date range, feature list (ordered), metrics,
  git commit. The bot refuses to load a model whose feature list differs from the live Feature Engine.

## 8. Runtime model serving

- Load all models at startup; run one dummy inference to warm up.
- Score a day as one numpy matrix (`tickers × features`) → single LightGBM call.
- FinBERT: batched; if it exceeds 120 ms at runtime, skip it for that day and use M1a + previous calibration.
- All inference wrapped in try/except → fallback + metric increment + journal flag.

## 9. Offline prep (do BEFORE leaving for the venue)

- [ ] `pip download -r requirements.txt -d wheels/` for the exact Python version on every laptop.
- [ ] `huggingface-cli download ProsusAI/finbert` → `models/finbert/`; export ONNX int8.
- [ ] Optional: NSE daily bars + Financial PhraseBank in `data/external/`.
- [ ] Verify the whole pipeline runs with networking disabled.
