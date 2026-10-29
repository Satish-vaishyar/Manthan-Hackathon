# model1 — Cross-Sectional Temporal Transformer (CSTT)

## 1. What the model does

Each simulated trading day **D**, after the candles arrive, model1 takes the last 64 days of every ticker in the live
universe and outputs, per ticker:

| Output | Meaning | Used by the bot for |
|---|---|---|
| `ret1` | Relative score for the excess return close(D) → close(D+1) | Ranking for event/rebalance trades |
| `ret5` | Relative score for the excess return over the next 5 days | Core book selection (lower turnover) |
| `p_up` | P(excess return tomorrow > 0) | Confidence / filters |
| `vol` | log mean \|r\| over next 5 days | Position sizing (inverse volatility) |
| `p_mkt_up` (per day) | P(equal-weight market up over next 5 days) | Regime / gross exposure |

Scores are **relative** (cross-sectional): we trade at the close of day D, so what matters is which tickers will do
better than the others, not the absolute level.

## 2. Design constraints that shaped it

1. **Live tickers ≠ training tickers** → the model never sees stock symbols:
   - symbols are used only to group rows when computing features and to label reports;
   - the input tensor holds only the 28 scale-free numeric features;
   - there is no embedding layer;
   - the cross-sectional encoder has no positional encoding, so it is permutation-equivariant;
   - ticker order is randomly shuffled in every training sample, and universe sizes vary (6–40).

   `src/check_no_ticker.py` verifies all of this.
2. **Small data** (~1,000 trading days × ≤ 40 tickers) → small model (≈ 157k parameters), strong regularisation,
   multi-task loss, ensemble of 5 seeds, LightGBM baseline that it must beat.
3. **3–5 s per simulated day** → CPU inference must be < 50 ms (measured ≈ 9 ms for 40 tickers via ONNX/PyTorch).
4. **Variable history** (stopped tickers, new tickers with short history) → padding masks plus training-time history
   truncation down to 20 days.

## 3. Input tensors

```
x      [B, N, L=64, F=28]  robust-scaled features, 0 where missing
tmask  [B, N, L]           True = real trading day for that ticker
nmask  [B, N]              True = real ticker (False = padding)
```
One training sample = **one trading day** with all eligible tickers (N ≤ 40). Batch = 32 days.

### 28 features per ticker-day (`src/features.py`, order fixed by `FEATURE_NAMES`)

| Group | Features |
|---|---|
| Returns | `r1`, `gap`, `intraday`, `r5`, `r10`, `r20`, `r60` (log) |
| Candle shape | `hl_range`, `clv`, `upper_wick`, `lower_wick` |
| Volume | `vol_z20`, `vol_ratio_5_20` |
| Volatility | `rv20`, `rv60`, `rv_ratio`, `atr14_pct` |
| Trend | `dist_sma20`, `dist_sma50`, `dist_sma200`, `macd_hist_n` |
| Oscillators | `rsi14`, `bb_pctb20` |
| 52-week position | `dist_252_high`, `dist_252_low` |
| Cross-sectional (same day) | `mkt_r1`, `excess_r1`, `cs_rank_r20` |

Scaling: `(x − median) / IQR`, clipped to ±5, fitted on train dates only, saved in `scaler.json`. The bot must use
the **same** `features.py` and `scaler.json` at inference.

## 4. Network

```
                    per ticker i (shared weights)                                 per day
x_i [L,F] ─► Linear(F→64)─GELU─Dropout─Linear(64→64) + learned positional emb [L,64]
          ─► Temporal Transformer encoder  (2 layers, 4 heads, FF 128, pre-norm, GELU, dropout 0.2,
                                            key-padding mask = ~tmask)
          ─► [ attention-pool over valid days ‖ hidden state of day D ] ─► Linear(128→64) = h_i
                                                                                     │
            [MKT] learned token  +  {h_1 … h_N} ──────────────────────────────────────┘
          ─► Cross-sectional Transformer encoder (2 layers, 4 heads, FF 128, no positional enc,
                                                  key-padding mask = ~nmask) ─► LayerNorm
          ─► z_MKT, z_1 … z_N
             (phase 2) z_i += sigmoid(W[z_i ‖ n_i]) ⊙ n_i   where n_i = MLP(news_i)   (gated news fusion)
Heads:  z_i ─► MLP(64→64→2)  = ret1, ret5
        z_i ─► Linear(64→1)   = dir logit  (p_up = sigmoid)
        z_i ─► Linear(64→1)   = vol
        z_MKT ─► Linear(64→1) = market-up logit
```

- Temporal attention learns which past days matter for each ticker (momentum, reversal, volatility patterns).
- Cross-sectional attention lets tickers "see" each other on the same day (relative strength, sector co-movement,
  market regime) — this is what a per-ticker model like LightGBM cannot do.
- The [MKT] token summarises the whole universe and drives the regime head.
- Parameters: **≈ 157k** (d_model 64). Do not scale up without more data.

## 5. Training objective

Targets are built from **close(D) → close(D+h)** log returns (we fill at close(D)), winsorised at ±15%, minus the same-day
cross-sectional mean (excess return). Per-day z-scores are used for the regression part.

```
L = − 1.0 · IC(ret1, y1)            # differentiable Pearson correlation per day, averaged over days
    − 0.5 · IC(ret5, y5)
    + 0.5 · [Huber(ret1, z(y1)) + Huber(ret5, z(y5))]
    + 0.3 · BCE(dir, 1[y1 > 0])
    + 0.2 · MSE(vol, log mean|r| next 5d)
    + 0.1 · BCE(mkt, 1[market 5d return > 0])
```
IC terms optimise exactly what we trade on (ranking); Huber keeps the scale stable; the auxiliary heads regularise.

### Optimisation and regularisation

| Item | Value |
|---|---|
| Optimiser | AdamW, lr 3e-4, weight decay 1e-2 |
| Schedule | linear warm-up 3 epochs, cosine decay |
| Epochs / early stop | up to 80; stop after 12 epochs without val IC (h1) improvement |
| Batch | 32 days |
| Precision | AMP (bf16 if supported, else fp16 + GradScaler); TF32 on |
| Grad clip | 1.0 |
| Augmentation | random ticker subset 6–40 per day; Gaussian feature noise 0.02; feature dropout 5%; random history truncation (30% of tickers, down to 20 days) |
| Ensemble | 5 seeds; predictions standardised per day, then averaged |

## 6. Phase 2 — news branch (off by default)

- Input per ticker-day: `news_i ∈ R^8` = [FinBERT p_pos, p_neg, p_neu, relevance score, n_items (log), event-tag
  polarity, decayed news memory, has_news], plus `news_mask`.
- Relevance is required because the stream contains many unrelated headlines (see DATA_REPORT §5): keep an item only
  if the company name/ticker or company keywords from its description appear, or a zero-shot relevance score > 0.5.
- Enable with `model.use_news: true` once ≥ 1 full sandbox loop of news is recorded. Train phase 2 by fine-tuning
  the phase-1 checkpoint with the news branch at lr 1e-4. The REST tickers have no news, so phase 2 is trained on
  the recorded Kafka tickers only.

## 7. Evaluation

Walk-forward by date with a 5-day embargo. Metrics per split (`metrics.json`):
- daily Spearman **IC** mean, std, ICIR, t-stat (h1 and h5)
- top-5 minus bottom-5 mean daily excess return
- hit rate

**Ship rule:** the model1 ensemble ships only if test IC(h1 or h5) > LightGBM baseline test IC and t-stat > 2.
Otherwise the bot uses LightGBM plus rules.

## 8. Serving in the bot

1. On warm-up, fetch ≥ 64 days of history per live ticker (or accumulate from Kafka; works from 20 days).
2. Each day: append the candle, recompute features with `features.py`, scale with `scaler.json`.
3. Build `x, tmask, nmask` for the current universe (batch = 1 day) → run every ensemble member (ONNX Runtime or
   `.pt`) → standardise per day → average.
4. Feed `ret5` (core) and `ret1` (events) into the strategy ensemble; `vol` into sizing; `p_mkt_up` into regime.
