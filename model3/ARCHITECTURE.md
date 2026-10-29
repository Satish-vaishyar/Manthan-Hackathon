# model3 — Deep Momentum Network (DMN)

## 1. Role in the system

| Model | Question it answers |
|---|---|
| model1 (CSTT) | Which tickers will beat the **others** tomorrow / over 5 days? (cross-sectional) |
| model2 (NRIM) | What does today's **news** mean for each ticker tomorrow? |
| **model3 (DMN)** | Is **this** ticker trending, and **how much** of it should we hold, after costs? (time-series momentum) |

model3 learns the position size directly. It is trained on the **net Sharpe ratio after trading costs**, so it
learns to follow trends and avoid churning, which matters because the scorecard rewards friction efficiency.
Its average signal across tickers is also a market trend gauge for the regime / gross-exposure logic.

Outputs per ticker per day:

| Output | Meaning | Bot use |
|---|---|---|
| `position` ∈ [−1, 1] | trend-following conviction (long > 0, short/avoid < 0) | core-book tilt, exits, rare shorts |
| `target_weight` | `position × min(σ_target/√252 / σ_t, max_leverage)` | volatility-targeted size before portfolio caps |
| `p_trend_up` | P(next 21-day return > 0) | confirmation / LTCG hold decisions |
| `momentum_regime` | mean `position` across the live universe | gross exposure (risk-on / off) |

## 2. Features (16, `src/features.py`, all volatility-normalised, no identity)

| Group | Features |
|---|---|
| Normalised returns | `nret_k = r(t−k, t) / (σ_t √k)` for k = 1, 5, 21, 63, 126, 252 |
| MACD trend speeds (Baz et al.) | `macd_8_24`, `macd_16_48`, `macd_32_96`: (EWMA_S − EWMA_L) / std₆₃(price), then / std₂₅₂ |
| Regime change (changepoint-lite) | `cpd_mean` (21-day vs 126-day mean return, z-score), `cpd_vol` (log 21d/126d vol), `dd_63`, `du_63` (distance from the 63-day high/low in σ units) |
| Market context (same day) | `mkt_nret_21`, `mkt_nret_63` (equal-weight market), `breadth_63` (share of tickers with a positive 63-day return − 0.5) |

σ_t is the EWMA (span 60) volatility of daily log returns. All features are clipped to ±5. A ticker becomes usable
once it has about 252 days of history (`nret_252`).

## 3. Network

```
x [B, L=63, 16]  ->  LSTM(hidden 32, 1 layer)  ->  Dropout(0.3)  ->  Linear(32→1) -> tanh  = position_t  (every step)
                                                                   ->  Linear(32→1)          = trend logit_t (auxiliary)
```
≈ 6.5k parameters. One shared network for every ticker; each ticker's output depends only on its own sequence.
It is small on purpose: ~24k usable ticker-days of training data, and Sharpe-trained networks overfit easily.

## 4. Loss (per batch of sequences, after a 21-step burn-in)

```
w_t      = position_t × min(0.15/√252 / σ_t, 2.0)                       volatility targeting
ret_t    = w_t · r_{t+1}  −  5 bps × |w_t − w_{t−1}|                     net of trading costs
L        = − Sharpe(ret) × √252
           + 0.1 × BCE(trend logit, 1[next 21-day return > 0])
           + 1.0 × (mean|position| − 0.5)²                                exposure anchor
```
The exposure anchor is needed because the Sharpe ratio doesn't change if every position is scaled. Without it the
network drifted towards near-zero positions in testing, which would make its sizes useless to the bot.

## 5. Training setup

| Item | Value |
|---|---|
| Sequences | 63 days, one ticker each, fully inside the train split; start offset rotates every epoch (stride 21) |
| Optimiser | AdamW lr 1e-3, weight decay 1e-4, grad clip 1.0, batch 256 |
| Epochs | ≤ 150, early stop (patience 20) on **validation net Sharpe** |
| Ensemble | 5 seeds; positions averaged |
| Splits | same dates as model1, with a 21-day embargo (covers the 21-day auxiliary target) |
| Precision | fp32 (tiny LSTM; mixed precision is not worth the instability in the Sharpe loss) |

## 6. Evaluation (`src/backtest.py`)

The daily portfolio averages the volatility-targeted, cost-adjusted returns of every ticker with a signal.
Reported for **long/short** and **long-only** (positions clipped at 0, which is how the bot mostly trades):
net and gross Sharpe, annual return and volatility, max drawdown, turnover, average |weight|, hit rate, cost share.

Baselines (`src/baselines.py`): buy-and-hold, 12-month time-series momentum (long/short and long-only), and the
MACD combination. Measured on this data (net Sharpe):

| Baseline | val | test |
|---|---|---|
| buy_and_hold | −0.03 | 1.40 |
| tsmom_12m | 0.75 | 1.42 |
| tsmom_12m_long | 0.30 | 1.54 |
| macd_combo | −1.47 | 0.79 |

**Ship rule:** ensemble val net Sharpe > best baseline val Sharpe, and test net Sharpe (long-only) ≥ the long-only
TSMOM baseline. Otherwise the bot uses `tsmom_12m_long` as its momentum signal.

## 7. Serving

1. Keep ≥ 63 days (better ≥ 315 = 252 + 63) of closes per live ticker; warm up from `/history` or the stream.
2. Each day compute features with `features.py` for the whole live universe (market features need all tickers).
3. Feed the last 63 rows per ticker (zero-fill missing early steps) into every member (ONNX or `.pt`), average.
4. ≈ 5 ms on CPU for 40 tickers.
