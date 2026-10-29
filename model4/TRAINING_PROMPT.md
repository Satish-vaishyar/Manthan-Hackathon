# model4 — Training Prompt

**N/A — model4 is parameter-free.** There is no training step.

The estimator is the textbook realised-volatility formula:

```
σ̂ = sqrt( Σ (r_t − r̄)² / n × 252 )
```

with `r_t = log(close_t / close_{t-1})` and a 20-day sliding window. There
are no weights to fit, no gradient, no validation set, no held-out data.

If you want a **trainable** vol model in the future, drop one of these
into `model4/src/` (each is a 30-line replacement behind the same
`VolEstimator.vol(ticker)` interface):

| Upgrade | Why | Cost |
|---|---|---|
| EWMA (RiskMetrics λ=0.94) | Reacts faster to recent shocks | Same O(1) update, no params |
| GARCH(1,1) | Captures volatility clustering | 3 params, MLE fit per ticker on warm-up data |
| HAR-RV (Hansen-Lunde) | Separates daily / weekly / monthly components | 3 params, OLS fit |
| Neural stochastic vol | Non-linear dynamics | Heavy; needs GPU + per-day retrain |

Until any of those ships, `model4/src/vol.py` is the canonical M4. The
upstream tests in `tests/test_vol.py` lock the contract; any future
replacement must keep them green.