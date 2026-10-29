# model5 — Market Regime Detector (MRD)

## 1. What the model does

Once per simulated trading day **D**, model5 ingests the day's `DayBundle`
(today's close per ticker) plus its own internal rolling history and emits
one of three regime labels:

| Label | Meaning | Gross-long multiplier (default) |
|---|---|---|
| `risk_on` | Calm + uptrend | 1.00 — full deployment |
| `neutral` | Mixed signals | 0.75 — mild de-risking |
| `risk_off` | High vol or deep drawdown | 0.40 — defensive |

The multiplier scales total long-side exposure in
`bot/portfolio/constructor.build_target_weights` (via the
`gross_long_target` lookup table in `configs/default.yaml`).

## 2. Design constraints that shaped it

1. **3-5 s per simulated day** → regime classification must run inside the
   800 ms hard deadline (docs/02). The detector uses three O(window)
   reductions over a 60-day market-returns buffer plus a 20-day
   dispersion buffer. Total work: < 1 ms for a 40-ticker universe.
2. **Explainability** → the portfolio team (and the portal reviewers) need
   to know *why* the bot is in risk-off mode at any given moment. The
   snapshot dict (`regime.snapshot()`) returns the four driving stats so
   the decision journal can log them.
3. **Tunable** → every threshold (vol_high, vol_low, trend_up, trend_dn,
   disp_high) is a constructor argument or a YAML field; no magic numbers
   in code (docs/10 §3).
4. **Robust to cold start** → for the first `min_history_days` (default
   20) the detector returns `neutral` (no regime opinion yet) so we
   don't de-risk on a 5-day noise spike.

## 3. Algorithm

### Inputs

- `bundle.ticks[t].close` for every ticker that reported today
- `_state.last_close[t]` from yesterday's bundle
- Internal rolling buffers: `market_returns` (60d), `cross_dispersion` (20d)

### Per-day market aggregates

```
market_return = mean_t ( log(close_t / last_close_t) )
dispersion    = stddev_t ( log(close_t / last_close_t) )
```

The first quantity is the **equal-weight market return**; the second is
**cross-sectional return dispersion** (how scattered today's ticker moves
are). Both are appended to their respective rolling windows.

### Classifier

```
if days_observed < min_history_days:
    return NEUTRAL                                    # warm-up

vol    = annualised_vol(market_returns)               # √252 scaling
cumret = sum(market_returns)                          # log-return sum
disp   = mean(cross_dispersion)

if vol >= vol_high:
    return RISK_OFF                                   # panic vol
if cumret <= trend_dn:
    return RISK_OFF                                   # deep drawdown
if disp >= disp_high and vol >= vol_low:
    return RISK_OFF                                   # choppy + active

if cumret >= trend_up and vol <= vol_low:
    return RISK_ON                                    # clean uptrend

return NEUTRAL
```

The risk-off check fires first so we never under-defend. Risk-on is the
strictest (requires both positive trend AND low vol).

### Output

- `current_regime` — string label
- `days_in_current_regime` — how many days we've been in this state
- `snapshot()` — diagnostic dict for the journal

## 4. Configuration

| Param | Default | Meaning |
|---|---|---|
| `market_window` | 60 | Days of market returns for vol + cumulative return |
| `disp_window` | 20 | Days of cross-sectional dispersion |
| `vol_low` | 0.10 | Below this → risk_on candidate |
| `vol_high` | 0.30 | Above this → risk_off |
| `trend_up` | +0.05 | Cumulative log return needed for risk_on |
| `trend_dn` | −0.10 | Cumulative log return needed for risk_off |
| `disp_high` | 0.025 | Cross-sectional dispersion considered "choppy" |
| `min_history_days` | 20 | Warm-up: stay neutral for this many days |

All values live in `configs/model5.yaml` and overlay `configs/default.yaml`
via the same `_merge(...)` path the rest of the config uses.

## 5. Tests

`tests/test_regime.py` — 11 tests:

1. Warmup is neutral for the first 19 days
2. Strong low-vol uptrend → `risk_on`
3. High-vol chaotic series → `risk_off`
4. Deep uniform drawdown → `risk_off`
5. Choppy low-vol sideways → `neutral`
6. Transition logging fires on regime change
7. `DEFAULT_GROSS_TARGET` matches docs/04 §4 (1.0 / 0.8 / 0.5)
8. `ALL_REGIMES` contains all three labels
9. `snapshot()` returns expected diagnostic keys
10. Empty bundle returns the current regime unchanged
11. Constructor scales weights correctly when given a regime

Plus the upstream integration test in `tests/test_signals_regime.py`
(exercises the `bot.signals.SignalsBundle` re-export).

## 6. Serving in the bot

```python
# bot/main.py — inside _on_day, after vol update
regime_label = self.regime.update(bundle)            # M5
gross_mult = self.settings.risk.gross_long_target.get(
    regime_label, DEFAULT_GROSS_TARGET[regime_label],
)
target_w = build_target_weights(
    scores, settings.strategy, equity,
    regime=regime_label,
    gross_long_target=self.settings.risk.gross_long_target,
    sector_caps=..., sector_of=...,
)
```

The scaled `target_w` then flows into `build_trade_deltas` exactly the same
way it does without regime gating — the regime multiplier is just a number
that shrinks the long book.

## 7. Why this is shipped live (no training step)

M5 has **no parameters** to fit — it's a threshold classifier on rolling
stats. There is no gradient, no validation set, no held-out data. The
"ship gate" is: does it produce sensible labels on the captured Kafka
data? Yes (see DATA_REPORT §4).

A future P2 upgrade could swap in an HMM (Markov-switching) or a small
Transformer classifier — both would replace the body of `_classify()`
without touching the public `update(bundle) → label` interface. Until
then, the rules detector ships.

## 8. Why this lives in `model5/` not `bot/models/`

The repo's `modelN/` layout is for **standalone, retrainable model
artifacts**. M5 is intentionally a **rules-based deterministic classifier**
(no weights, no gradients, no validation). Putting it at the repo root:

1. Makes the layout uniform — `model1/` `model2/` `model3/` `model4/`
   `model5/` `model6/` all sit at the repo root.
2. Lets a future P2 HMM upgrade replace just this folder without
   touching `bot/models/`.
3. The `bot/models/regime.py` shim re-exports everything, so the bot
   and tests don't care where the code lives.