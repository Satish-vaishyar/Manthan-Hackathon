# model4 — Realised-Volatility Estimator (RVE)

## 1. What the model does

For every ticker on every simulated trading day **D**, model4 estimates the
**annualised realised volatility** from the last N daily log-returns and
exposes:

| Output | Range | Used by the bot for |
|---|---|---|
| `vol(t)` | annualised σ of daily log-returns, √252 scaling | Inverse-vol position sizing in `bot/portfolio/constructor.py` |
| `ready(t)` | bool | Whether the estimator has ≥ 5 returns for ticker `t` |

The estimator is **per-ticker**, **incremental** (O(1) per update), and
**CPU-only**. No GPU, no pandas, no external state — the bot calls
`update(ticker, close)` once per day per ticker from the DayAssembler.

If a ticker has fewer than 5 returns, `vol(t)` returns the configured
`fallback_vol` (default `0.02` ≈ 32% annualised) so the bot still sizes
positions in early-life tickers rather than blowing up on a single quote.

## 2. Design constraints that shaped it

1. **3-5 s per simulated day** → the entire portfolio-size cycle (assemble →
   score → weight → submit) must finish in < 800 ms per docs/02. Vol
   computation per ticker is O(window) ≈ 20 multiplications; for 40 tickers
   this is ~10 µs total, well inside the budget.
2. **Incremental** → no `pandas.DataFrame`; no per-day array allocs. Welford-
   style or sliding-window sum + sum-of-squares refresh keeps the work
   constant per push.
3. **Per-ticker state** → each ticker gets its own `_Window`; tickers are
   independent so adding/removing one doesn't disturb the others.
4. **Tie-break with sizing** → when vol is exactly 0 (constant price
   series), inverse-vol weighting would divide by zero. The estimator
   returns `0.0` in that case and the constructor's `max(1e-9, vol)`
   guard prevents a crash.

## 3. Algorithm

For a single ticker, maintain a deque of the last `window` log-returns
`r_t = log(close_t / close_{t-1})`. The annualised realised vol is:

```
σ̂ = sqrt( (Σ r² − (Σ r)²/n) / n × 252 )
```

where `n = len(window)`. The estimator tracks `Σ r` and `Σ r²` incrementally
so each update is O(1); only the deque shift is O(1) amortised (ring buffer).

### Update rule

```
update(ticker, close_t):
    if prev_close is set:
        r = log(close_t / prev_close)
        if len(window) == window:
            old = window.popleft()          # ring buffer
            sum_r   -= old
            sum_r2  -= old * old
        window.append(r)
        sum_r   += r
        sum_r2  += r * r
    prev_close = close_t
```

### Read-out

```
vol(t) = sqrt(max(0, sum_r2/n − (sum_r/n)²) × 252)   if n ≥ 5
       = fallback_vol                                 otherwise
```

`max(0, ...)` guards against negative variance from floating-point error
when `sum_r2/n ≈ (sum_r/n)²`.

## 4. Configuration

| Param | Default | Meaning |
|---|---|---|
| `window` | 20 | Number of daily returns in the rolling window |
| `fallback_vol` | 0.02 | Returned when `n < 5` (early-life or no history) |

Both are wired through `bot/main.py::TradingBot.__init__` and exposed via
`bot/config.py` + `configs/default.yaml` so they can be tuned per env
without code edits.

## 5. Tests

- `tests/test_vol.py` — 9 tests:
  - empty state returns fallback
  - readiness after 5 obs
  - higher realised vol from noisier returns
  - window slide drops oldest
  - per-ticker independence (different series don't bleed)
  - unready tickers stay on fallback
  - `annualised_vol_from_closes` on constant / short / noisy series
- `tests/test_signals_volatility.py` — upstream tests for the
  `VolatilityModule` wrapper in `bot/signals/volatility.py`.

## 6. Serving in the bot

```python
# bot/main.py — inside _on_day
self.vol.update(ticker, tick.close)              # M4: feed the day's close
size = max(1e-9, self.vol.vol(ticker))            # used by constructor
```

The constructor (`bot/portfolio/constructor.py`) does
`raw_w[t] = max(0, score[t]) / vol[t]` then renormalises, so a ticker with
`vol = 0.04` gets twice the notional of one with `vol = 0.08` (same score).

## 7. Why this is shipped live (no training step)

M4 has **no parameters** to fit — it's the textbook realised-vol estimator.
There is no gradient, no validation set, no held-out data. The "ship
gate" is: does it match the textbook definition? Yes. Does it stay inside
the 800 ms deadline? Yes (< 1 ms for 40 tickers × 20-day window). Ship.

A future P2 upgrade could replace it with an EWMA vol (RiskMetrics-style
λ=0.94) or a GARCH(1,1) — both are 30-line replacements behind the same
`VolEstimator.vol(ticker)` interface.