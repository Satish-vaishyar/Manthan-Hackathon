# model4 — Data Report

## Input

M4 reads **only the daily close** of each ticker. Specifically:

| Field | Source | Cadence |
|---|---|---|
| `tick.close` | `DayBundle.ticks[ticker].close` from `bot/contracts.py::PriceTickData` | 1 per simulated day per ticker |

No volume, no order book, no news, no fundamentals. Just the daily candle.

## Origin

Each `price_tick` arrives on Kafka topic `market.ticks` from the gateway's
replay (development stage) or live sim (simulation stage). The DayAssembler
collects one tick per ticker per `simulated_date` and emits a `DayBundle`
with `ticks[ticker].close` for every ticker that reported on that day.

M4 ingests this stream via `bot/main.py::TradingBot._on_day`:

```python
for ticker, tick in bundle.ticks.items():
    self.vol.update(ticker, tick.close)
```

## Window length

Default: **20 daily returns**. Configurable via `configs/model4.yaml`:

```yaml
model4:
  window: 20            # rolling-window length
  fallback_vol: 0.02    # returned for tickers with < 5 returns
```

For a backtest that needs shorter history (≤ 5 returns), the fallback
guarantees every ticker still produces a finite vol.

## What's stored

Nothing on disk. M4 is fully in-memory:

| Field | Type | Lifetime |
|---|---|---|
| `_state[ticker]._ret` | `deque[float]` (maxlen=`window`) | Bot process lifetime |
| `_state[ticker]._sum` | `float` | Same |
| `_state[ticker]._sumsq` | `float` | Same |
| `_state[ticker]._prev_close` | `float \| None` | Same |

Per-ticker memory cost: `(20 × 8 bytes for deque + 3 × 8 bytes for scalars)
× 40 tickers ≈ 7 KB total`. Trivial.

## Edge cases

| Case | Behaviour |
|---|---|
| First tick ever (`prev_close=None`) | No return computed; only `prev_close` is recorded |
| Same close as previous day | Returns `log(1) = 0`; vol trends to 0 |
| Missing day (ticker didn't trade) | Skipped (DayAssembler doesn't emit a tick for it) |
| Ticker with fewer than 5 valid returns | `vol(t)` returns `fallback_vol` (0.02) |
| `close ≤ 0` (data error) | Silently ignored — no update, no return |
| Reset / restart | All state lost — the bot will warm up again from the next day |

## Output

Single float per ticker per day: annualised realised vol σ̂ in [0, ∞).

Typical values on a 10-ticker liquid universe:

- Trending stocks: σ̂ ≈ 0.10 – 0.20 (10–20% annualised)
- High-vol growth names: σ̂ ≈ 0.30 – 0.60
- Constant-price feed (data bug): σ̂ = 0.0 → constructor guard prevents
  div-by-zero in `raw_w[t] = max(0, score[t]) / max(1e-9, vol[t])`.

## Why no model artifact

The estimator is parameter-free; there's nothing to persist between
processes. Restart the bot, it warms up over 5 trading days, then converges
to the same vol it would have computed had it run continuously.

If we ever want to **share** state across processes (e.g., bot ↔ backtester)
we'd add a tiny `pickle.dumps(_state)` round-trip, but it's never been
needed in this repo.