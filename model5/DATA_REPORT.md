# model5 — Data Report

## Input

M5 reads the **`DayBundle`** that the bot's `DayAssembler` emits at the
close of each simulated day. Specifically:

| Field | Source | Cadence |
|---|---|---|
| `bundle.ticks[t].close` | `bot/contracts.py::PriceTickData.close` | 1 per simulated day per ticker |
| `bundle.simulated_date` | `bot/contracts.py::DayBundle.simulated_date` | Once per bundle |
| Yesterday's close (internal) | `_state.last_close[t]` | Persisted in-process |

No volume, no order book, no news, no fundamentals. Just the daily candle
plus yesterday's candle for the log-return calculation.

## Aggregates computed

| Stat | Formula | Window | Used by |
|---|---|---|---|
| `market_return` | `mean_t log(close_t / last_close_t)` | 1 day | Appended to `market_returns` buffer |
| `cross_dispersion` | `stddev_t log(close_t / last_close_t)` | 1 day | Appended to `cross_dispersion` buffer |
| `market_vol_annualised` | `sqrt(var(market_returns) × 252)` | 60d (`market_window`) | `vol` in classifier |
| `cumulative_return` | `sum(market_returns)` | 60d | `cumret` in classifier |
| `avg_dispersion` | `mean(cross_dispersion)` | 20d (`disp_window`) | `disp` in classifier |

## Origin

The DayAssembler collects one `price_tick` per ticker per `simulated_date`
from Kafka topic `market.ticks`. Once it has all the expected tickers (or
the 800 ms hard deadline fires, whichever comes first — see
`docs/02 §3`), it builds a `DayBundle` and invokes the bot's `_on_day`
callback, which calls `regime.update(bundle)`.

## Buffer state stored

| Field | Type | Lifetime | Cap |
|---|---|---|---|
| `_state.market_returns` | `deque[float]` | Bot process | `market_window` = 60 |
| `_state.cross_dispersion` | `deque[float]` | Bot process | `disp_window` = 20 |
| `_state.last_close` | `dict[str, float]` | Bot process | One per ticker ever seen |
| `_state.last_sim_date` | `date \| None` | Bot process | Single value |
| `_state.days_observed` | `int` | Bot process | Monotonic counter |
| `_state.current_regime` | `str` | Bot process | One of the 3 labels |
| `_state.days_in_current_regime` | `int` | Bot process | Monotonic counter |

Memory cost: trivially small (~2 KB regardless of universe size).

## Edge cases

| Case | Behaviour |
|---|---|
| First bundle ever (`last_close` empty) | `market_return = None`, not appended; `_state.last_close` is populated for tomorrow |
| Bundle with < 2 tickers that have history | `dispersion = None`, not appended (need ≥ 2 returns) |
| Bot restart / crash | State lost; detector warm-ups again for `min_history_days` |
| Bundle with no ticks at all | Returns current regime unchanged |
| Same bundle fed twice | Not idempotent; caller must feed each bundle exactly once |

## Output

Single string per bundle: one of `"risk_on"`, `"neutral"`, `"risk_off"`.

Distribution observed on the captured Kafka data (2026-10-29):

- `neutral` dominates during low-vol trending periods
- `risk_off` triggers when 5-day realised vol crosses 30% or cumulative
  return falls below −10%
- `risk_on` is rare in the captured 2-year dataset (only after extended
  low-vol uptrends)

## What's persisted

Nothing on disk. The classifier is in-memory only; restart the bot and it
re-warms from the next bundle.

If we ever want to **share** state across processes, we'd add a
`pickle.dumps(_state)` round-trip. Not currently needed.

## Why no model artifact

The classifier has zero parameters to fit — every threshold lives in
`configs/model5.yaml` and is loaded at process start. There is no
gradient, no validation set, no held-out data.

If we ever want a **trainable** classifier, drop one of these into
`model5/src/` (each is a drop-in replacement behind the same
`RegimeDetector.update(bundle) → label` interface):

| Upgrade | Why | Cost |
|---|---|---|
| HMM (2- or 3-state Markov-switching) | Learns regime boundaries from data | EM fit on warm-up data, ~200 lines |
| Small Transformer over rolling stats | Captures non-linear interactions | Heavy; needs GPU + retraining |
| GMM + threshold on posterior probability | Smooths transitions | 50 lines, fast |

Until any of those ships, `model5/src/regime.py` is the canonical M5.