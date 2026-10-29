# model3 — Data Report

## Data used

`data/raw/prices.csv` is the same cleaned REST history as model1 (49,654 rows, 40 tickers, 2021-02-04 → 2026-07-27).
It already passed every schema check (no nulls or duplicates, valid OHLC, no weekends). model3 uses only `date`,
`ticker` and `close`; symbols are used only to group rows per ticker. `kafka_prices.csv` (10 Kafka tickers, 37 days)
is included for inference smoke tests only, because momentum features need ~252 days of history.

## Built dataset (`python src/build_dataset.py`)

| Split | Dates | Days | Usable ticker-days |
|---|---|---|---|
| train | 2021-02-04 → 2024-06-28 | 855 | 24,012 |
| val | 2024-07-31 → 2025-03-31 | 167 | 6,680 |
| test | 2025-05-01 → 2026-07-27 | 310 | 7,202 |

- "Usable" means all 16 features exist, which needs ~252 days of history, so the first year of each ticker is
  warm-up only (72% of ticker-days are usable overall).
- 21-day embargo before val and test (the auxiliary target looks 21 days ahead).
- The 23 tickers that stop after 2025-08-29 simply drop out; the test period ends with 17 tickers.
- The 3 large crash days (FLXN, SYNM, CYNT) are kept; next-day returns are winsorised at ±15%.

## Checks run on this laptop (CPU)

- `check_no_ticker.py`: **PASS**. No identity columns; features identical after renaming all tickers
  (max diff 2.2e-16, floating-point rounding from summing the market average in a different order); numeric-only
  inputs; no embeddings; each ticker's output depends only on its own sequence.
- `baselines.py`, `train.py --smoke`, and `export.py` (ONNX + `.pt`, ≈ 5 ms for 40 tickers) all ran.
- A 60-epoch, 1-seed CPU run learned real signal: best epoch 35; net Sharpe val 0.98 (long/short), test 1.35
  long/short and 1.75 long-only. Single seed, not a final result. All test outputs were deleted.

## Hardware note

This model is small (~6.5k parameters) and trains in seconds per seed even on CPU. The GPU helps mostly with the
hyper-parameter search (many configs × 5 seeds).
