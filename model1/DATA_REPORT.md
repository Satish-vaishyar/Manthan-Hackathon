# model1 — Data Report (schema and training-readiness check)

Source folder: `E:\Manthan\data\data\` (copied into `model1/data/raw/`). Checked on 2026-10-29.

## 1. Files used by model1

| File in `model1/data/raw/` | Origin | Used for |
|---|---|---|
| `prices.csv` | REST `/stocks/:ticker/history`, cleaned | **Training data** for model1 |
| `tickers.csv` | REST `/stocks` + guessed sector | Reference only (the model doesn't use ticker identity) |
| `stopped_tickers.csv` | Derived | Reference: 23 tickers that stop after 2025-08-29 |
| `split_crash_flags.csv` | Derived | Reference: 3 big single-day drops |
| `meta.json` | Capture provenance | Reference |
| `kafka_prices.csv` | Kafka `market.ticks` capture | Not trainable yet (see §4) |
| `kafka_news.csv` | Kafka `market.news` capture | Phase-2 news branch (see §5) |

Generated: `data/processed/dataset.npz` + `data/processed/scaler.json` (by `src/build_dataset.py`).

## 2. `prices.csv` schema — PASS

| Column | Type | Check |
|---|---|---|
| `date` | YYYY-MM-DD (simulated) | parses 100%, no weekends |
| `ticker` | str | 40 unique |
| `open, high, low, close` | float, 2 decimals | > 0; high ≥ max(open, close, low); low ≤ min(open, close, high): **0 violations** |
| `volume` | int | > 0 everywhere |
| `ticker_active_in_live` | bool | 17 True / 23 False |
| `is_anomaly_crash` | bool | 3 rows |

- 49,654 rows, 2021-02-04 → 2026-07-27, 1,374 trading days, **0 nulls, 0 duplicate (date, ticker)**.
- Per-ticker files under `data/data/prices/` match `prices.csv` exactly; `prices_wide.csv` is 1,374 × 40.
- Matches the PS history schema (`date, open, high, low, close, volume`).

## 3. Training-readiness findings

| # | Finding | How model1 handles it |
|---|---|---|
| F1 | **REST universe (40 tickers) ≠ Kafka universe (10 tickers: AERX, BLUX, DSNY, HAVN, HMDP, MOTR, ORCX, PLNX, ROUT, SONA)** with no overlap | Model is **ticker-agnostic**: no ticker embedding, only scale-free features, trained with random universe sizes (6–40) so it works on any set of tickers |
| F2 | 23 tickers stop on 2025-08-29 → universe shrinks from 40 to 17 | Per-day ticker mask; samples require ≥ 20 valid days in the 64-day window |
| F3 | Tickers start on different dates in Feb 2021 | Same mask; first ~64 days of each ticker are skipped |
| F4 | 3 crashes > 25% (FLXN, SYNM, CYNT) with volume spikes 5–31× → real events, not splits | Kept in features; labels winsorised at ±15% |
| F5 | Dates are shifted (dataset says 2016–2025, dates show 2021–2026) | All logic uses simulated dates only; no calendar features in the model |
| F6 | Prices are on very different scales (≈ 20 to 600) | Only log returns, ratios and z-scores are used; robust scaler fitted on the train split only |
| F7 | NaN warm-up values (SMA200, 252-day high/low, r60) | 1.57% of valid rows; filled with the train median (0 after scaling) |

### Splits (by simulated date, 5-day embargo before val and test)

| Split | Dates | Days | Labelled ticker-days |
|---|---|---|---|
| train | 2021-02-04 → 2024-06-28 | 855 | 34,052 |
| val | 2024-07-09 → 2025-03-31 | 183 | 7,320 |
| test | 2025-04-08 → 2026-07-27 | 326 | 7,802 |

The test period includes the post-2025-08-29 era with only 17 tickers, which is closest to a small live universe.

## 4. Kafka prices — NOT enough for training

- 370 rows = 10 tickers × 37 days, 2024-10-25 → 2025-07-11, captured across **two sandbox loops (1 and 6)** with a 206-day jump.
- A 64-day window can't be built from it. **Use it only for a smoke check of the inference path.**
- To use those tickers for validation, record **one full sandbox loop** (500 days × 5 s ≈ 42 min) with the recorder,
  or fetch their history if REST `/history` accepts them during the live run.

## 5. Kafka news — NOT enough for training (phase 2)

- 689 rows, 10 tickers, 37 days. **324 duplicate (ticker, headline) rows**; one day has 335 items (a backlog burst).
- Very skewed: PLNX 406, DSNY 94, … ORCX 8.
- **Many headlines are unrelated to the company** (sports, politics, film reviews with a ticker attached), and some
  contain mojibake (`�`). So news is noisy by design: a relevance filter is required.
- Decision: model1 phase 1 is **price-only**. The news branch (`use_news: true`) is wired into the architecture but
  only becomes trainable after ≥ 1 full sandbox loop of news (~500 days) is recorded. See ARCHITECTURE.md §6.

## 6. Hardware check on this laptop

- No NVIDIA GPU (`nvidia-smi` not found); installed PyTorch is `2.6.0+cpu`.
- **Train on a GPU machine** (teammate laptop with CUDA, Kaggle or Colab). The code also runs on CPU
  (≈ 20 s per epoch here), so a full CPU run is possible as a fallback but slow for the 5-seed ensemble.

## 7. Smoke test results on CPU (pipeline check only, 2 epochs, 1 seed — NOT a real result)

- Dataset build, training, LightGBM baseline, ONNX export and onnxruntime inference all ran without errors.
- CPU inference latency: **≈ 9 ms for a 40-ticker day** (budget is 50 ms).
- LightGBM baseline (full training): val IC +0.035, test IC +0.032 (t = 2.1). **This is the bar model1 must beat.**
- Smoke artifacts were deleted so the GPU run starts clean.
