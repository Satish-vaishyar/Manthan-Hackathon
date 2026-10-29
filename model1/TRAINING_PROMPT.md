# model1 — Complete Training Prompt

Copy everything between the two `=====` lines and give it to the person or AI coding assistant (Claude Code,
Copilot, Cursor) running on the **GPU machine**. It is self-contained: it says what the data is, what the model is,
how to train it, how to check it and what to hand back.

=====================================================================================================

## ROLE
You are an ML engineer training **model1**, a GPU-trained PyTorch model for the MANTHAN 2026 autonomous trading
hackathon. The bot trades simulated stocks once per simulated day; orders fill at that day's **close**. model1 ranks
tickers by expected **next-day and next-5-day excess return** so the bot knows which stocks to buy, hold or avoid.

## WHAT YOU HAVE (folder `model1/`)
```
model1/
├── configs/model1.yaml        all hyper-parameters, paths and date splits
├── data/raw/prices.csv        49,654 rows of daily OHLCV, 40 tickers, 2021-02-04 → 2026-07-27 (simulated dates)
├── data/raw/…                 tickers, stopped tickers, crash flags, Kafka captures (reference only)
├── data/processed/            dataset.npz + scaler.json (re-buildable)
├── src/features.py            28 ticker-agnostic features (FEATURE_NAMES fixes the order)
├── src/build_dataset.py       prices.csv → dataset.npz [days, tickers, features] + labels + splits + scaler
├── src/dataset.py             one sample = one trading day with all eligible tickers; augmentation
├── src/model.py               CSTT: temporal Transformer per ticker + cross-sectional Transformer across tickers
├── src/train.py               multi-task training, AMP, early stopping, 5-seed ensemble, val/test metrics
├── src/baseline_lgbm.py       LightGBM baseline on the same splits (the bar to beat)
├── src/export.py              ONNX + .pt export, model1_meta.json, CPU latency check
├── ARCHITECTURE.md            full model description (read it first)
└── DATA_REPORT.md             schema checks and data caveats (read it first)
```

## DATA FACTS YOU MUST RESPECT
1. `prices.csv` columns: `date, ticker, open, high, low, close, volume, ticker_active_in_live, is_anomaly_crash`.
   It passed all schema checks (no nulls, no duplicates, valid OHLC, no weekends).
2. The **live trading universe is different from these 40 tickers**, so the model must stay ticker-agnostic:
   never add ticker IDs/embeddings, never use absolute price levels, never use calendar dates as features.
3. 23 tickers stop on 2025-08-29; tickers start on slightly different dates → always use the masks.
4. Labels are close(D) → close(D+h) log returns, winsorised at ±15%, minus the same-day cross-sectional mean.
   **No feature may use information after day D.** Cross-sectional features use only same-day values.
5. Splits are by date with a 5-day embargo: train ≤ 2024-06-28, val 2024-07-09 → 2025-03-31,
   test 2025-04-08 → 2026-07-27. **Never tune anything on the test split.**
6. News data is too small and noisy for now; keep `model.use_news: false`.

## MODEL (implemented in `src/model.py`, details in ARCHITECTURE.md)
- Input `x [B, N, 64, 28]`, `tmask [B, N, 64]`, `nmask [B, N]`.
- Per ticker: Linear→GELU→Linear projection + learned positional embedding → 2-layer Transformer encoder
  (d=64, 4 heads, FF 128, pre-norm, dropout 0.2) → [attention pool ‖ last day] → Linear → h_i.
- Per day: learned [MKT] token + all h_i → 2-layer cross-sectional Transformer (no positions, padding mask) → z.
- Heads: ret1 & ret5 (MLP), direction logit, volatility, market-regime logit (from [MKT]).
- ≈ 157k parameters. Loss = −IC(h1) − 0.5·IC(h5) + 0.5·Huber + 0.3·BCE(dir) + 0.2·MSE(vol) + 0.1·BCE(mkt).

## STEPS

### 1. Environment
```bash
python -m venv .venv && source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu124   # pick the CUDA build matching the driver
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
`cuda.is_available()` must print `True`. If not, fix the CUDA/torch install before continuing.
On Kaggle/Colab: upload the `model1/` folder, enable a GPU, skip the venv.

### 2. Build the dataset (from `model1/`)
```bash
python src/build_dataset.py --config configs/model1.yaml
```
Expected: `D=1374 days, N=40 tickers, F=28 features`; train 855 / val 183 / test 326 days.
"Mean of empty slice" warnings for the last days (no future label) are expected.

### 2b. Prove the model does not use stock symbols (must print PASS)
```bash
python src/check_no_ticker.py
```
It checks: no identity columns among inputs, features identical after renaming every ticker, numeric-only model
inputs, no embedding layers, and permutation-equivariance across tickers. Re-run it after any change to
`features.py` or `model.py`. If it prints FAIL, stop.

### 3. Smoke test (≈ 1 min on GPU)
```bash
python src/train.py --config configs/model1.yaml --smoke
```
It must run 2 epochs without errors and print `device: cuda (...)`. Delete `artifacts/` afterwards.

### 4. Baseline
```bash
python src/baseline_lgbm.py --config configs/model1.yaml
```
Record val/test IC from `artifacts/baseline_lgbm_metrics.json`. (Last measured on CPU: val IC ≈ +0.035,
test IC ≈ +0.032, t ≈ 2.1.)

### 5. Full training (5 seeds × ≤ 80 epochs; expect a few minutes per seed on a modern GPU)
```bash
python src/train.py --config configs/model1.yaml
```
Outputs: `artifacts/seed_<k>/best.pt`, `artifacts/seed_<k>/history.json`, `artifacts/metrics.json`,
`artifacts/test_predictions.csv`.

### 6. Hyper-parameter search (VAL split only; keep the search small)
Search only these knobs, ≤ 20 trials, choosing by **val IC h1 mean** (tie-break: val ICIR):

| Knob | Values |
|---|---|
| `model.d_model` | 32, 64, 96 |
| `model.dropout` | 0.1, 0.2, 0.3 |
| `data.window` | 32, 64 (if 32, also set `min_history` ≤ 20) |
| `train.lr` | 1e-4, 3e-4, 1e-3 |
| `train.loss.ic_h5` | 0.5, 1.0 |
| `train.ticker_subsample` | [6, 40], [8, 20] |

Change values in a copy of the config (`configs/model1_trialXX.yaml`), not in code. After choosing, run step 5
once with the chosen config and only then look at the test metrics.

### 7. Acceptance criteria (all must hold to ship)
- [ ] Test **IC h1 > baseline test IC** OR test **IC h5 > 0.03 with t-stat > 2**.
- [ ] Val and test IC have the same sign (no sign flip).
- [ ] Train IC is not wildly above val IC (if train IC > 3× val IC, increase dropout/noise, reduce d_model).
- [ ] Top-5 minus bottom-5 daily spread > 0 on test.
- [ ] Results hold on the post-2025-08-29 part of test (17-ticker universe): compute IC on
      `test_predictions.csv` rows with `date > 2025-08-29` and report it.
If the criteria fail, report it honestly — the bot will fall back to LightGBM + rules. Do not tune on test to pass.

### 8. Export for the bot
```bash
python src/export.py --config configs/model1.yaml
```
Produces `artifacts/model1_seed<k>.onnx`, `.pt`, and `model1_meta.json` (features, scaler, window, members, latency).
CPU latency for 40 tickers must be < 50 ms per member.

### 9. Leakage self-check (do it once)
Make a copy of `dataset.npz` where `y1`, `y5` are randomly shuffled across days (labels no longer belong to their
features), point a trial config at it and train one seed for 10 epochs. Val IC must drop to ≈ 0 (|IC| < 0.01).
If it doesn't, the features leak future information — stop and report.

## DELIVERABLES (send back to the team)
1. `artifacts/` folder: all `.onnx`, `.pt`, `model1_meta.json`, `metrics.json`, `baseline_lgbm_metrics.json`,
   `test_predictions.csv`, every `seed_*/history.json`.
2. The final config used (`configs/model1_final.yaml`).
3. A short `artifacts/REPORT.md` with: GPU used, training time, the hyper-parameter table tried with val IC,
   final val/test metrics vs baseline, post-2025-08-29 IC, leakage check result, acceptance checklist, and
   a plot of val IC per epoch per seed.

## RULES
- Do not change `src/features.py` feature definitions or order without also re-running `build_dataset.py`
  and telling the team (the bot uses the same file at inference).
- Do not add ticker identity, absolute prices or dates as inputs. Symbols may only be used to group rows when
  computing features and to label reports. `src/check_no_ticker.py` must print PASS before you hand back artifacts.
- Do not use external price data to identify the masked tickers (this is look-ahead and forbidden).
- Keep the model small; more parameters will overfit ~1,000 days of data.
- Fix seeds; everything must be reproducible from the config.

=====================================================================================================
