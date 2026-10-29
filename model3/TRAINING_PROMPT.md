# model3 — Complete Training Prompt

Copy everything between the two `=====` lines and give it to the person or AI coding assistant on the GPU machine.

=====================================================================================================

## ROLE
You are an ML engineer training **model3**, a Deep Momentum Network (LSTM) for the MANTHAN 2026 autonomous trading
hackathon. The exchange replays daily candles; orders fill at each day's close. model3 outputs, per ticker per day,
a trend-following position signal in [−1, 1]. It is trained directly on the **net Sharpe ratio after trading
costs** of volatility-targeted positions.

## WHAT YOU HAVE (folder `model3/`)
```
model3/
├── configs/model3.yaml       hyper-parameters, cost/vol targets, date splits
├── data/raw/prices.csv       40 tickers of daily OHLCV, 2021-02-04 → 2026-07-27 (simulated dates)
├── data/processed/           dataset.npz (re-buildable)
├── src/features.py           16 volatility-normalised momentum features (FEATURE_NAMES fixes the order)
├── src/build_dataset.py      prices → dataset.npz (features, next-day returns, σ, trend labels, splits)
├── src/dataset.py            63-day training sequences + evaluation windows
├── src/model.py              DMN: LSTM → tanh position + trend head
├── src/backtest.py           vectorised net-of-cost backtest (used for validation, baselines, reports)
├── src/train.py              Sharpe-loss training, early stopping on val net Sharpe, 5-seed ensemble
├── src/baselines.py          buy-and-hold, 12-month TSMOM, MACD combo
├── src/check_no_ticker.py    proves no stock symbols reach the model
├── src/export.py             ONNX + .pt + model3_meta.json, CPU latency
├── ARCHITECTURE.md / DATA_REPORT.md   read both first
```

## FACTS YOU MUST RESPECT
1. The live tickers differ from these 40, so the model stays ticker-agnostic: no IDs, no embeddings, no price
   levels, no dates as features. `check_no_ticker.py` must print PASS.
2. Labels are close(D) → close(D+1) log returns (winsorised ±15%). Features use data up to D only.
3. Splits by date: train ≤ 2024-06-28, val 2024-07-31 → 2025-03-31, test 2025-05-01 → 2026-07-27.
   **Choose everything on val. Look at test once, at the end.**
4. Keep the exposure anchor in the loss (the Sharpe ratio doesn't change if positions are scaled; without the
   anchor, positions collapse towards zero).

## STEPS (from `model3/`)

### 1. Environment
```bash
python -m venv .venv
.venv\Scripts\activate                      # Linux: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124   # match the driver's CUDA version
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 2. Data, check, baselines
```bash
python src/build_dataset.py          # expect D=1374, N=40, F=16; train/val/test usable 24012/6680/7202
python src/check_no_ticker.py        # must print PASS
python src/baselines.py              # writes artifacts/baseline_metrics.json
```

### 3. Smoke test, then full training
```bash
python src/train.py --smoke          # 3 epochs; then delete artifacts/seed_* and artifacts/metrics.json
python src/train.py                  # 5 seeds x <= 150 epochs (seconds to minutes per seed)
```

### 4. Hyper-parameter search (VAL net Sharpe only; ≤ 30 configs × 3 seeds; this is where the GPU helps)
Make config copies (`configs/model3_trialXX.yaml`) and run `python src/train.py --config <file>` with a separate
`export.out_dir` per trial.

| Knob | Values |
|---|---|
| `model.hidden` | 16, 32, 64 |
| `model.dropout` | 0.1, 0.3, 0.5 |
| `train.lr` | 1e-3, 3e-3 |
| `train.seq_len` | 63, 126 |
| `train.batch_size` | 128, 256 |
| `train.cost_bps` | 5, 10 (higher = smoother, lower-turnover positions) |
| `model.position` | tanh, sigmoid (long-only) |

Pick by mean val net Sharpe across seeds (tie-break: lower turnover, smaller max drawdown). Retrain the winner with
5 seeds, then read test once.

### 5. Acceptance criteria (all must hold to ship)
- [ ] Ensemble **val** net Sharpe (long/short) > best baseline val Sharpe (`tsmom_12m` = 0.75).
- [ ] **Test** long-only net Sharpe ≥ `tsmom_12m_long` (1.54).
- [ ] Test max drawdown not worse than buy-and-hold's.
- [ ] Average |position| between 0.2 and 0.8 (sizes are meaningful, not collapsed).
- [ ] Seeds agree: std of val Sharpe across seeds < 0.5.
If these fail, report it: the bot will use `tsmom_12m_long` as the momentum signal.

### 6. Export
```bash
python src/export.py                 # model3_seed*.onnx/.pt + model3_meta.json; CPU latency ~5 ms for 40 tickers
```

## DELIVERABLES
1. `artifacts/`: `model3_seed*.onnx`, `.pt`, `model3_meta.json`, `metrics.json`, `baseline_metrics.json`,
   `test_positions.csv`, `seed_*/history.json`.
2. The final config (`configs/model3_final.yaml`).
3. `artifacts/REPORT.md`: GPU, search table (config → mean val Sharpe ± std), final val/test vs baselines
   (long/short and long-only), turnover, drawdown, acceptance checklist, val-Sharpe-per-epoch plot.

## RULES
- Never tune on test. Never use external prices to unmask tickers.
- Don't remove the cost term or the exposure anchor from the loss.
- Keep the model small; fix seeds; everything reproducible from the config.

=====================================================================================================
