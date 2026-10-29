# model1 — Cross-Sectional Temporal Transformer (GPU)

A price-based ranking model for the MANTHAN 2026 trading bot. Each simulated day it scores every live ticker on
expected next-day and next-5-day excess return, plus volatility and market regime. It is ticker-agnostic, so it can be
trained on the 40 REST tickers and applied to whatever tickers the live Kafka stream sends.

| Read | For |
|---|---|
| [DATA_REPORT.md](DATA_REPORT.md) | Schema checks, data caveats, splits, hardware note |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Inputs, features, network, loss, evaluation, serving |
| [TRAINING_PROMPT.md](TRAINING_PROMPT.md) | **The complete prompt/instructions to train it on a GPU machine** |

## Quick start (from this folder)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python src/build_dataset.py            # data/raw/prices.csv -> data/processed/dataset.npz
python src/check_no_ticker.py          # proves no stock symbols reach the model (must print PASS)
python src/train.py --smoke            # 2-epoch pipeline check
python src/baseline_lgbm.py            # LightGBM bar to beat
python src/train.py                    # full 5-seed GPU training
python src/export.py                   # ONNX/.pt + model1_meta.json for the bot
```

## Layout

```
model1/
├── configs/model1.yaml
├── data/raw/            copied from data/data (prices, tickers, flags, Kafka captures)
├── data/processed/      dataset.npz, scaler.json (generated)
├── src/                 features, build_dataset, dataset, model, train, baseline_lgbm, export
├── artifacts/           training outputs (empty until trained)
└── requirements.txt
```

Status: the pipeline was verified end to end on CPU (smoke run). It has **not** been trained on a GPU yet.
