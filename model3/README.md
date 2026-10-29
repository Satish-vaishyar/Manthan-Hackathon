# model3 — Deep Momentum Network (momentum / trend model)

An LSTM that turns each ticker's recent price history into a trend-following position in [−1, 1]. It is trained
directly on **net Sharpe after trading costs**, with volatility targeting. It complements model1 (ranks tickers
against each other) and model2 (news), and its average signal is a market trend gauge. No stock symbols reach it.

| Read | For |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Features, network, loss, evaluation, baselines, serving |
| [DATA_REPORT.md](DATA_REPORT.md) | Data, splits, checks run, first CPU results |
| [TRAINING_PROMPT.md](TRAINING_PROMPT.md) | **Complete step-by-step training instructions** |

## Commands (from this folder)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python src/build_dataset.py
python src/check_no_ticker.py     # must print PASS
python src/baselines.py           # classic momentum bar to beat
python src/train.py --smoke       # pipeline check
python src/train.py               # 5-seed training
python src/export.py              # ONNX/.pt + model3_meta.json for the bot
```

Status: pipeline verified on CPU. A 1-seed CPU run already beat the 12-month momentum baseline on validation;
final training and the hyper-parameter search have not been run.
