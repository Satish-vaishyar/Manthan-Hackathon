# model2 — News Relevance & Impact Model (GPU)

Fine-tunes FinBERT to (1) tell which streamed headlines are really about a company and (2) predict that company's
next-day excess return from them. It outputs 8 news features per ticker per day for the strategy and for model1's
news input. Like model1, it never sees stock symbols.

| Read | For |
|---|---|
| [DATA_REPORT.md](DATA_REPORT.md) | What news data exists, its problems, pipeline check results |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Network, loss, training setup, serving |
| [TRAINING_PROMPT.md](TRAINING_PROMPT.md) | **Complete step-by-step instructions (record → train → export)** |

## Commands (from this folder)

```bash
# On the venue Wi-Fi: collect enough news first (≈ 45 min per sandbox loop)
python src/record_kafka.py --minutes 45

# On the GPU machine
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python src/consolidate.py      # merge captures, de-duplicate, mine aliases
python src/prepare.py          # labels + FinBERT teacher + weak relevance labels
python src/baseline.py         # zero-shot FinBERT bar to beat
python src/train.py --smoke    # pipeline check
python src/train.py            # full GPU fine-tuning (needs >= 300 train ticker-days)
python src/export.py           # package for the bot + CPU latency check
```

Status: the pipeline runs end to end on the existing 410 news items (CPU smoke run). **A real training run needs
at least one full sandbox loop of recorded news**; with the current data `train.py` stops with a clear message.
