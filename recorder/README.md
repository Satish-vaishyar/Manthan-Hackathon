# Recorder — Kafka → Parquet writer

This folder is reserved for `recorder/recorder.py`, which captures every
Kafka message and writes it to `data/raw/` as Parquet for offline training
(see `docs/05 §1`). It is a **separate workstream** from the trading bot —
the bot does NOT depend on this folder existing.

When you're ready to build it:

```python
# recorder/recorder.py (sketch)
# - subscribe to market.ticks and market.news
# - batch every 100 events or 5 seconds
# - write to data/raw/{date=YYYY-MM-DD}/{topic}.parquet
# - dedupe by (simulated_date, ticker, seq)
# - tag loop_id on sandbox-loop days
```

No code here yet — the model training data already lives in `data/data/`
from a previous capture, so this is only needed for fresh runs during the
event.