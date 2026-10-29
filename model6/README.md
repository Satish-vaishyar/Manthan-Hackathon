# model6 — Event Tagger (rules + calibrated impacts)

Recognises 33 types of corporate events in streamed headlines (earnings beats/misses, guidance, contract wins,
probes, capital raises, management exits…). It handles negation ("denies probe"), hedging ("reportedly"), and
magnitudes ("24% YoY", "$437M"), and outputs an explainable `event_score` per ticker per day. Event impacts start
from expert priors and are calibrated with an event study on recorded news. CPU only, ~17 ms per day, works
offline, and doubles as the bot's fallback news signal. No ticker symbols reach the matching.

| Read | For |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Pipeline, features, taxonomy, calibration |
| [DATA_REPORT.md](DATA_REPORT.md) | Data, current results, known issues |
| [TRAINING_PROMPT.md](TRAINING_PROMPT.md) | **Calibration and maintenance steps** |

## Commands (from this folder)

```bash
pip install -r requirements.txt
python -m pytest tests -q          # 30 unit tests
python src/calibrate.py            # event study + calibration + hold-out evaluation
python src/mine_keywords.py        # suggestions for new patterns
python src/export.py               # model6_meta.json + latency
```

## Using it in the bot

```python
import sys; sys.path.insert(0, "model6/src")
from tagger import EventTagger
tagger = EventTagger.from_files("model6")                 # uses artifacts/calibration.json if present
feats = tagger.score_day(todays_news, live_tickers, aliases, with_evidence=True)
# feats["PLNX"] -> {"event_score": 0.31, "rule_polarity": 0.76, "n_events": 2, ..., "evidence": [...]}
```

Status: complete and calibrated on the 410 news items recorded so far (impacts still close to their priors).
Re-run `calibrate.py` after each new Kafka recording.
