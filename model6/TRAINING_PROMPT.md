# model6 — Complete Calibration & Maintenance Prompt

model6 needs **no GPU**. "Training" means re-running the event study on newly recorded news, reviewing the tags,
and improving the patterns. Copy everything between the `=====` lines and give it to whoever owns model6.

=====================================================================================================

## ROLE
You maintain **model6**, the rule-based corporate event tagger for the MANTHAN 2026 trading bot. Each simulated day
the exchange streams news headlines per ticker over Kafka; orders fill at that day's close. model6 tags corporate
events in each headline, weights them (relevance, negation, hedging, magnitude), and outputs an explainable,
calibrated `event_score` per ticker-day. It is also the bot's fallback news signal if model2 is unavailable.

## FOLDER `model6/`
```
configs/events.yaml        33 event types: polarity, prior impact (bps), regex patterns; negation and hedging lists
configs/model6.yaml        weights, shrinkage, scoring scale, data paths
configs/aliases_manual.json  company brand names per ticker (shared idea with model2)
data/raw/                  news.csv, prices.csv, aliases_auto.json (copied from model2/data/processed)
src/text.py                cleaning, alias matching, masking (same as model2)
src/tagger.py              EventTagger: tag_item(), score_day()
src/labels.py              next-day excess-return labels
src/calibrate.py           event study + shrinkage calibration + hold-out evaluation
src/mine_keywords.py       pattern suggestions from data
src/export.py              model6_meta.json + latency check
tests/test_tagger.py       30 unit tests
```

## STEPS (from `model6/`, any laptop, CPU only)

### 1. Environment
```bash
pip install -r requirements.txt
python -m pytest tests -q                 # must show 30 passed
```

### 2. Refresh data after each Kafka recording
Run model2's `src/consolidate.py` first, then copy from `model2/data/processed/` into `model6/data/raw/`:
`news.csv`, `prices.csv`, `aliases_auto.json`. Copy any new names in `model2/configs/aliases_manual.json` into
`model6/configs/aliases_manual.json`.

### 3. Calibrate and evaluate
```bash
python src/calibrate.py
```
Read the printout and `artifacts/metrics.json`:
- `holdout_calibrated.event_score.ic_spearman` should be > 0, and ≥ the prior-only IC, once there are
  ≥ 100 hold-out ticker-days with an event.
- `mean_y_bps_when_positive` should be > `mean_y_bps_when_negative`.

### 4. Review tags (precision matters more than recall)
Open `artifacts/tagged_news.csv`. For the 5 most frequent events, read 20 tagged headlines each. For every false
tag, tighten that pattern in `configs/events.yaml` (add context words, or require "the company" as the subject).
Re-run `pytest` and `calibrate.py`. Target: ≥ 80% of tags on **relevant** items are correct.

### 5. Improve coverage
```bash
python src/mine_keywords.py
```
In `artifacts/keyword_suggestions.csv`:
- `coverage` rows: frequent phrases in relevant headlines with no tag. Add a pattern only for real corporate events.
- `lift` rows: phrases with |t_stat| > 2 and count ≥ 5. Consider a new pattern or event type.
For every new pattern, add a unit test (one headline that must match, one that must not).

### 6. Export
```bash
python src/export.py                      # artifacts/model6_meta.json, latency (< 50 ms per day expected)
```

## ACCEPTANCE (to use event_score as a primary signal, not only a fallback)
- [ ] ≥ 100 hold-out ticker-days with an event.
- [ ] Hold-out calibrated IC > 0.05 and positive-minus-negative mean return > 0.
- [ ] Tag precision on relevant items ≥ 80% (manual review of 100 tags).
- [ ] All unit tests pass.
Otherwise the bot still uses model6, but only as a small-weight signal and for journal explanations.

## RULES
- Never put ticker symbols or company names into patterns; use "the company" (masking does the rest).
- Don't change prior impacts to fit the hold-out; priors are expert guesses, and data enters only via calibration.
- Keep `events.yaml` strings quoted (in YAML, unquoted `no` becomes False and `?` breaks lists).

=====================================================================================================
