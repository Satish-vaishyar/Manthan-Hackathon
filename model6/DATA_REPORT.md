# model6 — Data Report

## Inputs (copied from model2's processed data)

| File | Content |
|---|---|
| `data/raw/news.csv` | 410 unique news items, 10 Kafka tickers, 41 simulated dates (deduplicated across sandbox loops) |
| `data/raw/prices.csv` | 370 Kafka candles (10 tickers × 37 dates) for next-day labels |
| `data/raw/aliases_auto.json` + `configs/aliases_manual.json` | company aliases for relevance and masking |

Schema and cleaning were already checked for model2 (see `model2/DATA_REPORT.md`). **Refresh these files from
model2 after every new Kafka recording** (`model2/src/consolidate.py` → copy `news.csv`, `prices.csv`,
`aliases_auto.json`).

## Results on the current data (run 2026-10-29)

| Item | Value |
|---|---|
| Items with at least one event | 54 of 410 (13%) |
| Items mentioning their company (relevant) | 62 of 410 (15%) |
| Hold-out period | dates ≥ 2025-06-24: 120 ticker-days, 14 with an event (coverage 12%) |
| Hold-out IC of `event_score` (prior-only impacts) | +0.13 on 14 ticker-days |
| Hold-out IC of `event_score` (calibrated) | +0.05 on 14 ticker-days |

**These numbers are far too small to mean anything yet.** They show only that the pipeline works end to end.
With so little data, calibration keeps every impact close to its expert prior, which is intended.

Event study on all dates (events with ≥ 3 labelled items) — directionally consistent where counts are largest:
REGULATORY_ACTION −275 bps mean (6 items, t −1.7, posterior −122 bps); DEMAND_STRONG +220 bps (5 items, t 2.2,
posterior +65 bps); PARTNERSHIP +9 bps (7 items). Full table: `artifacts/event_study.csv`.

## Known data issues

- Most streamed headlines are unrelated to the company; they get 0.25 relevance weight rather than being dropped.
- Some tags are still debatable. For example, "RouteWave driver indicted" is tagged as REGULATORY_ACTION
  for ROUT, because the masked brand is the subject. Review `artifacts/tagged_news.csv` after each recording.
- Coverage (~12% of ticker-days) is low because the stream is mostly general news. That's expected; model2 covers
  the softer sentiment cases.

## Checks

- `python -m pytest tests -q` → 30 passed.
- `src/export.py` → busiest recorded day (19 items) scored in ~17 ms on CPU.
