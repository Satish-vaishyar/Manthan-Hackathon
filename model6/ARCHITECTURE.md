# model6 — Event Tagger (rules + calibrated impacts)

## 1. Role

model6 is the planned **M1a event tagger**. It reads each streamed headline and description, recognises corporate
events such as earnings beats, contract wins, probes and capital raises, and turns them into a signed,
**explainable** impact score per ticker-day. It complements model2:

| | model2 (FinBERT NRIM) | model6 (Event Tagger) |
|---|---|---|
| Method | learned, GPU fine-tuned | rules + data-calibrated impacts, CPU |
| Speed | ~360 ms per member per day | ~17 ms for the busiest day |
| Explainability | attention weights | exact matched words, e.g. `CONTRACT_WIN: 'Awarded $437M Task Order' +72bps` |
| Data needed | ≥ 300 labelled ticker-days | works with zero data (expert priors); improves as data grows |
| Failure mode | can't run without the model files | always available; **fallback news signal** |

The bot uses `event_score` as a news signal (and as the fallback if model2 is unavailable), and puts the
`evidence` strings into the decision journal for the jury.

## 2. Pipeline per news item

```
headline, description, ticker
  → clean (mojibake, quotes, whitespace)
  → relevance = text mentions the ticker symbol or one of its aliases (configs/aliases_manual.json + CamelCase auto-aliases)
  → mask: the company's own name/aliases → "the company"; every other ticker symbol removed
  → for each of 33 event types: first matching regex in the headline, then in the description
       negated?  a negation word in the 4 words before the match ("denies probe")      → × −0.5
       hedged?   may / could / reportedly / considers / talks …                         → × 0.6
       magnitude: % figure for earnings/guidance/demand/stock events (1 + pct/100, cap 2)
                  $ / crore amount for contracts, raises, buybacks, fines (×1.2 million, ×1.5 billion)
  → weight = field (headline 1.0 / description 0.5) × relevance (1.0 / 0.25) × negation × hedging × magnitude
  → contribution_bps = impact_bps[event] × weight          (one tag per event per item; the headline wins)
```

## 3. Per ticker-day features (`EventTagger.score_day`)

| Feature | Meaning |
|---|---|
| `event_score` | tanh(Σ contribution_bps / 150) ∈ (−1, 1), using calibrated impacts |
| `rule_polarity` | tanh(Σ polarity × weight): uncalibrated, direction only |
| `n_events`, `n_relevant_items`, `has_event` | counts / flag |
| `max_abs_impact_bps` | the strongest single event of the day |
| `ev_<EVENT>` (16 types) | summed weights of the most important events |
| `evidence` (optional) | human-readable list of matched events, for the journal |

## 4. Event taxonomy (`configs/events.yaml`)

33 types, each with a polarity, an expert prior impact (bps of next-day excess return) and regex patterns:
EARNINGS_BEAT/MISS, GUIDANCE_UP/DOWN, DIVIDEND, BUYBACK, CONTRACT_WIN, PARTNERSHIP, PRODUCT_LAUNCH, APPROVAL,
REJECTION, MNA_TARGET, MNA_ACQUIRER, ANALYST_UPGRADE/DOWNGRADE, VALUATION_CONCERN, REGULATORY_ACTION, LEGAL_WIN,
MGMT_EXIT, MGMT_HIRE, LAYOFFS, DEBT_DISTRESS, CAPITAL_RAISE, INSIDER_SELL/BUY, SAFETY_BREACH, LABOR_STRIKE,
EXPANSION, DEMAND_STRONG/WEAK, STOCK_UP/DOWN, MACRO_TRADE.

Patterns were tightened after reviewing real tags. For example, "administration" (the US government) no longer
means bankruptcy, "opens" in an album review is not an expansion, "recalls" meaning "remembers" is not a product
recall, and generic crime words count as regulatory action only when the company is the subject.

## 5. Calibration ("training", `src/calibrate.py`)

Event study: for every event type, the weighted mean next-day excess return of the items it tagged. Labels are
built like model2's: next-day log return minus the cross-ticker mean, ±15% winsorised, and only when the next candle
is ≤ 5 days later.

```
posterior_bps = (n_eff × mean_bps + k × prior_bps) / (n_eff + k),    k = 20 pseudo-observations
```
- Sparse events stay near the expert prior; frequent events move towards what the data shows.
- Honest evaluation: fit on the first 70% of dates, score the last 30% (`calibration_holdout.json`, `metrics.json`).
- Deployment fit uses all dates (`calibration.json`).
- `mine_keywords.py` proposes new patterns: frequent phrases in relevant but untagged headlines, and phrases with
  a large mean return (t-stat). **Humans decide** what goes into `events.yaml`.

## 6. Guarantees

- No ticker symbol or company name reaches the matching (masked/stripped; unit-tested).
- 30 unit tests cover detection, unrelated headlines, negation, hedging, magnitude, relevance weighting,
  headline-over-description and output shapes.
- Deterministic and offline: no model downloads, no network.
