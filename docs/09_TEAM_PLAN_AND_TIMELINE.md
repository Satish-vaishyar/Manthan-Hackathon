# 09 — Team Plan & Timeline

## 1. Roles

| Role | Code | Owns | Backup for |
|------|------|------|-----------|
| Infra & Execution lead | **A** | Kafka, heartbeat, stage, API client, rate limiter, order manager, reconciliation, deployment | D |
| ML lead | **B** | Data pipeline, features, news model, ranker, regime, model serving | C |
| Strategy, Risk & Tax lead | **C** | Ledger, validator, constructor, risk manager, tax engine, strategy params | B |
| Backtest, Dashboard & Docs lead | **D** | Recorder, backtester, reports, Streamlit, journal, jury deck | A |

For a team of **3**: merge D into A (recorder/journal) and C (backtester). For a team of **2**: person 1 = A + D, person 2 = B + C.

**Captain** (pick one): tracks the timeline, calls checkpoints, owns the go/no-go at code freeze.

## 2. Before the event (prep)

- [ ] Everyone reads docs 01–06.
- [ ] Repo skeleton created (doc 10 layout), CI-free but `pytest` runnable.
- [ ] Offline bundle: wheels, FinBERT weights + ONNX, optional NSE data, Financial PhraseBank.
- [ ] Mock Kafka producer + mock REST server (FastAPI) replaying sample data → develop without the venue.
- [ ] Ledger, validator, tax engine **unit tests written** (pure logic, no network needed).
- [ ] Laptops: same Python version (3.11), power banks/chargers, LAN adapter if available.

## 3. 24-hour schedule

| Hour | A (Infra) | B (ML) | C (Strategy/Risk/Tax) | D (Backtest/Dash) | Checkpoint |
|------|-----------|--------|-----------------------|-------------------|------------|
| 0.0–0.5 | Wi-Fi, portal register, token | Watch video | Desk verification | Watch video, note open questions (doc 01 §14) | Token works on `/stage` |
| 0.5–2 | Consumer, heartbeat, stage watcher, API client | Pull `/history` for all tickers | Ledger + validator wired to live API | **Start recorder**, journal | Ticks visible; Ready on portal |
| 2–4 | Rate limiter, order sender, rejections | Dataset builder v0, inspect headlines | Rules strategy P0-11 | Journal viewer (simple) | **H4: `v0-fallback` tagged, trading with 0 rejections** |
| 4–9 | Reconciliation, retry safety, latency metrics | Features P1-01, event tagger, FinBERT ONNX | Constructor, risk manager, settlement calendar | Backtester + SimBroker | **H8 jury checkpoint**: architecture + fallback demo |
| 9–14 | Chaos tests, SAFE_MODE, kill switch | News head + ranker training, walk-forward | Tax engine, event book, parameter grid | Parity test, backtest report | **H14: `v1-ensemble` tagged**, backtest > rules |
| 14–18 | Hot-standby setup, perf tuning | Retrain on more data, SHAP, regime | Final param tuning (walk-forward only) | Dashboard, reason cards | **H18: CODE FREEZE → `v2-final`** |
| 18–20 | 30-min full dress rehearsal in sandbox; fix only bugs | Model freeze, meta files | Config freeze | Jury deck + diagrams | Go/no-go: 0 rejections, p95 latency < 1.2 s, 0 × 429 |
| 20–23 | Pre-flight: confirm Ready; run live bot | Monitor model health | Monitor risk/tax | Monitor dashboard, take screenshots | **Live hour: hands off the code** |
| 23–24 | Defense | Defense | Defense | Defense | Every member speaks |

## 4. Freeze rules

- **Code freeze at hour 18.** After that, only fixes for crashes/rejections, reviewed by a second person.
- **No parameter changes during the live hour.** Only kill switch / SAFE_MODE if something is clearly broken.
- Git tags: `v0-fallback`, `v1-ensemble`, `v2-final`. The live run uses `v2-final`; if it fails the
  dress rehearsal, the live run uses `v1-ensemble` or `v0-fallback`.

## 5. Communication

- 5-minute stand-up every 3 hours: done / next / blocked.
- Shared `NOTES.md` at repo root: venue facts (SERVER_IP, confirmed rules, answers to open questions).
- Any rule discovery (e.g. settlement anchor, FIFO) → update `docs/01_PROBLEM_SUMMARY.md` §14 immediately.

## 6. Rest plan

Stagger 60–90 minute naps between hours 10–17 so at least 3 members are always awake; nobody sleeps
during hours 18–24.
