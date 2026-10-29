# MANTHAN 2026 — Autonomous Trading Bot: Team Docs

These docs are the single source of truth for the team **before and during** implementation.
Read them in order on day one; afterwards use them as reference.

| # | Doc | Who must read it | What it covers |
|---|-----|------------------|----------------|
| 01 | [Problem Summary](01_PROBLEM_SUMMARY.md) | Everyone | Every rule from the official PS, condensed, plus the hidden implications |
| 02 | [Architecture](02_ARCHITECTURE.md) | Everyone | System design, components, data flow, timing budget |
| 03 | [API & Kafka Contracts](03_API_AND_KAFKA_CONTRACTS.md) | Infra, Execution | Exact message schemas, endpoints, error codes, rate limits |
| 04 | [Strategy](04_STRATEGY.md) | Strategy, ML | Core + satellite strategy, signal ensemble, sizing, decision loop |
| 05 | [Models & Features](05_MODELS_AND_FEATURES.md) | ML | Models to train, labels, features, validation, deployment |
| 06 | [Risk, Tax & Execution](06_RISK_TAX_AND_EXECUTION.md) | Strategy, Execution | Risk limits, Indian tax engine, order manager, rate limiter |
| 07 | [Backtester Spec](07_BACKTESTER_SPEC.md) | Backtest owner | Platform-faithful event-driven simulator |
| 08 | [Feature Backlog](08_FEATURE_BACKLOG.md) | Everyone | Prioritized P0/P1/P2 tickets with acceptance criteria |
| 09 | [Team Plan & Timeline](09_TEAM_PLAN_AND_TIMELINE.md) | Everyone | Roles, 24-hour schedule, checkpoints, freeze rules |
| 10 | [Repo & Coding Standards](10_REPO_AND_CODING_STANDARDS.md) | Everyone | Folder layout, config, logging, git workflow, offline setup |
| 11 | [Testing & Runbook](11_TESTING_AND_RUNBOOK.md) | Everyone | Tests, chaos drills, pre-flight checklist, live-hour runbook |
| 12 | [Jury Defense](12_JURY_DEFENSE.md) | Everyone | Evaluation pillars, talking points, expected questions |

## The one-paragraph brief

We build a single-process asyncio Python bot that consumes daily OHLCV candles and news from Kafka,
waits a few hundred milliseconds after each candle for that day's news, scores every ticker with an
ensemble (news-impact model + LightGBM cross-sectional ranker + momentum rules, gated by a market
regime model), converts scores into volatility-scaled, cost-aware target weights, and sends a small
number of validated, rate-limited market orders that fill at the day's close. A local ledger tracks
cash, lots, margin, short settlements and Indian capital-gains tax so we never get a rejection, never
get blocked, and actively optimise STCG/LTCG. Every decision is logged with its reasons so we can
explain every trade to the jury.

## Golden rules (memorise these)

1. **Zero rejected orders.** Validate every order client-side before sending it.
2. **Never exceed ~100 requests/minute.** Platform hard limit is 120/min, then a 30 s penalty box.
3. **Decide within 1 second of the candle.** A day lasts only 3–4 s.
4. **Stay mostly invested, trade rarely.** Cash utilisation is scored, and each order costs ₹20 (+₹15.34 on sells).
5. **Never hold a short through a 10-day settlement unless intended; always keep a cash buffer.**
6. **Always have a working fallback.** A rules-only bot must be tagged in git by hour 4.
7. **Everything runs offline.** Assume the venue Wi-Fi has no internet access.
8. **Log the reason for every decision, including HOLD.**
