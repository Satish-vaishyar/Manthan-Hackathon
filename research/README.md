# Research — offline scripts

This folder is for one-off scripts used during research/backtesting:

- `backtest.py` — replay historical data through the strategy stack
- `sim_broker.py` — `SimBroker` that mirrors platform fill semantics
  (used by the backtester in place of `bot.execution.api_client.APIClient`)
- `feature_analysis.py` — feature importance, SHAP, IC histograms
- `tax_report.py` — generate per-FY tax summary from a ledger dump

All code here imports from `bot/` directly — no duplicated strategy logic.
See `docs/07_BACKTESTER_SPEC.md` for the full spec.

No code here yet — separate workstream.