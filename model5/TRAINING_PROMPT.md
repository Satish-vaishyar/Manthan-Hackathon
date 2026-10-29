# model5 — Training Prompt

**N/A — model5 is parameter-free.** There is no training step.

The detector is a threshold classifier on rolling market aggregates
(volatility, cumulative return, cross-sectional dispersion). The thresholds
(`vol_low`, `vol_high`, `trend_up`, `trend_dn`, `disp_high`,
`min_history_days`) all live in `configs/model5.yaml` and are tuned by
hand from observed market behaviour, not by gradient descent.

If you want a **trainable** regime classifier in the future, drop one of
these into `model5/src/` (each is a drop-in replacement behind the same
`RegimeDetector.update(bundle) → label` interface):

| Upgrade | Why | Cost |
|---|---|---|
| HMM (2- or 3-state Markov-switching) | Learns regime boundaries from data | EM fit on warm-up data, ~200 lines |
| Small Transformer over rolling stats | Captures non-linear interactions | Heavy; needs GPU + retraining |
| GMM + threshold on posterior probability | Smooths transitions | 50 lines, fast |

Until any of those ships, `model5/src/regime.py` is the canonical M5. The
upstream tests in `tests/test_regime.py` lock the contract; any future
replacement must keep them green.

## Threshold tuning (manual, by hand)

For a real production deploy we'd want to tune the 7 thresholds against
historical data. Recommended procedure:

1. Capture ≥ 1 year of Kafka data with `python scripts/connect.py` (the
   bot already does this via the `data/` folder if started with the
   `recorder/` module).
2. Compute the three aggregates (`market_returns`, `cross_dispersion`)
   yourself in a notebook.
3. Manually label ~6 months of days as `risk_on` / `neutral` / `risk_off`
   from a market context you trust (e.g. Nifty 50 drawdown vs trailing
   vol).
4. Grid-search the 7 thresholds to maximise F1 on your labels.
5. Update `configs/model5.yaml` with the winning values.

This is a one-off exercise; we don't need a live retraining loop because
the aggregates don't drift over the 1-month hackathon window.