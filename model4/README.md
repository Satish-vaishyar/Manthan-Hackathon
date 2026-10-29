# model4 — Realised Volatility (M4)

Per-ticker rolling-window annualised volatility estimator. CPU-only, no
parameters to fit, no training step. Pure statistical estimator used by
the bot's inverse-vol position sizer.

## Layout

```
model4/
├── ARCHITECTURE.md      ← design + algorithm + bot integration
├── DATA_REPORT.md       ← what data the estimator consumes
├── README.md            ← this file
├── TRAINING_PROMPT.md   ← N/A — no training step (rules-based)
├── requirements.txt     ← only structlog + stdlib
├── configs/
│   └── model4.yaml      ← tunable thresholds (window, fallback_vol)
├── src/
│   └── vol.py           ← VolEstimator + annualised_vol_from_closes
└── tests/
    └── test_vol.py      ← 9 tests (in tests/, mirrored here is optional)
```

## Quick start

```python
from model4.src.vol import VolEstimator, annualised_vol_from_closes

est = VolEstimator(window=20, fallback_vol=0.02)
for ticker, close in day_closes.items():
    est.update(ticker, close)
v = est.vol("AAPL")           # annualised σ, or 0.02 if not ready
```

## Why this isn't in `bot/models/`

The repo's `modelN/` layout is for **standalone, retrainable model
artifacts**. M4 is intentionally a **rules-based deterministic estimator**
(no weights, no gradients, no validation). Giving it its own folder next
to model1/2/3/6:

1. Makes the layout uniform — `model1/` `model2/` `model3/` `model4/`
   `model5/` `model6/` all sit at the repo root.
2. Lets a future P2 GARCH/EWMA upgrade replace just this folder without
   touching `bot/models/`.
3. The `bot/models/vol.py` shim re-exports everything, so the bot and
   tests don't care where the code lives.

## Live status

- ✅ Production-ready; running live in `bot/main.py::TradingBot._on_day`.
- ✅ Wired into `bot.signals.volatility.VolatilityModule` (the M4
  wrapper exposed by the signals package).
- ✅ 9 unit tests + upstream's `tests/test_signals_volatility.py` pass.
- ✅ No GPU, no external data — feeds off `bundle.ticks[ticker].close`
  that the DayAssembler already produces.
- ⚠️ Warmup: first 5 days per ticker use the fallback (`0.02`). After
  that, the live window takes over.