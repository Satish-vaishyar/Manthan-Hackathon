# model5 — Market Regime Detector (M5)

Once per simulated day, classify the market into `risk_on` / `neutral` /
`risk_off` and scale the long-side gross exposure accordingly. Pure
statistical classifier on rolling market aggregates; no GPU, no training
step.

## Layout

```
model5/
├── ARCHITECTURE.md      ← design + algorithm + bot integration
├── DATA_REPORT.md       ← what data + aggregates the classifier consumes
├── README.md            ← this file
├── TRAINING_PROMPT.md   ← N/A — no training step (rules-based)
├── requirements.txt     ← only structlog + stdlib
├── configs/
│   └── model5.yaml      ← tunable thresholds (vol_low, vol_high, ...)
├── src/
│   └── regime.py        ← RegimeDetector + RegimeConfig + DEFAULT_GROSS_TARGET
└── tests/
    └── test_regime.py   ← 11 tests (mirrored from tests/)
```

## Quick start

```python
from model5.src.regime import RegimeDetector, RegimeConfig

det = RegimeDetector(RegimeConfig(min_history_days=20))
for bundle in day_bundles:
    label = det.update(bundle)              # "risk_on" / "neutral" / "risk_off"
    snap = det.snapshot()                   # diagnostic dict
```

## Why this isn't in `bot/models/`

The repo's `modelN/` layout is for **standalone, retrainable model
artifacts**. M5 is intentionally a **rules-based deterministic classifier**
(no weights, no gradients, no validation). Giving it its own folder next
to model1/2/3/6:

1. Makes the layout uniform — `model1/` `model2/` `model3/` `model4/`
   `model5/` `model6/` all sit at the repo root.
2. Lets a future P2 HMM/Markov-switching upgrade replace just this
   folder without touching `bot/models/`.
3. The `bot/models/regime.py` shim re-exports everything, so the bot
   and tests don't care where the code lives.

## Live status

- ✅ Production-ready; running live in `bot/main.py::TradingBot._on_day`.
- ✅ Wired into `bot/portfolio/constructor.build_target_weights` via
  the `regime` arg + `gross_long_target` lookup.
- ✅ 11 unit tests + upstream's `tests/test_signals_regime.py` pass.
- ✅ No GPU, no external data — feeds off `DayBundle.ticks` and an
  internal rolling-buffer state.
- ⚠️ Warmup: first 20 days return `neutral` (no opinion yet). The
  state machine engages after that.