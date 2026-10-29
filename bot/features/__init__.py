"""bot.features — feature engine (P1-01).

Public surface:
  RingBuffer       — per-ticker rolling OHLCV history
  build_model1_features  — [L=64, F=28] windows for M1 CSTT
  build_model3_features  — [L=63, F=16] windows for M3 DMN
"""
from .ring_buffer import RingBuffer, Candle
from .indicators import (
    MODEL1_FEATURE_NAMES, MODEL1_WINDOW, MODEL1_N_FEATURES,
    MODEL3_FEATURE_NAMES, MODEL3_WINDOW, MODEL3_N_FEATURES,
    build_model1_features, build_model3_features,
)

__all__ = [
    "RingBuffer", "Candle",
    "MODEL1_FEATURE_NAMES", "MODEL1_WINDOW", "MODEL1_N_FEATURES",
    "MODEL3_FEATURE_NAMES", "MODEL3_WINDOW", "MODEL3_N_FEATURES",
    "build_model1_features", "build_model3_features",
]