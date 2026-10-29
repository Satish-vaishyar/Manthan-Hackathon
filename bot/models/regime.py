"""bot.models.regime — backward-compat shim re-exporting model5.src.regime.

The canonical M5 implementation lives in `model5/src/regime.py` (matching
the `modelN/src/` layout used by model1/2/3/6). This file re-exports every
public symbol so existing imports (`from bot.models.regime import
RegimeDetector`) keep working without any code changes outside this
package.

If you want to edit M5: do it in `model5/src/regime.py`, then re-run the
tests to confirm the re-export still resolves cleanly.
"""
from __future__ import annotations

import os
import sys

# Make `model5.src` importable. model5/ sits at the repo root (a sibling
# of bot/), so its parent must be on sys.path.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_MODEL5_SRC = os.path.join(_REPO_ROOT, "model5", "src")
if _MODEL5_SRC not in sys.path:
    sys.path.insert(0, _MODEL5_SRC)

# Re-export everything from the canonical home.
from regime import (  # noqa: E402, F401
    ALL_REGIMES,
    DEFAULT_GROSS_TARGET,
    NEUTRAL,
    RISK_OFF,
    RISK_ON,
    RegimeConfig,
    RegimeDetector,
    RegimeState,
)

__all__ = [
    "ALL_REGIMES",
    "DEFAULT_GROSS_TARGET",
    "NEUTRAL",
    "RISK_OFF",
    "RISK_ON",
    "RegimeConfig",
    "RegimeDetector",
    "RegimeState",
]