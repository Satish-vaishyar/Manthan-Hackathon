"""bot.models.vol — backward-compat shim re-exporting model4.src.vol.

The canonical M4 implementation lives in `model4/src/vol.py` (matching the
`modelN/src/` layout used by model1/2/3/6). This file re-exports every
public symbol so existing imports (`from bot.models.vol import VolEstimator`)
keep working without any code changes outside this package.

If you want to edit M4: do it in `model4/src/vol.py`, then re-run the
tests to confirm the re-export still resolves cleanly.
"""
from __future__ import annotations

import os
import sys

# Make `model4.src` importable. model4/ sits at the repo root (a sibling
# of bot/), so its parent must be on sys.path.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_MODEL4_SRC = os.path.join(_REPO_ROOT, "model4", "src")
if _MODEL4_SRC not in sys.path:
    sys.path.insert(0, _MODEL4_SRC)

# Re-export everything from the canonical home.
from vol import (  # noqa: E402, F401
    EPS,
    VolEstimator,
    _Window,
    annualised_vol_from_closes,
)

__all__ = [
    "EPS",
    "VolEstimator",
    "_Window",
    "annualised_vol_from_closes",
]