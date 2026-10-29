"""bot.models.model1_onnx — wrapper around the exported model1 ONNX (M1).

Per docs/05 §4 and model1/src/export.py:
    - Input: x[B, N, L, F], tmask[B, N, L], nmask[B, N]  (F=28, L=64)
    - Output: ret1[B, N], ret5[B, N], p_up[B, N], vol[B, N], p_mkt[B, B]
    - Also needs scaler.json (median/IQR per feature, fitted on train only)

The bot calls this once per simulated day with:
    - the latest 64 days of every ticker's 28-feature window
    - ticker validity mask (which slots have data)
    - outputs ret1 (next-day cross-sectional score)

If the artifact doesn't exist yet, this falls back to the rules strategy
via a NotReadyError so the bot still runs on P0.

Usage:
    ranker = Model1Ranker("model1/artifacts")    # loads model1_seed0.onnx + scaler
    score = ranker(bundle, ring_buffer)          # dict[ticker -> float]
"""
from __future__ import annotations

import json
import math
import os
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import date as _date
from pathlib import Path
from typing import Optional

import numpy as np

from ..contracts import DayBundle


class Model1NotReady(RuntimeError):
    """Raised when the ONNX or scaler.json hasn't been exported yet."""


@dataclass
class Model1Meta:
    window: int = 64
    n_features: int = 28
    seed_paths: list[str] = field(default_factory=list)
    feature_names: list[str] = field(default_factory=list)
    scaler_median: list[float] = field(default_factory=list)
    scaler_iqr: list[float] = field(default_factory=list)


class Model1Ranker:
    """Wraps model1 ONNX runtime inference.

    Falls back to NaN scores (which the ensemble treats as 0.0) if the
    artifact isn't present. Call `.ready` to check.
    """

    def __init__(self, artifacts_dir: str | Path = "model1/artifacts",
                 ensemble_members: bool = True) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.ensemble_members = ensemble_members
        self.meta: Optional[Model1Meta] = None
        self._sessions: list = []
        self._loaded = False

    def _try_load(self) -> bool:
        if self._loaded:
            return True
        try:
            import onnxruntime as ort
        except ImportError:
            return False
        meta_path = self.artifacts_dir / "model1_meta.json"
        if not meta_path.exists():
            return False
        meta = json.loads(meta_path.read_text())
        self.meta = Model1Meta(
            window=int(meta.get("window", 64)),
            n_features=int(meta.get("n_features", 28)),
            seed_paths=[m.get("onnx", m.get("pt", "")) for m in meta.get("members", [])],
            feature_names=list(meta.get("feature_names", [])),
            scaler_median=list(meta.get("scaler_median", [])),
            scaler_iqr=list(meta.get("scaler_iqr", [])),
        )
        for rel in self.meta.seed_paths:
            path = self.artifacts_dir / rel
            if not path.exists() or not path.suffix == ".onnx":
                continue
            self._sessions.append(ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]))
        if not self._sessions:
            return False
        self._loaded = True
        return True

    @property
    def ready(self) -> bool:
        return self._try_load()

    def _robust_scale(self, x: np.ndarray) -> np.ndarray:
        """Apply the train-fit median/IQR scaler (model1/src/build_dataset.py)."""
        if not self.meta or not self.meta.scaler_median:
            return x
        med = np.asarray(self.meta.scaler_median, dtype=np.float32)
        iqr = np.asarray(self.meta.scaler_iqr, dtype=np.float32)
        # iqr may be 0; replace with 1 to avoid div-by-zero, mirroring model1
        iqr = np.where(iqr < 1e-9, 1.0, iqr)
        return (x - med) / iqr

    def score(self, bundle: DayBundle,
              feature_windows: dict[str, np.ndarray]) -> dict[str, float]:
        """Run model1 for the current day.

        Args:
            bundle: the day's DayBundle (used for ticker ordering + valid mask).
            feature_windows: ticker -> np.ndarray of shape [L, F] (raw features,
                unscaled). The caller (bot.features.ring_buffer) maintains this.

        Returns:
            dict[ticker -> float] of next-day cross-sectional scores in
            approximately [-1, +1] (tanh of model output).
        """
        if not self.ready:
            raise Model1NotReady(
                f"model1 artifacts not found in {self.artifacts_dir} — "
                "run model1/src/export.py first"
            )

        tickers = list(bundle.ticks.keys())
        N = len(tickers)
        L = self.meta.window
        F = self.meta.n_features

        # Build inputs [B=1, N, L, F]
        x = np.zeros((1, N, L, F), dtype=np.float32)
        tmask = np.zeros((1, N, L), dtype=bool)
        nmask = np.zeros((1, N), dtype=bool)

        for i, t in enumerate(tickers):
            win = feature_windows.get(t)
            if win is None or win.shape != (L, F):
                continue
            x[0, i] = self._robust_scale(win)
            tmask[0, i] = True
            nmask[0, i] = True

        if not nmask.any():
            return {t: 0.0 for t in tickers}

        # Average ensemble outputs across members
        ret1_acc: Optional[np.ndarray] = None
        for sess in self._sessions:
            outs = sess.run(None, {"x": x, "tmask": tmask, "nmask": nmask})
            ret1 = outs[0]  # [1, N]
            ret1_acc = ret1 if ret1_acc is None else ret1_acc + ret1
        ret1_mean = ret1_acc / len(self._sessions)

        out: dict[str, float] = {}
        for i, t in enumerate(tickers):
            v = float(ret1_mean[0, i])
            # Squash to [-1, +1] (model outputs are unbounded ret z-scores)
            out[t] = math.tanh(v)
        return out


# -----------------------------------------------------------------------------
# Fallback: a pure-rules ranker so the bot works WITHOUT the model artifact
# -----------------------------------------------------------------------------
class RulesRanker:
    """P0 fallback when model1 isn't exported yet.

    Wraps the existing rules.py logic so the ensemble can swap the rank_fn
    without changing its call signature.
    """

    def __init__(self) -> None:
        from .rules import RulesStrategy
        self._rules = RulesStrategy()

    def __call__(self, bundle: DayBundle) -> dict[str, float]:
        evals = self._rules.evaluate_day(bundle)
        # Mirror what ensemble does with _default_rank
        return {
            t: 0.7 * e.sub_scores.get("trend", 0.0) + 0.3 * e.sub_scores.get("rsi", 0.0)
            for t, e in evals.items()
        }