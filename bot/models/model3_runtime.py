"""bot.models.model3_runtime — runtime scorer for the trained M3 DMN ensemble.

M3 (Deep Momentum Network, LSTM) was trained in `model3/src/`. At export
time, `model3/src/export.py` wrote:
  - artifacts/model3_seed4{2-6}.pt   (5 PyTorch checkpoints)
  - artifacts/model3_meta.json       (scaler-free; LSTM is parameter-only)

Unlike M1 (which has ONNX exports), M3 has only `.pt`. We load the
PyTorch checkpoints directly (CPU) at startup, average across the 5 seeds,
and produce a per-ticker `position ∈ [-1, +1]` signal plus
`p_trend_up ∈ [0, 1]`.

Why load .pt instead of re-exporting to ONNX?
  - torch is already installed in the bot environment (verified).
  - Exporting to ONNX would require running the export script in the
    training environment first (which we don't have GPU access to from
    here). The .pt loading path runs entirely in CPU and is well under
    the 50 ms/day latency budget (meta reports 1-9 ms for 40 tickers).

API contract (matches Model1Ranker.score):
    windows: dict[ticker -> np.ndarray[L=63, F=16]]
    Returns: dict[ticker -> float]  (the `position` score, mapped to
    ensemble sub-score range via tanh or linear)
"""
from __future__ import annotations

import json
import math
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from ..contracts import DayBundle


class Model3NotReady(RuntimeError):
    """Raised when the M3 .pt artifacts aren't exported yet."""


@dataclass
class Model3Meta:
    model: str = "model3-DMN"
    seq_len: int = 63
    n_features: int = 16
    vol_span: int = 60
    sigma_target: float = 0.15
    max_leverage: float = 2.0
    seed_paths: list[str] = None  # type: ignore
    feature_names: list[str] = None  # type: ignore

    def __post_init__(self):
        if self.seed_paths is None:
            self.seed_paths = []
        if self.feature_names is None:
            self.feature_names = []


class Model3Scorer:
    """CPU inference for the trained M3 LSTM ensemble (5 seeds)."""

    def __init__(self,
                 artifacts_dir: str | Path = "model3/artifacts",
                 use_ensemble: bool = True) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.use_ensemble = use_ensemble
        self.meta: Optional[Model3Meta] = None
        self._models: list = []       # list[torch.nn.Module]
        self._loaded = False

    def _try_load(self) -> bool:
        """Idempotent loader: read model3_meta.json and load each seed's .pt
        checkpoint (CPU, eval mode). Any failure -> False (rules fallback)."""
        if self._loaded:
            return True
        try:
            meta_path = self.artifacts_dir / "model3_meta.json"
            if not meta_path.exists():
                return False
            meta = json.loads(meta_path.read_text())
            self.meta = Model3Meta(
                model=meta.get("model", "model3-DMN"),
                seq_len=int(meta.get("seq_len", 63)),
                n_features=len(meta.get("features", [])) or 16,
                vol_span=int(meta.get("vol_span", 60)),
                sigma_target=float(meta.get("sigma_target", 0.15)),
                max_leverage=float(meta.get("max_leverage", 2.0)),
                seed_paths=[m.get("pt", "") for m in meta.get("members", []) if m.get("pt")],
                feature_names=list(meta.get("features", [])),
            )
            # Lazy import the DMN class (matches model3/src/model.py).
            from model3.src.model import DMN  # type: ignore
            for rel in self.meta.seed_paths:
                path = self.artifacts_dir / rel
                if not path.exists():
                    continue
                ck = torch.load(str(path), map_location="cpu", weights_only=False)
                cfg = ck.get("model_cfg", {})
                m = DMN(**cfg).eval()
                m.load_state_dict(ck["state_dict"])
                self._models.append(m)
                if not self.use_ensemble:
                    break
            if not self._models:
                return False
            self._loaded = True
            return True
        except Exception:
            return False

    @property
    def ready(self) -> bool:
        return self._try_load()

    @property
    def n_members(self) -> int:
        return len(self._models)

    def score(self,
              windows: dict[str, np.ndarray],
              ) -> dict[str, float]:
        """Score every ticker.

        Args:
            windows: ticker -> np.ndarray[seq_len=63, n_features=16] (raw,
                unscaled — M3 LSTM doesn't need scaling).

        Returns:
            dict[ticker -> float] in [-1, +1]: averaged `position` head
            output across the ensemble (already tanh-bounded by the model).
        """
        if not self.ready:
            raise Model3NotReady(
                f"model3 artifacts not found in {self.artifacts_dir} — "
                "run model3/src/export.py first"
            )

        out: dict[str, float] = {}
        for t, win in windows.items():
            if win is None or win.shape != (self.meta.seq_len, self.meta.n_features):
                continue
            x = torch.from_numpy(win.astype(np.float32)).unsqueeze(0)  # [1, L, F]
            pos_acc = None
            with torch.no_grad():
                for m in self._models:
                    pos, _trend = m(x)
                    v = float(pos[0, -1].item())
                    pos_acc = v if pos_acc is None else pos_acc + v
            avg = pos_acc / len(self._models)
            out[t] = float(max(-1.0, min(1.0, avg)))
        return out


# -----------------------------------------------------------------------------
# Fallback: rules-based momentum when M3 isn't exported
# -----------------------------------------------------------------------------
class RulesMomentum:
    """P0 fallback momentum scorer — same shape as Model3Scorer.

    Returns a per-ticker score in [-1, +1] from the rolling momentum z-score.
    """
    def __init__(self) -> None:
        self._history: dict[str, deque[float]] = {}

    def push(self, ticker: str, close: float) -> None:
        buf = self._history.get(ticker)
        if buf is None:
            buf = deque(maxlen=64)
            self._history[ticker] = buf
        buf.append(close)

    def score(self, windows) -> dict[str, float]:
        out: dict[str, float] = {}
        for t, closes in self._history.items():
            if len(closes) < 20:
                continue
            arr = np.asarray(closes, dtype=np.float64)
            log_c = np.log(arr)
            r20 = log_c[-1] - log_c[-20]
            vol = float(np.std(np.diff(log_c[-20:]))) + 1e-9
            z = float(r20 / (vol * math.sqrt(20.0)))
            out[t] = float(max(-1.0, min(1.0, z / 2.0)))
        return out