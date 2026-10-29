"""bot.models.model1_onnx — runtime scorer for the trained M1 CSTT ensemble.

Per docs/05 §4 and model1/src/export.py:
    - Input: x[B, N, L, F], tmask[B, N, L], nmask[B, N]  (F=28, L=64)
    - Output: ret1[B, N], ret5[B, N], p_up[B, N], vol[B, N], p_mkt[B, B]
    - Also needs scaler.json (median/IQR per feature, fitted on train only)

The bot calls this once per simulated day with:
    - the latest 64 days of every ticker's 28-feature window
    - ticker validity mask (which slots have data)
    - outputs ret1 (next-day cross-sectional score)

Loader strategy (in priority order):
    1. PyTorch `.pt` checkpoints (always present) — preferred path
    2. ONNX runtime fallback if `.onnx` files exist (legacy)
    3. NaN scores → ensemble treats as missing → falls back to rules

Usage:
    ranker = Model1Ranker("model1/artifacts")    # loads model1_seed4{2-6}.pt + scaler
    score = ranker.score(feature_windows)        # dict[ticker -> float]
"""
from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from ..contracts import DayBundle


class Model1NotReady(RuntimeError):
    """Raised when neither .pt nor .onnx artifacts are present."""


@dataclass
class Model1Meta:
    window: int = 64
    n_features: int = 28
    seed_paths: list[str] = field(default_factory=list)
    feature_names: list[str] = field(default_factory=list)
    scaler_median: list[float] = field(default_factory=list)
    scaler_iqr: list[float] = field(default_factory=list)
    model_cfg: dict = field(default_factory=dict)


def _load_cstt_module(model_cfg: dict, window: int = 64):
    """Import the training-time CSTT class for PyTorch inference."""
    # model1/src/model.py defines CSTT
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "model1" / "src"))
    try:
        from model import CSTT  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"could not import CSTT: {e}") from e
    return CSTT(window=window, **model_cfg)


class _TorchMember:
    """Wrap a CSTT .pt checkpoint for CPU inference."""

    def __init__(self, ck_path: Path, model_cfg: dict, scaler_median, scaler_iqr,
                 window: int = 64):
        import torch
        self.torch = torch
        # If model_cfg didn't carry window, add it from the caller's meta.
        cfg = dict(model_cfg)
        self.model = _load_cstt_module(cfg, window=window)
        ck = torch.load(str(ck_path), map_location="cpu", weights_only=False)
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()
        self.med = np.asarray(scaler_median, dtype=np.float32) if scaler_median else None
        self.iqr = np.asarray(scaler_iqr, dtype=np.float32) if scaler_iqr else None
        if self.iqr is not None:
            self.iqr = np.where(self.iqr < 1e-9, 1.0, self.iqr)

    def run(self, x: np.ndarray, tmask: np.ndarray, nmask: np.ndarray) -> np.ndarray:
        with self.torch.no_grad():
            xt = self.torch.from_numpy(x.astype(np.float32))
            tmt = self.torch.from_numpy(tmask.astype(bool))
            nmt = self.torch.from_numpy(nmask.astype(bool))
            out = self.model(xt, tmt, nmt)
            return out["ret1"].cpu().numpy()  # [1, N]


class _OnnxMember:
    """Wrap a CSTT .onnx checkpoint via onnxruntime (fallback path)."""

    def __init__(self, ck_path: Path):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(str(ck_path), providers=["CPUExecutionProvider"])

    def run(self, x: np.ndarray, tmask: np.ndarray, nmask: np.ndarray) -> np.ndarray:
        outs = self.sess.run(None, {"x": x, "tmask": tmask, "nmask": nmask})
        return outs[0]


class Model1Ranker:
    """Wraps model1 inference (PyTorch preferred; ONNX fallback).

    Falls back to NaN scores (which the ensemble treats as 0.0) if no
    artifact is present. Call `.ready` to check.
    """

    def __init__(self, artifacts_dir: str | Path = "model1/artifacts",
                 ensemble_members: bool = True) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.ensemble_members = ensemble_members
        self.meta: Optional[Model1Meta] = None
        self._members: list = []  # list[_TorchMember | _OnnxMember]
        self._loaded = False

    def _try_load(self) -> bool:
        """Idempotent loader: read model1_meta.json, then load members —
        PyTorch .pt checkpoints first, ONNX as fallback. Any failure returns
        False (caller falls back to RulesRanker)."""
        if self._loaded:
            return True
        try:
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
                model_cfg=dict(meta.get("model_cfg", {})),
            )
            if not self.meta.scaler_median and "members" in meta and meta["members"]:
                # Try scaler from the first member's seed_*/best.pt if not in meta
                pass

            med = self.meta.scaler_median
            iqr = self.meta.scaler_iqr

            # Try torch .pt first (always present for M1)
            torch_loaded = False
            try:
                import torch  # noqa: F401
                window = self.meta.window
                for m in meta.get("members", []):
                    pt_rel = m.get("pt", "")
                    if not pt_rel:
                        continue
                    pt_path = self.artifacts_dir / pt_rel
                    if not pt_path.exists():
                        continue
                    # model1_meta.json doesn't carry model_cfg per member; the
                    # canonical source is the .pt file's own "model_cfg" key.
                    cfg = m.get("model_cfg") or self.meta.model_cfg
                    if not cfg:
                        ck = torch.load(str(pt_path), map_location="cpu", weights_only=False)
                        cfg = ck.get("model_cfg", {})
                        if not window:
                            window = ck.get("window", 64)
                    self._members.append(
                        _TorchMember(pt_path, cfg, med, iqr, window=window)
                    )
                    if not self.ensemble_members:
                        break
                torch_loaded = bool(self._members)
            except Exception as e:  # noqa: BLE001
                # torch not available or some other issue — fall through to ONNX
                torch_loaded = False

            if not torch_loaded:
                # Try ONNX fallback
                try:
                    for m in meta.get("members", []):
                        onx_rel = m.get("onnx", "")
                        if not onx_rel or not onx_rel.endswith(".onnx"):
                            continue
                        onx_path = self.artifacts_dir / onx_rel
                        if not onx_path.exists():
                            continue
                        self._members.append(_OnnxMember(onx_path))
                        if not self.ensemble_members:
                            break
                except Exception:
                    pass

            if not self._members:
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
        return len(self._members)

    def _robust_scale(self, x: np.ndarray) -> np.ndarray:
        """Apply the train-fit median/IQR scaler (model1/src/build_dataset.py)."""
        if not self.meta or not self.meta.scaler_median:
            return x
        med = np.asarray(self.meta.scaler_median, dtype=np.float32)
        iqr = np.asarray(self.meta.scaler_iqr, dtype=np.float32)
        # iqr may be 0; replace with 1 to avoid div-by-zero
        iqr = np.where(iqr < 1e-9, 1.0, iqr)
        return (x - med) / iqr

    def score(self,
              feature_windows: dict[str, np.ndarray],
              tickers: Optional[list[str]] = None) -> dict[str, float]:
        """Run model1 for the current day.

        Args:
            feature_windows: ticker -> np.ndarray of shape [L, F] (raw features,
                unscaled). The caller (bot.features.indicators.build_model1_features)
                maintains this.
            tickers: optional ordered list of tickers to score. If omitted,
                uses `feature_windows.keys()`.

        Returns:
            dict[ticker -> float] of next-day cross-sectional scores in
            approximately [-1, +1] (tanh of model output).
        """
        if not self.ready:
            raise Model1NotReady(
                f"model1 artifacts not found in {self.artifacts_dir} — "
                "run model1/src/export.py first"
            )
        if tickers is None:
            tickers = list(feature_windows.keys())
        if not tickers:
            return {}

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
        for member in self._members:
            ret1 = member.run(x, tmask, nmask)
            ret1_acc = ret1 if ret1_acc is None else ret1_acc + ret1
        ret1_mean = ret1_acc / len(self._members)

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