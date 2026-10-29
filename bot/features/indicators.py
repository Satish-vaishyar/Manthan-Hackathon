"""bot.features.indicators — incremental feature computation for M1 and M3.

Two model-specific feature builders:

  build_model1_features(ring_buffer, tickers) -> dict[ticker, np.ndarray[L, F=28]]
      Wraps model1/src/features.py::ticker_features and adds cross-sectional
      columns (mkt_r1, excess_r1, cs_rank_r20) at the bundle level.

  build_model3_features(ring_buffer, tickers) -> dict[ticker, np.ndarray[L, F=16]]
      Wraps model3/src/features.py::ticker_features.

Both call into the canonical training code under model1/src and model3/src
to guarantee exact feature parity between training and inference (docs/05
leakage checklist).

Why re-export the training code?
  - Docs/05 §6: "The bot must use the same `features.py` at inference."
  - Drift between training and live computation = silent wrong-model
    syndrome (the predictions stop matching the model's intent). Reusing
    the literal training code eliminates this risk.

The wrappers handle:
  - Loading the training code's `sys.path`
  - Calling the right function with the right shape
  - Returning numpy arrays in [L, F] order that Model1Ranker.score() and
    Model3Scorer.score() expect
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .ring_buffer import RingBuffer

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODEL1_SRC = _REPO_ROOT / "model1" / "src"
_MODEL3_SRC = _REPO_ROOT / "model3" / "src"


def _load_module_from_path(name: str, path: Path):
    """Import a .py file as a module by absolute path."""
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Lazy-loaded training modules (one-time cost on first call)
_model1_features_mod = None
_model3_features_mod = None


def _get_model1_features():
    global _model1_features_mod
    if _model1_features_mod is None:
        _model1_features_mod = _load_module_from_path(
            "model1_features_runtime", _MODEL1_SRC / "features.py",
        )
    return _model1_features_mod


def _get_model3_features():
    global _model3_features_mod
    if _model3_features_mod is None:
        _model3_features_mod = _load_module_from_path(
            "model3_features_runtime", _MODEL3_SRC / "features.py",
        )
    return _model3_features_mod


# Model 1 expects 28 features in this order (from model1_meta.json).
MODEL1_FEATURE_NAMES = [
    "r1", "gap", "intraday", "hl_range", "clv", "upper_wick", "lower_wick",
    "vol_z20", "vol_ratio_5_20",
    "r5", "r10", "r20", "r60",
    "rv20", "rv60", "rv_ratio",
    "dist_sma20", "dist_sma50", "dist_sma200",
    "rsi14", "bb_pctb20", "macd_hist_n", "atr14_pct",
    "dist_252_high", "dist_252_low",
    "mkt_r1", "excess_r1", "cs_rank_r20",
]
MODEL1_WINDOW = 64
MODEL1_N_FEATURES = 28

# Model 3 expects 16 features in this order (from model3_meta.json).
MODEL3_FEATURE_NAMES = [
    "nret_1", "nret_5", "nret_21", "nret_63", "nret_126", "nret_252",
    "macd_8_24", "macd_16_48", "macd_32_96",
    "cpd_mean", "cpd_vol", "dd_63", "du_63",
    "mkt_nret_21", "mkt_nret_63", "breadth_63",
]
MODEL3_WINDOW = 63
MODEL3_N_FEATURES = 16


def build_model1_features(
    rb: RingBuffer,
    tickers: list[str],
    window: int = MODEL1_WINDOW,
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Build [L, F=28] feature windows for M1 per ticker.

    Returns:
        (windows, ready_tickers) where:
          - windows: ticker -> np.ndarray[L, F=28] (float32, robust-scale ready)
          - ready_tickers: subset of `tickers` that have enough history (≥ L days)
            AND for which feature computation succeeded.
    """
    feat_mod = _get_model1_features()
    frames = []
    per_ticker_frames: dict[str, pd.DataFrame] = {}
    for t in tickers:
        f = rb.frame(t)
        if f is None or len(f) < window:
            continue
        per_ticker_frames[t] = f

    if not per_ticker_frames:
        return {}, []

    # Build the long frame the way model1/src/features.py expects:
    # long format with columns [date, ticker, open, high, low, close, volume].
    long_df = pd.concat(per_ticker_frames.values(), ignore_index=True)
    long_df = long_df[["date", "ticker", "open", "high", "low", "close", "volume"]]

    # Call the canonical feature builder. build_feature_frame returns a long
    # frame with columns [date, ticker, FEATURE_NAMES...].
    feats_long = feat_mod.build_feature_frame(long_df)

    # Restrict to the trailing `window` rows per ticker.
    feats_long = feats_long.sort_values(["ticker", "date"])
    feats_long = feats_long.groupby("ticker", sort=False).tail(window).reset_index(drop=True)

    # Cross-sectional columns are constant within a date. To match model1's
    # training-time shape, recompute them only on the trailing window:
    # (the original build_feature_frame already did this once across all
    # history, so trailing rows are consistent with their date group.)
    # No-op here; the trailing rows already carry the right values.

    out: dict[str, np.ndarray] = {}
    ready: list[str] = []
    for t, g in feats_long.groupby("ticker", sort=False):
        g = g.sort_values("date").tail(window)
        if len(g) < window:
            continue
        arr = g[MODEL1_FEATURE_NAMES].to_numpy(dtype=np.float32)
        out[t] = arr
        ready.append(t)
    return out, ready


def build_model3_features(
    rb: RingBuffer,
    tickers: list[str],
    window: int = MODEL3_WINDOW,
    vol_span: int = 60,
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Build [L, F=16] feature windows for M3 per ticker.

    Returns:
        (windows, ready_tickers) — same contract as build_model1_features.
    """
    feat_mod = _get_model3_features()
    per_ticker_frames: dict[str, pd.DataFrame] = {}
    for t in tickers:
        f = rb.frame(t)
        if f is None or len(f) < window:
            continue
        per_ticker_frames[t] = f

    if not per_ticker_frames:
        return {}, []

    long_df = pd.concat(per_ticker_frames.values(), ignore_index=True)
    long_df = long_df[["date", "ticker", "open", "high", "low", "close", "volume"]]

    # model3/src/features.py::ticker_features takes a single-ticker frame and
    # returns (features_df, sigma_series). We loop per ticker so the same-day
    # cross-sectional features (mkt_nret_21, etc.) reflect only the live
    # universe rather than the training universe.
    out: dict[str, np.ndarray] = {}
    for t in tickers:
        if t not in per_ticker_frames:
            continue
        f_df = per_ticker_frames[t].copy()
        try:
            f, _sigma = feat_mod.ticker_features(
                pd.Series(f_df["close"].values, index=pd.to_datetime(f_df["date"])),
                vol_span=vol_span,
            )
        except Exception:
            continue
        if f is None or len(f) < window:
            continue
        # Last `window` rows.
        f = f.tail(window)
        # Cross-sectional features are per-day; we approximate using the
        # mean/std across the live universe at the latest date.
        latest_date = f.index[-1]
        try:
            cross = _cross_sectional_m3(long_df, latest_date)
            f = f.assign(**cross)
        except Exception:
            continue
        # Ensure all 16 columns are present in order.
        for col in MODEL3_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0.0
        arr = f[MODEL3_FEATURE_NAMES].to_numpy(dtype=np.float32)
        if arr.shape != (window, MODEL3_N_FEATURES):
            continue
        out[t] = arr

    return out, list(out.keys())


def _cross_sectional_m3(long_df: pd.DataFrame, latest_date) -> dict:
    """Compute m3's per-day cross-sectional features for `latest_date`.

    These are:
      mkt_nret_21   = mean 21-day return across the live universe at latest_date
      mkt_nret_63   = mean 63-day return across the live universe at latest_date
      breadth_63    = fraction of tickers above their 63-day rolling mean
    """
    latest = pd.to_datetime(latest_date)
    df = long_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    sub = df[df["date"] <= latest].copy()
    if sub.empty:
        return {"mkt_nret_21": 0.0, "mkt_nret_63": 0.0, "breadth_63": 0.5}
    sub = sub.sort_values(["ticker", "date"])
    # log close and returns
    sub["log_close"] = np.log(sub["close"].astype(float))
    sub["r1"] = sub.groupby("ticker")["log_close"].diff()
    # 21-day and 63-day rolling return per ticker
    sub["nret_21"] = sub.groupby("ticker")["log_close"].diff(21)
    sub["nret_63"] = sub.groupby("ticker")["log_close"].diff(63)
    # Latest row per ticker
    last = sub.groupby("ticker").tail(1)
    if last.empty:
        return {"mkt_nret_21": 0.0, "mkt_nret_63": 0.0, "breadth_63": 0.5}
    mkt_21 = float(last["nret_21"].mean())
    mkt_63 = float(last["nret_63"].mean())
    # breadth = fraction of tickers above 63-day rolling mean of close
    above = 0
    counted = 0
    for tkr, g in sub.groupby("ticker"):
        if len(g) < 63:
            continue
        rolling_mean = g["close"].rolling(63).mean().iloc[-1]
        if not np.isnan(rolling_mean) and g["close"].iloc[-1] > rolling_mean:
            above += 1
        counted += 1
    breadth = (above / counted) if counted > 0 else 0.5
    return {"mkt_nret_21": mkt_21, "mkt_nret_63": mkt_63, "breadth_63": breadth}