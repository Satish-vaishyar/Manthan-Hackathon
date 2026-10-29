"""Vectorised daily backtest shared by training (validation), baselines and reporting.

Position signal pos[d, n] in [-1, 1] is decided at close(d) and earns r_next[d, n] (close d -> close d+1).
Volatility targeting: weight w = pos * min(sigma_target_daily / sigma, max_leverage).
Cost: cost_bps * |w_d - w_{d-1}| per ticker. Portfolio return = mean over tickers with a signal that day.
"""
from __future__ import annotations

import numpy as np


def weights(pos: np.ndarray, sigma: np.ndarray, sigma_target: float, max_lev: float) -> np.ndarray:
    lev = np.minimum(sigma_target / np.sqrt(252) / np.maximum(sigma, 1e-6), max_lev)
    return np.nan_to_num(pos) * lev


def run(pos: np.ndarray, r_next: np.ndarray, sigma: np.ndarray, mask: np.ndarray, *, sigma_target: float,
        max_leverage: float, cost_bps: float, ann: int = 252) -> dict:
    """pos/r_next/sigma/mask: [D, N] for the evaluation days (in date order)."""
    pos = np.where(mask, pos, 0.0)
    w = weights(pos, sigma, sigma_target, max_leverage)
    w_prev = np.vstack([np.zeros((1, w.shape[1])), w[:-1]])
    gross = w * np.where(mask, r_next, 0.0)
    cost = cost_bps * 1e-4 * np.abs(w - w_prev)
    n = mask.sum(1)
    keep = n > 0
    port = ((gross - cost).sum(1)[keep] / n[keep])
    port_gross = (gross.sum(1)[keep] / n[keep])
    if len(port) < 2 or port.std() == 0:
        return {"days": int(len(port)), "sharpe": 0.0}
    eq = np.cumsum(port)
    dd = eq - np.maximum.accumulate(eq)
    return {
        "days": int(len(port)),
        "sharpe": float(port.mean() / port.std() * np.sqrt(ann)),
        "sharpe_gross": float(port_gross.mean() / (port_gross.std() + 1e-12) * np.sqrt(ann)),
        "ann_return": float(port.mean() * ann),
        "ann_vol": float(port.std() * np.sqrt(ann)),
        "max_drawdown": float(dd.min()),
        "avg_turnover": float((np.abs(w - w_prev).sum(1)[keep] / n[keep]).mean()),
        "avg_abs_weight": float((np.abs(w).sum(1)[keep] / n[keep]).mean()),
        "hit_rate": float((port > 0).mean()),
        "cost_share_of_gross": float(cost.sum() / (np.abs(gross.sum()) + 1e-12)),
    }
