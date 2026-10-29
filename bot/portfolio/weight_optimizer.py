"""bot.portfolio.weight_optimizer — statistically-estimated ensemble weights.

Replaces hand-picked {rank, news, mom} weights with weights estimated from
the bot's own decision journals using four complementary estimators, then
combines them with a Monte Carlo bootstrap + shrinkage:

Estimators (each maps per-day cross-sectional ICs -> simplex weights)
---------------------------------------------------------------------
1. IR-weighted (Grinold & Kahn "fundamental law"):
       w_i ∝ max(IR_i, 0),  IR_i = mean(IC_i) / std(IC_i)
   Favours signals with consistent predictive power.

2. Inverse-variance (minimum-variance):
       w_i ∝ 1 / Var(IC_i)
   Favours stable signals regardless of sign consistency.

3. Mean-variance optimal (Markowitz on signal ICs):
       w* = argmax  w'μ - γ·w'Σw   s.t. w≥0, Σw=1
   Closed form for the unconstrained case: w = Σ⁻¹μ / (1'Σ⁻¹μ);
   we then project onto the simplex. This accounts for **correlation
   between signal ICs** — two signals that are right on the same days
   are worth less together than the IR weights suggest.

4. Monte Carlo bootstrap:
   Resample trading days with replacement (B=1000), recompute estimator 3
   per resample → distribution of optimal weights. Final weight = mean of
   bootstrap distribution; standard error → 95% CI. Signals whose CI
   includes 0 get shrunk toward the equal-weight prior:
       w_final = (1-λ_i)·w_MC_i + λ_i·(1/K)
       λ_i = P_boot(w_i ≤ 0)          (empirical, from the resamples)
   This is a data-driven James–Stein-style shrinkage: the less certain we
   are that a signal adds value, the more we fall back to diversification.

Information Coefficient
-----------------------
Per trading day d: IC_i(d) = Spearman( signal_i(·,d), fwd_ret(·,d) )
across the cross-section of tickers with both a score and a forward
return. Spearman = Pearson on ranks (robust to outliers, monotone-invariant).

All functions are pure numpy — no scipy, no network. The script
scripts/optimize_weights.py builds the input matrices from
runs/*/journal.jsonl (scores) and the HOLD/trade prices recorded in the
same journals (price matrix -> forward returns).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ----------------------------------------------------------------------
# Ranking / IC helpers
# ----------------------------------------------------------------------
def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks (1..n) of a 1-D array, ties get the mean rank."""
    x = np.asarray(x, dtype=float)
    n = x.size
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and x[order[j + 1]] == x[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman_ic(signal: np.ndarray, fwd_ret: np.ndarray) -> float:
    """Spearman rank correlation between signal and forward returns.

    NaN-safe: pairs with NaN on either side are dropped. Returns NaN if
    fewer than 3 valid pairs or either side is constant.
    """
    s = np.asarray(signal, dtype=float)
    r = np.asarray(fwd_ret, dtype=float)
    m = np.isfinite(s) & np.isfinite(r)
    if m.sum() < 3:
        return float("nan")
    s, r = s[m], r[m]
    if np.ptp(s) == 0 or np.ptp(r) == 0:
        return float("nan")
    rs, rr = _rankdata(s), _rankdata(r)
    rs = rs - rs.mean()
    rr = rr - rr.mean()
    denom = math.sqrt(float(rs @ rs) * float(rr @ rr))
    if denom == 0:
        return float("nan")
    return float(rs @ rr / denom)


def daily_ics(scores: np.ndarray, fwd: np.ndarray) -> np.ndarray:
    """Per-day ICs.

    scores: (D, T) signal values; fwd: (D, T) forward returns.
    Returns (D,) array of Spearman ICs (NaN where not computable).
    """
    D = scores.shape[0]
    out = np.full(D, np.nan)
    for d in range(D):
        out[d] = spearman_ic(scores[d], fwd[d])
    return out


# ----------------------------------------------------------------------
# Simplex projection (Duchi et al. 2008)
# ----------------------------------------------------------------------
def project_simplex(v: np.ndarray) -> np.ndarray:
    """Project v onto {w >= 0, sum(w) = 1}."""
    v = np.asarray(v, dtype=float)
    n = v.size
    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (css - 1))[0]
    if rho.size == 0:
        return np.full(n, 1.0 / n)
    rho = rho[-1]
    theta = (css[rho] - 1.0) / (rho + 1)
    return np.maximum(v - theta, 0.0)


# ----------------------------------------------------------------------
# Estimators
# ----------------------------------------------------------------------
def ir_weights(ics: np.ndarray) -> np.ndarray:
    """w_i ∝ max(mean/std, 0). ics: (D, K) daily ICs per signal."""
    K = ics.shape[1]
    w = np.zeros(K)
    for k in range(K):
        x = ics[:, k]
        x = x[np.isfinite(x)]
        if x.size < 3:
            continue
        sd = x.std(ddof=1)
        if sd <= 0:
            continue
        w[k] = max(x.mean() / sd, 0.0)
    if w.sum() <= 0:
        return np.full(K, 1.0 / K)
    return w / w.sum()


def inverse_variance_weights(ics: np.ndarray) -> np.ndarray:
    """w_i ∝ 1/Var(IC_i)."""
    K = ics.shape[1]
    w = np.zeros(K)
    for k in range(K):
        x = ics[:, k]
        x = x[np.isfinite(x)]
        if x.size < 3:
            continue
        var = x.var(ddof=1)
        if var <= 0:
            continue
        w[k] = 1.0 / var
    if w.sum() <= 0:
        return np.full(K, 1.0 / K)
    return w / w.sum()


def mean_variance_weights(ics: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Markowitz on ICs: w ∝ Σ⁻¹μ, projected onto the simplex.

    Uses Ledoit-Wolf-style shrinkage of the covariance toward its diagonal
    (δ = 0.5) — with few dozen days a raw Σ⁻¹ is numerically explosive.
    """
    x = ics[np.isfinite(ics).all(axis=1)]
    K = ics.shape[1]
    if x.shape[0] < 5:
        return np.full(K, 1.0 / K)
    mu = x.mean(axis=0)
    cov = np.cov(x, rowvar=False, ddof=1)
    if cov.ndim == 0:
        cov = cov.reshape(1, 1)
    # Shrink covariance toward diagonal (stabilises inversion).
    delta = 0.5
    diag = np.diag(np.diag(cov))
    cov_s = (1 - delta) * cov + delta * diag
    # Regularise.
    cov_s += np.eye(K) * 1e-8
    try:
        inv = np.linalg.solve(cov_s, mu / max(gamma, 1e-9))
    except np.linalg.LinAlgError:
        return np.full(K, 1.0 / K)
    return project_simplex(inv)


# ----------------------------------------------------------------------
# Monte Carlo bootstrap + shrinkage
# ----------------------------------------------------------------------
@dataclass
class WeightEstimate:
    labels: list[str]
    weights: dict[str, float]            # final shrunk weights
    raw: dict[str, dict[str, float]]     # per-estimator weights
    ci95: dict[str, tuple[float, float]] = field(default_factory=dict)
    p_nonpositive: dict[str, float] = field(default_factory=dict)
    n_days: int = 0
    ic_mean: dict[str, float] = field(default_factory=dict)
    ic_ir: dict[str, float] = field(default_factory=dict)


def monte_carlo_weights(
    ics: np.ndarray,
    labels: list[str],
    *,
    n_boot: int = 1000,
    gamma: float = 1.0,
    seed: int = 42,
) -> WeightEstimate:
    """Bootstrap the mean-variance estimator over trading days.

    Returns the shrunk final weights + diagnostics.
    """
    x = ics[np.isfinite(ics).all(axis=1)]
    D, K = x.shape
    if D < 5:
        eq = np.full(K, 1.0 / K)
        return WeightEstimate(
            labels=labels,
            weights={l: float(eq[i]) for i, l in enumerate(labels)},
            raw={}, n_days=D,
        )
    rng = np.random.default_rng(seed)
    boot = np.empty((n_boot, K))
    for b in range(n_boot):
        idx = rng.integers(0, D, size=D)
        boot[b] = mean_variance_weights(x[idx], gamma=gamma)
    w_mc = boot.mean(axis=0)
    # Empirical probability each weight is ~0 -> shrinkage intensity.
    p_nonpos = (boot <= 1e-6).mean(axis=0)
    eq = np.full(K, 1.0 / K)
    w_final = (1.0 - p_nonpos) * w_mc + p_nonpos * eq
    s = w_final.sum()
    if s > 0:
        w_final = w_final / s
    lo = np.percentile(boot, 2.5, axis=0)
    hi = np.percentile(boot, 97.5, axis=0)
    ic_mean = np.nanmean(ics, axis=0)
    ic_std = np.nanstd(ics, axis=0, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ic_ir = np.where(ic_std > 0, ic_mean / ic_std, 0.0)
    return WeightEstimate(
        labels=labels,
        weights={l: float(w_final[i]) for i, l in enumerate(labels)},
        raw={
            "ir": {l: float(v) for l, v in zip(labels, ir_weights(ics))},
            "inverse_variance": {l: float(v) for l, v in zip(labels, inverse_variance_weights(ics))},
            "mean_variance": {l: float(v) for l, v in zip(labels, mean_variance_weights(ics, gamma=gamma))},
            "monte_carlo_mean": {l: float(v) for l, v in zip(labels, w_mc)},
        },
        ci95={l: (float(lo[i]), float(hi[i])) for i, l in enumerate(labels)},
        p_nonpositive={l: float(p_nonpos[i]) for i, l in enumerate(labels)},
        n_days=D,
        ic_mean={l: float(ic_mean[i]) for i, l in enumerate(labels)},
        ic_ir={l: float(ic_ir[i]) for i, l in enumerate(labels)},
    )


def optimize(
    score_mats: dict[str, np.ndarray],
    fwd: np.ndarray,
    *,
    n_boot: int = 1000,
    gamma: float = 1.0,
    seed: int = 42,
) -> WeightEstimate:
    """Full pipeline.

    score_mats: {label: (D, T) scores}; fwd: (D, T) forward returns.
    All matrices must share the same (D, T) grid (NaN where missing).
    """
    labels = list(score_mats.keys())
    D = fwd.shape[0]
    ic_cols = []
    for l in labels:
        ic_cols.append(daily_ics(score_mats[l], fwd))
    ics = np.column_stack(ic_cols)  # (D, K)
    return monte_carlo_weights(ics, labels, n_boot=n_boot, gamma=gamma, seed=seed)
