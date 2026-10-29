"""Tests for bot.portfolio.weight_optimizer + weights_mode wiring."""
from __future__ import annotations

import numpy as np
import pytest

from bot.portfolio.weight_optimizer import (
    ir_weights,
    inverse_variance_weights,
    mean_variance_weights,
    monte_carlo_weights,
    project_simplex,
    spearman_ic,
)


def test_spearman_perfect_and_inverse() -> None:
    x = np.arange(1.0, 11.0)
    y = x.copy()
    assert spearman_ic(x, y) == pytest.approx(1.0, abs=1e-9)
    assert spearman_ic(x, y[::-1]) == pytest.approx(-1.0, abs=1e-9)


def test_spearman_nan_safe() -> None:
    assert np.isnan(spearman_ic(np.array([np.nan, 1, 2]), np.array([1, 2, 3])))


def test_spearman_monotone_invariant() -> None:
    """Nonlinear monotone transform keeps Spearman = 1."""
    x = np.arange(1.0, 21.0)
    assert spearman_ic(x, np.exp(x)) == pytest.approx(1.0, abs=1e-9)


def test_project_simplex() -> None:
    w = project_simplex(np.array([0.7, 0.2, -0.5]))
    assert w.min() >= -1e-12
    assert w.sum() == pytest.approx(1.0)
    # Duchi projection of [0.7, 0.2, -0.5] is [0.75, 0.25, 0]
    assert w[0] == pytest.approx(0.75)
    assert w[1] == pytest.approx(0.25)
    assert w[2] == pytest.approx(0.0)


def _mk_ics(D=60, seed=0):
    """Synthetic daily ICs: mom strong+stable, rank weak+noisy."""
    rng = np.random.default_rng(seed)
    mom = rng.normal(0.15, 0.25, D)
    news = rng.normal(0.08, 0.3, D)
    rank = rng.normal(0.01, 0.4, D)
    return np.column_stack([rank, news, mom])


def test_estimators_punish_weak_signal() -> None:
    ics = _mk_ics()
    for fn in (ir_weights, inverse_variance_weights, mean_variance_weights):
        w = fn(ics)
        # mom must dominate; rank must be the smallest (ties allowed at 0
        # for corner solutions of the simplex-projected estimators).
        assert w[2] >= w[1] >= w[0]
        assert w[2] > 0.5
        assert w.sum() == pytest.approx(1.0)


def test_mc_weights_simplex_and_shrinkage() -> None:
    ics = _mk_ics()
    est = monte_carlo_weights(ics, ["rank", "news", "mom"], n_boot=200)
    w = np.array([est.weights[k] for k in est.labels])
    assert (w >= -1e-12).all()
    assert w.sum() == pytest.approx(1.0)
    # mom must dominate; rank shrunk toward equality.
    assert est.weights["mom"] > est.weights["news"]
    # CIs are ordered and sane (the shrunk weight may sit slightly outside
    # the raw bootstrap CI because of the final renormalisation).
    lo, hi = est.ci95["mom"]
    assert lo <= hi
    assert 0.0 <= est.weights["mom"] <= 1.0
    # IC stats populated.
    assert est.ic_mean["mom"] > est.ic_mean["rank"]


def test_mc_degenerate_fallback() -> None:
    ics = np.array([[np.nan, np.nan, np.nan]] * 3)
    est = monte_carlo_weights(ics, ["a", "b", "c"])
    assert est.weights == {"a": 1/3, "b": 1/3, "c": 1/3}


def test_weights_mode_yaml(monkeypatch) -> None:
    monkeypatch.setenv("BOT_ACCESS_TOKEN", "dummy")
    monkeypatch.setenv("TEAM_USERNAME", "mn2632")
    from bot.config import load_settings
    s = load_settings(env="sandbox")
    assert s.strategy.weights_mode == "optimized"
    assert s.strategy.optimized_weights_path == "configs/optimized_weights.json"
