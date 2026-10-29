"""tests/test_signals_regime.py — coverage for bot.signals regime re-exports.

The M5 regime detector itself lives in `bot.models.regime` and is covered by
`tests/test_regime.py`. This file verifies that the regime symbols re-exported
from `bot.signals` line up with M5's labels and that the `gross_long_target`
helper there still works correctly.
"""
from __future__ import annotations

from datetime import date

import pytest

from bot.contracts import DayBundle, PriceTickData
from bot.signals import (
    ALL_REGIMES, NEUTRAL, RISK_OFF, RISK_ON,
    RegimeDetector, gross_long_target,
)


def _tick(t: str, c: float) -> PriceTickData:
    return PriceTickData(ticker=t, open=c, high=c, low=c, close=c, volume=1000)


def _bundle(ticks: dict[str, PriceTickData]) -> DayBundle:
    return DayBundle(
        simulated_date=date(2025, 1, 1),
        ticks=ticks,
        news=[],
        assembled_at=__import__("datetime").datetime(2025, 1, 1),
        assembly_latency_ms=0.0,
    )


# --------------------------------------------------------------- re-exports
def test_regime_labels_match_m5():
    """Re-exports must be the same labels M5 produces."""
    assert (RISK_ON, NEUTRAL, RISK_OFF) == ("risk_on", "neutral", "risk_off")
    assert ALL_REGIMES == ("risk_on", "neutral", "risk_off")


def test_regime_detector_instance_constructible():
    """bot.signals.RegimeDetector is the M5 class."""
    rd = RegimeDetector()
    # Same interface: .update(bundle) -> str, .current -> str
    assert rd.current == NEUTRAL  # until we feed it data
    bundle = _bundle({f"T{i:02d}": _tick(f"T{i:02d}", 100.0) for i in range(5)})
    label = rd.update(bundle)
    assert label in ALL_REGIMES


# ------------------------------------------------------------ gross_long_target
def test_gross_long_target_known_regimes():
    table = {"risk_on": 0.92, "neutral": 0.75, "risk_off": 0.40}
    assert gross_long_target("risk_on", table) == pytest.approx(0.92)
    assert gross_long_target("risk_off", table) == pytest.approx(0.40)
    assert gross_long_target("neutral", table) == pytest.approx(0.75)


def test_gross_long_target_unknown_regime_uses_default():
    assert gross_long_target("unknown", {}, default=0.5) == 0.5


def test_gross_long_target_clamps_out_of_range():
    table = {"risk_on": 1.5, "neutral": -0.1}  # invalid values
    assert gross_long_target("risk_on", table) == 1.0
    assert gross_long_target("neutral", table) == 0.0


# ------------------------------------------------------------ integration
def test_signals_ensemble_accepts_regime_label_string():
    """The new SignalsEnsemble.evaluate(bundle, regime_str) signature works."""
    from bot.config import Settings
    from bot.signals import SignalsEnsemble

    s = Settings(env="sandbox", server_ip="127.0.0.1", team_username="t",
                 bot_access_token="x")
    ens = SignalsEnsemble(s.signals)
    ticks = {f"T{i}": _tick(f"T{i}", 100.0 + i) for i in range(3)}
    evals = ens.evaluate(_bundle(ticks), regime=NEUTRAL)
    assert len(evals) == 3
    assert all(ev.regime == NEUTRAL for ev in evals)


def test_signals_ensemble_risk_off_changes_eval_regime_field():
    from bot.config import Settings
    from bot.signals import SignalsEnsemble

    s = Settings(env="sandbox", server_ip="127.0.0.1", team_username="t",
                 bot_access_token="x")
    ens = SignalsEnsemble(s.signals)
    ticks = {"T0": _tick("T0", 100.0)}
    evals = ens.evaluate(_bundle(ticks), regime=RISK_OFF)
    assert evals[0].regime == RISK_OFF
    # Reasons should mention the regime
    assert any("regime=risk_off" in r for r in evals[0].reasons)


# ------------------------------------------------------ bundle update path
def test_signals_bundle_update_volatility_only():
    """After M5 retirement, SignalsBundle only handles vol; regime is external."""
    from bot.config import Settings
    from bot.signals import SignalsBundle, SignalsEnsemble, EventTagger, RulesFallback, VolatilityModule

    s = Settings(env="sandbox", server_ip="127.0.0.1", team_username="t",
                 bot_access_token="x")
    bundle = SignalsBundle(
        tagger=EventTagger(),
        volatility=VolatilityModule(),
        rules=RulesFallback(),
        ensemble=SignalsEnsemble(s.signals),
    )
    ticks = {f"T{i}": _tick(f"T{i}", 100.0 + i) for i in range(3)}
    bundle.update_volatility(date(2025, 1, 1), ticks)
    # No regime field on the bundle anymore
    assert bundle.volatility.vol("T0") == pytest.approx(0.02)  # fallback pre-warmup