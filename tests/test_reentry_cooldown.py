"""tests/test_reentry_cooldown.py — P4-02 churn brake."""
from __future__ import annotations

import pytest

from bot.config import load_settings
from bot.signals.overlays import reentry_blocked


class TestReentryBlocked:
    def test_inside_cooldown_blocks(self) -> None:
        assert reentry_blocked(0, 3, 0.80, 0.90) is True
        assert reentry_blocked(2, 3, 0.80, 0.90) is True

    def test_cooldown_expired_allows(self) -> None:
        assert reentry_blocked(3, 3, 0.80, 0.90) is False
        assert reentry_blocked(10, 3, 0.10, 0.90) is False

    def test_high_conviction_override(self) -> None:
        # sold yesterday but the score is extreme -> re-entry allowed
        assert reentry_blocked(1, 3, 0.95, 0.90) is False
        assert reentry_blocked(1, 3, 0.90, 0.90) is False  # >= override passes

    def test_fail_soft(self) -> None:
        assert reentry_blocked(None, 3, 0.80, 0.90) is False   # never sold
        assert reentry_blocked(1, 0, 0.80, 0.90) is False      # cooldown off

    def test_negative_age_blocks(self) -> None:
        # clock-skewed sell date: conservatively inside cooldown
        assert reentry_blocked(-1, 3, 0.80, 0.90) is True


def test_cooldown_knobs_from_yaml() -> None:
    s = load_settings("configs/default.yaml")
    assert s.strategy.reentry_cooldown_days == 3
    assert s.strategy.reentry_score_override == pytest.approx(0.90)
    assert s.strategy.deployer_regime_cap is True
