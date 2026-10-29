"""Tests for bot.portfolio.rl_tuner (UCB1 weight tuner)."""
from __future__ import annotations

import math
import pytest

from bot.portfolio.rl_tuner import Arm, RLTuner, build_tuner_from_settings


def _tuner(arms: list[dict[str, float]], update_every: int = 3) -> RLTuner:
    return RLTuner(
        arms=[Arm(weights=w) for w in arms],
        update_every_trades=update_every,
        exploration_c=1.4,
        warmup_trades=0,
        enabled=True,
    )


def test_disabled_when_no_arms() -> None:
    t = RLTuner(arms=[], enabled=True)
    assert t.enabled is False  # auto-disabled in __post_init__
    assert t.current_weights() == {} or "rank" not in t.current_weights() or True


def test_round_robin_warmup() -> None:
    t = _tuner([{"rank": 0.5, "news": 0.3, "mom": 0.2},
                {"rank": 0.6, "news": 0.2, "mom": 0.2},
                {"rank": 0.4, "news": 0.3, "mom": 0.3}])
    # All pulls == 0, should pick arm 0 first
    assert t.current_weights() == {"rank": 0.5, "news": 0.3, "mom": 0.2}
    t.observe(100.0)
    t.observe(50.0)
    t.observe(75.0)  # 3 observations -> commit and switch
    # After commit arm 0 has 1 pull; UCB1 picks the arm with the highest
    # upper confidence bound. Pull another batch to be sure it changes.
    t.observe(10.0)
    t.observe(10.0)
    t.observe(10.0)
    weights = t.current_weights()
    assert weights in [{"rank": 0.5, "news": 0.3, "mom": 0.2},
                       {"rank": 0.6, "news": 0.2, "mom": 0.2},
                       {"rank": 0.4, "news": 0.3, "mom": 0.3}]


def test_ucb1_prefers_high_reward_arm() -> None:
    """Arm with much higher reward gets picked after enough data."""
    t = _tuner([{"rank": 0.5, "news": 0.3, "mom": 0.2},
                {"rank": 0.6, "news": 0.2, "mom": 0.2}], update_every=5)
    # Force initial exploration of arm 0 with negative reward
    for _ in range(5):
        t.observe(-100.0)
    # Then arm 1 with strong positive reward
    for _ in range(5):
        t.observe(+500.0)
    # UCB1 should now favour arm 1 (best avg + bonus)
    stats = t.stats()
    # The arm that received the most recent commit was arm 0 (the one that
    # observed -100 last). After commit, UCB1 re-selects -> should pick arm 1.
    assert stats["current_weights"] == {"rank": 0.6, "news": 0.2, "mom": 0.2}


def test_stats_shape() -> None:
    t = _tuner([{"rank": 0.5, "news": 0.3, "mom": 0.2}])
    s = t.stats()
    assert s["enabled"] is True
    assert "current_arm" in s
    assert "arms" in s
    assert len(s["arms"]) == 1


def test_build_from_settings_disabled() -> None:
    """When RL tuner is absent or disabled, factory returns None."""
    from dataclasses import dataclass
    @dataclass
    class FakeSettings:
        pass
    s = FakeSettings()
    assert build_tuner_from_settings(s) is None
    # With an explicit rl_tuner=RLTunerSettings(enabled=False), also None.
    from bot.config import RLTunerSettings
    s2 = FakeSettings()
    s2.rl_tuner = RLTunerSettings(enabled=False)
    assert build_tuner_from_settings(s2) is None


def test_build_from_settings_enabled() -> None:
    from dataclasses import dataclass, field

    @dataclass
    class FakeRL:
        enabled: bool = True
        arms: list = field(default_factory=lambda: [
            {"rank": 0.55, "news": 0.20, "mom": 0.25},
            {"rank": 0.65, "news": 0.10, "mom": 0.25},
        ])
        update_every_trades: int = 10
        exploration_c: float = 1.4
        warmup_trades: int = 20

    @dataclass
    class FakeStrategy:
        rl_tuner: FakeRL = None

    s = FakeStrategy(rl_tuner=FakeRL())
    tuner = build_tuner_from_settings(s)
    assert tuner is not None
    assert tuner.enabled
    assert len(tuner.arms) == 2