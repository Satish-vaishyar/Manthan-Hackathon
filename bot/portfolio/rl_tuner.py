"""bot.portfolio.rl_tuner — lightweight online weight tuner (no full RL).

Gradient-free multi-armed bandit (UCB1) over a hand-picked menu of weight
triples (rank/news/mom). Every N trades we update the reward for the arm
that was active based on the realized P&L, and the next day's ensemble
reads the current best arm.

Why UCB1 and not a full RL:
  * We don't have a dense reward signal (P&L arrives per fill, not per step).
  * No exploration budget — we can't simulate millions of episodes against
    the live platform.
  * UCB1 is provably near-optimal for finite stationary bandits and adds
    O(1) per arm per step.
  * This is per docs/04 §4 ("ensemble weights tunable from tradebook
    feedback") and is opt-in via `strategy.rl_tuner.enabled`.

Config (configs/default.yaml):
    strategy:
      rl_tuner:
        enabled: true
        arms:
          - { rank: 0.55, news: 0.20, mom: 0.25 }   # current best (analysis)
          - { rank: 0.65, news: 0.10, mom: 0.25 }   # ranker-heavy
          - { rank: 0.45, news: 0.30, mom: 0.25 }   # balanced
          - { rank: 0.50, news: 0.15, mom: 0.35 }   # momentum-heavy
          - { rank: 0.60, news: 0.10, mom: 0.30 }   # rank+mom (no news)
        update_every_trades: 10    # refresh reward estimate every 10 fills
        exploration_c: 1.4          # UCB1 exploration constant (sqrt(2) ~ 1.41)
        warmup_trades: 20           # use first arm (round-robin) for N fills
"""
from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import structlog

_log = structlog.get_logger("rl_tuner")


@dataclass
class Arm:
    weights: dict[str, float]
    pulls: int = 0            # how many trades we've observed under this arm
    reward_sum: float = 0.0   # sum of realised P&L (in rupees) since pull=0
    last_reward: float = 0.0  # most recent observed reward


@dataclass
class RLTuner:
    arms: list[Arm] = field(default_factory=list)
    update_every_trades: int = 10
    exploration_c: float = 1.4
    warmup_trades: int = 20
    enabled: bool = True

    # ---- runtime state ----
    _current_arm_idx: int = 0
    _trades_since_update: int = 0
    _pending_rewards: deque[float] = field(default_factory=lambda: deque(maxlen=200))
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _total_pulls: int = 0

    def __post_init__(self) -> None:
        if not self.arms:
            self.arms = [Arm(weights={"rank": 0.55, "news": 0.20, "mom": 0.25})]
            self.enabled = False  # disable if no arms configured

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def current_weights(self) -> dict[str, float]:
        """Return the weights the ensemble should use on the NEXT decision."""
        if not self.enabled:
            return dict(self.arms[self._current_arm_idx].weights)
        with self._lock:
            return dict(self.arms[self._current_arm_idx].weights)

    def observe(self, realized_pnl: float) -> None:
        """Feed back the realized P&L from a closed/filled trade.

        The tuner accumulates rewards; once we've seen `update_every_trades`
        rewards, it commits them to the active arm and re-selects the next
        arm via UCB1.
        """
        if not self.enabled:
            return
        with self._lock:
            self._pending_rewards.append(float(realized_pnl))
            self._trades_since_update += 1
            self._total_pulls += 1
            if self._trades_since_update >= self.update_every_trades:
                self._commit_and_select()

    def force_select(self) -> dict[str, float]:
        """Force a re-selection (useful at boot or on regime flip)."""
        with self._lock:
            self._select_ucb1()
            return dict(self.arms[self._current_arm_idx].weights)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _commit_and_select(self) -> None:
        """Credit accumulated rewards to the active arm, then re-select."""
        if not self._pending_rewards:
            self._select_ucb1()
            return
        avg_reward = sum(self._pending_rewards) / len(self._pending_rewards)
        arm = self.arms[self._current_arm_idx]
        arm.pulls += 1
        arm.reward_sum += avg_reward
        arm.last_reward = avg_reward
        _log.info("rl_tuner_commit",
                  arm_idx=self._current_arm_idx,
                  weights=arm.weights,
                  pulls=arm.pulls,
                  avg_reward=round(avg_reward, 2),
                  n_observations=len(self._pending_rewards),
                  component="rl_tuner")
        self._pending_rewards.clear()
        self._trades_since_update = 0
        self._select_ucb1()

    def _select_ucb1(self) -> None:
        n_arms = len(self.arms)
        if n_arms == 1:
            self._current_arm_idx = 0
            return
        # Warmup: round-robin until every arm has been pulled at least once.
        for i, a in enumerate(self.arms):
            if a.pulls == 0:
                self._current_arm_idx = i
                _log.info("rl_tuner_warmup", arm_idx=i, weights=a.weights,
                          component="rl_tuner")
                return
        # UCB1: argmax [ avg_reward_i + c * sqrt(2 ln N / n_i) ]
        N = max(1, sum(a.pulls for a in self.arms))
        c = float(self.exploration_c)
        best_idx, best_score = 0, -math.inf
        for i, a in enumerate(self.arms):
            avg = a.reward_sum / max(1, a.pulls)
            bonus = c * math.sqrt(max(1e-9, math.log(N) / a.pulls))
            score = avg + bonus
            if score > best_score:
                best_idx, best_score = i, score
        if best_idx != self._current_arm_idx:
            _log.info("rl_tuner_switch",
                      from_arm=self._current_arm_idx,
                      to_arm=best_idx,
                      new_weights=self.arms[best_idx].weights,
                      component="rl_tuner")
        self._current_arm_idx = best_idx

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def stats(self) -> dict:
        """Return per-arm stats for observability / journal."""
        return {
            "enabled": self.enabled,
            "current_arm": self._current_arm_idx,
            "current_weights": dict(self.arms[self._current_arm_idx].weights),
            "total_pulls": self._total_pulls,
            "arms": [
                {"idx": i,
                 "weights": dict(a.weights),
                 "pulls": a.pulls,
                 "avg_reward": (a.reward_sum / a.pulls) if a.pulls else 0.0,
                 "last_reward": a.last_reward}
                for i, a in enumerate(self.arms)
            ],
        }


def build_tuner_from_settings(strategy_settings) -> Optional[RLTuner]:
    """Construct an RLTuner from `strategy_settings.rl_tuner` config block.

    Returns None if RL tuner is disabled or absent.
    """
    rl_cfg = getattr(strategy_settings, "rl_tuner", None)
    if rl_cfg is None or not getattr(rl_cfg, "enabled", False):
        return None
    arms_cfg = getattr(rl_cfg, "arms", None) or []
    if not arms_cfg:
        return None
    arms = []
    for a in arms_cfg:
        w = {str(k): float(v) for k, v in dict(a).items()}
        arms.append(Arm(weights=w))
    return RLTuner(
        arms=arms,
        update_every_trades=int(getattr(rl_cfg, "update_every_trades", 10) or 10),
        exploration_c=float(getattr(rl_cfg, "exploration_c", 1.4) or 1.4),
        warmup_trades=int(getattr(rl_cfg, "warmup_trades", 20) or 20),
        enabled=True,
    )