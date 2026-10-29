"""Regression tests for the post-scaling per-name cap in build_target_weights."""
from bot.config import StrategySettings
from bot.portfolio.constructor import build_target_weights
from bot.portfolio.constructor import TickerScore  # type: ignore


def _scores(names):
    return [TickerScore(ticker=t, score=1.0, vol=0.3) for t in names]


def test_few_names_deploy_target_utilization():
    s = StrategySettings()
    s.top_k = 8
    s.max_weight_name = 0.20
    w = build_target_weights(_scores(["A", "B"]), s, equity=1_000_000,
                             current_holdings={}, current_shorts={},
                             last_close={"A": 100, "B": 100},
                             target_cash_utilization=0.95)
    # 2 names: cap relaxes to 0.475 each -> 95% deployed, not stuck at 40%.
    assert abs(sum(w.values()) - 0.95) < 1e-6, w
    assert all(abs(x - 0.475) < 1e-6 for x in w.values()), w


def test_many_names_static_cap_binds():
    s = StrategySettings()
    s.top_k = 8
    s.max_weight_name = 0.20
    names = [f"T{i}" for i in range(8)]
    w = build_target_weights(_scores(names), s, equity=1_000_000,
                             current_holdings={}, current_shorts={},
                             last_close={t: 100 for t in names},
                             target_cash_utilization=0.95)
    assert max(w.values()) <= 0.20 + 1e-9, w


def test_single_name_capped_at_100pct():
    s = StrategySettings()
    s.top_k = 8
    s.max_weight_name = 0.20
    w = build_target_weights(_scores(["ONLY"]), s, equity=1_000_000,
                             current_holdings={}, current_shorts={},
                             last_close={"ONLY": 100},
                             target_cash_utilization=0.95)
    assert w["ONLY"] <= 0.95 + 1e-9 and w["ONLY"] > 0.5, w
