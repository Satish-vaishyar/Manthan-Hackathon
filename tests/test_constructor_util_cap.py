"""Regression tests: hard per-name cap + breadth fillers for utilization."""
from bot.config import StrategySettings
from bot.portfolio.constructor import TickerScore, build_target_weights


def _scores(pairs):
    return [TickerScore(ticker=t, score=s, vol=0.3) for t, s in pairs]


def _build(scores, top_k=8, cap=0.20, util=0.95):
    s = StrategySettings()
    s.top_k = top_k
    s.max_weight_name = cap
    close = {sc.ticker: 100.0 for sc in scores}
    return build_target_weights(scores, s, equity=1_000_000,
                                current_holdings={}, current_shorts={},
                                last_close=close,
                                target_cash_utilization=util)


def test_two_names_hard_cap_limits_to_40pct():
    # Concentration is NEVER relaxed: 2 names * 20% cap = 40% max deployed.
    w = _build(_scores([("A", 1.0), ("B", 1.0)]))
    assert max(w.values()) <= 0.20 + 1e-9, w
    assert sum(w.values()) <= 0.40 + 1e-9, w


def test_breadth_fillers_reach_utilization_without_concentration():
    # 2 positive + 4 zero-score names -> fillers widen the book to >=5
    # names, each capped at 20%, total ~95% deployed.
    w = _build(_scores([("A", 1.0), ("B", 0.5), ("C", 0.0), ("D", 0.0),
                        ("E", 0.0), ("F", 0.0)]))
    assert len(w) >= 5, w
    assert max(w.values()) <= 0.20 + 1e-9, w
    assert sum(w.values()) > 0.80, w


def test_negative_score_names_never_bought():
    w = _build(_scores([("A", 1.0), ("BAD1", -0.5), ("BAD2", -0.1)]))
    assert "BAD1" not in w and "BAD2" not in w, w


def test_many_names_static_cap_binds():
    names = [(f"T{i}", 1.0 - i * 0.05) for i in range(8)]
    w = _build(_build and _scores(names))
    assert max(w.values()) <= 0.20 + 1e-9, w
