from datetime import date, timedelta
from bot.signals.overlays import NewsDecay

def test_negative_sentiment_dominates():
    nd = NewsDecay(half_life_days=2.0)
    d0 = date(2026, 1, 1)
    nd.update(d0, {"T": -0.8})
    nd.update(d0 + timedelta(days=2), {"T": -0.1})
    eff = nd.effective("T", d0 + timedelta(days=2))
    # -0.8 two days old decays to -0.4, which is stronger than -0.1.
    assert abs(eff - (-0.4)) < 1e-9, eff

def test_positive_still_max():
    nd = NewsDecay(half_life_days=2.0)
    d0 = date(2026, 1, 1)
    nd.update(d0, {"T": 0.8})
    nd.update(d0 + timedelta(days=2), {"T": 0.1})
    assert abs(nd.effective("T", d0 + timedelta(days=2)) - 0.4) < 1e-9
