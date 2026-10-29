"""tests/test_short_engine.py — Short Alpha Engine (P4-01)."""
from __future__ import annotations

import pytest

from bot.config import ShortEngineSettings, load_settings
from bot.portfolio.short_engine import ShortAlphaEngine
from bot.signals.ensemble import EnsembleEval
from bot.portfolio.constructor import build_trade_deltas, TradeDelta


def _settings(**kw) -> ShortEngineSettings:
    base = dict(enabled=True, mode="independent", trigger="blend",
                weights={"rank": 1/3, "news": 1/3, "mom": 1/3},
                entry_threshold=0.85, exit_threshold=0.40,
                adverse_stop_pct=0.10, max_names=3, max_weight_short=0.10)
    base.update(kw)
    return ShortEngineSettings(**base)


def _eval(t: str, r: float, n: float, m: float) -> EnsembleEval:
    return EnsembleEval(ticker=t, score=r + n + m, raw_score=r + n + m,
                        rank_score=r, news_score=n, momentum_score=m,
                        regime="neutral")


class TestBearishScore:
    def test_negates_components(self) -> None:
        e = ShortAlphaEngine(_settings())
        # strongly negative on every axis -> strongly bearish (~+1.0)
        assert e.bearish_score(-1.0, -1.0, -1.0, True, True, True) == pytest.approx(1.0)
        # positive axes -> negative bearish score
        assert e.bearish_score(1.0, 1.0, 1.0, True, True, True) == pytest.approx(-1.0)

    def test_renormalises_missing_axes(self) -> None:
        e = ShortAlphaEngine(_settings())
        # only momentum present: weight sum = 1/3 -> same value as full
        s = e.bearish_score(0.0, 0.0, -0.9, False, False, True)
        assert s == pytest.approx(0.9)

    def test_zero_when_no_axes(self) -> None:
        e = ShortAlphaEngine(_settings())
        assert e.bearish_score(-1, -1, -1, False, False, False) == 0.0


class TestEvaluate:
    def test_entry_threshold_gates(self) -> None:
        e = ShortAlphaEngine(_settings())
        evals = [_eval("AAA", -0.9, -0.9, -0.9),   # bs ~0.9 -> qualifies
                 _eval("BBB", -0.5, -0.5, -0.5)]   # bs 0.5 -> no
        out = e.evaluate(evals, regime="neutral", vol_fn=lambda t: 0.2)
        assert list(out) == ["AAA"]
        assert out["AAA"].bearish_score == pytest.approx(0.9)

    def test_held_short_kept_above_exit_threshold(self) -> None:
        e = ShortAlphaEngine(_settings())
        evals = [_eval("AAA", -0.5, -0.5, -0.5)]  # bs 0.5: below entry, above exit
        out = e.evaluate(evals, regime="neutral", vol_fn=lambda t: 0.2,
                         held_shorts={"AAA": 100})
        assert "AAA" in out
        # ...but below the exit threshold -> dropped (constructor COVERs)
        evals = [_eval("AAA", -0.1, -0.1, -0.1)]
        assert e.evaluate(evals, regime="neutral", vol_fn=lambda t: 0.2,
                          held_shorts={"AAA": 100}) == {}

    def test_hedge_mode_requires_risk_off(self) -> None:
        e = ShortAlphaEngine(_settings(mode="hedge"))
        evals = [_eval("AAA", -0.95, -0.95, -0.95)]
        assert e.evaluate(evals, regime="neutral", vol_fn=lambda t: 0.2) == {}
        assert "AAA" in e.evaluate(evals, regime="risk_off", vol_fn=lambda t: 0.2)

    def test_off_mode_no_candidates(self) -> None:
        e = ShortAlphaEngine(_settings(mode="off", enabled=True))
        assert e.enabled is False
        evals = [_eval("AAA", -0.99, -0.99, -0.99)]
        assert e.evaluate(evals, regime="risk_off", vol_fn=lambda t: 0.2) == {}

    def test_max_names_caps_entries(self) -> None:
        e = ShortAlphaEngine(_settings(max_names=2))
        evals = [_eval(f"T{i}", -0.9 - i * 0.01, -0.9, -0.9) for i in range(5)]
        out = e.evaluate(evals, regime="neutral", vol_fn=lambda t: 0.2)
        assert len(out) == 2  # strongest two bearish convictions

    def test_no_vol_no_candidate(self) -> None:
        e = ShortAlphaEngine(_settings())
        evals = [_eval("AAA", -0.9, -0.9, -0.9)]
        assert e.evaluate(evals, regime="neutral", vol_fn=lambda t: 0.0) == {}


class TestTargetWeights:
    def test_negative_inverse_vol_weights_capped(self) -> None:
        e = ShortAlphaEngine(_settings())
        evals = [_eval("AAA", -0.9, -0.9, -0.9), _eval("BBB", -0.88, -0.88, -0.88)]
        sigs = e.evaluate(evals, regime="neutral",
                          vol_fn=lambda t: 0.2 if t == "AAA" else 0.4)
        w = e.target_weights(sigs, equity=1_000_000, max_short_gross=0.20,
                             max_weight_name=0.20)
        assert all(v < 0 for v in w.values())
        # gross short exposure <= 20% of equity
        assert sum(abs(v) for v in w.values()) <= 0.20 + 1e-9
        # per-name cap respected
        assert max(abs(v) for v in w.values()) <= 0.10 + 1e-9
        # lower vol (AAA) gets the bigger short
        assert abs(w["AAA"]) > abs(w["BBB"])

    def test_empty_when_no_signals(self) -> None:
        e = ShortAlphaEngine(_settings())
        assert e.target_weights({}, equity=1e6, max_short_gross=0.2,
                                max_weight_name=0.2) == {}


class TestConstructorShortEmission:
    """Negative targets must emit SHORT (not a naked SELL); resizes emit
    COVER; a long held in a flipped name is SOLD first."""

    def _settings(self):
        from bot.config import StrategySettings
        s = StrategySettings()
        s.rebalance_threshold = 0.0
        s.min_trade_notional = 1_000.0
        return s

    def test_short_entry(self) -> None:
        out = build_trade_deltas(
            current_holdings={}, current_shorts={},
            last_close={"AAA": 100.0},
            target_weights={"AAA": -0.10}, equity=1_000_000,
            settings=self._settings(), max_orders_remaining=8,
            short_allowed=True,
        )
        assert len(out.trades) == 1
        d = out.trades[0]
        assert d.action == "SHORT" and d.qty == 1000

    def test_no_short_when_not_allowed(self) -> None:
        out = build_trade_deltas(
            current_holdings={}, current_shorts={},
            last_close={"AAA": 100.0},
            target_weights={"AAA": -0.10}, equity=1_000_000,
            settings=self._settings(), max_orders_remaining=8,
            short_allowed=False,
        )
        assert out.trades == []

    def test_long_flipped_is_sold_then_shorted(self) -> None:
        out = build_trade_deltas(
            current_holdings={"AAA": 500}, current_shorts={},
            last_close={"AAA": 100.0},
            target_weights={"AAA": -0.05}, equity=1_000_000,
            settings=self._settings(), max_orders_remaining=8,
            short_allowed=True,
        )
        acts = [(d.action, d.qty) for d in out.trades]
        assert ("SELL", 500) in acts and ("SHORT", 500) in acts

    def test_partial_cover_on_resize(self) -> None:
        out = build_trade_deltas(
            current_holdings={}, current_shorts={"AAA": 1000},
            last_close={"AAA": 100.0},
            target_weights={"AAA": -0.05}, equity=1_000_000,  # target 500
            settings=self._settings(), max_orders_remaining=8,
            short_allowed=True,
        )
        assert len(out.trades) == 1
        assert out.trades[0].action == "COVER" and out.trades[0].qty == 500

    def test_full_cover_when_target_gone(self) -> None:
        out = build_trade_deltas(
            current_holdings={}, current_shorts={"AAA": 1000},
            last_close={"AAA": 100.0},
            target_weights={}, equity=1_000_000,
            settings=self._settings(), max_orders_remaining=8,
            short_allowed=False,  # covers still allowed (risk-reducing)
        )
        assert len(out.trades) == 1
        assert out.trades[0].action == "COVER" and out.trades[0].qty == 1000


class TestAnyAxisTrigger:
    """The blend caps near ~0.36 (rank/news axes live in [-0.25,+0.25]), so
    a 0.63 BLEND bar never fires — 1,015 journal obs, max blend 0.355.
    any_axis enters when ONE axis shows >= threshold bearish conviction."""

    def test_any_axis_fires_on_strong_momentum(self) -> None:
        e = ShortAlphaEngine(_settings(trigger="any_axis", entry_threshold=0.63))
        # blend = (0.05 - 0.02 + 1.0)/3 ~ 0.34 -> blend mode would NOT fire
        ev = [_eval("AAA", -0.05, 0.02, -1.00)]
        out = e.evaluate(ev, regime="neutral", vol_fn=lambda t: 0.2)
        assert "AAA" in out

    def test_any_axis_ignores_weak_axes(self) -> None:
        e = ShortAlphaEngine(_settings(trigger="any_axis", entry_threshold=0.63))
        # every axis mildly bearish; blend ~0.4 but no axis >= 0.63
        ev = [_eval("AAA", -0.14, -0.25, -0.60)]
        assert e.evaluate(ev, regime="neutral", vol_fn=lambda t: 0.2) == {}

    def test_blend_mode_still_available(self) -> None:
        e = ShortAlphaEngine(_settings(trigger="blend", entry_threshold=0.30))
        ev = [_eval("AAA", -0.05, 0.02, -1.00)]  # blend ~0.34 >= 0.30
        assert "AAA" in e.evaluate(ev, regime="neutral", vol_fn=lambda t: 0.2)

    def test_any_axis_exit_still_blend_decay(self) -> None:
        e = ShortAlphaEngine(_settings(trigger="any_axis", entry_threshold=0.63,
                                       exit_threshold=0.40))
        # Held short whose axes all faded: blend (0.05-0.02+0.60)/3 = 0.21
        # < exit 0.40 AND no axis >= 0.63 entry -> COVER (dropped).
        ev = [_eval("AAA", -0.05, 0.02, -0.60)]
        assert e.evaluate(ev, regime="neutral", vol_fn=lambda t: 0.2,
                          held_shorts={"AAA": 100}) == {}

    def test_held_short_requalifying_via_entry_is_kept(self) -> None:
        e = ShortAlphaEngine(_settings(trigger="any_axis", entry_threshold=0.63,
                                       exit_threshold=0.40))
        # blend 0.343 < exit 0.40 but mom -1.00 still triggers entry: the
        # position RE-QUALIFIES, so it must be kept (never strand a short
        # whose thesis is still firing at full strength).
        ev = [_eval("AAA", -0.05, 0.02, -1.00)]
        out = e.evaluate(ev, regime="neutral", vol_fn=lambda t: 0.2,
                         held_shorts={"AAA": 100})
        assert "AAA" in out


def test_risk_per_name_short_cap_aligns_with_engine() -> None:
    """risk.max_short_single must cover strategy's max_weight_short, else
    every engine-sized short dies as RISK_SINGLE_SHORT (2026-10-30 bug)."""
    from bot.portfolio.risk import RiskManager
    from bot.portfolio.ledger import Ledger
    s = load_settings("configs/default.yaml")
    assert s.risk.max_short_single >= s.strategy.short_engine.max_weight_short
    rm = RiskManager(s.risk, None)  # no settlement calendar in test
    ledger = Ledger(cash=1_000_000.0, initial_cash=1_000_000.0)
    # a 9% single-name short passes; an 11% one is denied
    d = __import__("datetime").date(2026, 1, 5)
    assert rm.check_short(ledger, "AAA", 90, 1000.0, d).ok
    assert not rm.check_short(ledger, "AAA", 110, 1000.0, d).ok


def test_short_engine_block_parsed_from_yaml() -> None:
    s = load_settings("configs/default.yaml")
    se = s.strategy.short_engine
    assert se is not None
    # live-enabled in independent mode (user request 2026-10-30); per-axis
    # 0.63 trigger (user spec) + tight caps keep it a low-exposure module
    assert se.enabled is True and se.mode == "independent"
    assert se.trigger == "any_axis"
    assert se.entry_threshold == pytest.approx(0.63)  # user spec 2026-10-30
    assert se.exit_threshold == pytest.approx(0.40)
    assert se.adverse_stop_pct == pytest.approx(0.10)
    assert se.max_names == 3
    assert se.max_weight_short == pytest.approx(0.10)
    assert sum(se.weights.values()) == pytest.approx(1.0, abs=0.01)
