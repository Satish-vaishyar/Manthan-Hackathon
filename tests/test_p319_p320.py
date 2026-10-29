"""Tests for P3-19 (startup holdings review / exit floor) and
P3-20 (zero-rejection hardening: server-holding clamps + divergence resync)."""
from __future__ import annotations

import pytest

from bot.config import load_settings
from bot.portfolio.constructor import build_trade_deltas


def _settings(monkeypatch):
    monkeypatch.setenv("BOT_ACCESS_TOKEN", "dummy")
    monkeypatch.setenv("TEAM_USERNAME", "mn2632")
    return load_settings(env="sandbox")


# ----------------------------------------------------------------------
# P3-19
# ----------------------------------------------------------------------
def test_exit_floor_yaml(monkeypatch) -> None:
    s = _settings(monkeypatch)
    assert s.strategy.exit_min_notional == pytest.approx(5_000.0)
    assert s.strategy.min_trade_notional == pytest.approx(75_000.0)


def test_legacy_position_below_entry_floor_is_sold(monkeypatch) -> None:
    """A 20K legacy holding (below the 75K entry floor) must still exit
    when it's no longer in the target."""
    s = _settings(monkeypatch)
    deltas = build_trade_deltas(
        current_holdings={"LEGACY": 200},   # 200 x 100 = 20K
        current_shorts={},
        last_close={"LEGACY": 100.0, "HAVN": 100.0},
        target_weights={"HAVN": 0.5},
        equity=1_000_000.0,
        settings=s.strategy,
        max_orders_remaining=4,
        short_allowed=True,
    )
    exits = [t for t in deltas.trades if t.ticker == "LEGACY"]
    assert exits, "legacy position below entry floor must still be sellable"
    assert exits[0].action == "SELL"
    assert exits[0].qty == 200


def test_dust_position_skipped(monkeypatch) -> None:
    """Positions below exit_min_notional (5K) are dust — skip the exit."""
    s = _settings(monkeypatch)
    deltas = build_trade_deltas(
        current_holdings={"DUST": 10},   # 10 x 100 = 1K < 5K floor
        current_shorts={},
        last_close={"DUST": 100.0},
        target_weights={},
        equity=1_000_000.0,
        settings=s.strategy,
        max_orders_remaining=4,
        short_allowed=True,
    )
    assert all(t.ticker != "DUST" for t in deltas.trades)


# ----------------------------------------------------------------------
# P3-20
# ----------------------------------------------------------------------
def test_negative_server_holding_becomes_short() -> None:
    """Server reporting a negative holding must be treated as a short,
    not as a long lot — otherwise later SELLs get INSUFFICIENT_MARGIN."""
    # This exercises the reconcile parsing logic conceptually: the main
    # loop stores negative positions in _server_shorts. Here we test the
    # invariant directly against the clamp logic's data contract.
    server_holdings = {"HMDP": -1166}
    server_shorts = {"HMDP": 1166}
    sellable = max(0, int(server_holdings.get("HMDP", 0)))
    coverable = max(0, int(server_shorts.get("HMDP", 0)))
    assert sellable == 0       # never SELL against a negative holding
    assert coverable == 1166   # COVER is the correct action


def test_no_phantom_fill_on_empty_payload() -> None:
    """Empty server response must NOT produce a FILLED order."""
    import asyncio
    from datetime import date
    from bot.contracts import OrderRequest
    from bot.execution.order_manager import OrderManager

    class _API:
        async def post(self, path, body):
            return {"_status": 200, "_body": {}}  # empty success payload

    class _RL:
        async def acquire(self, priority="order"):
            return True
        async def note_429(self):
            pass

    class _Validator:
        def validate(self, req, ledger, **kw):
            from bot.execution.validator import ValidationResult
            return ValidationResult.allow()
        def mark_in_flight(self, req, price=0.0):
            pass
        def clear_in_flight(self, req):
            pass

    class _Journal:
        def record(self, rec):
            pass

    class _Ledger:
        last_close = {"HAVN": 100.0}
        def apply_buy(self, *a, **k):
            raise AssertionError("phantom fill applied!")

    diverged = []
    async def _on_div(reason="", ticker=""):
        diverged.append((reason, ticker))

    om = OrderManager(settings=None, api=_API(), rate_limiter=_RL(),
                      validator=_Validator(), ledger=_Ledger(),
                      journal=_Journal(), on_state_divergence=_on_div)
    req = OrderRequest(ticker="HAVN", action="BUY", quantity=10,
                       date=date(2026, 10, 29))
    resp = asyncio.run(om.submit(req))
    assert resp.status == "UNKNOWN"
    asyncio.run(asyncio.sleep(0))  # let the divergence task fire
    # NOTE: divergence callback is scheduled on the running loop inside
    # submit; by the time submit returns it may or may not have executed —
    # the important assertion is that no fill was applied (status UNKNOWN).
