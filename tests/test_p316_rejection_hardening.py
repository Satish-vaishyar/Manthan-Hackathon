"""Tests for P3-16 (rejection-0 hardening) and P3-17 (idle-cash deployer)."""
from __future__ import annotations

import pytest

from bot.config import load_settings


def _settings(monkeypatch):
    monkeypatch.setenv("BOT_ACCESS_TOKEN", "dummy")
    monkeypatch.setenv("TEAM_USERNAME", "mn2632")
    return load_settings(env="sandbox")


# ----------------------------------------------------------------------
# P3-16: in-flight cash reservation
# ----------------------------------------------------------------------
def test_inflight_cash_reservation_blocks_overdraw(monkeypatch) -> None:
    """Second BUY in a batch must validate against cash minus in-flight."""
    from datetime import date
    from bot.contracts import OrderRequest
    from bot.execution.validator import OrderValidator
    from bot.portfolio.ledger import Ledger
    from bot.portfolio.risk import RiskManager
    from bot.portfolio.settlement import SettlementCalendar

    s = _settings(monkeypatch)
    cal = SettlementCalendar.__new__(SettlementCalendar)
    risk = RiskManager(s.risk, cal)
    v = OrderValidator(s, risk)

    class _StubLedger:
        cash = 200_000.0
        last_close = {"HAVN": 100.0}
        shorts: dict = {}
        holdings: dict = {}
        equity = 1_000_000.0

        def holding_qty(self, t):
            return 0

        short_notional = 0.0

    ledger = _StubLedger()

    req1 = OrderRequest(ticker="HAVN", action="BUY", quantity=1500,  # 150k
                        date=date(2026, 10, 29))
    # First order validates against full 200k cash.
    r1 = v.validate(req1, ledger)
    # Mark in flight with the price -> reserve 150k + brokerage.
    v.mark_in_flight(req1, price=100.0)
    # Second identical order must now see only ~50k available and be denied
    # locally (never reaches the server -> server rejections stay at 0).
    req2 = OrderRequest(ticker="HAVN", action="BUY", quantity=1400,  # 140k
                        date=date(2026, 10, 29))
    r2 = v.validate(req2, ledger)
    # r1 may pass or fail on risk checks, but r2 must fail on cash or cap.
    assert not r2.ok, f"expected local denial, got ok (r1={r1.ok})"
    assert "reserved" in (r2.message or "") or "cap" in (r2.message or "")
    # Clearing releases the reservation.
    v.clear_in_flight(req1)
    v._in_flight.discard((req2.ticker, req2.action, req2.quantity))
    r3 = v.validate(req2, ledger)
    # After release the denial reason (if any) must not be the reservation.
    if not r3.ok:
        assert "reserved" not in (r3.message or "")


def test_mark_clear_reservation_symmetric(monkeypatch) -> None:
    from datetime import date
    from bot.contracts import OrderRequest
    from bot.execution.validator import OrderValidator
    from bot.portfolio.risk import RiskManager
    from bot.portfolio.settlement import SettlementCalendar

    s = _settings(monkeypatch)
    cal = SettlementCalendar.__new__(SettlementCalendar)
    v = OrderValidator(s, RiskManager(s.risk, cal))
    req = OrderRequest(ticker="HAVN", action="BUY", quantity=100,
                       date=date(2026, 10, 29))
    v.mark_in_flight(req, price=100.0)
    assert v._reserved_cash == pytest.approx(100 * 100 + s.tax.brokerage_per_order)
    v.clear_in_flight(req)
    assert v._reserved_cash == pytest.approx(0.0)
    # SELLs never reserve cash.
    req_s = OrderRequest(ticker="HAVN", action="SELL", quantity=100,
                         date=date(2026, 10, 29))
    v.mark_in_flight(req_s, price=100.0)
    assert v._reserved_cash == pytest.approx(0.0)


# ----------------------------------------------------------------------
# P3-17: idle-cash deployer config
# ----------------------------------------------------------------------
def test_idle_cash_deploy_yaml_roundtrip(monkeypatch) -> None:
    s = _settings(monkeypatch)
    assert s.strategy.idle_cash_deploy is not None
    assert s.strategy.idle_cash_deploy.enabled is True
    assert s.strategy.idle_cash_deploy.epsilon == pytest.approx(0.05)
