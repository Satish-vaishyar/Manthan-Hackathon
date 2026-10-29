"""tests/test_validator.py — every rejection code in docs/03 §3."""
from __future__ import annotations

from datetime import date

import pytest

from bot.config import (
    APISettings, KafkaSettings, RiskSettings, Settings, StrategySettings,
    TaxSettings, TimingSettings,
)
from bot.contracts import OrderRequest, RejectCode
from bot.execution.validator import OrderValidator
from bot.portfolio.ledger import Ledger
from bot.portfolio.risk import RiskManager
from bot.portfolio.settlement import SettlementCalendar


def _settings() -> Settings:
    return Settings(
        env="sandbox",
        server_ip="127.0.0.1",
        team_username="test",
        bot_access_token="x",
        api=APISettings(port=3002, timeout_s=1.5, rate_per_min=100),
        kafka=KafkaSettings(),
        timing=TimingSettings(),
        strategy=StrategySettings(min_trade_notional=50_000.0, max_orders_per_day=4),
        risk=RiskSettings(max_name_weight=0.15, max_sector_weight=0.30,
                          max_short_gross=0.20, cash_buffer=0.05,
                          dd_soft=0.08, dd_hard=0.15),
        tax=TaxSettings(brokerage_per_order=20.0, dp_charge_per_sell=15.34),
    )


def _validator() -> OrderValidator:
    s = _settings()
    cal = SettlementCalendar()
    risk = RiskManager(s.risk, cal)
    return OrderValidator(s, risk), risk


def _ledger(cash: float = 1_000_000.0) -> Ledger:
    L = Ledger(cash=cash, initial_cash=cash)
    L.last_close["AAA"] = 100.0
    return L


# ----------------------------------------------------------------- Happy path
def test_validate_buy_ok():
    v, _ = _validator()
    L = _ledger()
    req = OrderRequest(ticker="AAA", action="BUY", quantity=600, date=date(2024, 5, 1))
    res = v.validate(req, L)
    assert res.ok is True


# ----------------------------------------------------------------- INVALID_TICKER
def test_invalid_ticker_rejected():
    v, _ = _validator()
    L = _ledger()
    req = OrderRequest(ticker="ZZZZ", action="BUY", quantity=1, date=date(2024, 5, 1))
    res = v.validate(req, L, universe={"AAA"})
    assert not res.ok and res.code == RejectCode.INVALID_TICKER


# ----------------------------------------------------------------- INVALID_QUANTITY
def test_invalid_quantity_rejected():
    v, _ = _validator()
    L = _ledger()
    for q in (0, -5):
        req = OrderRequest(ticker="AAA", action="BUY", quantity=q, date=date(2024, 5, 1))
        res = v.validate(req, L)
        assert not res.ok and res.code == RejectCode.INVALID_QUANTITY


# ----------------------------------------------------------------- INSUFFICIENT_CASH
def test_insufficient_cash_rejected():
    v, _ = _validator()
    L = _ledger(cash=100.0)
    req = OrderRequest(ticker="AAA", action="BUY", quantity=600, date=date(2024, 5, 1))
    res = v.validate(req, L)
    assert not res.ok and res.code == RejectCode.INSUFFICIENT_CASH


# ----------------------------------------------------------------- INSUFFICIENT_HOLDINGS
def test_insufficient_holdings_rejected():
    v, _ = _validator()
    L = _ledger()
    req = OrderRequest(ticker="AAA", action="SELL", quantity=10, date=date(2024, 5, 1))
    res = v.validate(req, L)
    assert not res.ok and res.code == RejectCode.INSUFFICIENT_HOLDINGS


# ----------------------------------------------------------------- INSUFFICIENT_MARGIN
def test_insufficient_margin_rejected():
    v, _ = _validator()
    L = _ledger(cash=1000.0)
    req = OrderRequest(ticker="AAA", action="SHORT", quantity=10, date=date(2024, 5, 1))
    res = v.validate(req, L)
    assert not res.ok and res.code == RejectCode.INSUFFICIENT_MARGIN


# ----------------------------------------------------------------- SHORT_NOT_ALLOWED
def test_short_not_allowed_rejected():
    v, _ = _validator()
    L = _ledger()
    req = OrderRequest(ticker="AAA", action="SHORT", quantity=10, date=date(2024, 5, 1))
    res = v.validate(req, L, short_allowed=False)
    assert not res.ok and res.code == RejectCode.SHORT_NOT_ALLOWED


# ----------------------------------------------------------------- DUPLICATE_ORDER
def test_duplicate_in_flight_rejected():
    v, _ = _validator()
    L = _ledger()
    req = OrderRequest(ticker="AAA", action="BUY", quantity=600, date=date(2024, 5, 1))
    v.mark_in_flight(req)
    res = v.validate(req, L)
    assert not res.ok and res.code == RejectCode.DUPLICATE_ORDER
    v.clear_in_flight(req)


# ----------------------------------------------------------------- RISK_LIMIT (name cap)
def test_name_cap_rejected():
    s = _settings()
    s.risk.max_name_weight = 0.05
    cal = SettlementCalendar()
    risk = RiskManager(s.risk, cal)
    v = OrderValidator(s, risk)
    L = _ledger()
    # 600 @ 100 = 60k = 6% of 1M -> over 5% cap
    req = OrderRequest(ticker="AAA", action="BUY", quantity=600, date=date(2024, 5, 1))
    res = v.validate(req, L)
    assert not res.ok and res.code in ("RISK_NAME_CAP", "RISK_LIMIT")


# ----------------------------------------------------------------- Min notional
def test_min_notional_rejected():
    s = _settings()
    s.strategy.min_trade_notional = 1_000_000.0
    cal = SettlementCalendar()
    risk = RiskManager(s.risk, cal)
    v = OrderValidator(s, risk)
    L = _ledger()
    req = OrderRequest(ticker="AAA", action="BUY", quantity=100, date=date(2024, 5, 1))
    res = v.validate(req, L)
    assert not res.ok
    assert res.code in (RejectCode.RISK_LIMIT,)