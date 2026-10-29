"""tests/test_contracts.py — pydantic v2 schemas parse PS samples + ignore unknowns."""
from __future__ import annotations

from datetime import date, datetime

import pytest

from bot.contracts import (
    DayBundle,
    NewsFlashEvent,
    OrderRequest,
    OrderResponse,
    PortfolioResponse,
    PreflightTickEvent,
    PriceTickEvent,
    StageInfo,
)


def test_price_tick_parses():
    payload = {
        "event_type": "price_tick",
        "timestamp": "2026-10-19T10:00:00.000Z",
        "simulated_date": "2021-04-15",
        "data": {
            "ticker": "ALPHACORP",
            "open": 1440.0, "high": 1465.5, "low": 1435.2, "close": 1450.25,
            "volume": 1285000,
        },
    }
    e = PriceTickEvent.model_validate(payload)
    assert e.event_type == "price_tick"
    assert e.data.ticker == "ALPHACORP"
    assert e.data.close == 1450.25


def test_news_flash_parses():
    payload = {
        "event_type": "news_flash",
        "timestamp": "2026-10-19T10:00:00.000Z",
        "simulated_date": "2021-04-15",
        "data": {
            "ticker": "BLUX",
            "headline": "BLUX reports Q4 beat",
            "description": "details here",
            "source": "Wire",
        },
    }
    e = NewsFlashEvent.model_validate(payload)
    assert e.event_type == "news_flash"
    assert e.data.ticker == "BLUX"


def test_preflight_tick_parses():
    payload = {
        "event_type": "preflight_tick",
        "seq": 5,
        "timestamp": "2026-10-19T10:00:00.000Z",
        "simulated_date": "2021-04-15",
        "data": {},
    }
    e = PreflightTickEvent.model_validate(payload)
    assert e.seq == 5


def test_ignore_unknown_fields():
    payload = {
        "event_type": "price_tick",
        "timestamp": "2026-10-19T10:00:00.000Z",
        "simulated_date": "2021-04-15",
        "data": {"ticker": "X", "open": 1, "high": 2, "low": 1, "close": 1.5,
                 "volume": 100, "extra_field_ignored": "yes"},
    }
    e = PriceTickEvent.model_validate(payload)
    assert e.data.ticker == "X"


def test_order_request_validation():
    req = OrderRequest(ticker="AAA", action="BUY", quantity=10, date=date(2024, 5, 1))
    assert req.action == "BUY"
    assert req.quantity == 10


def test_stage_info_parses():
    s = StageInfo.model_validate({
        "stage": "development",
        "interval_seconds": 5.0,
        "total_days": 500,
        "active_dataset": "independent_test_2yr",
        "short_selling_enabled": False,
    })
    assert s.stage == "development"
    assert s.interval_seconds == 5.0


def test_portfolio_response_parses():
    p = PortfolioResponse.model_validate({
        "cash": 1000000.0,
        "equity": 1010000.0,
        "realised_pnl": 1000.0,
        "unrealised_pnl": 0.0,
        "positions": [{"ticker": "AAA", "quantity": 100, "avg_price": 100.0,
                        "current_price": 110.0, "unrealized_pnl": 1000.0}],
        "shorts": [],
        "timestamp": "2026-10-19T10:00:00.000Z",
    })
    assert p.cash == 1000000.0
    assert p.positions[0].ticker == "AAA"


def test_order_response_parses():
    o = OrderResponse.model_validate({
        "id": "ord_123",
        "ticker": "AAA", "action": "BUY", "quantity": 100,
        "price": 100.0, "brokerage": 20.0, "dp_charge": 0.0,
        "status": "FILLED",
        "date": "2024-05-01",
        "timestamp": "2026-10-19T10:00:00.000Z",
    })
    assert o.status == "FILLED"
    assert o.brokerage == 20.0