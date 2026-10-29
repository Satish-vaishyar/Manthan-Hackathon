"""tests/test_settlement.py — calendar + force-close guard."""
from __future__ import annotations

from datetime import date

import pytest

from bot.portfolio.settlement import SettlementCalendar


def test_calendar_counts_only_observed_days():
    cal = SettlementCalendar(interval_trading_days=10)
    # 2024-05-01 (Wed) through 2024-05-14 are 10 trading days if no holidays
    days = [date(2024, 5, d) for d in range(1, 15)]
    for d in days:
        cal.observe_trading_day(d)
    assert cal.trading_day_index(date(2024, 5, 1)) == 1
    assert cal.trading_day_index(date(2024, 5, 10)) == 10


def test_settlement_day_detection():
    cal = SettlementCalendar(interval_trading_days=10)
    days = [date(2024, 5, d) for d in range(1, 31)]
    for d in days:
        cal.observe_trading_day(d)
    cal.discover_anchor_from_event(date(2024, 5, 10))  # 10th trading day = settlement
    assert cal.is_settlement_day(date(2024, 5, 10)) is True
    assert cal.is_settlement_day(date(2024, 5, 20)) is True  # 20th = next settlement
    assert cal.is_settlement_day(date(2024, 5, 11)) is False


def test_days_to_next_settlement():
    cal = SettlementCalendar(interval_trading_days=10)
    days = [date(2024, 5, d) for d in range(1, 31)]
    for d in days:
        cal.observe_trading_day(d)
    cal.discover_anchor_from_event(date(2024, 5, 10))
    assert cal.days_to_next_settlement(date(2024, 5, 11)) == 9  # next at day 20


def test_should_force_close_short_lead_days():
    cal = SettlementCalendar(interval_trading_days=10)
    days = [date(2024, 5, d) for d in range(1, 31)]
    for d in days:
        cal.observe_trading_day(d)
    cal.discover_anchor_from_event(date(2024, 5, 10))
    # 2 days before next settlement (day 20) -> day 18
    assert cal.should_force_close_short(date(2024, 5, 18), lead_days=2) is True
    assert cal.should_force_close_short(date(2024, 5, 11), lead_days=2) is False


def test_loop_observation_does_not_break():
    cal = SettlementCalendar(interval_trading_days=10)
    # If we observe a date twice, no error
    cal.observe_trading_day(date(2024, 5, 1))
    cal.observe_trading_day(date(2024, 5, 1))
    assert cal.trading_day_index(date(2024, 5, 1)) == 1